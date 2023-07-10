from transformers import DataCollatorForLanguageModeling,DataCollatorForPermutationLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM,AutoConfig,EarlyStoppingCallback,AutoModel
import numpy as np 
import KmerTokenizer
from datasets import Dataset
import datetime
import time
import yaml
from itertools import product
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from pipeline_function import insert_row,fetch_table_as_dataframe,fetch_row_by_rowid,update_or_insert_column
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rowid', required=True, help='Row ID argument')
    parser.add_argument('--table_name', type=str, help="Table name")
    parser.add_argument('--db_file_path', type=str, help="Database name")
    
    args = parser.parse_args()
    return args


def trainer():
    
    try:
    
        
        args = parse_arguments()

        # Access the parameters
        rowid = args.rowid
        table_name = args.table_name
        db_file_path= args.db_file_path

        ### update state
        update_or_insert_column(db_file_path, table_name, rowid, column_name="current_status", column_value="running")

        param = fetch_row_by_rowid(db_file_path, table_name, rowid)
        
        

        directories = [
            "./workdir/hugg_log/",
            "./workdir/model_repo/",
            "./workdir/checkpoint_repo/",
            "./workdir/evaluate_results/"
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Hugginface-Face Log
        param['hugg_log'] = directories[0]

        # Model Repository
        param['model_repo'] = directories[1]

        # Checkpoint Repository
        param['checkpoint_repo'] = directories[2]

        # Embeddings Path
        param['evaluate_path'] = directories[3]        


        ## if @wandb is available, just uncomment the following lines and any @wandb properties
        # import wandb
        # wandb.init(project="Your_project", entity="Your_name", config=param, reinit=True)

        start = time.time()

        print("################################################\n")
        print( "Start Training  ......")
        print("################################################\n")



        print(param)         


        token_list = [''.join(x) for x in product(['A', 'C', 'T', 'G'], repeat=int(param['k_mer_size']))]  
        token_list.insert(0, "[UNK]")
        token_list.insert(1, "[SEP]")
        token_list.insert(2, "[PAD]")
        token_list.insert(3, "[CLS]")
        token_list.insert(4, "[MASK]")

        with open(str(param['k_mer_size'])+"mer_vocab.txt",'w') as file:
            for i,km in enumerate(token_list):
                       file.write(km+"\n")


        # Instantiate the tokenizer
        tokenizer=KmerTokenizer.KmerTokenizer(str(param['k_mer_size'])+"mer_vocab.txt",
                                              kmer_size=int(param['k_mer_size']),
                                              overlapping=bool(param['overlapping']),
                                             model_max_lengt=int(param['vector_size']))

        # Instantiate the collator

        if param['data_collator']=="DataCollatorForPermutationLanguageModeling":

            data_collator = DataCollatorForPermutationLanguageModeling(tokenizer=tokenizer,  return_tensors="pt")


        else:

             data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, return_tensors="pt",mlm_probability=float(param['mlm_probability']))



        start = time.time()



        def tokenize(element):
            outputs = tokenizer(
                element["seq"],
                truncation=True,
                padding='max_length',
                max_length=int(param['vector_size']),
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}




    ################################################
        with open(param['fragment_path']+'/process_data/All_fragments.npy', 'rb') as f:
            seq_list = np.load(f)


        train,test=train_test_split(seq_list, test_size=float(param['test_size']), random_state=42)

        seq_list=[]
        del seq_list

        print("after train and test split")

        train = Dataset.from_list([ {"seq": seq} for i,seq in enumerate(train)]) 
        test = Dataset.from_list([ {"seq": seq} for i,seq in enumerate(test)])    




        tokenized_train_datasets = train.map(tokenize,remove_columns=["seq"],batched=True,num_proc=12).shuffle(seed=42)


        tokenized_test_datasets = test.map(tokenize,remove_columns=["seq"],batched=True,num_proc=12).shuffle(seed=42)

        train=[]
        test= []
        del train
        del test

        param["tokenizing_time"]=(time.time()-start)/60

        print("#$# Tokenizing  time {} \n".format(param["tokenizing_time"]))




        if param['model_type']=="xlnet":     
                cfg = AutoConfig.for_model(param['model_type'],
                                     vocab_size=int(4**int(param['k_mer_size'])+5),
                                     d_model=int(param['EMBED_DIM']),
                                     d_head=39 ,
                                     n_layer =int(param['NUM_LAYERS']),
                                     n_head =int(param['NUM_HEAD']),
                                     max_position_embeddings=int(param["max_position_embeddings"]),      
                                     position_embedding_type=param['position_embedding_type'],      
                                     d_inner =int(param['FF_DIM'])) 
        elif param['model_type']=="bert":
                cfg = AutoConfig.for_model(param['model_type'],
                                             vocab_size=int(4**int(param['k_mer_size'])+5),
                                             hidden_size=int(param['EMBED_DIM']),    
                                             num_hidden_layers=int(param['NUM_LAYERS']),
                                             num_attention_heads=int(param['NUM_HEAD']),
                                             max_position_embeddings=int(param["max_position_embeddings"]),
                                             position_embedding_type=param['position_embedding_type'],  
                                             intermediate_size=int(param['FF_DIM']))
        elif param['model_type']=="roberta":
                cfg = AutoConfig.for_model(param['model_type'],
                                             vocab_size=int(4**int(param['k_mer_size'])+5),
                                             hidden_size=int(param['EMBED_DIM']),    
                                             num_hidden_layers=int(param['NUM_LAYERS']),
                                             num_attention_heads=int(param['NUM_HEAD']),
                                             max_position_embeddings=int(param["max_position_embeddings"]),
                                             position_embedding_type=param['position_embedding_type'], 
                                             intermediate_size=int(param['FF_DIM']))                         
        else:
                raise ValueError("It is required to choose your model")


        start = time.time()
        folder_name = datetime.datetime.now().strftime("%Y-%B-%d_%H-%M-%S_%p")
        # hugging face checkpoints folder for each run 
        log_hug = param['hugg_log']+folder_name

        if int(param['from_pretraied']) : 

            model = AutoModelForCausalLM.from_pretrained(param['model_repo']+param['pretrained_path'])

            # model = AutoModelForCausalLM.from_pretrained("../../hugg_log/2023-April-14_22-10_PM/checkpoint-111000")

            args = TrainingArguments(
                ## if @wandb is available, just uncomment the following lines and any @wandb properties
                # report_to="wandb",
                # run_name="my_training_run",
                output_dir=log_hug,
                per_device_train_batch_size=int(param['BATCH_SIZE']),
                per_device_eval_batch_size=int(param['BATCH_SIZE']),
                num_train_epochs=int(param['epochs']),
                evaluation_strategy="steps",
                eval_steps=1_000,
                logging_steps=5_00,
                # maybe load more data and make it faster 
                dataloader_num_workers=100,
                max_grad_norm=float(param['max_grad_norm']),
                adam_epsilon=float(param['adam_epsilon']),
                gradient_accumulation_steps=int(param['gradient_accumulation_steps']),
                weight_decay=0.01,
                warmup_steps=1_000,
                lr_scheduler_type="cosine",
                learning_rate=float(param['learning_rate']),
                save_steps=3_000,
                fp16=True,
                load_best_model_at_end =True,
                push_to_hub=False)

            trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    data_collator=data_collator,
                    #note in memory map version the  following look like tokenized_datasets["train"]
                    train_dataset=tokenized_train_datasets,
                    # callbacks=[EarlyStoppingCallback()],
                    eval_dataset=tokenized_test_datasets)

            trainer.train()

            # trainer.train(resume_from_checkpoint="../../hugg_log/2023-April-14_22-10_PM/checkpoint-111000")

        else:

            model = AutoModelForCausalLM.from_config(cfg)
                
            args = TrainingArguments(
                        ## if @wandb is available, just uncomment the following lines and any @wandb properties
                        # report_to="wandb",
                        # run_name="my_training_run",
                        output_dir=log_hug,
                        per_device_train_batch_size=int(param['BATCH_SIZE']),
                        per_device_eval_batch_size=int(param['BATCH_SIZE']),
                        num_train_epochs=int(param['epochs']),
                        evaluation_strategy="steps",
                        eval_steps=1_000,
                        logging_steps=5_00,
                        # maybe load more data and make it faster 
                        dataloader_num_workers=100,
                        max_grad_norm=float(param['max_grad_norm']),
                        adam_epsilon=float(param['adam_epsilon']),
                        gradient_accumulation_steps=int(param['gradient_accumulation_steps']),
                        weight_decay=0.01,
                        warmup_steps=1_000,
                        lr_scheduler_type="cosine",
                        learning_rate=float(param['learning_rate']),
                        save_steps=3_000,
                        fp16=True,
                        load_best_model_at_end =True,
                        push_to_hub=False)

            trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    data_collator=data_collator,
                    train_dataset=tokenized_train_datasets,
                   # callbacks=[EarlyStoppingCallback()],
                    eval_dataset=tokenized_test_datasets)


            trainer.train()




        param["traning_time"]=time.time()-start

        print("#$# Traning  time {} minute \n".format(param["traning_time"]/60))


        # the folder name of trainig 


        # I changed save_model to save_state to save every thing
        # trainer.output_dir = param['workdir']['model_repo']+folder_name
        trainer.save_model(param['model_repo']+folder_name)


        ## Save last checkpoint !! 
        if len(os.listdir(log_hug))!= 0 :
            if "checkpoint" in log_hug[0]:
                list_checkpoints=os.listdir(log_hug)
                list_checkpoints=[int(i.split("-")[1]) for i in list_checkpoints]
                final_checkpoint=max(list_checkpoints)
                final_checkpoint_name="/checkpoint-{}/*".format(str(final_checkpoint))
                os.mkdir(param['checkpoint_repo']+folder_name)
                os.system("cp -r {0} {1}".format(log_hug+final_checkpoint_name,param['checkpoint_repo']+folder_name))            
                os.system("rm -rf {}/*".format(log_hug))



        with open(param['model_repo']+folder_name+'/config_file.yml', 'w') as file:
                    config = yaml.dump(param, file)
   
           
        ### update table 
        update_or_insert_column(db_file_path, table_name, rowid, column_name="traning_time", column_value=param["traning_time"])
        update_or_insert_column(db_file_path, table_name, rowid, column_name="model_name", column_value=folder_name)
        update_or_insert_column(db_file_path, table_name, rowid, column_name="hugg_log", column_value=param['hugg_log'])
        update_or_insert_column(db_file_path, table_name, rowid, column_name="model_repo", column_value=param['model_repo'])
        update_or_insert_column(db_file_path, table_name, rowid, column_name="evaluate_path", column_value=param['evaluate_path'])
        update_or_insert_column(db_file_path, table_name, rowid, column_name="checkpoint_repo", column_value=param['checkpoint_repo'])                             
        update_or_insert_column(db_file_path, table_name, rowid, column_name="current_state", column_value="embedder") 
        update_or_insert_column(db_file_path, table_name, rowid, column_name="current_status", column_value="fail") 
                
    except Exception as e:
        
        exception_type = type(e).__name__
        exception_message = str(e)
        print(f"Exception Type: {exception_type}")
        print(f"Exception Message: {exception_message}")       
        # update the database to fail status
        update_or_insert_column(db_file_path, table_name, rowid, column_name="current_status", column_value="fail")
        
        
        

if __name__ == '__main__':
    
    trainer()            