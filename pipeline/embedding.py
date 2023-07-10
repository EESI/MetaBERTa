from transformers import AutoModelForCausalLM,AutoConfig,Trainer,BertConfig
from transformers import DataCollatorForLanguageModeling,DataCollatorForPermutationLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM,AutoConfig
import numpy as np 
from datasets import Dataset
from  KmerTokenizer import KmerTokenizer
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import yaml
from numpy import zeros, newaxis
import os
from pipeline_function import insert_row,fetch_table_as_dataframe,fetch_row_by_rowid,update_or_insert_column
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rowid', required=True, help='Row ID argument')
    parser.add_argument('--slurmid', type=str, help="Slurm ID")
    parser.add_argument('--table_name', type=str, help="Table name")
    parser.add_argument('--db_file_path', type=str, help="Database name")
    # Add more parameters as needed

    args = parser.parse_args()
    return args


def embedder():
    
    try:

        args = parse_arguments()

        rowid = args.rowid

        slurmid = args.slurmid

        table_name = args.table_name
        
        db_file_path= args.db_file_path

        update_or_insert_column(db_file_path, table_name, rowid, column_name="current_status", column_value="running")

        param = fetch_row_by_rowid(db_file_path, table_name, rowid)
                
        

        dataset_source = ["dairydb","its","s28s"]

        dt_name = dataset_source[int(slurmid)]

        output_name= param['evaluate_path']+param['model_name']+"/" + dt_name+"_embedding.npy"
        
        # I add this if it is already generated skip the process.
        if os.path.exists(output_name):
            return 0
        


        with open(param[dt_name]+'/process_data/All_fragments.npy', 'rb') as f:
                    seq_list = np.load(f)


        dataset = Dataset.from_list([ {"seq": seq} for i,seq in enumerate(seq_list)])    


        tokenizer=KmerTokenizer(str(param['k_mer_size'])+"mer_vocab.txt",
                                                  kmer_size=int(param['k_mer_size']),
                                                  overlapping=bool(param['overlapping']),
                                                 model_max_lengt=int(param['vector_size']))

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




        tokenized_datasets = dataset.map(tokenize, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns("seq")





        model = AutoModelForCausalLM.from_pretrained(param['model_repo']+param['model_name'])

        trainer = Trainer(model = model)


        class HFDataset(Dataset):
            def __init__(self, dset):
                self.dset = dset

            def __getitem__(self, idx):
                return self.dset[idx]

            def __len__(self):
                return len(self.dset)

        train_ds = HFDataset(tokenized_datasets)



        def get_features(name):
            def hook(model, input, output):
                features[name] = output
            return hook


        if param['model_type']=="xlnet":    
                model.transformer.layer[-2].register_forward_hook(get_features('feats'))


        elif param['model_type']=="bert":     
            model.bert.encoder.layer[-2].register_forward_hook(get_features('feats'))


        elif param['model_type']=="roberta":     
            model.roberta.encoder.layer[-2].register_forward_hook(get_features('feats'))

        else : 
            raise ValueError("It is required to choose your model")



        print(len(tokenized_datasets),"Number of datasets")

        data_loader = DataLoader(torch.IntTensor([tokenized_datasets[i]['input_ids']  for i in range(len(tokenized_datasets))]), 
                                 batch_size  = 8, 
                                 shuffle     = False, 
                                 num_workers = 5)


        # placeholders
        PREDS = []
        FEATS = []

        # placeholder for batch features
        features = {}
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        # loop through batches
        for idx, inputs in enumerate(data_loader):
            # move to device
            inputs = inputs.to(device)

            # forward pass [with feature extraction]
            preds = model(inputs)


            FEATS.append(np.mean(features['feats'][0].detach().cpu().numpy(),axis=1)[newaxis,...])



        x=np.concatenate(FEATS,axis=1)[0]

        print(x.shape)
        os.makedirs(param['evaluate_path']+param['model_name'], exist_ok=True)

        np.save(output_name, x)    

        print("Finished!!!")
        
    except Exception as e:
        
        exception_type = type(e).__name__
        exception_message = str(e)
        print(f"Exception Type: {exception_type}")
        print(f"Exception Message: {exception_message}")       
        update_or_insert_column(db_file_path, table_name, rowid, column_name="current_status", column_value="fail")        
        
if __name__ == '__main__':
    
    embedder()
