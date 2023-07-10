import pandas as pd
import yaml
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import cross_validate
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
from sklearn import manifold
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score,adjusted_rand_score,adjusted_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from sklearn.cluster import DBSCAN
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



import os 

print("Evaluating...............")



############################# PARAM #####################
args = parse_arguments()

rowid = args.rowid

slurmid = args.slurmid

table_name = args.table_name

db_file_path= args.db_file_path


update_or_insert_column(db_file_path, table_name, rowid, column_name="current_status", column_value="running")

param = fetch_row_by_rowid(db_file_path, table_name, rowid)


                
combinations = [
    ("phylum", "s28s"),
    ("phylum", "its"),
    ("phylum", "dairydb"),
    ("order", "s28s"),
    ("order", "its"),
    ("order", "dairydb"),
    ("family", "s28s"),
    ("family", "its"),
    ("genus", "dairydb")
]                
                

data_source = combinations[int(slurmid)][1]

taxon_class = combinations[int(slurmid)][0]
        

embedding_path = param['evaluate_path']+param['model_name']
model_name = param['model_name']


# import wandb                
project_name ="visualize-project"
    
knn_neighbors= 3           

RANDSEED = 42

# k-fold cross validation                
num_folds = 10




update_or_insert_column(db_file_path, table_name, rowid, column_name="current_status", column_value="running")





###################################### Reading dataset########################################


with open(embedding_path+"/"+data_source+"_embedding.npy", 'rb') as f:
            emb = np.load(f)

dataset_csv = "../../../downstream_dataset/"+data_source+"/process_data/200_fragmented_"+data_source+".csv"

# Load the dataset into a pandas DataFrame
data = pd.read_csv(dataset_csv)  

ignore_phyla_less_sample=data.groupby(taxon_class).filter(lambda x: len(x) < 10).index

emb = np.delete(emb, ignore_phyla_less_sample,axis=0)

data = data[data.groupby(taxon_class)[taxon_class].transform('size') >= 10]

data =data.reset_index(drop=True)

data["label"] = pd.Categorical(data[taxon_class])


data["label"] = data["label"].cat.codes  


X=emb

y=data["label"]



num_class = max(data['label'])+1
inp_sz = X.shape[1]
hidden_sz = int(inp_sz/2)  



 

del data



# os.environ['WANDB_DISABLE_SERVICE']="true"


    

    ###################################### KNN classification########################################
def KNN(X_train, X_test,y_train, y_test):    
    print("KNN")

    clf = KNeighborsClassifier(n_neighbors=knn_neighbors)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred)


    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)

    print("################ KNN #############")
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, accuracy_train,f1_score(y_test, y_pred, average='macro'),f1_score(y_test, y_pred, average='micro')


    ##################################### RF classification########################################
def RF(X_train, X_test,y_train, y_test):    
    print("RF")
    
#     class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
    
    # class_weights = dict(zip(np.unique(y_train), class_weights))
    
    # class_weight=class_weights
    
    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred)


    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)

    print("################ RF #############")
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, accuracy_train,f1_score(y_test, y_pred, average='macro'),f1_score(y_test, y_pred, average='micro')






    

if __name__ == '__main__':

    try :



        # Create the cross-validation object
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)



        acc_test_rf_list = []
        acc_test_rf_list_train = []
        micro_test_rf_list = []
        macro_test_rf_list = []        
        
        acc_test_knn_list = []
        acc_test_knn_list_train = []
        micro_test_knn_list = []
        macro_test_knn_list = []


            
            
        accuracy_test_RF,accuracy_train_RF,macro_RF,micro_RF= RF(X_train, X_test,y_train, y_test)




        accuracy_test_KNN,accuracy_train_KNN,macro_KNN,micro_KNN= KNN(X_train, X_test,y_train, y_test)


#                 # Append accuracy for RF and KNN classifiers

        acc_test_knn_list.append(accuracy_test_KNN)

        acc_test_rf_list.append(accuracy_test_RF)



        # Append micro-average and macro-average F1-score for KNN classifier
        micro_test_knn_list.append(micro_KNN)
        macro_test_knn_list.append(macro_KNN)


        # Append micro-average and macro-average F1-score for RF classifier
        micro_test_rf_list.append(micro_RF)
        macro_test_rf_list.append(macro_RF)


#                    # Append accuracy for RF and KNN classifiers
        acc_test_knn_list_train.append(accuracy_train_KNN)
        acc_test_rf_list_train.append(accuracy_train_RF)
            
            
            

        # os.environ['WANDB_DISABLE_SERVICE']="true"

        
        
        # wandb.init(project=project_name, name="RF_" + taxon_class +model_name+data_source, tags=[model_name,taxon_class,model_name,data_source], config={"n_units": 1}, reinit=True)
        


        accuracy_train = np.mean(acc_test_rf_list_train)
        accuracy_test = np.mean(acc_test_rf_list)

        micro_test = np.mean(micro_test_rf_list)

        macro_test = np.mean(macro_test_rf_list)

#         wandb.summary["accuracy_train_mean"] = accuracy_train  
#         wandb.summary["accuracy_mean"] = accuracy_test

#         wandb.summary["micro_mean"] = micro_test

#         wandb.summary["macro_mean"] = macro_test

        accuracy_train = np.std(acc_test_rf_list_train)
        accuracy_test = np.std(acc_test_rf_list)

        micro_test = np.std(micro_test_rf_list)
        macro_test = np.std(macro_test_rf_list)

#         wandb.summary["accuracy_train_std"] = accuracy_train  
#         wandb.summary["accuracy_std"] = accuracy_test

#         wandb.summary["micro_std"] = micro_test

#         wandb.summary["macro_std"] = macro_test
        
        


        # wandb.init(project=project_name, name="KNN_" + taxon_class +model_name+data_source, tags=[model_name,taxon_class,model_name,data_source], config={"n_units": 2}, reinit=True)
        



        accuracy_train = np.mean(acc_test_knn_list_train)
        accuracy_test = np.mean(acc_test_knn_list)

        micro_test = np.mean(micro_test_knn_list)

        macro_test = np.mean(macro_test_knn_list)

#         wandb.summary["accuracy_train_mean"] = accuracy_train  
#         wandb.summary["accuracy_mean"] = accuracy_test

#         wandb.summary["micro_mean"] = micro_test

#         wandb.summary["macro_mean"] = macro_test

        accuracy_train = np.std(acc_test_knn_list_train)
        accuracy_test = np.std(acc_test_knn_list)

        micro_test = np.std(micro_test_knn_list)
        macro_test = np.std(macro_test_knn_list)

#         wandb.summary["accuracy_train_std"] = accuracy_train  
#         wandb.summary["accuracy_std"] = accuracy_test

#         wandb.summary["micro_std"] = micro_test

#         wandb.summary["macro_std"] = macro_test        

    except Exception as e:

        exception_type = type(e).__name__
        exception_message = str(e)
        print(f"Exception Type: {exception_type}")
        print(f"Exception Message: {exception_message}")       
        update_or_insert_column(db_file_path, table_name, rowid,
                                column_name="current_status", column_value="fail")  
