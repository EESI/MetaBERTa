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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import wandb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
from sklearn import manifold
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import completeness_score, homogeneity_score,adjusted_rand_score,adjusted_mutual_info_score

from sklearn.cluster import DBSCAN


import os 

print("Evaluating...............")

############################# PARAM #####################
with open('config.yml', 'r') as file:
     param = yaml.safe_load(file)

embedding_path = param['evaluate']['embedding_path']
model_name = param['evaluate']['model_name']

project_name ="visualize-project"
    
knn_neighbors= 3 
kernel_svm= 'linear'
taxon_class = "order"   #"phyla" #"order" #   "genus"
batch_size = 32
num_epochs = 25
over_sampling= "without_over_sampling" # "over_sampling"
dataset_source = "silva" #"its" #"28s" #"dairydb" #"silva" 









###################################### Reading dataset########################################
if dataset_source=="silva":
    
    with open(embedding_path+"/silva_train_output_embedding.npy", 'rb') as f:
            train = np.load(f)
    
    train_label= pd.read_csv("/ifs/groups/rosenMRIGrp/sr3622/datasets/silva_train_1.7m_6mer_200-fragmnets/process_data/meta_data.csv",header=None)
    train_label=train_label.rename(columns={0:'seq_id', 1:'frag_id'})
    
    with open(embedding_path+"/silva_test_output_embedding.npy", 'rb') as f:
            test = np.load(f)
            
    test_label= pd.read_csv("/ifs/groups/rosenMRIGrp/sr3622/datasets/silva_test_1.7m_6mer_200-fragmnets/process_data/meta_data.csv",header=None)
    test_label=test_label.rename(columns={0:'seq_id', 1:'frag_id'})
    
    metadata= pd.read_csv("/ifs/groups/rosenMRIGrp/sr3622/SILVA/sliva_dataset.csv",delimiter=",") 
    metadata.drop(columns=['seq', 'count'])
    
    train_label=pd.merge(metadata[["seq_id","phylum label"]], train_label, on='seq_id', how='right').dropna()
    test_label=pd.merge(metadata[["seq_id","phylum label"]], test_label, on='seq_id', how='right').dropna()
    

    train_label["phylum label"] = pd.Categorical(train_label["phylum label"])
    
    categories = train_label["phylum label"].cat.categories
    
    train_label['label'] = train_label["phylum label"].cat.codes      
    

    test_label["phylum label"] = pd.Categorical(test_label["phylum label"],categories)
    
    test_label['label'] = test_label["phylum label"].cat.codes   
    
    test = np.delete(test, test_label[test_label["label"]==-1].index,axis=0)

    test_label=test_label[test_label["label"]!=-1].reset_index(drop=True)

    X_train, X_test, y_train, y_test=train,test, train_label['label'],test_label["label"]

    num_class = max(train_label['label'])+1
    inp_sz = X_train.shape[1]
    hidden_sz = int(X_train.shape[1]/2)    
        
elif dataset_source=="dairydb":    
    
    ############################################# dairy db
    with open(embedding_path+"/output_embedding.npy", 'rb') as f:
            emb = np.load(f)

    ## dairydb       
    metadata= pd.read_csv("/ifs/groups/rosenMRIGrp/sr3622/datasets/dairy_db_6mer_200/process_data/meta_data.csv",delimiter=",", on_bad_lines='warn', header=None)     

    ## dairydb bad lines
    bad_line=[33205,33206,33207,33208,33209,33210,33211,33212,78254,78255,78256,78257,78258,78259,78260,78261]

    emb = np.delete(emb, bad_line,axis=0)

    taxon_level = {"phyla" : 3 , "order": 7 , "genus" : 11 }


    # keep clases with more than 10 samples
    ignore_phyla_less_sample=metadata.groupby(taxon_level[taxon_class]).filter(lambda x: len(x) < 10).index

    emb = np.delete(emb, ignore_phyla_less_sample,axis=0)

    metadata = metadata[metadata.groupby(taxon_level[taxon_class])[taxon_level[taxon_class]].transform('size') >= 10]

    # remove  " ' " from the name of class ( typo issue)
    metadata[taxon_level[taxon_class]]=metadata[taxon_level[taxon_class]].apply(lambda x : x[2:])

    metadata =metadata.reset_index()

    # Create dataset
    metadata[taxon_level[taxon_class]] = pd.Categorical(metadata[taxon_level[taxon_class]])

    labels =pd.Categorical(metadata[taxon_level[taxon_class]]).categories

    metadata['label'] = metadata[taxon_level[taxon_class]].cat.codes

    if over_sampling == "over_sampling":

        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(emb, metadata['label']) 
    else:
        X_res, y_res = emb, metadata['label']


    print("AFTER OVERSMAPLING",X_res.shape)

    X_train, X_test, y_train, y_test=train_test_split(X_res, y_res,test_size=0.1,random_state=42, stratify=y_res)

    print("Dataset size:" , X_train.shape )

    num_class = max(metadata['label'])+1
    inp_sz = X_train.shape[1]
    hidden_sz = int(X_train.shape[1]/2)


    df= pd.DataFrame(emb)
    df["label"]=metadata[taxon_level[taxon_class]]
    df=df.groupby("label").mean()
    
elif dataset_source=="28s":
    #
    with open(embedding_path+"/fungi_embedding.npy", 'rb') as f:
            emb = np.load(f)
            
    metadata= pd.read_csv("../../datasets/fungi_6mer_200/process_data/meta_data.csv",delimiter=",", \
                          on_bad_lines='warn', header=None)
    metadata=metadata.rename(columns={0:'seq_id', 1:'frag_id'})

    metadata_2=pd.read_csv("../../FUNGI/FungiLSU_train_1400bp_8506_mod.tax",delimiter="	",header=None)
    metadata_2=metadata_2.rename(columns={0:'seq_id', 1:'info'})
    dataPd = pd.DataFrame(np.array(list(metadata_2["info"].apply(lambda x : x.split(";")))))
    metadata_2=pd.concat([metadata_2,dataPd],axis=1)
    meta_data=pd.merge(metadata, metadata_2, on='seq_id', how='inner')
    del metadata_2
    
    if taxon_class=="phyla":
        level=2
    else :
        level=4
    
    
    ignore_phyla_less_sample=meta_data.groupby(level).filter(lambda x: len(x) < 10).index

    emb = np.delete(emb, ignore_phyla_less_sample,axis=0)
    meta_data = meta_data[meta_data.groupby(level)[level].transform('size') >= 10]
    meta_data=meta_data.reset_index()
    
    meta_data["label"] = pd.Categorical(meta_data[level])
    
    
    meta_data["label"] = meta_data["label"].cat.codes  
    
    X_res=emb
    
    y_res=meta_data["label"]
    
    del emb
    
    X_train, X_test, y_train, y_test=train_test_split(X_res, y_res,test_size=0.1,\
                                                      random_state=42, stratify=y_res)
    num_class = max(y_res)+1
    inp_sz = X_train.shape[1]
    hidden_sz = int(X_res.shape[1]/2)  
    
else: 
    
    with open(embedding_path+"/its_embedding.npy", 'rb') as f:
            emb = np.load(f)
            
    metadata= pd.read_csv("../../datasets/ITS_6mer_200/process_data/meta_data.csv",delimiter=",,",on_bad_lines='warn', header=None)
    metadata=metadata.rename(columns={0:'seq_id', 1:'frag_id'})

    metadata_2=pd.read_csv("../../FUNGI/ITS_fungi.csv",delimiter=",")
    metadata_2["seq_id"]=metadata_2["acc_id"]
    # metadata_2=metadata_2.rename(columns={0:'seq_id', 1:'info'})

    meta_data=pd.merge(metadata, metadata_2, on='seq_id', how='inner')
    
    del metadata_2
    
    if taxon_class=="phyla":
        level="2"
    else :
        level="4"
    
    
    ignore_phyla_less_sample=meta_data.groupby(level).filter(lambda x: len(x) < 10).index

    emb = np.delete(emb, ignore_phyla_less_sample,axis=0)
    meta_data = meta_data[meta_data.groupby(level)[level].transform('size') >= 10]
    
    meta_data=meta_data.reset_index()
    
    meta_data["label"] = pd.Categorical(meta_data[level])
    
    
    meta_data["label"] = meta_data["label"].cat.codes  
    
    X_res=emb
    
    y_res=meta_data["label"]
    
    del emb
    
    X_train, X_test, y_train, y_test=train_test_split(X_res, y_res,test_size=0.1,random_state=42, stratify=y_res)

    num_class = max(y_res)+1
    inp_sz = X_train.shape[1]
    hidden_sz = int(X_res.shape[1]/2)    
 

os.environ['WANDB_DISABLE_SERVICE']="true"




# del metadata
# del emb
###################################### DNN classification########################################

def DNN():
    print("DNN")

    wandb.init(project=project_name, name="DNN_" + taxon_class +model_name+dataset_source, tags=[model_name,str(over_sampling),taxon_class,model_name,dataset_source], config={"n_units": 1}, reinit=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    # Define the data
    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X[index], self.y[index]



    train_dataset = MyDataset(X_train, list(y_train))
    test_dataset = MyDataset(X_test, list(y_test))


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    def get_accuracy(logit, target, batch_size):
        ''' Obtain accuracy for training round '''
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects/batch_size
        return accuracy.item()

    # Define the model
    class MyModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MyModel, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(p=0.2)
            self.output = torch.nn.Linear(hidden_dim, output_dim)


        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.output(x)
            # x =torch.softmax(x, dim=1)
            return x

    model = MyModel(inp_sz, hidden_sz, num_class)
    model.to(device)
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0002)


    progress_bar_train = tqdm(range(num_epochs*int(len(X_train)/batch_size)))
    y_pred_train = []
    

    for epoch in range(num_epochs):
        
        train_running_loss = 0.0
        train_acc = 0.0
        model = model.train()
        
        for i,(X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # _, predicted = torch.max(y_pred.data, 1)
            # if num_epochs-1 == epoch:
            #     y_pred_train.append(predicted)
            # loss = criterion(y_pred, y_batch.long())
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(y_pred, y_batch, batch_size)
            
            progress_bar_train.update(1)
        # model.eval()    
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f'%(epoch, train_running_loss / i, train_acc/i))
        
    # y_pred= torch.cat(y_pred_train).cpu()        

    # accuracy = accuracy_score(y_train, y_pred)

    # wandb.log({"accuracy": accuracy})
    wandb.summary["accuracy_train"] = (train_acc/i)/100  

    # Evaluate the model
    y_pred=[]
    test_acc=0.0
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        i=0
        for X_batch, y_batch in test_loader:
            # print(y_batch)
            # print(type(y_batch))
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)        
            pred = model(X_batch)

            _, predicted = torch.max(pred.data, 1)
            # print(predicted)
            # # total += y_batch.size(0)
            y_pred.append(predicted)
            i=i+1
            test_acc += get_accuracy(pred, y_batch, batch_size)
    # y_pred= torch.cat(y_pred).cpu()        

    # accuracy = accuracy_score(y_test, y_pred)
    # wandb.log({"accuracy": accuracy})
    y_pred=torch.cat(y_pred).cpu()
    
    wandb.summary["accuracy"] = (test_acc/i)/100
    
    wandb.summary["macro"] = f1_score(y_test, y_pred, average='macro')
    wandb.summary["micro"] = f1_score(y_test, y_pred, average='micro')
    
 

    print("################ DNN #############")
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    
    
    if dataset_source=="silva":    

        test_dataset = MyDataset(X_test_unseen, list(y_test_unseen))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            # Evaluate the model
        y_pred=[]
        test_acc=0.0
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            i=0
            for X_batch, y_batch in test_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)        
                pred = model(X_batch)

                _, predicted = torch.max(pred.data, 1)

                # y_pred.append(predicted)
                i=i+1
                test_acc += get_accuracy(pred, y_batch, batch_size)

        # y_pred=torch.cat(y_pred).cpu()

        wandb.summary["accuracy_non_seen"] = (test_acc/i)/100
    
    
    
    
    
    

    ###################################### KNN classification########################################
def KNN():    
    wandb.init(project=project_name, name="KNN_" + taxon_class +model_name+dataset_source, tags=[model_name,str(over_sampling),taxon_class,model_name,dataset_source], config={"n_units": 2}, reinit=True)

    print("KNN")

    clf = KNeighborsClassifier(n_neighbors=knn_neighbors)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)

    # wandb.log({"accuracy": accuracy})
    wandb.summary["accuracy_train"] = accuracy  


    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)

    print("################ KNN #############")
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    # wandb.log({"accuracy": accuracy})
    wandb.summary["accuracy"] = accuracy
    wandb.summary["macro"] = f1_score(y_test, y_pred, average='macro')
    wandb.summary["micro"] = f1_score(y_test, y_pred, average='micro')    


# wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred, y_probas, labels,
#                                                          model_name='KNN', feature_names=None)

###################################### SVM classification########################################
# wandb.init(project="visualize-project", name="SVM_" + taxon_class +model_name, tags=[model_name,str(over_sampling)], config={"n_units": 3}, reinit=True)

# print("SVM")

# clf = svm.SVC(kernel=kernel_svm, random_state=0,probability=True)

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_train)
# accuracy = accuracy_score(y_train, y_pred)

# # wandb.log({"accuracy": accuracy})
# wandb.summary["accuracy_train"] = accuracy  


# y_pred = clf.predict(X_test)
# y_probas = clf.predict_proba(X_test)


# accuracy = accuracy_score(y_test, y_pred)
# # wandb.log({"accuracy": accuracy})

# wandb.summary["accuracy"] = accuracy

# print("################ SVM #############")
# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))

# wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred, y_probas, labels,
#                                                          model_name='SVC', feature_names=None)

##########################################  Cosine similarity ################################
def COS_SIM():
    wandb.init(project=project_name, name="Cosine_sim_" + taxon_class +model_name+dataset_source, tags=[model_name,str(over_sampling),taxon_class,model_name,dataset_source], config={"n_units": 4}, reinit=True)

    cos_sim=cosine_similarity(df)
    fig, ax = plt.subplots(figsize=(30, 30), dpi=80)
    ax = sns.heatmap(cos_sim, linewidth=0.5,xticklabels=df.index, yticklabels=df.index)

    fig.savefig("cosine_sim.png")
    wandb.log({"cosine_sim": wandb.Image("cosine_sim.png")})


##########################################  TSNE ################################

def TSNE():

    wandb.init(project=project_name, name="TSNE_" +taxon_class +model_name+dataset_source, tags=[model_name,str(over_sampling),taxon_class,model_name,dataset_source], config={"n_units": 5}, reinit=True)
    
    if dataset_silva:
        
        most_common =list(train_label.groupby("phylum label").count().sort_values("seq_id",ascending=False)[:10].index)
        
        sample=train_label[train_label["phylum label"].isin(most_common)]

        tsne_result = manifold.TSNE(n_components=2, n_jobs=20, perplexity=30, verbose=True,
                    learning_rate='auto', init='pca').fit_transform(train[sample.index])

        fig, ax = plt.subplots()
        fig.set_size_inches(20,20)

        
        sample=sample.reset_index(drop=True)


        for g in list(metadata["phylum label"].unique()):


            if g not in most_common  :
                continue
            s = sample[sample["phylum label"]==g].index

            ax.scatter(tsne_result[s,0], tsne_result[s,1], label=g, marker='o', alpha=1)

        plt.title("t-SNE")
        ax.legend(bbox_to_anchor=(1.1, 0.9))
        fig.savefig("tsne.png")

        wandb.log({"t-sne": wandb.Image("tsne.png")}) 
        
        
    else:    
        tsne_result = manifold.TSNE(n_components=2, n_jobs=20, perplexity=30, verbose=True,
                    learning_rate='auto', init='pca').fit_transform(emb)

        fig, ax = plt.subplots()
        fig.set_size_inches(20,20)

        most_common = list(metadata.groupby(taxon_level[taxon_class]).count().sort_values(0,ascending=False)[:5].index)

        for g in metadata[taxon_level[taxon_class]].unique():


            if g not in most_common  :
                continue
            s = metadata[metadata[taxon_level[taxon_class]]==g].index

            ax.scatter(tsne_result[s,0], tsne_result[s,1], label=g, marker='o', alpha=1)

        plt.title("t-SNE")
        ax.legend(bbox_to_anchor=(1.1, 0.9))
        fig.savefig("tsne.png")

        wandb.log({"t-sne": wandb.Image("tsne.png")})


        
def clustering():

    wandb.init(project=project_name, name="clustering_" +taxon_class +model_name+dataset_source, tags=[model_name,str(over_sampling),taxon_class,model_name,dataset_source], config={"n_units":6}, reinit=True)
    
    os.environ['OPENBLAS_NUM_THREADS'] = '3'


    # Create a clustering model
    clustering= DBSCAN(eps=1, min_samples=2)

    # Fit the model to your data
    clustering.fit(X_res)

    # Predict the clusters
    labels = clustering.labels_

    # Calculate the completeness and homogeneity scores
    completeness = completeness_score(y_res, labels)
    homogeneity = homogeneity_score(y_res, labels)
        # Calculate the completeness and homogeneity scores
    AMI = adjusted_mutual_info_score(y_res, labels)
    ARI = adjusted_rand_score(y_res, labels)

    df_min = pd.DataFrame()
    df_min["label"]=y_res
    df_min["predicted"]=labels
    
    df_min=df_min[df_min["predicted"]!=-1]
    # Print the completeness and homogeneity scores
    print("Completeness score: ", completeness)
    print("Homogeneity score: ", homogeneity)
        # Print the completeness and homogeneity scores
    print("AMI score: ", AMI)
    print("ARI score: ", ARI)
    
    # add without -1 
    # Calculate the completeness and homogeneity scores
    completeness_without_min = completeness_score(df_min["label"], df_min["predicted"])
    homogeneity_without_min = homogeneity_score(df_min["label"], df_min["predicted"])
        # Calculate the completeness and homogeneity scores
    AMI_without_min = adjusted_mutual_info_score(df_min["label"], df_min["predicted"])
    ARI_without_min = adjusted_rand_score(df_min["label"], df_min["predicted"])
    
    wandb.summary["number_of_unknown"] = len(labels)-len(df_min)
                                                 
    wandb.summary["completeness_without_mi"] = completeness_without_min
    wandb.summary["homogeneity_without_mi"] = homogeneity_without_min
    wandb.summary["AMI_without_mi"] = AMI_without_min
    wandb.summary["ARI_without_mi"] = ARI_without_min                                                 
    
    wandb.summary["completeness"] = completeness
    wandb.summary["homogeneity"] = homogeneity
    wandb.summary["AMI"] = AMI
    wandb.summary["ARI"] = ARI
    
    
    # final_meta_data = df_min[df_min.groupby("label")["label"].transform('size') >= 200]
    # sample_df=final_meta_data.groupby(level).apply(lambda x: x.sample(n=200, replace=True))
    # selected_index= [i[1] for i in sample_df.index]

    ###################
    





#########
# DNN()
KNN()
# clustering()
# TSNE()
# COS_SIM()
#########
