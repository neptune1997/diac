import pandas as pd
import os
import random




train_df=pd.read_csv("./train_ten.csv")
# dev_df = pd.read_csv("./data_second/dev.csv")
# train_df = pd.concat([train_df,dev_df],ignore_index=True)
test_df=pd.read_csv("./nlp_new_train0/test.csv")
train_df['label']=train_df['label'].astype(int)
# test_df['label']=0

# test_df['content']=test_df['content'].fillna('')
# train_df['content']=train_df['content'].fillna('')
# test_df['title']=test_df['title'].fillna('')
# train_df['title']=train_df['title'].fillna('')

train_df.dropna(axis=0,how='any')
index=set(range(train_df.shape[0]))
K_fold=[]
for i in range(5):
    if i == 4:
        tmp=index
    else:
        tmp=random.sample(index,int(1.0/5*train_df.shape[0]))
    index=index-set(tmp)
    print("Number:",len(tmp))
    K_fold.append(tmp)
    

for i in range(5):
    print("Fold",i)
    os.system("mkdir data_{}".format(i))
    dev_index=list(K_fold[i])
    train_index=[]
    for j in range(5):
        if j!=i:
            train_index+=K_fold[j]
    train_df[['id','question1','question2','label']].iloc[train_index].to_csv("data_{}/train.csv".format(i), index=False)
    train_df[['id','question1','question2','label']].iloc[dev_index].to_csv("data_{}/dev.csv".format(i), index=False)
    test_df.to_csv("data_{}/test.csv".format(i), index=False)
