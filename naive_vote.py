import pandas as pd
import numpy as np
import argparse
def count_func(b):
    def func(x):
        count= 0
        for ele in x:
            if ele==b:
                count += 1
        return count
    return func


def vote(x):
    if x[0] == x[1]:
        return 1
    else:
        x = np.array(x)
        return x.argmax()


def main(file_list):
    df = pd.read_csv(file_list[0],delimiter="\t",header=None,names=['id','label'])
    df_result = pd.DataFrame()
    df_result['id'] = df['id']
    for file in file_list[1:]:
        dftmp = pd.read_csv(file,delimiter="\t",header=None,names=['id','label'])
        df = pd.merge(df, dftmp, how='inner', on='id', suffixes=('_x', '_y'))
    for i in range(2):
        df_result[i] = df[df.columns[1:]].apply(count_func(i), axis=1)
    df_result['label'] = df_result[[0,1]].apply(lambda x:vote(x),axis=1)
    df_result[['id','label']].to_csv(args.out_name, index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_list",type=str, nargs="+",required=True, help="please input the naive vote file list")
    parser.add_argument("--out_name",type=str, required=True, default="./submit_naive_vote.csv", help="please input the naive vote file list")
    args = parser.parse_args()
    file_list = args.file_list
    main(file_list)