import pandas as pd
import numpy as np
import argparse
def main(file_list, out_path):
    df = pd.read_csv(file_list[0])
    df_result = pd.DataFrame()
    df_result['id'] = df['id']
    for file in file_list[1:]:
        dftmp = pd.read_csv(file)
        df = pd.merge(df, dftmp, how='inner', on='id', suffixes=('_x', '_y'))
    score_matrix = np.array(df[df.columns[1:]])
    result_matrix = np.zeros(score_matrix.shape)
    for i in range(score_matrix.shape[1]):
        label = i%3
        result_matrix[:,label] = result_matrix[:,label]+score_matrix[:,i]
    labels = result_matrix.argmax(axis=1)
    df_result['label'] = labels
    df_result[['id','label']].to_csv(out_path, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_list",type=str, nargs="+",required=True, help="please input the naive vote file list")
    parser.add_argument("--out_path",type=str,default="./submit_average_ensemble.csv",required=False)
    args = parser.parse_args()
    file_list = args.file_list
    out_path = args.out_path
    main(file_list, out_path)