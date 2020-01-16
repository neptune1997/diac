import pandas as pd
import random
import argparse
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="please input the training file path")
    parser.add_argument("--output_path", type=str, required=True, help="please input the output path")
    args = parser.parse_args()
    if os.path.isfile(args.file):
        df = pd.read_csv(os.path.join(args.file))
    else:
        logger.warning("the file path not exist %s", args.file)
        return 0
    example_num = df.shape[0]
    indexs = list(range(example_num))
    logger.info(str(example_num))
    dev_indexs = random.sample(indexs, int(1/5 * example_num))
    train_indexs = list(set(indexs) - set(dev_indexs))
    if os.path.isdir(args.output_path):
        out_dir = args.output_path
    else:
        os.mkdir(args.output_path)
        out_dir = args.output_path
    df.iloc[dev_indexs,:].to_csv(os.path.join(out_dir, "dev.csv"), index=False)
    df.iloc[train_indexs,:].to_csv(os.path.join(out_dir, "train.csv"), index=False)
if __name__=="__main__":
    main()
