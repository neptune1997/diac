import logging
import os
from torch import nn
import pandas as pd
import numpy as np
import torch
import time
from torch.utils.data import Dataset
from itertools import cycle



def get_logger(args=None):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    return logging.getLogger(__name__)
logger=get_logger()


class Sample(object):
    def __init__(self,id,question1,question2,label):
        self.id = id
        self.question1 = question1
        self.question2 = question2
        self.label = label

class Corpus(Dataset):
    def __init__(self, args, file_name, transform=None):
        super(Corpus,self).__init__()
        if os.path.isfile(os.path.join(args.data_dir,file_name)):
            self.df = pd.read_csv(os.path.join(args.data_dir, file_name))
            if not 'label' in self.df.columns: ##fill the public_set
                self.df['label'] = [0] * self.df.shape[0]
        else:
            print("the %s does not exist "%(os.path.join(args.data_dir,file_name)))
            self.df = None
        self.transform = transform
        logger.info("columns of the dataframe %s",str(self.df.columns))


    def __len__(self, ):
        if isinstance(self.df,pd.DataFrame):
            return self.df.shape[0]
        else:
            return 0


    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        sample = Sample(row['id'], row['question1'], row["question2"],row['label'])
        if self.transform:
            sample = self.transform(sample)
        return sample


    def get_feature(self, feature_name):
        n_sample = len(self)
        features = []
        for i in range(n_sample):
            features.append(self[i][feature_name])
        return features

class split_transform(object):
    def __init__(self,tokenizer,args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, sample):
        id = sample.id
        if type(sample.question1) == float:
            logger.info("the error type of text_a {}".format(sample.question1))
            sample.question1 = ""
        if type(sample.question2) == float:
            logger.info("the error type of id ".format(sample.id))
            logger.info("the error type of text_a {}".format(sample.question1))
            logger.info("the error type of text_b {}".format(sample.question2))
            sample.question2 = ""
        if sample.question2=="" and sample.question1 !="":
            sample.question2 = sample.question1
        if sample.question1 =="" and sample.question1 !="":
            sample.question1 = sample.question2
        question1_tokens = self.tokenizer.tokenize(sample.question1)
        question2_tokens = self.tokenizer.tokenize(sample.question2)
        label = sample.label
        if len(question1_tokens) > self.args.question1_seq_len -1 :
            question1_tokens = question1_tokens[:self.args.question1_seq_len-1]
        if len(question2_tokens) > self.args.question2_seq_len -1:
            question2_tokens = question2_tokens[:self.args.question2_seq_len -1]
        question1_tokens += ['CLS']
        question2_tokens += ['CLS']
        question1_ids = self.tokenizer.convert_tokens_to_ids(question1_tokens)
        question2_ids = self.tokenizer.convert_tokens_to_ids(question2_tokens)
        question1_mask = [1]*(len(question1_ids))
        question2_mask = [1]*(len(question2_ids))
        question1_padding_len = self.args.question1_seq_len - len(question1_ids)
        question2_padding_len = self.args.question2_seq_len - len(question2_ids)
        question1_ids += [0]*question1_padding_len
        question2_ids += [0]*question2_padding_len
        question1_mask += [0]*question1_padding_len
        question2_mask += [0]*question2_padding_len
        label = int(sample.label)
        question1_ids = torch.tensor(question1_ids)
        question1_mask = torch.tensor(question1_mask)
        question2_ids = torch.tensor(question2_ids)
        question2_mask = torch.tensor(question2_mask)
        label = torch.tensor(label, dtype=torch.long)
        # input_ids = torch.tensor(input_ids).reshape(1, -1)
        # segment_ids = torch.tensor(segment_ids).reshape(1, -1)
        # input_mask = torch.tensor(input_mask).reshape(1, -1)
        # label = torch.tensor(label, dtype=torch.long)
        assert question1_ids.shape[0] == question1_mask.shape[0]
        assert  question2_ids.shape[0] == question2_mask.shape[0], "the length goes wrong{}\n{}\n{}\n".format(question2_ids.shape, question2_mask, question1_ids.shape,question1_mask.shape)
        return {"title_ids":question1_ids,
                "title_mask":question1_mask,
                "content_ids": question2_ids,
                "content_mask":question2_mask,
                "label": label}


class base_transform(object):
    def __init__(self,tokenizer,args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, sample):
        id = sample.id
        if type(sample.question1) == float:
            logger.info("the error type of text_a {}".format(sample.question1))
            sample.question1 = ""
        if type(sample.question2) == float:
            logger.info("the error type of id {}".format(sample.id))
            logger.info("the error type of text_a {}".format(sample.question1))
            logger.info("the error type of text_b {}".format(sample.question2))
            sample.question2 = ""
        question1_tokens = self.tokenizer.tokenize(sample.question1)
        question2_tokens = self.tokenizer.tokenize(sample.question2)
        label = sample.label
        if len(question1_tokens)+len(question2_tokens) > self.args.max_seq_len -3:
            if len(question1_tokens) > self.args.max_seq_len -3 :
                question1_tokens = question1_tokens[:self.args.max_seq_len-3]
                question2_tokens = []
            else:
                question2_tokens = question2_tokens[:self.args.max_seq_len-3-len(question1_tokens)] ##truncate question2
        tokens = ["[CLS]"] + question1_tokens + ["[SEP]"] + question2_tokens+ ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0]*(len(question1_tokens)+2)+[1]*(len(question2_tokens)+1)
        input_mask = [1]*(len(input_ids))
        padding_len = self.args.max_seq_len - len(input_ids)
        input_ids += [0]*padding_len
        segment_ids += [0]*padding_len
        input_mask += [0]*padding_len
        input_ids = torch.tensor(input_ids)
        segment_ids = torch.tensor(segment_ids)
        input_mask = torch.tensor(input_mask)
        label = torch.tensor(label, dtype=torch.long)
        # input_ids = torch.tensor(input_ids).view(1,-1)
        # segment_ids = torch.tensor(segment_ids).view(1,-1)
        # input_mask = torch.tensor(input_mask).view(1,-1)
        # label = torch.tensor(label, dtype=torch.long).view(1,-1)
        # assert input_ids.shape[1]==input_mask.shape[1]==segment_ids.shape[1]==self.args.max_seq_len, "the length goes wrong{}\n{}\n{}\n".format(input_ids.shape[1], input_mask.shape[1], segment_ids.shape[1])
        return {"input_ids":input_ids, "segment_ids":segment_ids, "input_mask":input_mask,"label":label}

class no_loss_transform(object):
    def init(self,args):
        self.args = args


    def __call__(self, index,):
        pass

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
