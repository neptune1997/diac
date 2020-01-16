from __future__ import absolute_import
import argparse
import logging
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from transformers.modeling_bert import *
from transformers.modeling_xlnet import *
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers import AdamW, WarmupLinearSchedule
from models import SplitLinearModel, LinearBertModel, RnnBertModel,SPRnnBertModel,CnnBertModel,RCnnBertModel, MyRnnBertModel, MLPBertModel, PoolingBertModel
from torch import nn
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler, WeightedRandomSampler, TensorDataset
from itertools import cycle
from data_utils import Corpus, split_transform, base_transform
import json

# from ipdb import launch_ipdb_on_exception,set_trace

import torch
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'LayerNorm' not in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'LayerNorm' not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



MODEL_MAP = {
'bert_SPLi':(BertConfig,SplitLinearModel,BertTokenizer,split_transform),
'bert_SPRNN':(BertConfig,SPRnnBertModel,BertTokenizer,split_transform),
'bert_RCNN':(BertConfig,RCnnBertModel,BertTokenizer,split_transform),
'bert_Li':(BertConfig,LinearBertModel,BertTokenizer,base_transform),
'bert_RNN':(BertConfig,RnnBertModel,BertTokenizer,base_transform),
'bert_MyRNN':(BertConfig,MyRnnBertModel,BertTokenizer,base_transform),
'bert_MLP':(BertConfig,MLPBertModel,BertTokenizer,base_transform),
'bert_Pool':(BertConfig,PoolingBertModel,BertTokenizer,base_transform),
'bert_CNN':(BertConfig,CnnBertModel,BertTokenizer,base_transform),
'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,base_transform),
}


def get_logger(args=None):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    return logging.getLogger(__name__)
logger=get_logger()


def set_seed(args):
    random.seed(args.seed) ##设置种子
    np.random.seed(args.seed) ##
    torch.manual_seed(args.seed) ##setting random seed for torch
    if args.n_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_model(Model, args, config):
    if os.path.isfile(args.resume_model_path) and args.to_resume_model:
        # BertForSequenceClassification.from_pretrained("./model_roberta_full/pytorch_model.bin",args,config=config)
        model = Model(config=config)
        logger.info("resuming model from {} ...".format(args.resume_model_path))
        model.load_state_dict(torch.load(args.resume_model_path))
    else:
        # model = Model.from_pretrained(args.pretrained_model_path, args,config=config)
        model = Model.from_pretrained(args.pretrained_model_path, config=config)
    return model

def add_args_to_config(args, config):
    config.linear_hidden_size = args.linear_hidden_size
    config.lstm_hidden_size = args.lstm_hidden_size
    config.lstm_layers = args.lstm_layers
    config.lstm_dropout = args.lstm_dropout
    config.kmax = args.kmax
    config.kernel_sizes =args.kernel_sizes
    config.out_channels = args.out_channels
    config.weight = args.weight
    config.output_hidden_states=args.output_hidden_states
    return config


def stacking_models(args, stacker):
    model_list = args.ensemble_models
    origin_out_dir = args.out_dir
    train_outs = []
    label_outs = []
    test_outs = []
    for idx, model in enumerate(model_list):
        k = args.folds
        args.out_dir = os.path.join(origin_out_dir, "model_{}_{}".format(model,idx))
        args.model_name = model
        args.resume_model_path = os.path.join(args.out_dir,"pytorch_model.bin")
        args.to_resume_model=False
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
        for i in range(k):
            args.data_dir = "./data/data_{}".format(i)
            dev_set_list = []
            dev_label_list = []
            test_set_list = []
            dev_set, dev_label, test_set = train(args) ##
            dev_set_list.append(dev_set)
            dev_label_list.append(dev_label)
            test_set_list.append(test_set.reshape(test_set.shape[0],test_set.shape[1],1))
            args.to_resume_model = True
        train_features = np.concatenate(dev_set_list, axis=0)
        train_labels = np.concatenate(dev_label_list, axis=0)
        test_features = np.concatenate(test_set_list, axis=2)
        test_features = test_features.mean(axis=2) ##calculate the mean score of five fold
        train_outs.append(train_features)
        label_outs.append(train_labels)
        test_outs.append(test_features)
    train_set = np.concatenate(train_outs,axis=1)
    assert all(label_outs[0]==label_outs[1])
    train_labels = label_outs[0]
    test_set = np.concatenate(test_outs,axis=1)
    ###output the middle features for better use
    train_df = pd.DataFrame(train_set)
    label_df = pd.DataFrame(train_labels)
    test_df = pd.DataFrame(test_set)
    train_df.to_csv(os.path.join(origin_out_dir,'train.csv'),index=False)
    label_df.to_csv(os.path.join(origin_out_dir,'label.csv'),index=False)
    test_df.to_csv(os.path.join(origin_out_dir,'test.csv'),index=False)
    ##train test split
    train_x, test_x,train_y,test_y = train_test_split(train_set, train_labels, test_size=0.33, shuffle=True,random_state=676)
    stacker = stacker.fit(train_x,train_y)
    pre_label = stacker.predict(test_set)
    filename = time.ctime().replace(' ', '-')
    label_filename = "label-" + filename
    label_filename = label_filename.replace(':', '-') + ".csv"
    df = pd.read_csv("./data/data_0/test.csv")
    df['label_pre']=pre_label
    df[['id', 'label_pre']].to_csv(os.path.join(origin_out_dir, label_filename), index=False)
    ###evaluate model###
    test_pre = stacker.predict(test_x)
    f1 = f1_score(test_y,test_pre )
    logger.info("the f1 score over test set is: %10f"%f1)



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    best_f1 = 0
    logger.info("the current config is :\n {}".format(str(vars(args))))
    set_seed(args)
    if args.model_name in MODEL_MAP:
        Config, Model, Tokenizer, Transform = MODEL_MAP[args.model_name]
        config = BertConfig.from_pretrained(args.pretrained_model_path, num_labels=args.num_labels)
        config = add_args_to_config(args,config) ##add customized args
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=args.do_lower_case)
        model = load_model(Model, args, config)
        model = model.to(device)
        if args.n_gpus >1:
            model = nn.DataParallel(model)
        ###adv training
        fgm = FGM(model)
        ###adv training
        transform = base_transform(tokenizer,args)
        train_data = Corpus(args, "train.csv", transform)
        dev_data = Corpus(args, 'dev.csv', transform)
        dev_sampler = SequentialSampler(dev_data)
        dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size, sampler=dev_sampler)

        # Run prediction for full data
        eval_sampler = SequentialSampler(dev_data)
        dev_loader = DataLoader(dev_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        train_sampler = RandomSampler(train_data)
        test_sampler = SubsetRandomSampler(np.random.randint(low=0,high=(len(train_data)),size=len(dev_data)))
        train_loader = DataLoader(train_data,batch_size=args.batch_size, sampler= train_sampler,drop_last=True)

        test_loader = DataLoader(train_data,batch_size=args.eval_batch_size, sampler= test_sampler)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", args.epochs)
        logger.info("  Early Stopping dev_loss = %f",args.dev_loss)
        bar = tqdm(total=len(train_loader)*args.epochs)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0 , t_total=len(bar))
        steps = 0
        total_train_loss = 0
        set_seed(args)
        for _ in range(args.epochs):
            for step, data_batch in enumerate(train_loader):
                bar.update(1)
                model.train()
                for k,v in data_batch.items():
                    data_batch[k] = v.to(device)
                loss = model(batch=data_batch,feed_labels = True)
                if args.n_gpus>1:
                    loss = loss.mean()
                loss.backward()
                ###adv training
                if steps>2600:
                    fgm.attack()
                    loss_adv = model(batch=data_batch, feed_labels=True)
                    if args.n_gpus > 1:
                        loss_adv = loss_adv.mean()
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数
                ###adv training
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
                scheduler.step()
                ##setting bar
                steps += 1
                total_train_loss += loss.item()
                bar.set_description("training loss {}".format(loss.item()))
                if (steps)%args.eval_steps==0:
                    logits, loss, dev_labels= do_inference(model, dev_loader, device)
                    test_logits, test_loss,test_labels = do_inference(model, test_loader, device)
                    inference_labels = logits.argmax(axis=1)
                    test_inference_labels = test_logits.argmax(axis=1)
                    f1 = f1_score(y_true=dev_labels, y_pred=inference_labels)
                    test_f1 = f1_score(y_true=test_labels, y_pred=test_inference_labels)
                    acc = accuracy_score(dev_labels, inference_labels)
                    logger.info("=========eval report =========")
                    logger.info("step : %s ", str(steps))
                    logger.info("average_train loss: %s" %(str(total_train_loss/steps)))
                    logger.info("subset train loss: %s" %(str(test_loss)))
                    logger.info("subset train f1 score: %s", str(test_f1))
                    logger.info("eval loss: %s",str(loss))
                    logger.info("eval acc: %s", str(acc))
                    logger.info("eval f1 score: %s",str(f1))
                    output_eval_file = os.path.join(args.out_dir, "eval_records.txt")
                    with open(output_eval_file, "a") as writer:
                        if steps==args.eval_steps:
                            writer.write("\n%s\n"%(args.memo))
                        writer.write("=========eval report =========\n")
                        writer.write("step : %s \n" %(str(steps)))
                        writer.write("average_train loss: %s\n" % (str(total_train_loss/steps)))
                        writer.write("subset train loss: %s\n" % (str(test_loss)))
                        writer.write("subset f1 score: %s\n" % (str(test_f1)))
                        writer.write("eval loss: %s\n" %(str(loss)))
                        writer.write("eval f1 score: %s\n"%( str(f1)))
                        writer.write('\n')
                    if f1 > best_f1:
                        logger.info("we get a best dev f1 %s saving model....",str(f1))
                        output_path = os.path.join(args.out_dir,"pytorch_model.bin")
                        if hasattr(model, 'module'):
                            logger.info("model has module")
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), output_path)
                        logger.info("model saved")
                        best_f1 = f1
        save_config(args)
        logger.info("args saved")
        ##load the final model
        args.to_resume_model=True
        model = load_model(Model, args, config)
        model = model.to(device)
        if args.n_gpus > 1:
            model = nn.DataParallel(model)
        dev_logits, loss, dev_labels = do_inference(model, dev_loader, device) ##do the inference for dev set
        pub_data = Corpus(args, 'test.csv', transform)
        pub_sampler = SequentialSampler(pub_data)
        pub_loader = DataLoader(pub_data, batch_size=args.eval_batch_size, sampler=pub_sampler)
        # logits, loss, dev_labels = do_inference(model, dev_loader, device)
        test_logits, _, _ = do_inference(model, pub_loader, device)
        return dev_logits, dev_labels, test_logits
    else:
        logger.info("the model %s is not registered", args.model_name)
        return








# def kfold_train(args): ##get oom error
#     k = args.folds
#     origin_data_dir = args.data_dir
#     origin_out_dir = args.out_dir
#     # if len(args.weight_list)==k: ##using weight list as weight
#     #     weight_list = args.weight_list
#     for i in range(k):
#         # args.weight = weight_list[i]
#         args.data_dir = origin_data_dir.format(i)
#         args.out_dir = os.path.join(origin_out_dir,"model_{}".format(i))
#         args.resume_model_path = os.path.join(args.out_dir,"pytorch_model.bin")
#         if not os.path.isdir(args.out_dir):
#             os.mkdir(args.out_dir)
#         logger.info("training fold %d"%i)
#         args.to_resume_model = False
#         try:
#             train(args)
#         except:
#             train(args)
#         ###get the evaluation##
#         args.to_resume_model=True ##set to resume the model
#         predict(args,is_eval=True) ##output the dev result
#         ###do the test set prediction###
#         predict(args,is_eval=False)
#         torch.cuda.empty_cache() ##empty cache for each training





def save_config(args):
    filename = time.ctime().replace(' ', '-')
    filename = "config_"+filename.replace(':', '-')+".json"
    filename = os.path.join(args.out_dir,filename)
    with open(filename,'w') as file:
        json.dump(vars(args), file)

def predict(args,is_eval=False):
    args.to_resume_model = True
    if is_eval:
        input_file_name = "dev.csv"
    else:
        input_file_name = "test.csv"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.model_name in MODEL_MAP:
        Config, Model, Tokenizer,Tansform = MODEL_MAP[args.model_name]
        config = Config.from_pretrained(args.pretrained_model_path, num_labels=args.num_labels)
        config = add_args_to_config(args,config)
        tokenizer = Tokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=args.do_lower_case)
        transform = Tansform(tokenizer,args)
        model = load_model(Model, args, config)
        model = model.to(device)
        if args.n_gpus >1:
            model = nn.DataParallel(model)
        pub_data = Corpus(args,input_file_name,transform)
        pub_sampler = SequentialSampler(pub_data)
        pub_loader = DataLoader(pub_data, batch_size=args.eval_batch_size,sampler= pub_sampler)
        logits, _ , _ = do_inference(model, pub_loader, device)
        df = pd.read_csv(os.path.join(args.data_dir, input_file_name))
        inference_label = logits.argmax(axis =1)
        df['label_pre'] = inference_label
        if is_eval:
            df['label_0'] = logits[:, 0]
            df['label_1'] = logits[:, 1]
            df[['id', 'label', 'label_pre', 'label_0', 'label_1', 'question1', 'question2']].to_csv(
                os.path.join(args.out_dir, "dev_sub.csv"), index=False)
        else:
            df['label_0'] = logits[:, 0]
            df['label_1'] = logits[:, 1]
            filename = time.ctime().replace(' ', '-')
            label_filename = "label-" + filename
            filename = filename.replace(':', '-') + ".csv"
            label_filename = label_filename.replace(':', '-') + ".csv"
            df[['id', 'label_0', 'label_1', 'question1', 'question2']].to_csv(os.path.join(args.out_dir, filename),
                                                                              index=False)
            df[['id', 'label_pre']].to_csv(os.path.join(args.out_dir, label_filename), index=False, header=False,
                                           sep='\t')


def do_inference(model, dataloader, device):
    model.eval()
    dataloader = iter(dataloader)
    eval_loss = 0
    scores_list = []
    inf_steps = 0
    steps = tqdm(list(range(len(dataloader))), total=len(dataloader))
    total_label = []
    for step in steps:
        data_batch = next(dataloader)
        total_label.append(data_batch.get("label").numpy())
        for k, v in data_batch.items():
            data_batch[k] = v.to(device)
        with torch.no_grad():  ##runing inference with in the no_grad() context
            tmp_eval_loss = model(batch = data_batch, feed_labels=True)
            logits = model(batch = data_batch)
        eval_loss += tmp_eval_loss.mean().item()
        logits = logits.detach().cpu().numpy()
        scores_list.append(logits)
        inf_steps += 1
        steps.set_description("eval loss {}:".format(eval_loss/inf_steps))
    total_label = np.concatenate(total_label,axis=0)
    scores = np.concatenate(scores_list, axis=0)##concatenate each batch scores
    return scores, eval_loss/inf_steps, total_label




def main():
    parser = argparse.ArgumentParser()
    ##required parameter
    parser.add_argument("--memo", default='running adv training with whole data drop 2000 steps ', type=str, required=False)
    parser.add_argument("--model_name", type=str, default='bert_Li', required=False)
    parser.add_argument("--data_dir", type=str, default="./data/data_0", required=False)
    parser.add_argument("--out_dir", type=str, default="./adv_Limodel_bert_test", required=False)
    parser.add_argument("--pretrained_model_path", type=str, default="./bert-base-chinese", required=False)
    parser.add_argument("--to_resume_model",type=bool,default=False, required=False)
    parser.add_argument("--resume_model_path", type=str, default="./adv_Limodel_bert_test", required=False)
    parser.add_argument("--num_labels", type=int,default=2, required=False)
    ##other parameters
    parser.add_argument("--output_hidden_states", type=bool, default=True, required=False)
    parser.add_argument("--do_kfold", action="store_true", default=False, required=False)
    parser.add_argument("--do_ensemble", action="store_true", default=False, required=False)
    parser.add_argument("--do_train", action="store_true",default=True, required=False)
    parser.add_argument("--do_eval", action="store_true",default=True, required=False)
    parser.add_argument("--test", action="store_true", default=True, required=False)
    parser.add_argument("--folds",type=int,default=5,required=False)
    parser.add_argument("--epochs", type=int, default=3, required=False)
    parser.add_argument("--weight", default=[1.0, 1.0], type=list, required=False, help='the weight for crossentropy')
    parser.add_argument("--ensemble_models",default=["bert_SPRNN", "bert_RCNN", "bert_RNN", "bert_CNN", "bert_Li"],required=False,help='the ensemble model names')
    parser.add_argument("--weight_list", default=[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]],type=list,required=False,help='weight list used in the ensemble mode')
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--eval_batch_size", default=64, type=int, required=False)
    parser.add_argument("--max_seq_len",type=int, default=150,required=False)
    parser.add_argument("--title_seq_len", type=int, default=71, required=False)
    parser.add_argument("--content_seq_len", type=int, default=71, required=False)
    parser.add_argument("--no_cuda", default=False, action="store_true", required=False)
    parser.add_argument("--log_dir", default=None, type=str, required=False)
    parser.add_argument("--dev_loss",default=0,type=float,required=False)
    parser.add_argument("--seed",default=42,type=int,required=False)
    parser.add_argument("--do_lower_case", action="store_true", default=False, required=False)
    parser.add_argument("--optimize_steps", type= int, default=20000, required=False)
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="L2 regularization.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--eval_steps", default=200, type=int,required=False,
                        help="")
    parser.add_argument("--lstm_hidden_size", default=512, type=int,
                        help="")
    parser.add_argument("--lstm_layers", default=1, type=int,
                        help="")
    parser.add_argument("--lstm_dropout", default=0.5, type=float,
                        help="")
    parser.add_argument("--linear_hidden_size", default=1024, type=float,
                        help="")
    parser.add_argument("--kernel_sizes", default=[2,3,4,5], type=list,
                        help="set the kernel sizes for cnn model")
    parser.add_argument("--out_channels", default=256, type=int,
                        help="set the out channel for cnn model")
    parser.add_argument("--kmax", default=2, type=float,
                        help="set the features from kmax")
    parser.add_argument("--meta_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list:")

    args = parser.parse_args()
    args.n_gpus = torch.cuda.device_count()
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir) #prepare output directory
    if args.do_ensemble:
        stacker = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',class_weight={0:5.0, 1:1.0, 2:1.0})
        stacking_models(args,stacker)
        return
    # if args.do_kfold:
    #     kfold_train(args)
    #     return
    if args.do_train:
        args.to_resume_model = False
        train(args)
    if args.do_eval:
        args.to_resume_model=True
        predict(args, is_eval=True)  ##output the dev set result
    if args.test:
        args.to_resume_model=True
        predict(args)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    main()
