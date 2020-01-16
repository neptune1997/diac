'''
from pytorch_transformers.modeling_bert import *
from torch import nn
import torch
# #weight=torch.tensor([10.0,1.0,1.0])
# weight=torch.tensor([5.0,1.0,1.0])
# #weight=torch.tensor([1.0,1.0,1.0])

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)



class LinearBertModel(BertPreTrainedModel):
    def __init__(self,config):
        super(LinearBertModel,self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.config.hidden_size,self.config.num_labels)
        self.apply(self.init_weights)

    def forward(self, batch, feed_labels = False):
        input_ids = batch.get("input_ids")
        token_type_ids = batch.get("segment_ids")
        attention_mask = batch.get("input_mask")
        labels = batch.get("label")
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output) ## dropout at 0.5
        logits = self.classifier(pooled_output)
        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits

class SplitLinearModel(BertPreTrainedModel):
    def __init__(self,config):
        super(SplitLinearModel,self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(nn.Linear(self.config.hidden_size*2, self.config.linear_hidden_size),
                                        nn.Dropout(0.5),
                                        #nn.BatchNorm1d(self.config.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels)) ###MLP
        self.apply(self.init_weights)

    def forward(self, batch, feed_labels = False):
        title_ids, title_mask, content_ids, content_mask, labels = batch.values()
        _, title_pooled_output = self.bert(title_ids, attention_mask = title_mask)
        _, content_pooled_output = self.bert(content_ids, attention_mask=content_mask)
        pooled_output = torch.cat((title_pooled_output,content_pooled_output), dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits


class CnnBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CnnBertModel, self).__init__(config)
        self.config=config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.apply(self.init_weights)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                               out_channels=self.config.out_channels,
                                               kernel_size= (size,self.config.hidden_size),
                                               stride=1) for size in self.config.kernel_sizes])
        self.classifier = nn.Sequential(nn.Linear(self.config.out_channels*self.config.kmax*len(self.config.kernel_sizes), self.config.linear_hidden_size),
                                        nn.Dropout(0.5),
                                        #nn.BatchNorm1d(self.config.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels))

    def forward(self,batch,feed_labels=False):
        input_ids = batch.get("input_ids")
        token_type_ids = batch.get("segment_ids")
        attention_mask = batch.get("input_mask")
        labels = batch.get("label")
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        text_em = outputs[0]
        text_em = self.dropout(text_em)  ## dropout at 0.5
        text_em.unsqueeze_(dim=1)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(text_em)
            conv_out = conv_out.squeeze(dim=3)
            pooled_out = kmax_pooling(conv_out,dim=2,k=self.config.kmax)
            conv_outs.append(pooled_out)
        feats = torch.cat(conv_outs,dim=1)
        feats = feats.view(feats.size(0),-1)
        assert feats.size(0)==labels.size(0), "shape dont match"
        logits = self.classifier(feats)

        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits


class RCnnBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(RCnnBertModel, self).__init__(config)
        self.config=config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.lstm_dropout = nn.Dropout(0.5)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                               out_channels=self.config.out_channels,
                                               kernel_size= (size,self.config.hidden_size),
                                               stride=1) for size in self.config.kernel_sizes])

        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.lstm_hidden_size,
                            num_layers=config.lstm_layers, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(nn.Linear(self.config.out_channels*self.config.kmax*len(self.config.kernel_sizes)+self.config.lstm_hidden_size*2*self.config.kmax, self.config.linear_hidden_size),
                                        nn.BatchNorm1d(self.config.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels))
        self.apply(self.init_weights)


    def forward(self,batch,feed_labels=False):
        title_ids, title_mask, content_ids, content_mask, labels = batch.values()
        title_em, _ = self.bert(title_ids, attention_mask=title_mask)
        content_em, _ = self.bert(content_ids, attention_mask=content_ids)
        title_em = self.dropout(title_em)  ## dropout at 0.5
        content_em = self.dropout(content_em)
        title_em.unsqueeze_(dim=1)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(title_em)
            conv_out = conv_out.squeeze(dim=3)
            pooled_out = kmax_pooling(conv_out,dim=2,k=self.config.kmax)
            conv_outs.append(pooled_out)

        title_feats = torch.cat(conv_outs,dim=1)
        title_feats = title_feats.view(title_feats.size(0),-1)
        self.lstm.flatten_parameters()
        lstm_out, states = self.lstm(content_em)
        #lstm_out = self.lstm_dropout(lstm_out)
        pooled = kmax_pooling(lstm_out, dim=1, k=self.config.kmax)
        content_feats = pooled.view(pooled.size(0), -1)
        feats = torch.cat((content_feats,title_feats),dim=1)
        assert feats.size(0)==labels.size(0), "shape dont match"
        feats = self.lstm_dropout(feats)
        logits = self.classifier(feats)
        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits






class RnnBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(RnnBertModel, self).__init__(config)
        self.config=config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = RnnClassifier(config)
        self.apply(self.init_weights)

    def forward(self,batch,feed_labels=False):
        input_ids = batch.get("input_ids")
        token_type_ids = batch.get("segment_ids")
        attention_mask = batch.get("input_mask")
        labels = batch.get("label")
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        text_em = outputs[0]
        text_em = self.dropout(text_em)  ## dropout at 0.5
        logits = self.classifier(text_em)
        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits



class SPRnnBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SPRnnBertModel, self).__init__(config)
        self.config=config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.title_encoder = nn.LSTM(config.hidden_size,config.lstm_hidden_size,num_layers=config.lstm_layers,bidirectional=True, batch_first=True)
        self.content_encoder = nn.LSTM(config.hidden_size,config.lstm_hidden_size,num_layers=config.lstm_layers,bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(self.config.lstm_hidden_size*2*2*config.kmax, self.config.linear_hidden_size),
                                        nn.Dropout(0.5),
                                        #nn.BatchNorm1d(self.config.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels))
        self.apply(self.init_weights)

    def forward(self,batch,feed_labels=False):
        self.title_encoder.flatten_parameters()
        self.content_encoder.flatten_parameters()
        title_ids, title_mask, content_ids, content_mask, labels = batch.values()
        title_em, _ = self.bert(title_ids, attention_mask=title_mask)
        content_em, _ =self.bert(content_ids,attention_mask = content_ids)
        title_em = self.dropout(title_em)  ## dropout at 0.5
        content_em = self.dropout(content_em)
        title_lstm_hidden, _ = self.title_encoder(title_em)
        content_lstm_hidden, _ = self.content_encoder(content_em)
        title_pooled = kmax_pooling(title_lstm_hidden,dim=1,k=self.config.kmax)
        content_pooled = kmax_pooling(content_lstm_hidden,dim=1,k=self.config.kmax)
        features = torch.cat([title_pooled,content_pooled],dim=2)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        if feed_labels:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits





class FTBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(FTBertModel, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = FTClassifier(config)
        self.apply(self.init_weights)

    def forward(self,batch,feed_labels=False):
        title_ids, title_mask, content_ids, content_mask, labels = batch.values()
        title_em, title_pooler_out = self.bert(title_ids,attention_mask = title_mask)
        content_em, content_pooler_out = self.bert(content_ids,attention_mask = content_mask)
        logits = self.classifier(title_em,content_em,title_pooler_out,content_pooler_out)
        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits


class RnnClassifier(nn.Module):
    def __init__(self, config):
        super(RnnClassifier,self).__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.lstm_hidden_size, num_layers=config.lstm_layers, bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(config.lstm_dropout)
        self.fc = nn.Sequential(nn.Linear(self.config.lstm_hidden_size*2*config.kmax, self.config.linear_hidden_size),
                                        nn.Dropout(0.5),
                                        #nn.BatchNorm1d(self.config.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels))

    def forward(self, text_em):
        self.lstm.flatten_parameters()
        lstm_out, states = self.lstm(text_em)
        # hidden = states[0]
        pooled = kmax_pooling(lstm_out,dim=1,k=self.config.kmax)
        # hidden = self.dropout(hidden)
        features = pooled.view(pooled.size(0), -1)
        logits = self.fc(features)
        return logits





class FTClassifier(nn.Module):
    def __init__(self, config):
        super(FTClassifier, self).__init__()
        self.pre1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size ),
            nn.BatchNorm1d(config.hidden_size ),
            nn.ReLU(True)
        )

        self.pre2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU(True)
        )
        # self.pre_fc = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        # self.bn = nn.BatchNorm1d(opt.embedding_dim*2)
        # self.pre_fc2 = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        # self.bn2 = nn.BatchNorm1d(opt.embedding_dim*2)

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.linear_hidden_size, config.num_labels)
        )

    def forward(self, title_em, content_em,title_pooler_out, content_pooler_out):
        title_size = title_em.size()
        content_size = content_em.size()
        title_2 = self.pre1(title_em.contiguous().reshape(-1,title_size[-1])).view(title_size[0], title_size[1], -1)
        content_2 = self.pre2(content_em.contiguous().reshape(-1,content_size[-1])).view(content_size[0], content_size[1], -1)

        title_ = torch.mean(title_2, dim=1)
        content_ = torch.mean(content_2, dim=1)
        inputs = torch.cat((title_, content_,title_pooler_out,content_pooler_out), 1)
        out = self.fc(inputs)
        # content_out=self.content_fc(content.view(content.size(0),-1))
        # out=torch.cat((title_out,content_out),1)
        # out=self.fc(out)
        return out

'''


