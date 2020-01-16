from transformers.modeling_bert import *
from torch import nn
import torch



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)



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