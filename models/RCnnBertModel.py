from transformers.modeling_bert import *
from torch import nn
import torch



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)



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
                                        # nn.BatchNorm1d(self.config.hidden_size),
                                        nn.Dropout(0.5),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels))


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
