from transformers.modeling_bert import *
from torch import nn
import torch


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
