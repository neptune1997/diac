from transformers.modeling_bert import *
from torch import nn
import torch


class MLPBertModel(BertPreTrainedModel):
    def __init__(self,config):
        super(MLPBertModel,self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.linear_hidden_size),
                                        nn.Dropout(0.5),
                                        # nn.BatchNorm1d(self.config.linear_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels))




    def forward(self, batch, feed_labels = False):
        input_ids = batch.get("input_ids")
        token_type_ids = batch.get("segment_ids")
        attention_mask = batch.get("input_mask")
        labels = batch.get("label")
        text_em, _ = self.bert(input_ids, attention_mask,token_type_ids )
        pooled_output = text_em[:,0,:]
        # pooled_output = self.dropout(pooled_output) ## dropout at 0.5
        logits = self.classifier(pooled_output)
        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits
