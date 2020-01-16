from transformers.modeling_bert import *
from torch import nn
import torch




# class LinearBertModel(BertPreTrainedModel):
#     def __init__(self,config):
#         super(LinearBertModel,self).__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(0.2)
#         self.classifier = nn.Linear(self.config.hidden_size,self.config.num_labels)
#
#
#     def forward(self, batch, feed_labels = False):
#         input_ids = batch.get("input_ids")
#         token_type_ids = batch.get("segment_ids")
#         attention_mask = batch.get("input_mask")
#         labels = batch.get("label")
#         outputs = self.bert(input_ids, attention_mask,token_type_ids )
#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output) ## dropout at 0.5
#         logits = self.classifier(pooled_output)
#         if feed_labels:
#             loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
#             loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
#             return loss
#         else:
#             logits = nn.functional.softmax(logits, -1)
#             return logits
class LinearBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(LinearBertModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, batch, feed_labels = False):
        input_ids = batch.get("input_ids")
        token_type_ids = batch.get("segment_ids")
        attention_mask = batch.get("input_mask")
        labels = batch.get("label")
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        outputs = logits
        if feed_labels:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs
            outputs = loss
        return outputs  # (loss), logits, (hidden_states), (attentions)
