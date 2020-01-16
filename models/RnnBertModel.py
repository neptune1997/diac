from pytorch_transformers.modeling_bert import *
from torch import nn
import torch

# from ipdb import launch_ipdb_on_exception,set_trace


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)



class RnnBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(RnnBertModel, self).__init__(config)
        self.config=config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = RnnClassifier(config)
        self.apply(self.init_weights)

    def forward(self,batch,feed_labels=False):
        input_ids = batch.get("input_ids")
        token_type_ids = batch.get("segment_ids")
        attention_mask = batch.get("input_mask")
        labels = batch.get("label")
        # set_trace()
        # logger.info("input_ids %s"%' '.join(map(str,input_ids)))
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask,token_type_ids = token_type_ids)
        # text_em = outputs[0]
        text_em = outputs[1]
        text_em = self.dropout(text_em)  ## dropout at 0.5
        logits = self.classifier(text_em)
        if feed_labels:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits


    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



# class RnnClassifier(nn.Module):
#     def __init__(self, config):
#         super(RnnClassifier,self).__init__()
#         self.config = config
#         self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.lstm_hidden_size, num_layers=config.lstm_layers, bidirectional=True,batch_first=True)
#         self.dropout = nn.Dropout(config.lstm_dropout)
#         self.fc = nn.Sequential(nn.Linear(self.config.lstm_hidden_size*2, self.config.linear_hidden_size),
#                                         nn.Dropout(0.5),
#                                         #nn.BatchNorm1d(self.config.hidden_size),
#                                         nn.ReLU(inplace=True),
#                                         nn.Linear(self.config.linear_hidden_size,self.config.num_labels))
#
#     def forward(self, text_em):
#         pooled_out = kmax_pooling(text_em, dim=1, k=self.config.kmax)
#         self.lstm.flatten_parameters()
#         lstm_out, states = self.lstm(pooled_out)
#         hidden = states[0]
#
#         # hidden = self.dropout(hidden)
#         features = hidden.view(hidden.size(1), -1)
#         logits = self.fc(features)
#         return logits


class RnnClassifier(nn.Module):
    def __init__(self, config):
        super(RnnClassifier,self).__init__()
        self.dropout = nn.Dropout(config.lstm_dropout)
        self.config = config
        self.classifier = nn.Sequential(nn.Linear(self.config.lstm_hidden_size*2, 1024),
                                        nn.Dropout(0.5),
                                        #nn.BatchNorm1d(self.config.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024,self.config.num_labels))
        self.W = []
        self.gru = []
        for i in range(config.lstm_layers):
            self.W.append(nn.Linear(config.lstm_hidden_size * 2, config.lstm_hidden_size * 2))
            self.gru.append(
                nn.GRU(config.hidden_size if i == 0 else config.lstm_hidden_size * 4, config.lstm_hidden_size,
                       num_layers=1, bidirectional=True, batch_first=True).cuda())
        self.W = nn.ModuleList(self.W)
        self.gru = nn.ModuleList(self.gru)

    def forward(self, text_em):
        text_em = text_em.unsqueeze(axis=1)
        # logger.info("text_em shape %s\n" % ' '.join(map(str,text_em)))
        output = text_em.reshape(text_em.size(0), text_em.size(1), -1).contiguous()
        # logger.info("text_em %s\n" % str(text_em))
        for w, gru in zip(self.W, self.gru):
            try:
                gru.flatten_parameters()
            except:
                pass
            output, hidden = gru(output)
            output = self.dropout(output)
        hidden = hidden.permute(1, 0, 2).reshape(text_em.size(0), -1).contiguous()
        # hidden=output.mean(1)
        # hidden=nn.functional.tanh(self.pooling(hidden))
        # hidden=self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits