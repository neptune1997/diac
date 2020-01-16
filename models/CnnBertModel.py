from transformers.modeling_bert import *
from torch import nn
import torch



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)



class CnnBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CnnBertModel, self).__init__(config)
        self.config=config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                               out_channels=self.config.out_channels,
                                               kernel_size= (size,self.config.hidden_size),
                                               stride=1) for size in self.config.kernel_sizes])
        self.classifier = nn.Sequential(nn.Linear(self.config.out_channels*self.config.kmax*len(self.config.kernel_sizes), self.config.linear_hidden_size),
                                        nn.Dropout(0.5),
                                        # nn.BatchNorm1d(self.config.linear_hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.config.linear_hidden_size,self.config.num_labels))
        self.init_weights()

    def forward(self,batch,feed_labels=False):
        input_ids = batch.get("input_ids")
        token_type_ids = batch.get("segment_ids")
        attention_mask = batch.get("input_mask")
        labels = batch.get("label")
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_layers_outs = outputs[2]
        assert len(outputs) == 3, "bert outputs not right"
        assert len(all_layers_outs) == 13, " the len of all_layers_outs is not right"
        cls_list = [out[:, 0, :].unsqueeze(dim=1) for out in all_layers_outs]  ##extract all the cls output
        cls_out = torch.cat(cls_list, axis=1)
        text_em = cls_out
        #text_em = self.dropout(text_em)  ## dropout at 0.5
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