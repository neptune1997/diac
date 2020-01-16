from transformers.modeling_bert import *
from torch import nn
import torch



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)



class PoolingBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(PoolingBertModel, self).__init__(config)
        self.config=config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(nn.Linear(self.config.kmax*self.config.hidden_size+self.config.hidden_size, self.config.linear_hidden_size),
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
        pooler_out = outputs[1]
        all_layers_outs = outputs[2]
        assert len(outputs)==3,"bert outputs not right"
        assert len(all_layers_outs)==13," the len of all_layers_outs is not right"
        cls_list = [ out[:,0,:].unsqueeze(dim=1) for out in all_layers_outs[1:12] ] ##extract all the cls output
        cls = torch.cat(cls_list,axis=1)
        # logger.info("shape of cls %s"%str(cls.shape))
        feats = kmax_pooling(cls,dim=1,k=self.config.kmax)
        # logger.info("pooling out shape %s"%str(feats.shape))
        feats = feats.reshape(feats.size(0), -1)
        feats = torch.cat((pooler_out,feats),axis=1)
        assert feats.size(0)==labels.size(0), "shape dont match"
        logits = self.classifier(feats)
        if feed_labels:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.config.weight).cuda())
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss
        else:
            logits = nn.functional.softmax(logits, -1)
            return logits