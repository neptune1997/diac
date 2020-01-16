from transformers.modeling_bert import *
from torch import nn
import torch
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.lstm_dropout)
        self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.lstm_hidden_size*2, self.config.num_labels)

        self.W=[]
        self.gru=[]
        for i in range(config.lstm_layers):
            self.W.append(nn.Linear(config.lstm_hidden_size*2, config.lstm_hidden_size*2))
            self.gru.append(nn.GRU(config.hidden_size if i==0 else config.lstm_hidden_size*4, config.lstm_hidden_size,num_layers=1,bidirectional=True,batch_first=True).cuda() )
        self.W=nn.ModuleList(self.W)
        self.gru=nn.ModuleList(self.gru)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None


        outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]


        output = pooled_output.reshape(input_ids.size(0),input_ids.size(1),-1).contiguous()


        for w,gru in zip(self.W,self.gru):
            try:
              gru.flatten_parameters()
            except:
              pass
            output, hidden = gru(output)
            output = self.dropout(output)




        hidden=hidden.permute(1,0,2).reshape(input_ids.size(0),-1).contiguous()
        #hidden=output.mean(1)
        #hidden=nn.functional.tanh(self.pooling(hidden))
        #hidden=self.dropout(hidden)
        logits = self.classifier(hidden)



        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = loss
        else:
            outputs = nn.functional.softmax(logits,-1)
        return outputs  # (loss), logits, (hidden_states), (attentions)
