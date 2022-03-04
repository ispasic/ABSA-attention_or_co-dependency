import torch
from transformers import BertForSequenceClassification

class BertSentimentExploreAttention(torch.nn.Module):
    
    def __init__(self, PRE_TRAINED_BERT_MODEL_NAME):
        super(BertSentimentExploreAttention, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(PRE_TRAINED_BERT_MODEL_NAME)
        self.config = self.bert.config
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, output_hidden_states, output_attentions):
        out= self.bert(
            input_ids=input_ids,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
            )
        
        hidden_states = out.hidden_states
        attention = out.attentions
        
        
        return out, hidden_states, attention