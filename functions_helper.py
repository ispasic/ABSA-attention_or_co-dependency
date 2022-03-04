import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


PRE_TRAINED_BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_BERT_MODEL_NAME)

def to_torch_sentiment(rating):
    torch_sentiment = []    
    for i in rating:
        if i == -1:
            torch_sentiment.append(0)
        elif i == 1:
            torch_sentiment.append(1)
        else: 
            print('Sentiment labels -1 or 1.')
    return torch_sentiment    

class DrugReviewDataset(Dataset):
  
  def __init__(self, reviews, aspects, targets, tokenizer):
    self.reviews = reviews
    self.aspects = aspects
    self.targets = targets
    self.tokenizer = tokenizer
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    aspect = str(self.aspects[item])
    target = self.targets[item]
    encoded = tokenizer(review, aspect, padding='max_length', max_length=70, return_tensors='pt')
    position_ids = torch.arange(len(encoded.input_ids[0]), dtype=torch.long)
    encoded['position_ids'] = torch.stack([position_ids for i in range(len(encoded.input_ids))], dim=0)
    return {
      'review_text': review,
      'aspect': aspect,
      'input_ids': encoded['input_ids'].flatten(),
      'token_type_ids': encoded['token_type_ids'].flatten(),
      'attention_mask': encoded['attention_mask'].flatten(),
      'position_ids': encoded['position_ids'].flatten(),
      'labels': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, batch_size):
    ds = DrugReviewDataset(
        reviews = df.review.to_list(), 
        aspects = df.symptom.to_list(), 
        targets = df.torch_sentiment.to_numpy(), 
        tokenizer = tokenizer
        )
    
    return DataLoader(
        ds, 
        batch_size = batch_size, 
        num_workers = 0
        )