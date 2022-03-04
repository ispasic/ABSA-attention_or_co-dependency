import torch 
from model import BertSentimentExploreAttention
import pickle
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from functions_helper import to_torch_sentiment, create_data_loader
import torch.nn.functional as F
from torch import nn
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

infile = open('data/data_train.pkl', 'rb')
data_train = pickle.load(infile)
data_train.reset_index(inplace=True)
data_train.drop(['index'], axis=1, inplace=True)

infile_validation = open('data/data_validation.pkl', 'rb')
data_val = pickle.load(infile_validation)
data_val.reset_index(inplace=True)
data_val.drop(['index'], axis=1, inplace=True)

# test data
infile_test = open('data/data_test.pkl', 'rb')
data_test = pickle.load(infile_test)
data_test.reset_index(inplace=True)
data_test.drop(['index'], axis=1, inplace=True)


data_train['torch_sentiment'] = to_torch_sentiment(data_train.sentiment)
data_val['torch_sentiment'] = to_torch_sentiment(data_val.sentiment)
data_test['torch_sentiment'] = to_torch_sentiment(data_test.sentiment)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    
    model = model.train()
    model = model.to(device)

    
    losses = []
    correct_predictions = 0
    hidden_states_ALL = []
    attention_ALL = []
    
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        token_type_ids = d['token_type_ids'].to(device)
        position_ids = d['position_ids'].to(device)
        labels = d['labels'].to(device)
        
        output, hidden_st, attention = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, output_hidden_states=True, output_attentions=True)
        
        predictions = output.logits
        
        predictions = F.softmax(predictions, dim=1)
        
        predicted_labels = torch.max(predictions, dim=1).indices
        correct_predictions += torch.sum(predicted_labels == labels) #Summing up the corectly predicted labels to calculate the metrics 

        
        loss = loss_fn(predictions, labels)
        losses.append(loss.item())
        
        hidden_states_ALL.append(hidden_st)
        attention_ALL.append(attention)


        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses), hidden_states_ALL, attention_ALL



def eval_epoch(model, data_loader, loss_fn, device, n_examples):
    
    model = model.eval()
    model = model.to(device)

    
    losses_val = []
    correct_predictions_val = 0
    hidden_states_ALL = []
    attention_ALL = []
    predicted_labels_all = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            token_type_ids = d['token_type_ids'].to(device)
            position_ids = d['position_ids'].to(device)
            labels = d['labels'].to(device)
            
            output, hidden_st, attention = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, output_hidden_states=True, output_attentions=True)
            
            predictions = output.logits
            
            predictions = F.softmax(predictions, dim=1)

            
            predicted_labels_val = torch.max(predictions, dim=1).indices
            correct_predictions_val += torch.sum(predicted_labels_val == labels)
            
            loss = loss_fn(predictions, labels)
            losses_val.append(loss.item())
            
            hidden_states_ALL.append(hidden_st)
            attention_ALL.append(attention)
            predicted_labels_all.append(predicted_labels_val)
            
        return correct_predictions_val.double() / n_examples, np.mean(losses_val), hidden_states_ALL, attention_ALL, predicted_labels_all


PRE_TRAINED_BERT_MODEL_NAME = 'bert-base-cased'
EPOCHS = 4
BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_BERT_MODEL_NAME)

train_data_loader = create_data_loader(data_train, tokenizer, BATCH_SIZE)
val_data_loader = create_data_loader(data_val, tokenizer, BATCH_SIZE)
test_data_loader = create_data_loader(data_test, tokenizer, BATCH_SIZE)

device = torch.device("cpu")

model = BertSentimentExploreAttention(PRE_TRAINED_BERT_MODEL_NAME)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss().to(device)   
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)


total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)


os.mkdir('files')
model_save_name = 'files/'+str(PRE_TRAINED_BERT_MODEL_NAME)+'_explore_attention_'+str(EPOCHS)+'epochs_bs'+str(BATCH_SIZE)+'_config.bin'


history = defaultdict(list)
best_accuracy = 0

hidden_train_ALL =[]
hidden_val_ALL = []
attention_train_ALL = []
attention_val_ALL = []

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss, hidden_train, attention_train = train_epoch(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(data_train)
  )
  
  hidden_train_ALL.append(hidden_train)
  attention_train_ALL.append(attention_train)

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss, hidden_val, attention_val, predicted_labels_evaluation = eval_epoch(
    model,
    val_data_loader,
    loss_fn, 
    device, 
    len(data_val)
  )
  
  hidden_val_ALL.append(hidden_val)
  attention_val_ALL.append(attention_val)

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), model_save_name)
    best_accuracy = val_acc

x = range(1,5)    
plt.plot(x, history['train_acc'], label='train accuracy')
plt.plot(x, history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim([0, 1]);    

# saving the attention weigth in a separate file
torch.save(attention_val_ALL, 'files/tensor_val_attention_bs16_4ep_config.pt')
torch.save(attention_train_ALL, 'files/tensor_train_attention_bs16_4ep_config.pt')



# Run the model on test dataset
model_path = 'files/'+str(PRE_TRAINED_BERT_MODEL_NAME)+'_explore_attention_'+str(EPOCHS)+'epochs_bs'+str(BATCH_SIZE)+'_config.bin'
model_test = BertSentimentExploreAttention(PRE_TRAINED_BERT_MODEL_NAME)#, dropout)
model_test.load_state_dict(torch.load(model_path))
model_test = model_test.to(device)
model_test.eval()
model_test.zero_grad()

test_acc, test_loss, hidden_test, attention_test, pred_labels_testset = eval_epoch(
    model_test,
    test_data_loader,
    loss_fn, 
    device, 
    len(data_test)
  )


a = []
for i in range(len(pred_labels_testset)):
    for j in range(len(pred_labels_testset[i])):
        a.append(pred_labels_testset[i][j])

b = data_test.torch_sentiment.to_list()

precision_recall_fscore_support(b, a, average='weighted')



torch.save(attention_test, 'files/tensor_test_attention.pt')
torch.save(hidden_test, 'files/tensor_test_hidden.pt')


'''
########## Predicting on raw text ##############

class_names = ['negative', 'positive']
raw_text = 'I am still sleepy, but the pain went away.'
raw_aspect = 'pain'

encoded_raw_text = tokenizer(raw_text, raw_aspect, padding='max_length', max_length=70, return_tensors='pt')

raw_input_ids = encoded_raw_text['input_ids'].to(device)
raw_attention_mask = encoded_raw_text['attention_mask'].to(device)
raw_token_type_ids = encoded_raw_text['token_type_ids'].to(device)
raw_position_ids = torch.arange(len(encoded_raw_text.input_ids[0]), dtype=torch.long).to(device)#, device=input_ids.device)

raw_output = model_test(input_ids=raw_input_ids, attention_mask=raw_attention_mask, token_type_ids=raw_token_type_ids, position_ids=raw_position_ids, output_attentions=False, output_hidden_states=False)
raw_output_softmax = F.softmax(raw_output[0].logits, dim=1)
raw_predicted_label = torch.max(raw_output_softmax, dim=1).indices

print(f'Review text: {raw_text}')
print(f'Aspect: {raw_aspect}')
print(f'Sentiment: {class_names[raw_predicted_label]}')

'''