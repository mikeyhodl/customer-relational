import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import random
import json

f = open("static/data/intents.json")
data = json.load(f)

# specify GPU
device = torch.device("cpu")

dt = []
for i in range(0, 6):
    for txt in range(len(data['intents'][i]['text'])):
            dt.append([data['intents'][i]['text'][txt], data['intents'][i]['intent']])

df = pd.DataFrame(dt, columns=['Text', 'intent'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['intent'] = le.fit_transform(df['intent'])

df['intent'].value_counts(normalize = True)

Xtrain, ytrain = df['Text'], df['intent']



from transformers import AutoModel, BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

bert = AutoModel.from_pretrained('distilbert-base-uncased')

max_seq_len = 12

tokens_train = tokenizer(
            Xtrain.tolist(),
            max_length=max_seq_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=False )

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(ytrain.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class BERT_ARCH(nn.Module):
    def __init__(self, bert):
        super(BERT_ARCH, self).__init__()
        self.trainable = False
        
        
        self.bert = bert
        
        self.dropout = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(768, 512)
        
        self.fc2 = nn.Linear(512, 256)
        
        self.fc3 = nn.Linear(256, 6)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]
        
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        x = self.softmax(x)
        
        return x
        
for param in bert.parameters():
    param.requires_grad = False
    
model = BERT_ARCH(bert)

# push the model to GPU
model = model.to(device)

from transformers import AdamW
# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

from sklearn.utils.class_weight import compute_class_weight
from torch.optim import lr_scheduler

class_wts = compute_class_weight(classes=np.unique(ytrain), y = ytrain, class_weight='balanced')

weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)
cross_entropy = nn.NLLLoss(weight=weights)

train_losses = []

epochs = 1

model.load_state_dict(torch.load('static/weights/bot.pt'))

def train():
  model.train()

  total_loss = 0

  total_preds = []

  for step, batch in enumerate(train_dataloader):
    if step % 50 == 0 and not step == 0:
      print('Batch {:>5,} of {:>5}.'.format(step, len(train_dataloader)))

    batch = [r.to(device) for r in batch]
    sent_id, mask, labels = batch

    preds = model(sent_id, mask)

    loss = cross_entropy(preds, labels)

    total_loss = total_loss + loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    optimizer.zero_grad()

    preds = preds.detach().cpu().numpy()

    total_preds.append(preds)

  avg_loss = total_loss / len(train_dataloader)

  total_preds = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

#lr_sch = lr_scheduler.SetpLR(optimizer, step_size=100, gamma=0.1)

#model = torch.load("../static/weights/bot.pt"


if model.trainable:
  for epoch in range(epochs):
    print("\nEpoch {:} / {:}".format(epoch + 1, epochs))

    train_loss, _ = train()

    train_losses.append(train_loss)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.banchmark = False

  print(f"\nTraing Loss : {train_loss:.3f}")


# model.eval()

def get_prediction(str):
        str = re.sub(r'[^a-zA-Z ]+', '', str)
        test_text = [str]
        model.eval()
        
        tokens_test_data = tokenizer(
        test_text,
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
        )
        test_seq = torch.tensor(tokens_test_data['input_ids'])
        test_mask = torch.tensor(tokens_test_data['attention_mask'])
        
        preds = None
        with torch.no_grad():
            preds = model(test_seq.to(device), test_mask.to(device))
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis = 1)
            #print("Intent Identified: ", le.inverse_transform(preds)[0])
            return le.inverse_transform(preds)[0]
def get_response(message): 
    intent = get_prediction(message)
    for i in data['intents']: 
            if i["intent"] == intent:
                result = random.choice(i["responses"])
                if intent == 'name':
                    user_name = " ".join(i for i in message.split() if i not in " ".join(data['intents'][1]['text']).split())
                    result = str(result).format(user_name)
                break
                
    print(f"Response :\n {result}")
    return result #"Intent: "+ intent + '\n' + "Response: " + result

get_response("my name is jane doe")
print(type(model))
