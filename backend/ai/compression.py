"""
import torch 
from transformers import BertForSequenceClassification 

torch.serialization.add_safe_globals([BertForSequenceClassification])
model = torch.load('textClassifier', weights_only = False).to('cpu')
torch.save(model.state_dict(), 'textClassifierStateDict')

"""
import torch 
from transformers import BertForSequenceClassification 
from collections import OrderedDict

torch.serialization.add_safe_globals([BertForSequenceClassification])
model_state = torch.load('textClassifierStateDict')
model_state_chunks = [OrderedDict()]
model_chunk_size = len(model_state) // 20

i = 0
for x, y in model_state.items(): 
    if i % model_chunk_size == 0:
        model_state_chunks.append(OrderedDict())
    model_state_chunks[i // model_chunk_size][x] = y
    i += 1

print(i == len(model_state))

"""
i = 0
for chunk in model_state_chunks: 
    torch.save(chunk, 'model_state_chunk_' + str(i))
    i += 1
"""
