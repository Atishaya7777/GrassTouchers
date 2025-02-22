from os import walk
import torch 
from transformers import BertForSequenceClassification

from transformers import BertModel, BertForSequenceClassification 
from transformers import models
from transformers import activations
from torch import nn
from collections import OrderedDict 

torch.serialization.add_safe_globals([BertForSequenceClassification])
torch.serialization.add_safe_globals([BertModel])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertEmbeddings])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertEncoder])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertLayer])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertAttention])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertSdpaSelfAttention])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertSelfOutput])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertIntermediate])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertOutput])
torch.serialization.add_safe_globals([models.bert.modeling_bert.BertPooler])
torch.serialization.add_safe_globals([models.bert.configuration_bert.BertConfig])
torch.serialization.add_safe_globals([nn.modules.sparse.Embedding])
torch.serialization.add_safe_globals([nn.modules.normalization.LayerNorm])
torch.serialization.add_safe_globals([nn.modules.dropout.Dropout])
torch.serialization.add_safe_globals([nn.modules.container.ModuleList])
torch.serialization.add_safe_globals([nn.modules.linear.Linear])
torch.serialization.add_safe_globals([nn.modules.activation.Tanh])
torch.serialization.add_safe_globals([activations.GELUActivation])
torch.serialization.add_safe_globals([torch._C._nn.gelu])

model_state = OrderedDict()

for i in range(22):
    curr_chunk = torch.load('model_state_chunk_' + str(i))
    for x, y in curr_chunk.items():
        model_state[x] = y

model = BertForSequenceClassification.from_pretrained( 
    "bert-base-cased",
    num_labels = 6, 
    output_attentions = False, 
    output_hidden_states = False
)

model.load_state_dict(model_state)
torch.save(model, 'textClassifier')
