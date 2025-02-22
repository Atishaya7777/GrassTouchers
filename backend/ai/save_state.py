import torch 
from transformers import BertModel, BertForSequenceClassification 
from transformers import models
from transformers import activations
from torch import nn

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

model = torch.load('textClassifier', weights_only = True).to('cpu')
torch.save(model.state_dict(), 'textClassifierStateDict')

