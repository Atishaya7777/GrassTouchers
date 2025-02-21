import nltk
import torch
import re
import transformers 
from transformers import BertTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords


def removeHashtags(x):
    if "#" in x:
        hashIndex = x.index("#")
        return x[0:hashIndex]
    else:
        return x


sw = stopwords.words('english')

def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r"http\S+", "",text) #Removing URLs 
    #text = re.sub(r"http", "",text)
    
    html=re.compile(r'<.*?>') 
    
    text = html.sub(r'',text) #Removing html tags
    
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,'') #Removing punctuations
        
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    
    text = " ".join(text) #removing stopwords
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) #Removing emojis
     
    return text

 
f = open("data.txt", "r")
currLabel = -1
data = []
input_ids = []
attention_masks = []
labels = []
maxLen = 0

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

for x in f:
    if x[0] == '*':
        currLabel = int(x[2])
    else:
        if x != "\n":
            if currLabel >= 0:
                x = removeHashtags(x)
                x = x.strip()
                data.append((x, currLabel))
            else:
                print("ERR: no label defined")

for x, y in data:
    input_ids = tokenizer.encode(x, add_special_tokens=True)
    maxLen = max(maxLen, len(input_ids))

for text, label in data:
    encodedDict = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = maxLen,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    input_ids.append(encodedDict['input_ids'])
    attention_masks.append(encodedDict['attention_mask'])
    labels.append(label)

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
    
