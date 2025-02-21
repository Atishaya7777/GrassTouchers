import nltk
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

for text, label in data:
    print(label, text)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)





