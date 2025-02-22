import torch
from rebuilding import create_single_state
from preprocessing import removeHashtags 
from transformers import BertTokenizer

# categories are indexed as follows: (dad joke, nerdy, positive, negative, neutral, brainrot)
def classify(message):
    return (1, 2, 3, 4, 5, 0)

def real_classify(message):
    model = create_single_state()
    message = removeHashtags(message)
    message = message.strip()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    encoded_dict = tokenizer.encode_plus(
                        message,                     
                        add_special_tokens = True, 
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    logits = 0

    with torch.no_grad():
        output=model(input_id, token_type_ids=None, attention_mask=attention_mask)
        logits = output.logits 
    
    return logits

print(real_classify("I love physics and math and computer science"))
