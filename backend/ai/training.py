import time
import datetime
import torch 
import torch.nn as nn 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import transformers 
from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_ids = torch.load("dataTensor//input_ids.pt")
attention_masks = torch.load("dataTensor//attention_masks.pt")
labels = torch.load("dataTensor//labels.pt")

print(input_ids.size())

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32

train_dataloader = DataLoader(
    train_dataset, 
    sampler = RandomSampler(train_dataset), 
    batch_size = batch_size 
)

validation_dataloader = DataLoader(
    val_dataset, 
    sampler = SequentialSampler(val_dataset),
    batch_size = batch_size
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels = 6,
    output_attentions = False, 
    output_hidden_states = False
)

model = model.to(device)

max_epochs = 4
optim = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps = len(train_dataloader)*max_epochs)
epochStart = 0
batchStart = 0
total_train_loss = 0
best_eval_accuracy = 0

model.train()

def printBatchSum(loss, batchStart, batchSize):
    elapsed = time.time() - batchStart
    elapsed_rounded = int(round((elapsed)))
    avg_train_loss = loss / batchSize
    print(datetime.timedelta(seconds=elapsed_rounded), "|", "{0:.4f}".format(avg_train_loss))

def top1Accuracy(predictions, labels):
    predictedCategory = [] 
    correctCount = 0
    numTested = 0

    for x in predictions:
        predictedCategory.append(torch.argmax(x))

    for i in range(predictions.size(dim=0)):
        if predictedCategory[i] == labels[i]:
            correctCount += 1
        numTested += 1

    return correctCount/numTested

for i in range(max_epochs):
    epochStart = time.time()

    print("---STARTING EPOCH ", i, "---")
    for step, batch in enumerate(train_dataloader):
        batchStart = time.time()

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        optim.zero_grad()
        output = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss = output.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        printBatchSum(loss, batchStart, len(b_labels))

    print("VALIDATION")
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            output = model(b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask,
                labels=b_labels)
        loss = output.loss 
        total_eval_loss += loss.item() 
        logits = output.logits 
        logits = logits.to('cpu')
        label_ids = b_labels.to('cpu') 
        total_eval_accuracy += top1Accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

    if avg_val_accuracy > best_eval_accuracy:
        torch.save(model, 'textClassifier')
        best_eval_accuracy = avg_val_accuracy

    print("---FINISHED EPOCH", i, "---")
    printBatchSum(total_train_loss, epochStart, len(train_dataloader))
    print("Training accuracy: ", avg_val_accuracy*100, "%")
    



