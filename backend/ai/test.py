import torch 
import numpy as np 

def top1Accuracy(predictions, labels):
    predictedCategory = [] 
    correctCount = 0

    for x in predictions:
        print(x)
        predictedCategory.append(np.argmax(x))

    for i in range(len(predictions)):
        if predictedCategory[i] == labels[i]:
            correctCount += 1

    return correctCount/len(predictions)

labels = torch.load("dataTensor//labels.pt")
labels = labels[0:5]
predictions = [0, 4, 5, 1, 3] 
print(labels)
print(predictions)
print(top1Accuracy(predictions, labels))

