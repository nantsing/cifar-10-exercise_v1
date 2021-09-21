import torch
import model
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.utils as us
import argparse

import data

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('--EPOCH','-E',type=int,default=5)
parser.add_argument('--BATCH_SIZE','-B' ,type=int,default=30)
parser.add_argument('--LR','-L',type=float,default=0.01)
args = parser.parse_args()

cnn = model.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=args.LR)

train_data = data.train_data_load()
train_data = us.data.DataLoader(train_data, batch_size = args.BATCH_SIZE, shuffle=True)

test_data = data.test_data_load()
test_data = us.data.DataLoader(test_data, batch_size = 500)

print(test_data)

for epoch in range(args.EPOCH):
    print(epoch,':')
    cnn.train()
    for index, (data, label) in enumerate(train_data):
        #print(data)
        optimizer.zero_grad()
        output = cnn(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if (index%100 == 0):
            print('index:',index//100,'  ','loss=',loss.item())

    cnn.eval()
    for index, (data, label) in enumerate(test_data) :
        output = cnn(data)
        _,pred = torch.max(output,1)
        accuracy = sum(pred == label)/len(label) * 100
        print('test_batch_',index,':Accuracy is ', accuracy.item())

    
    
        
        
