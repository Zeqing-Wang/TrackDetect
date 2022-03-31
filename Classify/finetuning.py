# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 15:36
# @Author  : naptmn
# @File    : finetuning.py
# @Software: PyCharm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                # print(phase,len(dataloaders[phase]))
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(len(inputs))
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss =+ loss.item()*inputs.size(0)
                # print(len(preds))
                # print(preds)
                # print(labels.data)
                running_corrects += torch.sum(preds == labels.data)
                # print(running_corrects)
            epoch_loss = running_loss/dataset_sizes[phase]
            # print(running_corrects)
            epoch_acc = running_corrects.double()/dataset_sizes[phase]

            print('{}Loss:{:.4f}Acc:{:.4f}'.format(
                phase, epoch_loss, epoch_acc))
    return model
if __name__=='__main__':
    data_dir = './data/mit'
    model_name = "resnet"
    num_classes = 2
    batchsize = 16
    num_epochs = 20
    # 是否微调整个模型
    feature_extract = True # False时微调整个模型
    data_transform = {
        'train':transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val':transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x),
                                             data_transform[x])
                      for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x:len(image_datasets[x])for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(dataset_sizes,class_names,device)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=200)