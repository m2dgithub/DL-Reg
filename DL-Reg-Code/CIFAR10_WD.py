'''Train CIFAR10 with PyTorch using the Weight-Decay regularizer.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
import os
import argparse
from models import *


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(28, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

#trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1280, shuffle=False, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx)
#)

set_seed(42)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2000, shuffle=True, num_workers=0, worker_init_fn=np.random.seed(42))


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
set_seed(42)
testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=0, worker_init_fn=np.random.seed(42))

# Model
print('==> Building model..')
net = mlp_relu(in_ch=3072, num_classes=10)          # Running model is a simple MLP with a ReLu activation

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
MSE_criterion = torch.nn.MSELoss().cuda()
criterion = nn.CrossEntropyLoss().cuda()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
#optimizer = optim.Adadelta(net.parameters(), lr=args.lr, rho=0.9, eps=1e-05)
decayRate = 0.96
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# Training
print('Training on CIFAR-10 using the Weight_Decay regularizer...')
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets).float()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return (train_loss / (batch_idx + 1)), (100. * correct / total)
    
    
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('============= Current Test Acc: '+str(acc))
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    print('============= Best Test Acc: ' + str(best_acc))
    return acc

for epoch in range(start_epoch, start_epoch+1250):
    trainLoss, trainAcc = train(epoch)
    if np.mod(epoch,30)==0:
        my_lr_scheduler.step()
    
    print('TrainAcc: '+str(trainAcc)+'    TrainLoss: '+str(trainLoss))
    testAcc = test(epoch)
    file1 = open("WD_CIFAR10_TrainAcc.txt", "a")
    file2 = open("WD_CIFAR10_TrainLoss.txt", "a")
    file3 = open("WD_CIFAR10_TestAcc.txt", "a")
    file1.write(str(trainAcc)+'\n')
    file2.write(str(trainLoss) + '\n')
    file3.write(str(testAcc)+ '\n')
    file1.close()
    file2.close()
    file3.close()
