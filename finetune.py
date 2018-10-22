'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import numpy as np

from models import *
from utils import progress_bar



pretrain = True


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--workdir', type=str, default='./exp1')
parser.add_argument('--data', type=str, default='/tempspace2/lcai/fly/small_data_2')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''global variable'''
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_acc = []
'''global variable'''


# Data
print('==> Preparing data..')

traindir = os.path.join(args.data , 'train')
valdir = os.path.join(args.data, 'val')

trainset = datasets.ImageFolder(traindir,
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.746878, 0.7491242, 0.79941636), (0.22726493, 0.22574322, 0.15550046))]))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size = args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

valset = datasets.ImageFolder(valdir,
    transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.746878, 0.7491242, 0.79941636), (0.22726493, 0.22574322, 0.15550046))]))

testloader = torch.utils.data.DataLoader(valset,batch_size = args.batch_size, shuffle=False, num_workers=4)


def rank_loss(similarity, label):
    val = similarity[range(similarity.size(0)), label]
    loss = -torch.sum(val)
    return loss

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch<10:
        lr = 0.001
    elif epoch<20:
        lr = 0.00001
    elif epoch<30:
        lr = 0.000001
    else:
        lr = 0.0000001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def trainResNet(epoch, net, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    adjust_learning_rate(optimizer, epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def trainRankNet(epoch, net, trainloader, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_cls = 0
    train_rank = 0
    correct = 0
    ranked_correct = 0
    total = 0
    adjust_learning_rate(optimizer, epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        reference = load_reference()
        optimizer.zero_grad()
        outputs, similarity = net(inputs, reference)
        loss_cls = criterion(outputs, targets)
        loss_rank = rank_loss(similarity, targets)
        loss = loss_cls + loss_rank
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_cls += loss_cls.item()
        train_rank += loss_rank.item()
        _, predicted = outputs.max(1)
        _, ranked = similarity.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        ranked_correct += ranked.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: (%.3f | %.3f + %.3f) | Acc: %.3f%% , %.3f%% (%d | %d/%d)'
            % (train_loss/(batch_idx+1), train_cls/(batch_idx+1), train_rank/(batch_idx+1), 100.*correct/total, 100.*ranked_correct/total,  correct, ranked_correct, total))

def test(epoch, net, testloader, optimizer, path):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total,  correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    total_acc.append(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, path + '/ckpt.t7')
        best_acc = acc


def testRank(epoch, net, testloader, optimizer, path):
    global best_acc
    net.eval()
    test_loss = 0
    test_cls = 0
    test_rank = 0
    correct = 0
    total = 0
    ranked_correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            reference = load_reference()
            outputs, similarity = net(inputs, reference)
            loss_cls = criterion(outputs, targets)
            loss_rank = rank_loss(similarity, targets)
            loss = loss_cls + loss_rank

            test_loss += loss.item()
            test_cls += loss_cls.item()
            test_rank += loss_rank.item()
            _, predicted = outputs.max(1)
            _, ranked = similarity.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            ranked_correct += ranked.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | %.3f + %.3f | Acc: %.3f%% , %.3f%% (%d | %d/%d)'
                % (test_loss/(batch_idx+1), test_cls/(batch_idx+1), test_rank/(batch_idx+1), 100.*correct/total, 100.*ranked_correct/total,  correct, ranked_correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    total_acc.append(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, path+'/ckpt.t7')
        best_acc = acc


def load_reference_idx(idx):
    ref_name = './reference/reference'+str(idx)+'.npy'
    ref = np.load(ref_name)
    ref = torch.FloatTensor(ref).cuda()
    return ref

def load_reference():
    idx = np.random.randint(100, size=1)
    ref_name = args.workdir + '/reference/reference'+str(idx[0])+'.npy'
    ref = np.load(ref_name)
    ref = torch.FloatTensor(ref).cuda()
    return ref

def find_reference(file_name, net, trainloader, path):
    net.eval()
    img = torch.FloatTensor(14,3,128,320).zero_().cuda()
    count = np.zeros(14)
    for batch_idx,  (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        res = torch.topk(outputs, 2)
        if predicted.eq(targets).sum().item() == 1:
            if res[0][0][0] - res[0][0][1] > 3 and count[targets] == 0:
                img[targets] = inputs
                count[targets] = 1
                #print(np.sum(count))
            if np.sum(count)==14:
                np.save(path+'/'+file_name,img)
                break

criterion = nn.CrossEntropyLoss()




net = RankNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
print('==> Resuming from checkpoint..')
checkpoint = torch.load(args.workdir+'/res_pretrain/ckpt.t7')
model_dict = net.state_dict()
state = checkpoint['net']
model_dict.update(state)
net.load_state_dict(model_dict)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
for epoch in range(start_epoch, start_epoch+50):
    trainRankNet(epoch, net, trainloader, optimizer)
    testRank(epoch, net, testloader, optimizer, args.workdir + '/rank_model')








