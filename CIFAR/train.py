#coding: utf-8
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import os
import argparse
from tqdm import tqdm
from loss import LDAMLoss, compute_adjustment, FCEloss_alpha, FCEloss_beta
from net import resnet32
from dataset import IMBALANCECIFAR10, IMBALANCECIFAR100

### train ###
def train(epoch):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):

        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        targets = targets.long()

        output = model(inputs)

        if args.losstype=='CE':
            loss = criterion(output, targets)
        elif args.losstype=='LDAM':
            loss = criterion(output, targets)
        elif args.losstype=='LogitAdjust':
            output_ad = output + logit_ajust
            loss = criterion(output_ad, targets)
        elif args.losstype=='FCE_a' or args.losstype=='FCE_b':
            loss = criterion(output, targets) + criterion_fce(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = F.softmax(output, dim=1)
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        sum_loss += loss.item()
 
    return sum_loss/(batch_idx+1), float(correct)/float(total)


### test ###
def test(epoch):
    model.eval()
    sum_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, leave=False)):

            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()

            output = model(inputs)

            if args.losstype=='CE':
                loss = criterion(output, targets)
            elif args.losstype=='LDAM':
                loss = criterion(output, targets)
            elif args.losstype=='LogitAdjust':
                loss = criterion(output, targets)
            elif args.losstype=='FCE_a' or args.losstype=='FCE_b':
                loss = criterion(output, targets) + criterion_fce(output, targets)


            output = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            sum_loss += loss.item()


    return sum_loss/(batch_idx+1), float(correct)/float(total)


def adjust_learning_rate(optimizer, epoch, lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCEloss')
    parser.add_argument('--dataset', '-d', type=str, default='CIFAR10')
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--Tbatchsize', '-t', type=int, default=50)
    parser.add_argument('--num_epochs', '-e', type=int, default=200)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--ratio', '-r', type=float, default=0.01)
    parser.add_argument('--losstype', '-l', type=str, default='CE')
    args = parser.parse_args()
    gpu_flag = args.gpu


    if not os.path.exists("{}".format(args.out)):
      	os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "{}".format(args.ratio))):
      	os.mkdir(os.path.join("{}".format(args.out), "{}".format(args.ratio)))
    if not os.path.exists(os.path.join("{}".format(args.out), "{}".format(args.ratio), "model")):
      	os.mkdir(os.path.join("{}".format(args.out), "{}".format(args.ratio), "model"))

    PATH_1 = "{}/{}/trainloss.txt".format(args.out, args.ratio)
    PATH_2 = "{}/{}/testloss.txt".format(args.out, args.ratio)
    PATH_3 = "{}/{}/trainaccuracy.txt".format(args.out, args.ratio)
    PATH_4 = "{}/{}/testaccuracy.txt".format(args.out, args.ratio)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass
    with open(PATH_4, mode = 'w') as f:
        pass


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if args.dataset=='CIFAR10':
        n_classes = 10
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type='exp', imb_factor=args.ratio, rand_number=args.seed, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset=='CIFAR100':
        n_classes = 100
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type='exp', imb_factor=args.ratio, rand_number=args.seed, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.Tbatchsize, shuffle=False, drop_last=True)

    cls_num_lists = train_dataset.get_cls_num_list()

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


    ### CE loss ###
    if args.losstype=='CE':
        criterion = nn.CrossEntropyLoss().cuda(device)
        model = resnet32(num_classes=n_classes, use_norm=False).cuda(device)
    ### LDAM loss ###
    elif args.losstype=='LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_lists).cuda(device)
        model = resnet32(num_classes=n_classes, use_norm=True).cuda(device)
    ### LogitAdjust ###
    elif args.losstype=='LogitAdjust':
        criterion = nn.CrossEntropyLoss().cuda(device)
        model = resnet32(num_classes=n_classes, use_norm=False).cuda(device)
        logit_ajust = compute_adjustment(train_loader)
    ### FCEloss type alpha ###
    elif args.losstype=='FCE_a':
        criterion = nn.CrossEntropyLoss().cuda(device)
        criterion_fce = FCEloss_alpha(cls_num=cls_num_lists).cuda(device)
        model = resnet32(num_classes=n_classes, use_norm=False).cuda(device)
    ### FCEloss type beta
    elif args.losstype=='FCE_b':
        criterion = nn.CrossEntropyLoss().cuda(device)
        criterion_fce = FCEloss_beta(n_classes=n_classes, cls_num=cls_num_lists).cuda(device)
        model = resnet32(num_classes=n_classes, use_norm=False).cuda(device)


    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)


    sample_acc = 0
    sample_loss = 1000000

    ##### training & test #####
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch, 0.1)

        loss_train, acc_train = train(epoch)
        loss_test, acc_test = test(epoch)

        print("Epoch{:3d}/{:3d}  TrainLoss={:.4f}  TestAccuracy={:.2f}%".format(epoch+1, args.num_epochs, loss_train, acc_test*100))

        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, acc_train))
        with open(PATH_4, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, acc_test))

        if acc_test >= sample_acc:
           sample_acc = acc_test
           PATH_best ="{}/{}/model/model_bestacc.pth".format(args.out, args.ratio)
           torch.save(model.state_dict(), PATH_best)

        if loss_train <= sample_loss:
           sample_loss = loss_train
           PATH_best ="{}/{}/model/model_bestloss.pth".format(args.out, args.ratio)
           torch.save(model.state_dict(), PATH_best)


