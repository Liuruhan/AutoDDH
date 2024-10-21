import argparse
import torch
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report
from itertools import cycle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchnet import meter

# models
from models.AutoDDH import AutoDDH

from dataset import BasicDataset
from dataloaderX import DataPrefetcher, DataLoaderX
from loss import FocalLoss

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch AutoDDH',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', type=int, default=4, metavar='B',
                        help='input ba tch size for training (default : 64)', dest='batch_size')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=4, metavar='TB',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('-nc', '--ncls', type=int, default=4, metavar='NC',
                        help='number of DDH degrees')
    parser.add_argument('-ns', '--nseg', type=int, default=8, metavar='NS',
                        help='number of key structures')
    parser.add_argument('-l', '--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('-c', '--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-d', '--device', type=str, default='0,1,2', metavar='device',
                        help='CUDA device ID')
    parser.add_argument('-ml', '--model-list', type=list, default=['autoDDH'], metavar='model_list',
                        help='training model list')
    parser.add_argument('-log', '--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-cp', '--save-cp', type=bool, default=True, metavar='save_cp',
                        help='is save the model?')
    parser.add_argument('-trdir', '--trdir', type=str, default='./data/DDH/train/', metavar='trdir',
                        help='the path of training models')
    parser.add_argument('-tedir', '--tedir', type=str, default='./data/DDH/test/', metavar='tedir',
                        help='the path of testing models')
    parser.add_argument('-valdir', '--valdir', type=str, default='./data/DDH/val/', metavar='valdir',
                        help='the path of validation models')
    parser.add_argument('-dir', '--cp-dir', type=str, default='./checkpoints/', metavar='cp_dir',
                        help='the path of saving models')
    parser.add_argument('-result', '--rd', type=str, default='./model_result/', metavar='rd',
                        help='the path of saving vis figures for models')
    return parser.parse_args()

def plot_AUC(pred, target, n_classes, epoch, path, din_color):
    n_target = np.eye(n_classes)[target.astype(int)]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(n_target[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(n_target.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    lw = 2

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='chocolate', linestyle=':', linewidth=3)
    plt.legend(loc="lower right")


    if din_color == plt.cm.Blues:
        colors = cycle(['lightskyblue', 'dodgerblue', 'royalblue', 'midnightblue', 'slategrey'])
    elif din_color  == plt.cm.Reds:
        colors = cycle(['mistyrose', 'lightcoral', 'firebrick', 'maroon', 'rosybrown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    if din_color == plt.cm.Blues:
        plt.savefig(path + '/' + str(epoch)+'_test_AUC.jpg')
    elif din_color == plt.cm.Reds:
        plt.savefig(path + '/' + str(epoch) + '_extraVal_AUC.jpg')
    plt.close()
    return

def plot_Matrix(cm, classes, epoch, path, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '0.4f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(float(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if cmap == plt.cm.Blues:
        plt.savefig(path + '/' + str(epoch) + 'test_cm.jpg', dpi=300)
    elif cmap == plt.cm.Reds:
        plt.savefig(path + '/' + str(epoch) + 'extraVal_cm.jpg', dpi=300)
    plt.close()
    return

def testdata_cal(model, args, loader, png_flag, epoch, path_str, n_classes, color):
    model.eval()
    test_loss = 0
    confusion_matrix = meter.ConfusionMeter(n_classes)
    out_np = np.zeros((len(loader)*args.batch_size, n_classes))
    tgt_np = np.zeros(len(loader)*args.batch_size)

    data_idx = 0
    criterion = nn.CrossEntropyLoss()
    prefetcher = DataPrefetcher(loader)
    batch = prefetcher.next()
    iter_id = 0
    while batch is not None:
        iter_id += 1
        img = batch['img']
        seg = batch['seg']
        target = batch['target'][:, 0]

        if args.cuda:
            img, seg, target = img.cuda(), seg.cuda(), target.cuda()
        img, seg, target = Variable(img), Variable(seg), Variable(target)

        seg_out, cls_out = model(img)
        # output = model(data)
        test_loss += criterion(cls_out, target).item()

        confusion_matrix.add(cls_out.data, target.data)
        size = cls_out.data.cpu().numpy().shape[0]
        out_np[data_idx:data_idx+size] = cls_out.data.cpu().numpy()
        tgt_np[data_idx:data_idx+size] = target.data.cpu().numpy()
        data_idx += size

        batch = prefetcher.next()
    
    pred_np = np.argmax(out_np, axis=1)
    print('Epoch:', epoch, 'Test_loss:', test_loss / len(loader))
    result = classification_report(tgt_np, pred_np, target_names=['Normal', 'Mild DDH', 'Severe DDH', 'Hip dislocation'], digits=4)
    print(result)

    cm_value = confusion_matrix.value()
    if png_flag == True:
        if not os.path.exists(path_str):
            os.mkdir(path_str)
        plot_Matrix(cm_value, [1, 2, 3, 4], epoch, path_str, title=None, cmap=color)
        plot_AUC(out_np, tgt_np, n_classes, epoch, path_str, color)
    return 

def train_net(args, test_loader, extraVal_loader):
    if args.model_name == 'autoDDH':
        model = AutoDDH(seg_nclasses=8, cls_nclasses=4, n_channels=3, cuda_device=True)

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()
    if not os.path.exists(args.rd):
        os.mkdir(args.rd)
    if not os.path.exists(args.cp_dir):
        os.mkdir(args.cp_dir)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-12, momentum=0.95)
    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-12)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)
    # softmax = nn.Softmax()

    # class loss
    # weights = [0.04, 0.1, 0.43, 0.43]
    weights = [0.1, 0.2, 0.35, 0.35]
    class_weights = torch.FloatTensor(weights).cuda()
    cls_loss = nn.CrossEntropyLoss(weight=class_weights)

    # seg loss
    weight = 0.5
    seg_loss1 = nn.CrossEntropyLoss()
    seg_loss2 = FocalLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        prefetcher = DataPrefetcher(train_loader)
        batch = prefetcher.next()
        iter_id = 0
        while batch is not None:
            iter_id += 1
            img = batch['img']
            seg = batch['seg']
            target = batch['target'][:, 0]

            if args.cuda:
                img, seg, target = img.cuda(), seg.cuda(), target.cuda()
            img, seg, target = Variable(img), Variable(seg), Variable(target)

            optimizer.zero_grad()
            seg_out, cls_out = model(img)

            cls_loss_value = cls_loss(cls_out, target)
            seg_loss_value1 = seg_loss1(seg_out, seg)
            seg_loss_value2 = weight * seg_loss2(seg_out, seg)
            loss = cls_loss_value + seg_loss_value1 + seg_loss_value2

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            batch = prefetcher.next()

            # if (iter_id+1) % args.log_interval == 0:
            #     print("####################################################")
            #     print('Percentage:', (iter_id+1) / len(train_loader), 'Train_loss:', train_loss / iter_id)
            #     testdata_cal(model, args, test_loader, True, epoch*10000+iter_id, model_name, n_classes, plt.cm.Blues)
            #     testdata_cal(model, args, extraVal_loader, True, epoch*10000+iter_id, model_name, n_classes, plt.cm.Reds)

        print('Epoch:', epoch, 'Train_loss:', train_loss / len(train_loader))
        testdata_cal(model, args, test_loader, True, epoch, args.rd + args.model_name, args.ncls, plt.cm.Blues)
        testdata_cal(model, args, extraVal_loader, True, epoch, args.rd + args.model_name, args.ncls, plt.cm.Reds)
        print("####################################################")

        if epoch % 10 == 0:
           for p in optimizer.param_groups:
               p['lr'] *= 0.95

        if args.save_cp:
            try:
                os.mkdir(args.cp_dir)
            except OSError:
                pass
            torch.save(model.state_dict(),
                       args.cp_dir + args.model_name + f'CP_epoch{epoch + 1}.pth')
            print(f'Checkpoint {epoch + 1} saved !')
    
    del model
    
    return

if __name__ == "__main__":
    args = get_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cuda_device = torch.cuda.is_available()
    print('cuda available:', args.cuda)

    train = BasicDataset(args.trdir, args.ncls, args.nseg)
    val = BasicDataset(args.tedir, args.ncls, args.nseg)
    test = BasicDataset(args.valdir, args.ncls, args.nseg)
    n_val = len(val)
    n_train = len(train)
    n_test = len(test)
    print('samples:', 'tr', n_train, 'val', n_val, 'te', n_test)

    train_loader = DataLoaderX(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoaderX(val, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    extraVal_loader = DataLoaderX(test, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for i in range(len(args.model_list)):
        print(args.model_list[i])
        args.model_name = args.model_list[i]
        train_net(args, test_loader, extraVal_loader)


