import argparse
import datetime
import numpy as np
import math
import os
import gc

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloader import ChalDataset
from network import Network 

parser = argparse.ArgumentParser()

parser.add_argument_group('Optimization related arguments')
parser.add_argument('-num_epochs', default=20, type=int, help='Epochs')
parser.add_argument('-batch_size', default=64, type=int, help='Batch size')
parser.add_argument('-lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('-lr_decay_rate', default=0.9997592083, type=float, help='Decay for lr')
parser.add_argument('-min_lr', default=5e-5, type=float, help='Minimum learning rate')
parser.add_argument('-weight_init', default='xavier', choices=['xavier', 'kaiming'], help='Weight initialization strategy')
parser.add_argument('-overfit', action='store_true', help='Overfit on 5 examples, meant for debugging')
parser.add_argument('-gpuid', default=0, type=int, help='GPU id to use')
        
parser.add_argument('-input_csv', default='./training_data.csv')
parser.add_argument('-normalize', default=True)
parser.add_argument('-test_size', default=0.33)

parser.add_argument_group('Checkpointing related arguments')
parser.add_argument('-load_path', default='', help='Checkpoint to load path from')
parser.add_argument('-save_path', default='checkpoints/', help='Path to save checkpoints')
parser.add_argument('-save_step', default=2, type=int, help='Save checkpoint after every save_step epochs')

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_path == 'checkpoints/':
    args.save_path += start_time

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# transfer all options to model
model_args = args

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

if args.load_path != '':
    components = torch.load(args.load_path)
    model_args = components['model_args']
    model_args.gpuid = args.gpuid
    model_args.batch_size = args.batch_size

    # this is required by dataloader
    args.normalize = model_args.normalize

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = ChalDataset(args)
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate_fn)

# ----------------------------------------------------------------------------
# setting model args
# ----------------------------------------------------------------------------
# iterations per epoch
setattr(args, 'iter_per_epoch', math.ceil(dataset.num_data_points['train'] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

net = Network(model_args)
optimizer = optim.Adam(net.parameters())
criterion = nn.MultiLabelSoftMarginLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay_rate)

if args.load_path != '':
    net.load_state_dict(components['net'])
    print("Loaded model from {}".format(args.load_path))

if args.gpuid >= 0:
    net = net.cuda()
    criterion = criterion.cuda()

# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

net.train()
os.makedirs(args.save_path, exist_ok=True)

running_loss = 0.0
train_begin = datetime.datetime.utcnow()
print("Training start time: {}".format(datetime.datetime.strftime(train_begin, '%d-%b-%Y-%H:%M:%S')))

log_loss = []
for epoch in range(1, model_args.num_epochs + 1):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        for key in batch:
            batch[key] = Variable(batch[key])
            if args.gpuid >= 0:
                batch[key] = batch[key].cuda()

        # --------------------------------------------------------------------
        # forward-backward pass and optimizer step
        # --------------------------------------------------------------------
        net_out = net(batch['features'])

        cur_loss = criterion(net_out, batch['outputs'])
        cur_loss.backward()

        optimizer.step()
        gc.collect()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        #train_loss = cur_loss.data[0]
        train_loss = cur_loss.item()
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * train_loss
        else:
            running_loss = train_loss 

        if optimizer.param_groups[0]['lr'] > args.min_lr:
            scheduler.step()


        # --------------------------------------------------------------------
        # print after every few iterations
        # --------------------------------------------------------------------
        if i % 10 == 0:
            test_losses = []
            accuracy = []
            for i in range(int(dataset.num_data_points['test']/args.batch_size)):
                test_feat = dataset.X_test[i*args.batch_size:(i+1)*args.batch_size, :]
                test_labels = dataset.y_test[i*args.batch_size:(i+1)*args.batch_size, :]
                test_feat = Variable(test_feat)
                test_labels = Variable(test_labels)
                if args.gpuid >= 0:
                    test_feat = test_feat.cuda()
                    test_labels = test_labels.cuda()
                net_out = net(test_feat)
                cur_loss = criterion(net_out, test_labels)
                test_losses.append(cur_loss.item())
                
                y_pred = torch.sigmoid(net_out).data > 0.5
                y_pred = y_pred.cpu().numpy()
                accuracy.append((test_labels.cpu().numpy() == y_pred).all(axis=1))

            validation_loss = np.mean(test_losses)
            
            accuracy = np.mean(accuracy)

            iteration = (epoch - 1) * args.iter_per_epoch + i

            log_loss.append((epoch,
                             iteration,
                             running_loss,
                             train_loss,
                             validation_loss,
                             accuracy,
                             optimizer.param_groups[0]['lr']))

            # print current time, running average, learning rate, iteration, epoch
            print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][val loss: {:6f}][accuracy: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                    iteration, running_loss, validation_loss, accuracy,
                    optimizer.param_groups[0]['lr']))

    # ------------------------------------------------------------------------
    # save checkpoints and final model
    # ------------------------------------------------------------------------
    if epoch % args.save_step == 0:
        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': net.args
        }, os.path.join(args.save_path, 'model_epoch_{}.pth'.format(epoch)))

torch.save({
    'net': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': net.args
}, os.path.join(args.save_path, 'model_final.pth'))

np.save(os.path.join(args.save_path, 'log_loss'), log_loss)
