import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils
import torch.nn as nn
from torchsummary import summary
from sklearn.model_selection import KFold


from model import set_parameter_requires_grad,initialize_model
from CornellDataset import CornellDataset
import grasp
from transformation import RandomTranslate,CentralCrop,RandomRotate,Rescale,Normalize,ToTensor

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='alexnet',
                        help='Network name in inference/models')
    parser.add_argument('--criterion', type=str, default='MSE',
                        help='metric error to evaluate the error')
    # Datasets
    parser.add_argument('--dataset-path', type=str,default='Dataset',
                        help='Path to dataset')     
    parser.add_argument('--random-rotate', type=float, default=True,
                        help='Choose to random rotate')    
    parser.add_argument('--random-translate', type=float, default=True,
                        help='Choose to random translate')   
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--n-splits', type=float, default=5,
                        help='Number of folds for training and validation (remainder is validation)') 
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')    
    parser.add_argument('--epochs', type=int, default=25,
                        help='Training epochs')    
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()
    
    return args

def training(epoch, net, device, criterion, train_data, optimizer, vis):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param criterion: function to evaluate the loss
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    
    results = 0
    batch_idx = 0
    # change flag of training
    net.train()
    
    for  sample in train_data:        
        
        batch_idx +=1

        # load of the normalized image
        img = sample['image'].to(device)
        
        # load of the ground bounding boxes
        ground_bb = sample['bb'].to(device)
        
        # forward propagation
        pred_bb = net(img)
        
        # evaluate the loss
        loss = criterion(ground_bb,pred_bb)
        
        results += loss
        
        #  gradients are zeroed
        optimizer.zero_grad()
        
        # backward propagation
        loss.backward()
        
        # optimization of the parameters
        optimizer.step()        
        
        logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss))
    
    return results/batch_idx
    
    
def validate(net, device, val_loader):
    
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_loader: Validation Dataset
    :return: Successes, Failures 
    
    """
    
    net.eval()
    
    results = {
        'correct': 0,
        'failed': 0        
    }
    batch_idx = 0

    with torch.no_grad():
        
        for sample in val_loader:
            
            batch_idx +=1

            # load of the normalized image
            img = sample['image'].to(device)            
            
            # load of the ground bounding boxes
            ground_bb = sample['bb'].to(device).squeeze().cpu()
            
            # forward propagation
            pred_bb = net(img).to(device).squeeze().cpu()
            
            ground_gr = grasp.Grasp((ground_bb[0],ground_bb[1]),np.arctan(ground_bb[3]/ground_bb[2]),ground_bb[4],ground_bb[5])
            ground_rect = ground_gr.as_gr
            
            pred_gr = grasp.Grasp((pred_bb[0],pred_bb[1]),np.arctan(pred_bb[3]/pred_bb[2]),pred_bb[4],pred_bb[5])
            pred_rect = pred_gr.as_gr
            
            iou_param = ground_rect.iou(pred_rect)
            
            if iou_param > 0.25:
                results['correct'] += 1
            else:
                results['failed'] += 1
    
    return results

def run():
    
    args = parse_args()
    
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))
    
    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)
    
    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)
            
    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
     # Load the network
    logging.info('Loading Network...')
    
    net, _ = initialize_model(args.network, 6, feature_extract=False, use_pretrained=True)
    net = net.to(device)
    
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(),lr=0.0005, weight_decay = 0.001)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.0005, weight_decay = 0.001)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))
    
    if args.criterion.lower() == 'mse':
        criterion = nn.MSELoss()
    elif args.criterion.lower() == 'rmse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError('Criterion {} is not implemented'.format(args.criterion))
    
    logging.info('Done')  
    
     # Print model architecture.
    summary(net, (3, 224, 224))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (3, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()
    
    # Load Dataset
    logging.info('Loading Cornell Dataset...')
    
    dataset = CornellDataset(args.dataset_path,transform=transforms.Compose([RandomTranslate(),
                                                            CentralCrop(),
                                                            RandomRotate(),
                                                            Rescale(),
                                                            Normalize(),
                                                            ToTensor()
                                                            ]))
    
    logging.info('Done')  
    
    logging.info('Dataset size is {}'.format(dataset.len))   
    

    kf = KFold(n_splits=args.n_splits)
    
    fold = 0
    best_iou = 0.0
    
    for train_indices, val_indices in kf.split(dataset):
        
        fold+=1
        
        
        logging.info('Beginning fold {:2d}'.format(fold))

        train_data = torch.utils.data.Subset(dataset, train_indices)
        val_data = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=1,
            num_workers=args.num_workers
        )       
        
        logging.info('Training size: {}'.format(len(train_indices)))
        logging.info('Validation size: {}'.format(len(val_indices)))
        
        for epoch in range(args.epochs):
            
            np.random.seed(20)
            random.seed(20)     
            
            logging.info('Beginning Epoch {:02d}'.format(epoch))
            train_results = training(epoch, net, device, criterion, train_loader, optimizer, vis=args.vis)
            # Log training losses to tensorboard
            tb.add_scalar('loss/train_loss', train_results, epoch, fold)
            
            # Run Validation
            logging.info('Validating...')
            test_results = validate(net, device, val_loader)
            logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
            
            # Log validation results to tensorbaord
            tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
            
            # Save best performing network
            iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
            
            if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
                torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
                best_iou = iou
            
if __name__ == '__main__':
    run()