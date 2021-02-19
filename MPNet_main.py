import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from MPNet_dataset import ObstacleDataSet, PathDataSet

from MPNet_models import MLP, CAE

from tqdm import tqdm
from os.path import join
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import argparse
import datetime

import neptune
RAEYO_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNDkwNTM5ODktODFiOC00OWI3LWE1YmYtYmQxMjYxZTliZjMwIn0="
from fileApi import *



mpnet_data_root = "/home/raeyo/dev_tools/MotionPlanning/MPNet/MPNetDataset"
S2D_data_path = join(mpnet_data_root, "S2D")

S3D_data_path = join(mpnet_data_root, "S3D")

def train_cae(args):
    train_dataset = ObstacleDataSet(S2D_data_path)
    val_dataset = ObstacleDataSet(S2D_data_path, is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    now = datetime.now()
    output_folder = args.output_folder + '/' + now.strftime('%Y-%m-%d_%H-%M-%S')
    check_and_create_dir(output_folder)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CAE().to(device)

    optimizer = torch.optim.Adagrad(model.parameters())
    # criterion = nn.MSELoss() the loss function inside model

    for epoch in range(args.max_epoch):
        model.train()

        for i, data in enumerate(tqdm(train_loader)):
            # get data
            x = data.to(device)
            # reconstruct
            recons_x = model(x)
            # get loss
            loss = model.get_loss(x, recons_x)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            neptune.log_metric("batch_loss", loss.item())

        print('\ncalculate validation accuracy..')
        
        model.eval()
        with torch.no_grad():
            losses = []
            for i, data in enumerate(tqdm(val_loader)):
                # get data
                x = data.to(device)
                # reconstruct
                recons_x = model(x)
                # get loss
                loss = model.get_loss(x, recons_x)
                losses.append(loss.item())


            val_loss = np.mean(losses) 
            neptune.log_metric("val_loss", val_loss)

        print("validation result, epoch {}: {}".format(epoch, val_loss))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), '{}/epoch_{}.tar'.format(output_folder, epoch))

def test_cae(args):
    test_dataset = ObstacleDataSet(S2D_data_path, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    now = datetime.now()
    output_folder = args.output_folder + '/' + now.strftime('%Y-%m-%d_%H-%M-%S')
    check_and_create_dir(output_folder)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CAE().to(device)
    
    assert args.load_weights, "No trained weight"
    model.load_state_dict(torch.load(args.load_weights))

    model.eval()
    losses = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            # get data
            x = data.to(device)
            # reconstruct
            recons_x = model(x)
            # get loss
            loss = model.get_loss(x, recons_x)
            losses.append(loss.item())

            x = x.cpu().reshape(-1, 2)            
            recons_x = recons_x.cpu().reshape(-1, 2)
            plt.scatter(x[:,0], x[:,1], c='blue')
            plt.scatter(recons_x[:,0], recons_x[:,1], c='red')
            
            plt.show()


    test_loss = np.mean(losses)
        
    print("test result: {}".format(test_loss))

def train_mlp(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    assert args.cae_weight, "No trained cae weight"
    encoder = CAE().encoder.to(device)
    encoder.eval()
    encoder.load_state_dict(torch.load(args.cae_weight), strict=False)

    train_dataset = PathDataSet(S2D_data_path, encoder)
    val_dataset = PathDataSet(S2D_data_path, encoder, is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    now = datetime.now()
    output_folder = args.output_folder + '/' + now.strftime('%Y-%m-%d_%H-%M-%S')
    check_and_create_dir(output_folder)
    
    model = MLP(args.input_size, args.output_size).to(device)
    if args.load_weights:
        print("Load weight from {}".format(args.load_weights))
        model.load_state_dict(torch.load(args.load_weights))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters())

    for epoch in range(args.max_epoch):
        model.train()

        for i, data in enumerate(tqdm(train_loader)):
            # get data
            input_data = data[0].to(device) # B, 32
            next_config = data[1].to(device) # B, 2

            # predict
            predict_config = model(input_data)
            
            # get loss
            loss = criterion(predict_config, next_config)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            neptune.log_metric("batch_loss", loss.item())

        print('\ncalculate validation accuracy..')
        
        model.eval()
        with torch.no_grad():
            losses = []
            for i, data in enumerate(tqdm(val_loader)):
                # get data
                input_data = data[0].to(device) # B, 32
                next_config = data[1].to(device) # B, 2

                # predict
                predict_config = model(input_data)
                
                # get loss
                loss = criterion(predict_config, next_config)

                losses.append(loss.item())

            val_loss = np.mean(losses) 
            neptune.log_metric("val_loss", val_loss)

        print("validation result, epoch {}: {}".format(epoch, val_loss))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), '{}/epoch_{}.tar'.format(output_folder, epoch))

def test_mlp(args):
    pass 


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # training hyperparameter
    parser.add_argument("-e","--max_epoch", type=int, default=200, help="number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("-bs","--batch_size", type=int, default=4, help="batch size")
    
    # data loader
    parser.add_argument("--num_workers", type=int, default=6, help="number of workers for data loader")
    
    # model setting
    parser.add_argument("-m","--model", type=str, default='MLP', help="CAE or MLP")
    parser.add_argument("--load_weights", type=str, default=None, help="path to the pretrained weights")

    # cae setting
    parser.add_argument("--cae_weight", type=str, default="./results/2021-02-18_18-44-22/epoch_100.tar", help="path to the pretrained weights")
    # mlp setting
    parser.add_argument('--input_size', type=int , default=32, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
    parser.add_argument("--mlp_weight", type=str, default=None, help="path to the pretrained weights")

    # inference
    parser.add_argument("--output_folder", type=str, default='results', help="path to the dataset root")
    parser.add_argument("--test", action='store_true')


    args = parser.parse_args()
    
    check_and_create_dir(args.output_folder)

    if args.model == "CAE":
        if not args.test:
            neptune.init(api_token=RAEYO_TOKEN, project_qualified_name='raeyo/MPNet')
            neptune.create_experiment('Train CAE S2D')
            train_cae(args)
        else:
            test_cae(args)
    elif args.model == "MLP":
        if not args.test:
            neptune.init(api_token=RAEYO_TOKEN, project_qualified_name='raeyo/MPNet')
            neptune.create_experiment('Train MLP S2D')
            train_mlp(args)
        else:
            test_mlp(args)
    else:
        assert False, "Check model name"

