import argparse
import torch
import numpy as np
import random

from torch.nn import Module, Linear
from opacus.grad_sample import GradSampleModule

from utils import criterion, clip_grad_by_l2norm, load_tensor_by_batch, str2bool

from loader import load_data

import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.nn import Linear, Module, init

logging.getLogger().setLevel(logging.INFO)
class LR(Module):
    def __init__(self, n_in, n_out, bias=False):
        super(LR, self).__init__()
        self.linear = Linear(n_in, n_out, bias=bias)

    def forward(self, x):
        return self.linear(x)

    

@hydra.main(version_base=None,  config_path="config", config_name="config")
def main(args: DictConfig) -> None:   
    # fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # get device
    if str(args.device).lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    
    # load data
    train_tensor, test_tensor, n, d_in, d_out = load_data(
        split_size=args.split_size, row_clip=args.row_clip, state=args.folk_state, gaussian_sigma=args.gaussian_sigma, flip_y=args.flip_y)
    batch_size = int(n*args.q)
    print('batch size per iteration', batch_size)
    test_loader = load_tensor_by_batch(test_tensor)
    train_loader = load_tensor_by_batch(train_tensor, batch_size)
    
    # model
    model = LR(d_in, d_out)
    model.to(device)
    # optimizer 
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch_i in range(1, args.epoch + 1):
        model.train()
        sum_loss = 0.
            
        for x, y in train_loader:
            # backward
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            # forward
            pred = model(x)            
            
            if len(y.size()) == 1:
                y = torch.unsqueeze(y, dim=-1)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item() * len(y)
            
        if epoch_i % args.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                sum_loss, sum_correct = 0., 0.
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    pred = model(x)
                    if len(y.size()) == 1:
                        y = torch.unsqueeze(y, dim=-1)
                    loss = criterion(pred, y)

                    predict = 1. / (1. + torch.exp(- pred))

                    sum_correct += torch.sum(torch.sign(predict - 0.5) * (y - 0.5) * 2 == 1).item()
                    sum_loss += loss.item() * len(y)

                logging.info(f"State {args.folk_state}, Datatype {args.data_type} Evaluation at Epoch #{epoch_i}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}")

"""
Example:
    python3 ldp_sgd.py folk_state=$dataset q=0.001 sigma_multi=0 epoch=15 seed=4 gaussian_sigma=13.285 flip_y=0.437
"""


if __name__ == "__main__":
    main()