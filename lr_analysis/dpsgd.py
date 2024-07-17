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

logging.getLogger().setLevel(logging.INFO)

class LR(Module):
    def __init__(self, n_in, n_out, bias=False):
        super(LR, self).__init__()
        self.linear = GradSampleModule(Linear(n_in, n_out, bias=bias))

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
        split_size=args.split_size, row_clip=args.row_clip, state=args.folk_state, gaussian_sigma=args.gaussian_sigma, gaussian_sigma_y=args.gaussian_sigma_y)
    batch_size = int(n*args.q)
    print('batch size per iteration', batch_size)
    test_loader = load_tensor_by_batch(test_tensor)
    train_loader = load_tensor_by_batch(train_tensor, batch_size)
    
    # model
    model = LR(d_in, d_out).to(device)
    # optimizer 
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch_i in range(1, args.epoch + 1):
        model.train()
        sum_loss = 0.
            
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # forward
            pred = model(x)
            if len(y.size()) == 1:
                y = torch.unsqueeze(y, dim=-1)
            loss = criterion(pred, y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # clip
            if args.c > 0:
                for param in model.parameters():
                    # clip each sample
                    for i, grad_sample in enumerate(param.grad_sample):
                        param.grad_sample[i] = clip_grad_by_l2norm(grad_sample, args.c)
                    # aggregate gradients
                    param.grad.data = torch.mean(param.grad_sample, dim=0)

            # inject noise
            if args.perturb_grad:
                for param in model.parameters():
                    param.grad += 1. / float(len(y)) * torch.normal(mean=0, std=args.sigma_multi*args.c, size=param.size(),device=param.device)

            optimizer.step()

             # clip model weights
            if args.model_c > 0:
                for param in model.parameters():
                    param = clip_grad_by_l2norm(param, args.model_c)

            # clear gradient samples to avoid memory leakage
            for param in model.parameters():
                param.grad_sample = None

            sum_loss += loss.item() * len(y)

        # logging.info(f"Epoch #{epoch_i}: Training loss {sum_loss /n}")
            
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

                logging.info(f"State {args.folk_state}, Evaluation at Epoch #{epoch_i}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}")

"""
Example:
    python3 dpsgd.py folk_state=CA q=0.001 sigma_multi=1.15 epoch=2 device=$device seed=4
"""
if __name__ == "__main__":
    main()