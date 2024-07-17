import argparse
import torch
import numpy as np
import random
import pandas as pd
from torch.nn import Module, Linear

from utils import criterion, clip_grad_by_l2norm, load_tensor_by_batch

from loader import load_data

import logging
import time
import math
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

logging.getLogger().setLevel(logging.INFO)

class LR(Module):
    def __init__(self, n_in, n_out, bias=False):
        super(LR, self).__init__()
        self.linear = Linear(n_in, n_out, bias=bias)

    def forward(self, x):
        return self.linear(x)

def discretize_tensor(x, device, gamma):
    x = x.to(device)
    floor = torch.floor(gamma * x)
    p_ceil = x - floor

    random_nums = torch.rand(x.size(), device=device, requires_grad=False)
    choice_floor = (random_nums > p_ceil).type(torch.float32)

    discrete_x = choice_floor * floor + (1. - choice_floor) * (floor + 1.)
    return discrete_x

def no_discretize_tensor(x, device, num_exp):
    x = x.to(device)
    return x


def skellam_noise(size, poisson_mu, device):
    rate = torch.full(size, poisson_mu, dtype=torch.float).to(device)
    noise = torch.poisson(rate) - torch.poisson(rate)

    return noise#.to(device)


def noisy_gradient_approx(weight, x, targets, device, gamma=100):
    """
        return the approx gradient sum without the noise.
    """
    dis_weight = discretize_tensor(0.25*weight, device, gamma) / gamma
    dis_x = discretize_tensor(x, device, gamma) / gamma
    dis_targets = discretize_tensor(targets, device, gamma) / gamma
    dis_b1 = discretize_tensor(torch.ones_like(x)*0.5, device, gamma) / gamma
    s = dis_b1*dis_x + torch.sum(dis_x * dis_weight, dim=1).unsqueeze(1) * dis_x - dis_targets*dis_x

    return s

"""
python3 lr_sqm.py data_type=folktable_income folk_state=CA is_onehot=True gamma=100 device=3 q=0.001 epoch=1 mu_normalized=2.11
"""
@hydra.main(version_base=None,  config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    # fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # run = wandb.init(project=f"skellam_{args.data_type}", config=args)

    # get device
    if str(args.device).lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    # load data
    logging.info(f"loading data...")

    train_tensor, test_tensor, n, d_in, d_out = load_data(
        data_type=args.data_type, split_size=args.split_size, row_clip=args.row_clip, state=args.folk_state,is_onehot=args.is_onehot)

    batch_size = int(n*args.q)
    logging.info(f"loading tensor by batch...")
    print('batch size per iteration', batch_size)
    test_loader = load_tensor_by_batch(test_tensor)
    train_loader = load_tensor_by_batch(train_tensor, batch_size)
    logging.info(f"training start...")

    # model
    model = LR(d_in, d_out).to(device)
    # optimizer
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch_i in range(1, args.epoch + 1):
        model.train()
        sum_loss = 0.
        start_time = time.time()
        batch_count = 0
        batch_start_time = time.time()
        epoch_start_time = time.time()
        for x, y in train_loader:
            # print(x.shape)
            x, y = x.to(device), y.to(device)
            # forward
            pred = model(x)
            if len(y.size()) == 1:
                y = torch.unsqueeze(y, dim=-1)
            loss = criterion(pred, y)

            loss.backward()
            for param in model.parameters():
                if args.gaussian_approx:
                    noisy_grad = noisy_gradient_approx(param, x, y, device, args.gamma)
                    noisy_grad = noisy_grad.sum(dim=0).reshape(1,-1)
                    # print(noisy_grad.shape)
                    noisy_grad = noisy_grad + torch.normal(mean=0, std=math.sqrt(2*args.mu_normalized), size=noisy_grad.size(), device=device) / (n*args.q)
                    param.grad = noisy_grad
                else:
                    noisy_grad = noisy_gradient_approx(param, x, y, device, args.gamma)
                    noisy_grad = noisy_grad.sum(dim=0).reshape(1,-1)
                    noisy_grad = noisy_grad + skellam_noise(noisy_grad.size(), args.mu_normalized, device) / (n*args.q)
                    param.grad = noisy_grad

            optimizer.step()

            # clip model weights
            if args.model_c > 0:
                for param in model.parameters():
                    param = clip_grad_by_l2norm(param, args.model_c)

            # clear gradient samples to avoid memory leakage
            for param in model.parameters():
                param.grad_sample = None
            noisy_grad = None
            sum_loss += loss.item() * len(y)

            batch_count = batch_count + 1
            # if batch_count % 10 == 0:
            #     print(f'{batch_count} batches finished in time: {time.time()-batch_start_time}')


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

        log_info = {
                "mu": args.mu_normalized,
                "state": args.folk_state,
                "Task": args.data_type,
                "epoch": epoch_i,
                "test_loss": sum_loss / len(test_loader.dataset),
                "test_acc": sum_correct / len(test_loader.dataset),
                "time": time.time()-start_time,
            }
        pd.DataFrame([log_info]).to_csv("lr_skellam.csv", mode='a', header=False, index=False)

        # TODO: add wandb db logging
        logging.info(f" mu is #{args.mu_normalized}, State is #{args.folk_state}, Task is #{args.data_type}: Epoch #{epoch_i}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}, Finished in time: {time.time()-epoch_start_time}")

if __name__ == "__main__":
    main()
