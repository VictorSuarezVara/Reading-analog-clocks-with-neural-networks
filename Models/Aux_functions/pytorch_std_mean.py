import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm

def get_mean_std(loader):

    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ , _ in tqdm(loader):
        channels_sum += torch.mean(torch.mean(data, dim=[0,2,3]))
        channels_sqrd_sum += torch.mean(torch.mean(data ** 2, dim=[0,2,3]))
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    if True in torch.isnan(data):
        debugg_here = 0
        print("There is a nan in hours")

    return mean, std



