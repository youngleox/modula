import os
import pickle
import requests
import numpy as np
import torch

from torchvision import datasets, transforms

from data.sampler import RandomSampler


def getDataset(dataset):

    if dataset == "cifar10":

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

        input_dim = (3,32,32)
        output_dim = 10
        
        return trainset, testset, input_dim, output_dim

    elif dataset == "shakespeare":

        if not os.path.exists('data/shakespeare/train.bin'):
            exec(open('data/shakespeare.py').read())

        trainset = np.memmap('data/shakespeare/train.bin', dtype=np.uint16, mode='r')
        testset  = np.memmap('data/shakespeare/val.bin',   dtype=np.uint16, mode='r')

        vocab_size = 65

        return trainset, testset, vocab_size, None


def getIterator(dataset, device, batch_size, context=None):

    trainset, testset, input_dim, output_dim = getDataset(dataset)

    if dataset == 'cifar10':

        train_sampler = RandomSampler(trainset, batch_size)
        test_sampler = RandomSampler(testset, batch_size)

        train_loader = torch.utils.data.DataLoader( trainset, num_workers=8, pin_memory=True, batch_sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(  testset,  num_workers=8, pin_memory=True, batch_sampler=test_sampler)

        train_iterator = iter(train_loader)
        test_iterator  = iter(test_loader)

        _getBatch = lambda train: next(train_iterator if train else test_iterator)

    elif dataset == 'shakespeare':

        def _getBatch(train):
            data = trainset if train else testset
            ix = torch.randint(len(data) - context, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+context]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+context]).astype(np.int64)) for i in ix])
            if device != "cpu":
                x, y = x.pin_memory(), y.pin_memory()
            return x, y

    def getBatch(train):
        data, target = _getBatch(train)
        return data.to(device, non_blocking=True), target.to(device, non_blocking=True)

    return getBatch, input_dim, output_dim
