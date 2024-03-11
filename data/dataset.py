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


def getIterator(dataset, batch_size):

    trainset, testset, input_dim, output_dim = getDataset(dataset)

    train_sampler = RandomSampler(trainset, batch_size)
    test_sampler = RandomSampler(testset, batch_size)

    train_loader = torch.utils.data.DataLoader( trainset, num_workers=8, pin_memory=True, batch_sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(  testset,  num_workers=8, pin_memory=True, batch_sampler=test_sampler)

    train_iterator = iter(train_loader)
    test_iterator  = iter(test_loader)

    getBatch = lambda train: next(train_iterator if train else test_iterator)

    return getBatch, input_dim, output_dim
