import torch
from torchtext.datasets import AG_NEWS


def main_sample():
    train_iter = AG_NEWS(split='train')
    print(train_iter)
    print(type[train_iter])
    print(next(train_iter))


if __name__ == '__main__':
    main_sample()
