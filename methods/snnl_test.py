import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from snnl import *


def test_pairwise_euclidean_dist():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_ = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    dist = pairwise_euclid_distance(x, x_)
    assert torch.tensor(13.86) >= torch.sum(dist) >= torch.tensor(13.85), "Incorrect Euclidean dist"


def test_pairwise_cos_dist():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_ = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    dist = pairwise_cos_distance(x, x_)
    assert torch.tensor(0.045) >= torch.sum(dist) >= torch.tensor(0.044), "Incorrect cosine dist"


def test_fits():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_ = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    temp = 0.5
    sol = fits(x, x_, temp, True)
    assert torch.tensor(3.92) >= torch.sum(sol) >= torch.tensor(3.91), "Incorrect fits"


def test_pick_prob():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    temp = 0.5
    sol = pick_probability(x, temp, True)
    assert torch.tensor(4.01) >= torch.sum(sol) >= torch.tensor(3.99), "Incorrect pick prob"


def test_label_mask():
    y = torch.tensor([1.0, 1.0, 0.0, 2.0])
    sol = same_label_mask(y, y)
    assert torch.sum(sol) == torch.tensor(6.0), "Incorrect label mask"


def test_SNNL():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    temp = 0.5
    y = torch.tensor([1.0, 1.0, 0.0, 2.0])
    sol = SNNL(x, y, temp, True)
    assert torch.tensor(6.32) >= sol >= torch.tensor(6.31), "Incorrect SNNL"


if __name__ == "__main__":
    print("Tests: ")
    test_pairwise_euclidean_dist()
    test_pairwise_cos_dist()
    test_fits()
    test_pick_prob()
    test_label_mask()
    test_SNNL()