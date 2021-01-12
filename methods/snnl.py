import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

STABILITY_EPS = 0.00001


def pairwise_euclid_distance(A, B):
    """Pairwise Euclidean distance between two matrices."""
    pdist = torch.cdist(A, B, p=2)
    return pdist


def pairwise_cos_distance(A, B, eps=1e-8):
    """Pairwise cosine distance between two matrices."""
    w1 = A.norm(p=2, dim=1, keepdim=True)
    w2 = B.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(A, B.t()) / (w1 * w2.t()).clamp(min=eps)


def fits(A, B, temp, cos_distance):
    """Exponentiated pairwise distance between each element of A and
    all those of B."""
    if cos_distance:
      distance_matrix = pairwise_cos_distance(A, B)
    else:
      distance_matrix = pairwise_euclid_distance(A, B)
    return torch.exp(-(distance_matrix / temp))


def pick_probability(x, temp, cos_distance):
    """Row normalized exponentiated pairwise distance between all the elements
    of x. Conceptualized as the probability of sampling a neighbor point for
    every element of x, proportional to the distance between the points."""
    f = fits(
        x, x, temp, cos_distance) - torch.eye(x.shape[0], device='cuda:0')
    return f / (STABILITY_EPS + f.sum(1).unsqueeze(1))


def same_label_mask(y, y2):
    """Masking matrix such that element i,j is 1 iff y[i] == y2[i]."""
    return torch.squeeze(torch.eq(y, y2.unsqueeze(1))).float()


def masked_pick_probability(x, y, temp, cos_distance):
    """The pairwise sampling probabilities for the elements of x for neighbor
    points which share labels."""
    return pick_probability(x, temp, cos_distance) * same_label_mask(y, y)


def SNNL(x, y, temp, cos_distance):
    summed_masked_pick_prob = masked_pick_probability(x, y, temp, cos_distance).sum(1)
    return torch.mean(-torch.log(STABILITY_EPS + summed_masked_pick_prob))


def optimized_temp_SNNL(x, y, initial_temp, cos_distance):
    """The optimized variant of Soft Nearest Neighbor Loss. Every time this
    tensor is evaluated, the temperature is optimized to minimize the loss
    value, this results in more numerically stable calculations of the SNNL."""
    t = tf.Variable(1, dtype=tf.float32, trainable=False, name="temp")

    def inverse_temp(t):
        # pylint: disable=missing-docstring
        # we use inverse_temp because it was observed to be more stable when optimizing.
        return tf.math.divide(initial_temp, t)

    ent_loss = SNNL(x, y, inverse_temp(t), cos_distance)
    updated_t = tf.assign(t, tf.subtract(t, 0.1*tf.gradients(ent_loss, t)[0]))
    inverse_t = inverse_temp(updated_t)
    return SNNL(x, y, inverse_t, cos_distance)

# define soft nearest neighbor loss
def snn_loss(temperature, cos_distance):
    def loss(y_pred, y_true):
        #print(tf.keras.backend.shape(y_pred))
        #print(tf.keras.backend.shape(y_true))
        #y_true = K.print_tensor(y_true, message='y_true = ')
        #y_pred = K.print_tensor(y_pred, message='y_pred = ')
        return SNNL(y_pred, y_true, temperature, cos_distance)
    return loss