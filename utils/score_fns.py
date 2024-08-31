from torch import softmax, exp
from torch import max as tmax
from torch import sum as tsum
from torch import argmax as targmax

# from torch import sum
from math import log
import numpy as np



def max_probability(logits):
    probs = softmax(logits, dim=1)
    max_prob = tmax(probs).item()
    return max_prob


def max_probability2(logits):
    probs = softmax(logits, dim=1)
    label = targmax(logits, dim=1)
    max_prob = tmax(probs).item()
    return label, max_prob

def calibrated_max_probability(logits, T):
    # logits = logits.to('cpu')
    probs = exp(logits / T) / tsum(exp(logits / T))
    max_prob = tmax(probs).item()
    return max_prob


def difference(logits):
    probs = softmax(logits, dim=1)
    probs = probs[0].tolist()
    first = max(probs)
    probs[probs.index(max(probs))] = 0
    second = max(probs)
    difference = first - second
    return difference


def entropy(logits):
    probs = softmax(logits, dim=1)
    entropy = -sum([x.item() * log(x.item()) for x in probs[0]])
    return entropy


def entropy_temp(logits, t=1):
    probs = softmax(logits / t, dim=1)
    entropy = -sum([x.item() * log(x.item()) for x in probs[0]])
    return entropy