from __future__ import division  # floating point division
import numpy as np


def sigmoid(xvec):
    """ Compute the sigmoid function """
    # Cap -xvec, to avoid overflow
    # Undeflow is okay, since it get set to zero
    try:
        xvec[xvec < -100] = -100
    except RuntimeWarning:
        print(xvec[xvec < - 100])

    vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))

    return vecsig


def dsigmoid(xvec):
    """ Gradient of standard sigmoid 1/(1+e^-x) """
    vecsig = sigmoid(xvec)
    return vecsig * (1 - vecsig)


def relu(xvec):
    """ Compute the ReLU function """
    return np.maximum(0, xvec)


def drelu(xvec):
    """ Gradient of ReLU """
    grad = np.zeros(xvec.shape)
    grad[xvec > 0] = 1
    return grad


def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes


def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2 """
    for k in dict1:
        if k in dict2:
            dict1[k]=dict2[k]
