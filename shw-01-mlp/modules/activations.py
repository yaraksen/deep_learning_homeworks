import numpy as np
from .base import Module
import scipy


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * np.maximum(np.sign(input), 0)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # return 1 / (1 + np.exp(-input))
        return scipy.special.expit(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sf = scipy.special.expit(input)
        return grad_output * sf * (1 - sf)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # return np.exp(input) / np.exp(input).sum(axis=1)[:, None]
        return scipy.special.softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # отсюда взял идею с np.einsum
        # https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-activation-function-and-its-derivative-jacobian-in-python/
        softmax = self.compute_output(input)
        dfdx = np.einsum('ij,jk->ijk', softmax, np.eye(input.shape[1])) - np.einsum('ij,ik->ijk', softmax, softmax)
        return np.einsum('ij,ijk->ik', grad_output, dfdx)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # return np.log(np.exp(input) / np.exp(input).sum(axis=1)[:, None])
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # softmax = np.exp(input) / np.exp(input).sum(axis=1)[:, None]
        sf = scipy.special.softmax(input, axis=1)
        dfdx = -np.einsum('ij,ik->ijk', np.ones_like(input), sf) + np.eye(input.shape[1])
        return np.einsum('ij,ijk->ik', grad_output, dfdx)
