from random import shuffle

import numpy as np
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # TODO(gh): implement the softmax again
    num_train = X.shape[0]
    num_classes = W.shape[1]

    S = X.dot(W)
    S -= np.max(S)
    exp = np.exp(S)
    P = exp / np.sum(exp, axis=1, keepdims=True)
    for i in range(num_train):
        loss += -np.log(P[i][y[i]])
        temp = P[i]
        temp[y[i]] = 0
        temp[y[i]] = -np.sum(temp)
        dW += X[i].T.dot(np.matrix(temp))

    loss /= num_train
    loss += reg * np.sum(W**2)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    S = X.dot(W)
    S -= np.max(S)
    exp = np.exp(S)
    P = exp / np.sum(exp, axis=1, keepdims=True)
    loss = -np.sum(np.log(P[np.arange(num_train), y]))
    loss /= num_train
    loss += reg * np.sum(W**2)

    temp = P
    temp[np.arange(num_train), y] = 0
    temp[np.arange(num_train), y] = -np.sum(temp, axis=1)

    dW = X.T.dot(temp)
    dW /= num_train
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW
