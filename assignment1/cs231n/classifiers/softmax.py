import numpy as np
from random import shuffle
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    f = X[i].dot(W)
    f -= np.max(f)
    p = np.zeros_like(f)
    for j in xrange(num_classes):
        p[j] = np.exp(f[j])
    p /= np.sum(p)
    loss += -np.log(p[y[i]])

    for j in xrange(num_classes):
      softmax_score = p[j]

      # Gradient calculation.
      if j == y[i]:
        dW[:, j] += (-1 + softmax_score) * X[i]
      else:
        dW[:, j] += softmax_score * X[i]


  loss /= num_train
  loss += reg*np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W


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
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= np.max(f,axis=1,keepdims=True)
  xf = np.exp(f)
  p = xf / np.sum(xf,axis=1,keepdims=True)
  loss = np.sum(-np.log(p[range(num_train),y])) / num_train
  loss += reg*np.sum(W*W)
  p[range(num_train),y] -= 1
  dW = (X.T).dot(p)

  dW /= num_train
  dW += 2*reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
