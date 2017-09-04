import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):

      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # Gradient for non correct class weight.
        loss += margin
        dW[:,j] += X[i]
        dW[:, y[i]] -= X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Average our gradient across the batch and add gradient of regularization term.
  dW = dW/num_train + 2*reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #from lecture:
  num_train = X.shape[0]

  #get all scores with big matrix multiplication
  scores = X.dot(W)

  #create a 2D Index array to get class scores for each training example
  correct_class_idx = range(num_train),y
  correct_scores = scores[correct_class_idx]
  correct_scores = np.reshape(correct_scores,(num_train,1))

  ###LOSS
  # get margins by subtracting score of correct class
  margins = scores - correct_scores + 1

  #hinge-loss
  margins = np.maximum(0,margins)

  #correct class scores don't contribute to loss
  margins[correct_class_idx] = 0

  #average loss across training example
  loss = np.sum(margins) / num_train

  #reguarlization
  loss += reg * np.sum(W * W)

  ###GRADIENT

  #look at non-zero indexes
  #for each non-zero index:
  #add column to row

  nonzero = np.copy(margins)
  nonzero[nonzero > 0] = 1
  nonzero[correct_class_idx] = -(np.sum(nonzero, axis=1))

  dW = (X.T).dot(nonzero)
  dW /= num_train
  dW += 2*reg*W

  return loss, dW
