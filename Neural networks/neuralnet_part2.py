# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 2)
        self.optimizer = optim.SGD(self.parameters(), lr = lrate)
        self.loss_fn = loss_fn
        #raise NotImplementedError("You need to write this part!")


    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #raise NotImplementedError("You need to write this part!")
        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #raise NotImplementedError("You need to write this part!")
        return loss.item()

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    loss = nn.CrossEntropyLoss()
    net = NeuralNet (0.07, loss, train_set.shape[1], 2)
    losses = []
    yhats = []
    train_set = (train_set - train_set.mean()) / train_set.std()
    dev_set = (dev_set - dev_set.mean()) / dev_set.std()
    for i in range (n_iter):
        losses.append(net.step(train_set[(i * batch_size)% train_set.shape[0]: ((i+1)*batch_size) % train_set.shape[0]], train_labels[(i * batch_size) % train_set.shape[0] : ((i+1) * batch_size) % train_set.shape[0]]))
    for item in dev_set:
        arr = net.forward(item)
        if arr[0][0] > arr[0][1]:
            yhats.append(0)
        else:
            yhats.append(1)
    #raise NotImplementedError("You need to write this part!")
    return losses,yhats,net

