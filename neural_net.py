#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:36:34 2019

@author: jonathan
"""

import settings
import data_gathering

import torch
import pickle
import matplotlib

# This class defines the neural net being used to predict confidences in stock prices.
# Rather than attempting to predict how much a stock will go up or down by,
# the model is trained to give a number roughly coincing with its confidence that
# a stock will go up or down. i.e. return ~0 if it has no clue, return a small positive
# number if it thinks it might go up, return a large negative number if it's certain that
# it will go down.
class nnet:
    # Initializes a new neural network with random weights and biases, with a structure
    # defined by the appropriate values in settings.py, and tracking the gradients
    # of the weights and biases.
    def __init__(self, input_count = settings.input_count,
                 hidden_shape = settings.hidden_shape,
                 output_count = settings.output_count):
        self.shape = [input_count] + hidden_shape + [output_count]
        self.l = len(self.shape) - 1
        self.W = [torch.autograd.Variable(torch.randn(self.shape[i], self.shape[i+1]) / 10, requires_grad = True) for i in range(self.l)]
        self.B = [torch.autograd.Variable(torch.randn(i) / 10, requires_grad = True) for i in self.shape[1:]]
        
    # Gives the neural network an input, and then takes the output of the network.
    # Uses ReLU as its activation function.
    def predict(self, x):
        x_prime = torch.tensor(x, dtype = torch.float32)
        for i in range(self.l):
            x_prime = (x_prime.mm(self.W[i]) + self.B[i])
            if i + 1 < self.l:
                x_prime = x_prime.clamp(min = 0)
        return(x_prime)
    
    # Uses standard stochastic gradient descent with a predefined constant
    # learning rate in order to update the behavior of the neural net.
    # I plan to start using something like AdamOptimizer in the future,
    # once I've read up on it more and understand how it works.
    def train(self, x, y, learning_rate = settings.learning_rate):
        y_hat = self.predict(x)
        loss = -torch.tensor(y, dtype = torch.float32).t().mm(y_hat).mean()
        loss.backward()
        for i in self.W + self.B:
            i.data -= learning_rate * i.grad.data
            i.grad.zero_()
        return(loss)
            
# The following two functions save and load the model, respectively.
# However, pickle is tempermental for some reason, and fails to work consistently.
# Working on that
def save_model(model, loc = settings.model_loc):
    with open(loc, 'wb') as file:
        pickle.dump(model, file)
    
def load_model(loc = settings.model_loc):
    with open(loc, 'rb') as file:
        return(pickle.load(file))
        
# Trains a given model by collecting an epoch from the data, preprocessing it,
# turning it into a single matrix, doing predictions, doing the gradient descent,
# you know the drill
def train(model, loc = settings.model_loc, epoch = settings.examples_per_epoch, files = settings.n_files):
    X, Y = data_gathering.prep(data_gathering.get_training_examples(epoch, files))
    loss = model.train(X, Y)
    save_model(model, loc)
    return(loss)
    
# Basically calculates the same loss that the neural net is using, but outside
# of that particular class, in order to more readily calculate validation loss
# to better understand the behavior of the model during training
def validate(model):
    X, Y = data_gathering.prep(data_gathering.get_training_examples())
    Y_hat = model.predict(X)
    return(-torch.tensor(Y, dtype = torch.float32).t().mm(Y_hat).mean(), Y[0], Y_hat[0])
    
# Trains the model, calculates loss, and produces a graph of loss versus epoch
def training(model, epochs = settings.epochs_to_train):
    training_loss = []
    validation_loss = []
    for i in range(epochs):
        print("Beginning Epoch " + str(i))
        training_loss.append(train(model))
        l, y, y_h = validate(model)
        validation_loss.append(l.data)
        print("Training loss: " + str(training_loss[-1].data))
        print("Validation loss: " + str(l.data))
        print(y)
        print(y_h)
    plt = matplotlib.pyplot
    plt.plot(training_loss, label = "Training Loss")
    plt.plot(validation_loss, label = "Validation Loss")
    plt.legend()
    plt.show()