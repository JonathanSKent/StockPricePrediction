#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:31:02 2019

@author: Jonathan S. Kent
"""


data_directory = '<DIRECTORY WHERE THE STOCK DATA FILES ARE KEPT>'
model_loc = '<LOCATION OF THE STORED NEURAL NETWORK>'

n_files = 200 # Number of files from which training examples are sampled
examples_per_epoch = 5000 # Number of training examples that are sampled per epoch
length_of_example = 30 # Number of trading days used to produce an example
epochs_to_train = 10 # Number of epochs upon which to train a given network

input_count = 29 # Number of inputs to a network
hidden_shape = [200, 200, 100, 10] # Number of neurons in fully connected hidden layers
output_count = 4 # Number of outputs a network should have
learning_rate = 1e-7 # Coefficient by which gradients are multiplied before descent

days_per_sim = 200 # Number of trading days over which to simulate fund management