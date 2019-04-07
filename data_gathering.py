#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:30:16 2019

@author: Jonathan S. Kent
"""

import settings

import os
import random
import csv
import numpy as np
from copy import deepcopy as dc

# The stock price training data is stored in a folder, where each file
# represents the history of the stock through the date that the data was
# gathered. In order to make collection of a subset of that data easier
# for the purpose of producing a training epoch, this function
# returns the names of a small number of files
def get_n_files(loc = settings.data_directory, n = settings.n_files):
    file_names = os.listdir(loc)
    i = list(range(len(file_names)))
    new_file_names = []
    for _ in range(n):
        k = random.randint(0, len(i) - 1)
        new_file_names.append(file_names[i[k]])
        del(i[k])
    return(new_file_names)

# This function takes the names given by get_n_files and randomly selects
# periods of data from each file, with the probability that a file is selected
# being based on the length of the file, e.g. if one file only has one 30 day period
# to choose from, and another file has 60 days total, and thus 31 possible 30 day
# periods, it will select random months from the second file 31x as often
def get_training_examples(example_count = settings.examples_per_epoch, file_count = settings.n_files):
    locs = get_n_files(n = file_count)
    data_main = {}
    for loc in locs:
        with open(os.path.join(settings.data_directory, loc), 'r') as file:
            reader = csv.reader(file)
            data_main[loc] = [line for line in reader][1:]
    volumes = np.array([(len(data_main[loc]) - settings.length_of_example) for loc in locs])
    for i in range(len(volumes)):
        volumes[i] *= volumes[i] > 0
    volumes = volumes / np.sum(volumes)
    examples = []
    for _ in range(example_count):
        curr_loc = np.random.choice(locs, p = volumes)
        start = random.randint(0, len(data_main[curr_loc]) - settings.length_of_example)
        examples.append(data_main[curr_loc][start : start + settings.length_of_example])
    return(examples)

# This function takes a single example from get_training_examples, and does
# preprocessing to turn it into a 1xN vector that can more easily be understood
# by the neural net. Four numbers represent how much up the price was a month ago,
# twenty numbers represent the percentage change day to day in the last five days,
# and five numbers are the logarithm of the number of shares traded those days.
# This serves to normalize the data.
def prep_(example):
    data = np.array(np.array(example)[:, 1:6], dtype = np.float32)
    prices = data[:, :4]
    volumes = data[:, -1]
    mean = prices.mean()
    base = dc(prices[0, :])
    base -= mean
    prices = (prices[1:, :] - prices[:-1, :])
    prices /= mean * 0.01
    base /= mean * 0.01
    X = np.concatenate((np.concatenate((np.array([base]), prices[-6:-1])).reshape([-1, 1]), np.array([np.log(volumes[-6:-1] + 1)]).reshape([-1, 1]))).reshape([1, -1])
    Y = np.array([prices[-1, :]])
    return(X, Y)
   
# This function takes a list of examples, and turns them into a single array
# that can be passed to the neural net and trained on in one fell swoop.
# This also helps significantly with using GPU acceleration to speed up both
# training and neural net operation
def prep(examples):
    X_, Y_ = [], []
    for i in examples:
        x, y = prep_(i)
        X_.append(x)
        Y_.append(y)
    return(np.concatenate(X_), np.concatenate(Y_))