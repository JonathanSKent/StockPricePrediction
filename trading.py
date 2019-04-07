#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:13:26 2019

@author: Jonathan S. Kent
"""

import neural_net
import settings
import data_gathering

import math
import csv
import matplotlib

# Given a model and a dictionary with pre-processed stock data for a given day,
# returns a dictionary wherein each stock is valued with the proportion of holdings
# that are held in that stock for the day.
def portfolio(model, stocks_data_dict):
    preds = {stock : (model.predict(stocks_data_dict[stock][0]).data[0][-1]) for stock in stocks_data_dict}
    s = sum([math.e ** preds[i] for i in preds if preds[i] > 0])
    port = {stock : (preds[stock] > 0) * (math.e ** preds[stock]) / s for stock in preds}
    return(port)
    
# In a loop, gathers daily data, prepares the day's portfolio, updates total fund
# value based on performance for the day, and repeats. Returns the fund's value over time.
def trade(model, stocks, days = settings.days_per_sim):
    data = {}
    for stock in stocks:
        with open(settings.data_directory + '/' + stock + '.us.txt', 'r') as file:
            reader = csv.reader(file)
            data[stock] = [line for line in reader][1:]
    v = [1]
    for i in range(days, 0, -1):
        stock_data_dict = {stock : data_gathering.prep_(data[stock][-(i + settings.length_of_example):-i]) for stock in stocks}
        port = portfolio(model, stock_data_dict)
        s = 0
        for stock in port:
            s += port[stock] * (float(data[stock][(-i)+1][4]) / float(data[stock][-i][4]))
        if not s:
            s = 1
        v.append(float(v[-1] * s))
    return(v[:-1])
   
# Makes a pretty graph
def plot_fund_value(vals):
    matplotlib.pyplot.plot(vals, label = "Returns Relative to Initial Capital")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("Trading Days Elapsed")
    matplotlib.pyplot.ylabel("Relative Fund Value (x-times initial capital)")
    matplotlib.pyplot.title("Simulated Fund Value Over Time")