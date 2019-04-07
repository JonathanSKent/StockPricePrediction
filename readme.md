######################
STOCK PRICE PREDICTION
######################

This project was done more or less as an exercise in PyTorch, as practice for
implementing and using neural networks for analyses.

HOW DOES IT WORK?

Fairly low-level descriptions of each element of the functionality is found
commented ahead of each function in the code itself. Here, I will provide a 
high-level overview of how the whole thing works.

It begins with a dataset stored in a directory, wherein each folder represents
the stock price history of a given stock. The dataset I used was obtained from
Kaggle, but in the past I have collected my own dataset using tools like
urllib2 and BeautifulSoup.

This dataset is collected into random training epochs by data_gathering.py. A
neural net is initialized randomly by neural_net.py, and is then trained via
gradient descent on these epochs to yield levels of confidence in the stock
going up or down.

Using trading.py, it simulates a fund managed according to the predictions made
by the model; no capital is invested in stocks predicted to go down, and the
entirety of the capital is apportioned according to the softmax of the closing
price delta confidences. The code then updates the value of the fund, and
then produces a nice graph.

Having the fund trade on a handful of blue chip tech stocks yielded a simulated
return of ~25% over 200 days, or ~+.1% fund capital per day.

I would caution any would-be financiers against using my model in order to
decide what stocks to buy, on the basis that these results are extremely limited
in their generalizability.
