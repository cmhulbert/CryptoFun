# CryptoFun

set up virtual environment and/or install pip requirements
`pip install -r requirements.txt` should work.

###To test Orderbook:

```
import GDAXWrapper as gd
ob = gd.Orderbook()
ob.plot(live=True, fraction=.01)
```
Note, the orderbook live plot can only be canceled currently through a KeyboardInterupt error (Ctrl+C)


### To test the Genetic Algorithm
Note, this currently doesn't actually give better results than doing nothing, but it does work (albeit poorly).
```
import GDAXWrapper as gd
import pandas as pd
import matplotlib.pyplot as plt

## read data set
data_set = pd.read_csv('two_years.tsv', sep='\t')
## split data to training and testing set
cutoff = int((2/3)*len(data_set))
train_set = data_set.iloc[:cutoff]
test_set = data_set.iloc[cutoff:];
## initialize first model
dataframe, params = gd.simulateModel(train_set)
## Run for multiple generations, it will run for max of 100 if no children in any generation improve
previous_best_value = float(dataframe.value.iloc[-1])
best_dataframe, best_params, p = gd.simulateGeneration(train_set, params, previous_best=previous_best_value)
## when done, you can view the results in a plot
best_dataframe.plot(x='time', y=['currency1', 'currency2', 'value'], logy=True)
plt.show()
```

Important note, I occasionally get an overflow error (and you end up with extreme numbers for the value).
I don't currently have a solution (mostly just haven't looked into), but for now, just rerun the gd.simulateGeneration()
line until it doesn't happen. It is fairly rare, in general.

