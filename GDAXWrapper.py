import gdax
from datetime import timedelta, date
from datetime import datetime as DT
from time import time, sleep
import pandas as pd
from copy import deepcopy
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from pylab import *
from sklearn.svm import SVR


def getHistoricRate(crypto='BTC', currency='USD',  granularity=200, start=None, end=None):
    client = gdax.PublicClient()
    call_letters = str(crypto + '-' + currency)
    # print(call_letters, start, end)
    if start is None:
        data = client.get_product_historic_rates(call_letters, granularity=granularity)
    elif end is None:
        data = client.get_product_historic_rates(call_letters,start=start, granularity=granularity)
    else:
        data = client.get_product_historic_rates(call_letters, start=start, end=end, granularity=granularity)
    return data


def dayRange(start_date, end_date):
    start_date = date(start_date[0], start_date[1], start_date[2])
    end_date = date(end_date[0], end_date[1], end_date[2])
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


def bulkHistoryicalRate(crypto='BTC', currency='USD', start=None, end=None, granularity=200):
    client = gdax.PublicClient()
    dateGen = dayRange(start, end)
    Historic_data = []
    for day in dateGen:
        print(day)
        day1 = str(day)
        day2 = day1 + 'T12:00:00'
        day_rate = getHistoricRate(crypto, currency, granularity=granularity, start=day1, end=day2)
        # print(day_rate)
        if type(day_rate) is list :
            Historic_data += day_rate
        else:
            sleep(.1)
            day_rate = getHistoricRate(crypto, currency, granularity=granularity, start=day1, end=day2)
            if type(day_rate) is list :
                Historic_data += day_rate
            else:
                print(day_rate)
        print(str(day)+'T12:00:00')
        day1 = str(day) + 'T12:00:00'
        day2 = str(day + timedelta(1))
        day_rate = getHistoricRate(crypto, currency, granularity=granularity, start=day1, end=day2)
        # print(day_rate)
        if type(day_rate) is list:
            Historic_data += day_rate
        else:
            sleep(.1)
            day_rate = getHistoricRate(crypto, currency, granularity=granularity, start=day1, end=day2)
            if type(day_rate) is list:
                Historic_data += day_rate
            else:
                print(day_rate)
    dataframe = pd.DataFrame(Historic_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'], index=None)
    dataframe = dataframe.sort_values(by='time')
    return dataframe


def simulateModel(dataframe, window=241, principal=1, topFraction=.001, bottomFraction=.001, tradeFraction=.9, by='open'):
    ## Start with equal value of both currency
    currency1 = principal
    currency2 = principal*dataframe.iloc[0][by]
    currency_change = [[dataframe.iloc[0].time, currency1, currency2]]
    sums = [currency1*dataframe.iloc[0][by] + currency2]
    ## initial fraction to trade with when a condition is met
    tradeFraction = tradeFraction
    previous_mean = None
    ## move through timeseries data one value at a time
    values = [currency1 + currency2 / dataframe.iloc[0][by]]
    norm_values = [(currency2 - dataframe.iloc[0][by]) + ((currency1 * dataframe.iloc[0][by]) - dataframe.iloc[0][by])]
    for i in range(0,int(len(dataframe)-window-1),window):
        if previous_mean is None:
            start = int(1)
            end = int(start + window)
            window_df = dataframe.iloc[start:end]
            window_mean = window_df[by].mean()
            previous_mean = window_mean
            continue
        start = int(i + window +1)
        end = int(start+window)
        window_df = dataframe.iloc[start:end]
        window_mean = window_df[by].mean()
        if window_mean > previous_mean*(1+topFraction):
            mean_exchange_rate = window_df[by].mean()
            last_exchange_rate = window_df[by].iloc[-1]
            ## trade curr1 -> curr2
            new_currency1 = currency1*(1-tradeFraction)
            new_currency2 = currency2 + currency1*tradeFraction*mean_exchange_rate
            #tradeFraction = tradeFraction*.9
            values.append(new_currency1 + new_currency2 / float(last_exchange_rate))
            norm_values.append((new_currency2 - last_exchange_rate) +((new_currency1 * last_exchange_rate) - last_exchange_rate))
            currency_change.append([window_df.iloc[-1].time, new_currency1, new_currency2])
            sums.append(new_currency1 * window_df.iloc[0][by] + new_currency2)
            currency1 = new_currency1
            currency2 = new_currency2
        elif window_mean < previous_mean*(1-bottomFraction):
            mean_exchange_rate = window_df[by].mean()
            last_exchange_rate = window_df[by].iloc[-1]
            ## trade curr@ -> curr1
            new_currency2 = currency2 * (1 - tradeFraction)
            new_currency1 = currency1 + currency2*tradeFraction*(1.0/mean_exchange_rate)
            #tradeFraction = tradeFraction + (1-tradeFraction)*.1
            values.append(new_currency1 + new_currency2 / float(last_exchange_rate))
            norm_values.append(
                (new_currency2 - last_exchange_rate) + ((new_currency1 * last_exchange_rate) - last_exchange_rate))
            currency_change.append([window_df.iloc[-1].time, new_currency1, new_currency2])
            sums.append(new_currency1 * window_df.iloc[0][by] + new_currency2)
            currency1 = new_currency1
            currency2 = new_currency2
        else:
            pass
    params = {'window':window, 'topFraction':topFraction, 'bottomFraction':bottomFraction, 'tradeFraction':tradeFraction}
    values_df = pd.DataFrame(values)
    sums_df = pd.DataFrame(sums)
    norm_vals_df = pd.DataFrame(norm_values)
    currency_change_df = pd.DataFrame(currency_change, columns=['time', 'curr1', 'curr2'])
    currency_change_df['value'] = values_df
    currency_change_df['sums'] = sums_df
    currency_change_df['norm'] = norm_vals_df
    return currency_change_df, params


def simulateGeneration(dataframe, params, generationSize=10, previous_best = None, col_to_max='norm'):
    if previous_best is None:
        best_max = pd.DataFrame([0.0])
        best_sums = pd.DataFrame([0.0])
    best_local_max = 0
    best_local_parameters = None
    best_parameters = None
    # params['window'] = 100
    best_currency_change = None
    for i in range(generationSize):
        print("New Generation! {}".format(i))
        NextGen = True
        sibling_count = 0
        local_sums = pd.DataFrame([0.0])
        while NextGen:
            sibling_count += 1
            mutated_params = deepcopy(params)
            # to_change = np.random.choice(list(mutated_params.keys()))
            # mutated_params[to_change] = params[to_change] + np.random.choice([-1, 1]) * np.random.random()* .5 * params[to_change]
            for j in params.keys():
                mutate = np.random.choice([0,1])
                if mutate:
                    mutated_params[j] = params[j] + np.random.choice([-1,1])*np.random.random()*params[j]
            if mutated_params['window'] < 67:
                mutated_params['window'] = 67
            # mutated_params['window'] = 241#np.random.choice([400,500,600])
            # mutated_params['window'] = np.random.choice([int(x*params['window']) for x  in [.5, .6, .7, .8, .9 , 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]])
            currency_change, new_params = simulateModel(dataframe, window=mutated_params['window'],
                                           topFraction=mutated_params['topFraction'],
                                           bottomFraction=mutated_params['bottomFraction'],
                                           tradeFraction=mutated_params['tradeFraction'])
            max_column = currency_change[col_to_max]
            if previous_best is None:
                best_max.iloc[-1] = max_column.iloc[-1]
                previous_best = 0.0
            # values = currency_change['value']
            sums = currency_change['sums']
            print("Sibling %s: %s %s" % (sibling_count, round(float(max_column.iloc[-1]),5), round(float(best_max.iloc[-1]), 5)))
            if float(max_column.iloc[-1]) > float(best_max.iloc[-1]):
            # if float(values.mean()) > float(best_values.mean()):
            #     print(sums.iloc[-1], best_sums.iloc[-1])
                best_parameters = new_params
                params = new_params
                best_max = pd.DataFrame(max_column)
                best_currency_change = currency_change
                NextGen = False
            if sibling_count > 99:
                print(' Reached Sibling Max Limit.')
                return best_currency_change, best_parameters
    return best_currency_change, best_parameters


def SupportVectorRegression(xdata, ydata, kernel='rbf', C=1e3, gamma=False, testx=None, testy=None):
    from time import time
    t1 = time()
    if not gamma:
        svr = SVR(kernel=kernel, C=C)
    else:
        svr = SVR(kernel=kernel, C=C, gamma=gamma)
    svr.fit(xdata, ydata)
    prediction = svr.predict(xdata)
    print(time() - t1)
    plt.subplot(211)
    plt.scatter(xdata, ydata, color='black', label='Training Data')
    plt.plot(xdata, prediction, color = 'red', label = '{} Training'.format(kernel))
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    # plt.show()
    if testx is not None and testy is not None:
        plt.subplot(212)
        plt.scatter(testx, testy, color='black', label="Test Data")
        plt.plot(testx, svr.predict(testx), color='red', label = '{} Validation'.format(kernel))
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
    plt.show()
    return svr, plt


class WebsocketClient(object):
    def __init__(self, currencyPair='ETH-BTC'):
        self.client = gdax.WebsocketClient(url="wss://ws-feed.gdax.com", products=currencyPair)

class OrderBook(object):
    def __init__(self, currencyPair='ETH-BTC'):
        self.client = gdax.PublicClient()
        self.currency_pair = currencyPair
        self.plt = plt
        self.time = time()

        self.update(1)

    def update(self, Hz=None):
        # while True:
        self.updateOrderBook()
            # sleep(Hz)

    def timeStamptoUTCDateTime(self, timestamp):
        date_time = DT.utcfromtimestamp(timestamp)
        return date_time.__str__()


    def formatOrderbook(self, orderbook):
        volume_data_list = []
        tempRow = []
        for order in orderbook:
            tempRow = []
            tempRow.append(float(order[0]))
            if len(volume_data_list) == 0:
                tempRow.append(float(order[1]))
            else:
                tempRow.append(float(order[1]) + volume_data_list[-1][-2])
            tempRow.append(float(order[1]))
            volume_data_list.append(tempRow)
        volume_data = pd.DataFrame(volume_data_list, columns=['price', 'cumulative_volume', 'volume'])
        volume_data['time'] = self.time
        return volume_data

    def updateOrderBook(self):
        self.time = self.timeStamptoUTCDateTime(time())
        orderBook = self.client.get_product_order_book(self.currency_pair, level=3)
        self.sells = self.formatOrderbook(orderBook['asks'])
        self.buys = self.formatOrderbook(orderBook['bids'])
        self.aggregated = pd.DataFrame(self.client.get_product_order_book(self.currency_pair, level=2), columns=['bids', 'asks', 'sequence'])
        self.bids = pd.DataFrame(list(self.aggregated.bids), columns=['price', 'volume', 'orders'])
        self.asks = pd.DataFrame(list(self.aggregated.asks), columns=['price', 'volume', 'orders'])



    def plot(self, fraction=1, minprice=None, maxprice=None, log=True, live=False):
        plt.ion()
        fig, ax = plt.subplots()
        if fraction== 'all':
            ax.plot(self.buys.price, self.buys.cumulative_volume)
            ax.plot(self.sells.price, self.sells.cumulative_volume)
        else:
            if (minprice is None or maxprice is None):
                buys = self.buys[self.buys.price >= self.buys.price.min() + self.buys.price.max()*(1-fraction)]
                sells = self.sells[self.sells.price <= self.sells.price.min() + self.buys.price.max()*fraction]
            else:
                buys = self.buys[self.buys.price >= minprice]
                sells = self.sells[self.sells.price <= maxprice]
        # return buys, sells
        line1, = ax.plot(sells.price, sells.cumulative_volume, c='red')
        line2, = ax.plot(buys.price, buys.cumulative_volume, c='green')
        # ax.fill(buys.price, buys.cumulative_volume, 'green', sells.price, sells.cumulative_volume, red')
        labels = ['offers', 'bids']
        lines, _ = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')
        if log:
            plt.yscale('log')
        if live:
            while True:
                try:
                    line1, line2, fig = self.updatePlot(line1, line2, fig, fraction=fraction)
                except KeyboardInterrupt:
                    break;
        else:
            return fig, line1, line2


    def updatePlot(self, line1, line2, fig, fraction=.005):
        self.updateOrderBook()
        # fraction = .02
        if fraction != 'all':
            buys = self.buys[self.buys.price >= self.buys.price.min() + self.buys.price.max() * (1 - fraction)]
            sells = self.sells[self.sells.price <= self.sells.price.min() + self.buys.price.max() * fraction]
        else:
            buys = self.buys
            sells = self.sells
        line1.set_data(buys.price, buys.cumulative_volume)
        line2.set_data(sells.price, sells.cumulative_volume)
        fig.canvas.draw()
        return line1, line2, fig




    def logVolumeData(self, outputFile, log_time=60, loops=10):
        prev_df = None
        try:
            while True:
                print('starting')
                log_list = []
                for i in range(loops):
                    self.updateOrderBook()
                    log_list.append([self.buys, self.sells])
                    sleep(log_time)
                log_df = pd.DataFrame(log_list, columns=['buys', 'sells'])
                if prev_df is None:
                    log_df.to_csv(outputFile+'.tsv', sep='\t', index=False)
                else:
                    log_df = prev_df.append(log_df)
                    log_df.to_csv(outputFile + '.tsv', sep='\t', index=False, header=True)
                prev_df = log_df.copy(deep=True)
                print('Done!')
        except KeyboardInterrupt as e:
            log_df = pd.DataFrame(log_list, columns=['buys', 'sells'])
            if prev_df is None:
                log_df.to_csv(outputFile + '.tsv', sep='\t', index=False)
            else:
                log_df = prev_df.append(log_df)
                log_df.to_csv(outputFile + '.tsv', sep='\t', index=False)
            print('Logging Canceled"')

def aggregatePreviousWindow(data, window):
    data2 = []
    for i in range(window, len(data)):
        data2.append(data.iloc[i-window:i].mean())
    data2_df = pd.DataFrame(data2)
    return data2_df


def main(test=False):
    o = pd.read_csv('two_years.tsv', sep='\t')
    o = o[o.time > int(o[o.open > .04].iloc[0].time)]
    o_train = o.iloc[:int(len(o) * (1 / 2))]
    o_validate = o.iloc[int(len(o) * (1 / 2)) + 1:]
    time = o_train.time
    time = np.array(time)
    time = time.reshape(-1,1)
    time = time/time.mean()
    time_filt = [time[x] for x in range(len(time)) if x%100==0]
    price_df = o_train.open
    price = np.array(price_df)
    price = price.reshape(-1,1)
    price = price/price.mean()
    price_filt = [price[x] for x in range(len(price)) if x%100==0]
    # price_ag = aggregatePreviousWindow(price_df, window=50)
    # price_ag = np.array(price_ag)
    # price_ag = price_ag.reshape(-1, 1)
    # price_ag = price_ag / price_ag.mean()
    # mod = int(len(price_ag)/len(price_filt))
    price_ag_filt = [price[x-window:x].mean() for x in range(len(price_ag)) if x % mod == 0]
    if len(price_filt) != len(price_ag_filt):
        price_filt = price_filt[:len(price_ag_filt)]
    if test:
        validate_time = o_validate.time
        validate_time = np.array(validate_time)
        validate_time = validate_time.reshape(-1, 1)
        validate_time = validate_time / validate_time.mean()
        validate_time_filt = [validate_time[x] for x in range(len(validate_time)) if x % 100 == 0]
        validate_price_df = o_validate.open
        validate_price = np.array(validate_price_df)
        validate_price = validate_price.reshape(-1, 1)
        validate_price = validate_price / validate_price.mean()
        validate_price_filt = [validate_price[x] for x in range(len(validate_price)) if x % 100 == 0]

        validate_price_ag = aggregatePreviousWindow(validate_price_df, window=50)
        validate_price_ag = np.array(validate_price_ag)
        validate_price_ag = validate_price_ag.reshape(-1, 1)
        validate_price_ag = validate_price_ag / validate_price_ag.mean()
        mod = int(len(validate_price_ag) / len(validate_price_filt))
        validate_price_ag_filt = [validate_price_ag[x] for x in range(len(validate_price_ag)) if x % mod == 0]
        if len(validate_price_filt) != len(validate_price_ag_filt):
            validate_price_filt = validate_price_filt[:len(validate_price_ag_filt)]

        svr, plt = SupportVectorRegression(price_ag_filt, price_filt, 'rbf', .1, 1e6, testx=validate_price_ag_filt, testy=validate_price_filt)
    else:
        svr, plt = SupportVectorRegression(time_filt, price_filt, 'rbf', 1, 10000)

if __name__=='__main__':
    main(True)
    # ob = OrderBook()
    # bulkHistoryicalRate('ETH', 'BTC', (2016,1,1), (2016,1,2), granularity=100)
    # getHistoricRate(start='2015-01-01T12:00:00')
