import gdax
from datetime import timedelta, date
from datetime import datetime as DT
from time import time, sleep
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *

def getHistoricRate(crypto='BTC', currency='USD',  granularity=200, start=None, end=None):
    client = gdax.PublicClient()
    call_letters = str(crypto + '-' + currency)
    print(call_letters, start, end)
    if start is None:
        if end is None:
            data = client.get_product_historic_rates(call_letters, granularity=granularity)
        else:
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
        day1 = str(day)
        day2 = str(day + timedelta(1))
        data = getHistoricRate(crypto, currency, granularity=granularity, start=day1, end=day2)
        print(data)
        try:
            data = [x.insert(0, day1) for x in data]
        except Exception as error:
            print(error)
        Historic_data += data
    return Historic_data


class OrderBook(object):
    def __init__(self, currencyPair='ETH-BTC'):
        self.client = gdax.PublicClient()
        self.currency_pair = currencyPair
        self.plt = plt
        self.time = time()

        self.update(1)

    def update(self, Hz):
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
            try:
                tempRow.append(float(order[1]) + volume_data_list[-1][-1])
            except:
                tempRow.append(float(order[1]))
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



    def plot(self, fraction=1, minprice=None, maxprice=None):
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
        labels = ['buys', 'sells']
        lines, _ = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')
        return fig, line1, line2


    def updatePlot(self, line1, line2, fig, fraction=.005):
        self.updateOrderBook()
        fraction = .02
        buys = self.buys[self.buys.price >= self.buys.price.min() + self.buys.price.max() * (1 - fraction)]
        sells = self.sells[self.sells.price <= self.sells.price.min() + self.buys.price.max() * fraction]
        line1.set_data(buys.price, buys.cumulative_volume)
        line2.set_data(sells.price, sells.cumulative_volume)
        # fig.canvas.draw()
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


if __name__=='__main__':
    o = OrderBook('BTC-USD')
    o.plot().show()