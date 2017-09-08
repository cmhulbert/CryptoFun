import gdax
from datetime import timedelta, date
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt

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
        # self.time =

        self.update(1)

    def update(self, Hz):
        # while True:
        self.updateOrderBook()
            # sleep(Hz)


    def formatOrderbook(self, orderbook):
        cumulative_volume_data_list = []
        tempRow = []
        for order in orderbook:
            tempRow = []
            tempRow.append(float(order[0]))
            try:
                tempRow.append(float(order[1]) + cumulative_volume_data_list[-1][-1])
            except:
                tempRow.append(float(order[1]))
            cumulative_volume_data_list.append(tempRow)
        cumulative_volume_data = pd.DataFrame(cumulative_volume_data_list, columns=['price', 'cumulative_volume'])
        return cumulative_volume_data

    def updateOrderBook(self):
        orderBook = self.client.get_product_order_book(self.currency_pair, level=3)
        self.sells = self.formatOrderbook(orderBook['asks'])
        self.buys = self.formatOrderbook(orderBook['bids'])


    def plot(self, minprice=None, maxprice=None, fraction=1):
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
        ax.plot(buys.price, buys.cumulative_volume)
        ax.plot(sells.price, sells.cumulative_volume)
        return fig


if __name__=='__main__':
    o = OrderBook('BTC-USD')
    o.plot().show()