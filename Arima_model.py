from yfinance import download
from statsmodels.tsa.arima.model import ARIMA
from threading import Thread
from pandas import DataFrame
import matplotlib.pyplot as plt


class ArimaModel:
    def __init__(self, ticker, date_start, date_end):
        self.ticker = ticker
        self.date_start = date_start
        self.date_end = date_end
        self.predictions = []

    def pars_data(self, ticker, date_start, date_end):
        try:
            df = download(ticker, date_start, date_end)
            df['EWMA_Close'] = df['Close'].ewm(com=5).mean()
            print(df['EWMA_Close'].tail())
            return df
        except TypeError:
            assert False, 'Enter different types downloads parameters'

    def arima_rea(self, ticker, date_start, date_end):
        df = self.pars_data(ticker, date_start, date_end)
        print(df.columns)
        close_price, ewm_close_price = df['Close'], df['EWMA_Close']
        model = ARIMA(ewm_close_price, order=(1, 0, 1))
        model_fit = model.fit()
        print(model_fit.summary())
        output = model_fit.forecast()
        res = [0] * (len(close_price) - 1)
        res.append(output)
        df['ARIMA'] = res
        self.predictions = DataFrame(res)
        print(self.predictions.describe())
        print('ARIMA prediction = ', output)
        plt.style.use("ggplot")
        plt.plot(close_price, label='Цена закрытия')
        plt.plot(df['Open'], label='Цена открытия')
        plt.scatter(df['ARIMA'].index, df['ARIMA'], label='Прогноз модели ARIMA(1, 0, 1)', color='green')
        plt.title('Прогноз модели для акций Nvidia\nна 2022-03-09')
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    tickers = ['NVDA', 'Z', 'AA']
    list_of_threads = []
    for ticker in tickers:
        Arima = ArimaModel(ticker, '2021-03-07', '2022-03-13')
        th = Thread(target=Arima.arima_rea(ticker, '2021-03-09', '2022-03-09'), args=(tickers.index(ticker),))
        list_of_threads.append(th)
        th.start()

    for th in list_of_threads:
        th.join()


if __name__ == '__main__':
    main()
