from pandas import DataFrame
import pandas_datareader as pdr


def get_data_moex_ticker(ticker, date_start, date_end) -> list:
    if type(ticker) is not str or type(date_start) is not str or type(date_end) is not str:
        print('Type Error! You enter not str for function.')
        exit(1)
    try:
        prices = []
        df = pdr.data.DataReader(ticker, 'moex', date_start, date_end)['CLOSE']
        for price in df:
            prices.append(price)
        return prices
    except ImportError:
        assert False, 'Bad try to parsing data. Check your ticker name...'


def get_data_to_ticker_list(tickers, date_start, date_end) -> DataFrame:
    close_prices = DataFrame(columns=tickers)
    for ticker in tickers:
        close_prices[ticker] = get_data_moex_ticker(ticker, date_start, date_end)
    return close_prices
