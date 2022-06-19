from shutil import rmtree
from os import path
from os.path import exists
from parsing_data import get_data_to_ticker_list
from Models import MarkovModel
from Models import SharpModel


def is_exists():
    if exists("graphs"):
        rmtree(path.join(path.abspath(path.dirname(__file__)), 'graphs'))


def model_calculations(tickers, date_start, date_end):
    df_close = get_data_to_ticker_list(tickers, date_start, date_end)
    MarkovModel(df_close).make_graphs_print_result()
    SharpModel(df_close).print_result(df_close)


def main():
    """ list of tickers from moex exchange """
    tickers = []
    date_start, date_end = '', ''
    is_exists()
    model_calculations(tickers, date_start, date_end)


if __name__ == '__main__':
    main()
