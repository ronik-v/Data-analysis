import numpy as np
import matplotlib.pyplot as plt
import pypfopt.plotting as pplt

from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
from os import chdir, mkdir
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.cla import CLA


class MarkovModel:
    def __init__(self, df_close):
        self.df_close = df_close
        self.df_close_data = df_close.pct_change()
        self.df_close_mean = self.df_close_data.mean()
        self.cov_matrix = self.df_close_data.cov()
        self.tickers_amount = len(self.df_close.columns)

    def random_portfolio(self):
        result = np.exp(np.random.randn(self.tickers_amount))
        result = result / result.sum()
        return result

    def profitability_of_portfolio(self, random_port):
        return np.matmul(self.df_close_mean.values, random_port)

    def risk_of_portfolio(self, random_port):
        return np.sqrt(np.matmul(np.matmul(random_port, self.cov_matrix.values), random_port))

    def make_graphs_print_result(self):
        mkdir("graphs")
        chdir("graphs")
        iterations = 1000
        risk = np.zeros(iterations)
        doh = np.zeros(iterations)
        portf = np.zeros((iterations, self.tickers_amount))

        for it in range(iterations):
            r = self.random_portfolio()
            portf[it, :] = r
            risk[it] = self.risk_of_portfolio(r)
            doh[it] = self.profitability_of_portfolio(r)

        fig = plt.figure(figsize=(10, 8))
        plt.style.use('seaborn-whitegrid')

        plt.scatter(risk * 100, doh * 100, c='y', marker='.')
        plt.xlabel('риск, %')
        plt.ylabel('доходность, %')
        plt.title("Облако портфелей")

        min_risk = np.argmin(risk)
        plt.scatter([(risk[min_risk]) * 100], [(doh[min_risk]) * 100], c='r', marker='*', label='минимальный риск')

        max_sharp_koef = np.argmax(doh / risk)
        plt.scatter([risk[max_sharp_koef] * 100], [doh[max_sharp_koef] * 100], c='g', marker='o',
                    label='максимальный коэффициент Шарпа')

        r_mean = np.ones(self.tickers_amount) / self.tickers_amount
        risk_mean = self.risk_of_portfolio(r_mean)
        doh_mean = self.profitability_of_portfolio(r_mean)
        plt.scatter([risk_mean * 100], [doh_mean * 100], c='b', marker='x', label='усредненный портфель')

        plt.legend()
        fig.savefig('Облако_портфелей.png')

        for ticker in self.df_close.columns:
            fig = plt.figure()
            plt.subplot(2, 1, 1)
            self.df_close[ticker].plot()
            plt.grid(True)
            plt.title(ticker)
            plt.xlabel("Индекс дня")
            plt.ylabel("Цена закрытия")
            plt.subplot(2, 1, 2)
            self.df_close_data[ticker].plot()
            plt.grid(True)
            plt.xlabel("Индекс дня")
            plt.ylabel("Относительные изменения курсов")
            fig.savefig(ticker + '.png')

        def print_result():
            print('============= Портфель по Маркову =============')
            print('============= Минимальный риск =============', "\n")
            print("риск = %1.2f%%" % (float(risk[min_risk]) * 100.))
            print("доходность = %1.2f%%" % (float(doh[min_risk]) * 100.), "\n")
            print(DataFrame([portf[min_risk] * 100], columns=self.df_close.columns, index=['доли, %']).T, "\n")
            print('============= Максимальный коэффициент Шарпа =============', "\n")
            print("риск = %1.2f%%" % (float(risk[max_sharp_koef]) * 100.))
            print("доходность = %1.2f%%" % (float(doh[max_sharp_koef]) * 100.), "\n")
            print(DataFrame([portf[max_sharp_koef] * 100], columns=self.df_close.columns, index=['доли, %']).T, "\n")
            print('============= Средний портфель =============', "\n")
            print("риск = %1.2f%%" % (float(risk_mean) * 100.))
            print("доходность = %1.2f%%" % (float(doh_mean) * 100.), "\n")
            print(DataFrame([r_mean * 100], columns=self.df_close.columns, index=['доли, %']).T, "\n")
            print('=======================================', '\n')

        print_result()


class SharpModel:
    def __init__(self, df_close):
        self.df_close = df_close

    def print_format_portfolio(self, obj):
        for ob in obj.items():
            print(" ", ob[0], " --- ", round(ob[1] * 100, 2), "%")

    def print_result(self, df_close):
        print('============= Портфель по Шарпу =============', '\n')
        mu = expected_returns.mean_historical_return(df_close)
        sigma = risk_models.sample_cov(df_close)
        ef = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))
        sharpe_portfolio = ef.max_sharpe()
        ef.portfolio_performance(verbose=True)
        ef1 = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))
        min_vol = ef1.min_volatility()
        min_vol_pwt = ef1.clean_weights()
        ef1.portfolio_performance(verbose=True, risk_free_rate=0.27)
        cl_obj = CLA(mu, sigma)

        fig, ax = plt.subplots()
        pplt.plot_efficient_frontier(cl_obj, ax=ax, show_assets=True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        fig.savefig('График_эффективных_границ.png', dpi=200)

        print('--------------------------------------')
        self.print_format_portfolio(min_vol)
        print('--------------------------------------')
        self.print_format_portfolio(min_vol_pwt)
        print('--------------------------------------')
        self.print_format_portfolio(sharpe_portfolio)
        print('=======================================', '\n')
