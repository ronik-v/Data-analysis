import tensorflow as tf
import matplotlib.pyplot as plt
from yfinance import download
from numpy import cov, var, std, array, arange
from math import log1p
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, Activation, Dropout


def parsing_data(ticker, date_start, date_end):
    try:
        df = download(ticker, date_start, date_end)
        df['SMA5 Open'] = df['Open'].rolling(5).mean()
        df['SMA12 Open'] = df['Open'].rolling(12).mean()
        df['EWMA'] = df['Open'].ewm(com=5).mean()
        return df
    except ImportError:
        assert False, 'bad try to pars dataframe'


class RegressionModels:
    def __init__(self, prices):
        self.prices = prices

    def regress_coefficient_log(self, prices):
        data_x = []
        for i in range(0, len(prices)):
            data_x.append(log1p(prices[i]))
        data_y = list(range(0, len(data_x)))
        b = cov(data_x, data_y)[0][1] / var(data_x)
        a = (sum(data_y) / len(data_y)) - b * (sum(data_x) / len(data_x))
        return a, b

    def regress_coefficient_deg(self, prices):
        data_x = []
        for i in range(0, len(prices)):
            data_x.append(prices[i])
        data_y = list(range(0, len(data_x)))
        b = cov(data_x, data_y)[0][1] / var(data_x)
        a = (sum(data_y) / len(data_y)) - b * (sum(data_x) / len(data_x))
        return a, b

    def LogRegress(self, a, b, t, Std):
        return a + b*log1p(t) + Std

    def DegRegress(self, a, b, t, Std):
        return a + t**b - Std * 2.5

    def RegressionsPredictions(self, prices):
        a_log, b_log = self.regress_coefficient_log(prices)
        a_deg, b_deg = self.regress_coefficient_deg(prices)
        print("log: ", a_log, b_log)
        print("deg: ", a_deg, b_deg)
        Std = std(prices)
        Log_reg_prediction, Deg_reg_prediction = self.LogRegress(a_log, b_log, len(prices), Std), self.DegRegress(a_deg, b_deg, len(prices), Std)
        return Log_reg_prediction, Deg_reg_prediction


class DataPreparation:
    def __init__(self, ticker, train_date_start, train_date_end, prediction_date_start, prediction_date_end):
        self.ticker = ticker
        self.train_date_start = train_date_start
        self.train_date_end = train_date_end
        self.prediction_date_start = prediction_date_start
        self.prediction_date_end = prediction_date_end
        self.df_train = parsing_data(ticker, train_date_start, train_date_end)
        self.df_prediction = parsing_data(ticker, prediction_date_start, prediction_date_end)

    def get_numbers_prediction(self):
        prices = []
        last_day = len(self.df_prediction) - 1
        for price in self.df_prediction['Open']:
            prices.append(price)
        Log_const, Deg_const = RegressionModels(prices).RegressionsPredictions(prices)
        SMA5, SMA12, EWMA = self.df_prediction['SMA5 Open'][last_day], self.df_prediction['SMA12 Open'][last_day], self.df_prediction['EWMA'][last_day]
        return Log_const, Deg_const, SMA5, SMA12, EWMA

    def get_np_arrays(self):
        train_open, train_close = [], []
        prediction_open, prediction_close = [], []
        for day_index in range(len(self.df_prediction)):
            train_open.append(self.df_train['Open'][day_index])
            train_close.append(self.df_train['Close'][day_index])
            prediction_open.append(self.df_prediction['Open'][day_index])
            prediction_close.append(self.df_prediction['Close'][day_index])
        return array(train_open), array(train_close), array(prediction_open), array(prediction_close)


class NeuronPrediction:
    def __init__(self, train_open, train_close, prediction_open, prediction_close, Log_const, Deg_const, SMA5, SMA12, EWMA):
        self.train_open = train_open
        self.train_close = train_close
        self.prediction_open = prediction_open
        self.prediction_close = prediction_close
        self.Log_const = Log_const
        self.Deg_const = Deg_const
        self.SMA5 = SMA5
        self.SMA12 = SMA12
        self.EWMA = EWMA

    def Neuro_model(self, train_open, train_close, prediction_open, prediction_close, Log_const, Deg_const, SMA5, SMA12, EWMA):
        epoch = 150
        model = Sequential()
        model.add(Dense(16, input_dim=1, activation='relu'))
        model.add(Dropout(0.15))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(1))
        model.add(Activation('relu'))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.65,
                                                         patience=5, min_lr=0.0001)
        model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

        history = model.fit(train_open, train_close,
                            epochs=epoch,
                            batch_size=4,
                            verbose=1,
                            validation_data=(prediction_open, prediction_close),
                            shuffle=True,
                            callbacks=[reduce_lr], use_multiprocessing=True)

        N = arange(0, epoch)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, history.history["loss"], label="loss")
        plt.plot(N, history.history["val_loss"], label="val_loss")
        plt.plot(N, history.history["mean_absolute_error"], label="mean_absolute_error")
        plt.plot(N, history.history["val_mean_absolute_error"], label="val_mean_absolute_error")
        plt.title('Model errors for ticker prediction')
        plt.xlabel("epoch")
        plt.ylabel("Loss/mean_absolute_error")
        plt.legend()
        last_price = prediction_close[len(prediction_close) - 1]

        def prediction(Log_const, Deg_const, SMA5, SMA12, EWMA):
            print("\n")
            print("==============Forecast values for the next day with the calculation of increments==============")
            print("Log regression: ", model.predict([Log_const]), " increment = ", (model.predict([Log_const])/last_price - 1)*100, "%")
            print("Deg regression: ", model.predict([Deg_const]), " increment = ",
                  (model.predict([Deg_const]) / last_price - 1) * 100, "%")
            print("SMA5: ", model.predict([SMA5]), " increment = ", (model.predict([SMA5])/last_price - 1)*100, "%")
            print("SMA12: ", model.predict([SMA12]), " increment = ", (model.predict([SMA12]) / last_price - 1) * 100, "%")
            print("EWMA: ", model.predict([EWMA]), " increment = ", (model.predict([EWMA]) / last_price - 1) * 100, "%")
            print("===============================================================================================")
        prediction(Log_const, Deg_const, SMA5, SMA12, EWMA)
        plt.show()


def main():
    ticker, train_date_start, train_date_end = 'NVDA', '2020-03-08', '2021-09-08'
    prediction_date_start, prediction_date_end = '2021-03-08', '2022-03-08'
    D = DataPreparation(ticker, train_date_start, train_date_end, prediction_date_start, prediction_date_end)
    train_open, train_close, prediction_open, prediction_close = D.get_np_arrays()
    Log_const, Deg_const, SMA5, SMA12, EWMA = D.get_numbers_prediction()
    NP = NeuronPrediction(train_open, train_close, prediction_open, prediction_close, Log_const, Deg_const, SMA5, SMA12, EWMA)
    NP.Neuro_model(train_open, train_close, prediction_open, prediction_close, Log_const, Deg_const, SMA5, SMA12, EWMA)


if __name__ == '__main__':
    main()
