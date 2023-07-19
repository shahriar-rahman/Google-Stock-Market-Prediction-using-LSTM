import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class GoogleStock:
    def __init__(self):
        self.train = pd.read_csv('../dataset/Google_Stock_Price_Train.csv')  # keras only takes numpy array

    def stock_price(self):
        print(self.train.head(10))

        df_open = self.train.loc[:, ["Open"]].values
        print(df_open)

        plt.plot(df_open)
        plt.xlabel("Time")
        plt.ylabel("Open")
        plt.title("Stock Price")
        plt.show()

        # Preprocessing Data
        # Reshape
        data = df_open.reshape(-1, 1)
        data = data.astype("float32")
        print(data.shape)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

        train_size = int(len(data) * 0.7)
        test_size = len(data) - train_size

        train = data[0:train_size, :]
        test = data[train_size:len(data), :]
        print("train size: {}, test size: {} ".format(len(train), len(test)))

        time_steps = 5
        x_data = []
        y_data = []

        for i in range(len(train) - time_steps - 1):
            a = train[i:(i + time_steps), 0]
            x_data.append(a)
            y_data.append(train[i + time_steps, 0])

        x_train = np.array(x_data)
        y_train = np.array(y_data)

        x_data = []
        y_data = []

        for i in range(len(test) - time_steps - 1):
            a = test[i:(i + time_steps), 0]
            x_data.append(a)
            y_data.append(test[i + time_steps, 0])

        x_test = np.array(x_data)
        y_test = np.array(y_data)

        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        model = Sequential()
        model.add(LSTM(100, input_shape=(1, time_steps)))  # 50 LSTM neuron (block)

        model.add(Dense(units=1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(x_train, y_train, epochs=100, batch_size=1)

        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        # invert predictions
        train_pred = scaler.inverse_transform(train_pred)
        y_train = scaler.inverse_transform([y_train])
        test_pred = scaler.inverse_transform(test_pred)
        y_test = scaler.inverse_transform([y_test])

        # calculate root mean squared error
        train_score = math.sqrt(mean_squared_error(y_train[0], train_pred[:, 0]))
        print("Train Score: %.2f " % (train_score))

        test_score = math.sqrt(mean_squared_error(y_test[0], test_pred[:, 0]))
        print("Test Score: %.2f " % (test_score))

        train_pred_plot = np.empty_like(data)
        train_pred_plot[:, :] = np.nan
        train_pred_plot[time_steps:len(train_pred) + time_steps, :] = train_pred

        # shifting test predictions for plotting
        test_prep_plot = np.empty_like(data)
        test_prep_plot[:, :] = np.nan
        test_prep_plot[len(train_pred) + (time_steps * 2) + 1:len(data) - 1, :] = test_pred

        # plot base line and predictions
        plt.plot(scaler.inverse_transform(data))
        plt.plot(train_pred_plot)
        plt.plot(test_prep_plot)
        plt.show()


if __name__ == "__main__":
    main = GoogleStock()
    main.stock_price()

