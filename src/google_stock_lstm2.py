import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")


class GoogleStock:
    def __init__(self):
        self.train_data = pd.read_csv('../dataset/Google_Stock_Price_Train.csv')  # keras only takes numpy array
        self.test_data = pd.read_csv("../dataset/Google_Stock_Price_Test.csv")

    def stock_price(self):
        print(self.train_data.head(10))

        stock_data = self.train_data[['Date', 'Open']]
        stock_data['Date'] = pd.to_datetime(self.train_data['Date'].apply(lambda x: x.split()[0]))
        stock_data.set_index('Date', drop=True, inplace=True)
        stock_data.head()

        plt.figure(figsize=(20, 7))
        plt.plot(stock_data['Open'], label='Open', color='indigo')
        plt.xlabel('Date', size=15)
        plt.ylabel('Price', size=15)
        plt.legend()
        plt.show()

        stock_data.info()
        print(stock_data.shape)

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_training_set = scaler.fit_transform(stock_data)
        print(scaled_training_set)

        X_train = []
        y_train = []

        for i in range(60, 1258):
            X_train.append(scaled_training_set[i - 60:i, 0])
            y_train.append(scaled_training_set[i, 0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(X_train.shape, y_train.shape)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        print(X_train.shape)

        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.summary()

        model.fit(X_train, y_train, epochs=100, batch_size=32)

        actual_stock_price = self.test_data.iloc[:, 1:2].values
        print(actual_stock_price)

        dataset_total = pd.concat((stock_data["Open"], self.test_data["Open"]), axis=0)
        inputs = dataset_total[len(dataset_total) - len(self.test_data) - 60:].values

        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(60, 80):
            X_test.append(inputs[i - 60:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_prices = model.predict(X_test)
        predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)
        print(predicted_stock_prices)

        plt.figure(figsize=(9, 7))
        plt.plot(actual_stock_price, color='red', label='Actual Google Stock Price')
        plt.plot(predicted_stock_prices, color="blue", label="Predicted Google Stock Price")
        plt.title("Google Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Google Stock Price")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main = GoogleStock()
    main.stock_price()

