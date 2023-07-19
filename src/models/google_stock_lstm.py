import pandas as pd
import numpy as np
import warnings
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import joblib as jb
from sklearn.preprocessing import MinMaxScaler
import os
import sys
sys.path.append(os.path.abspath('../visualization'))
warnings.filterwarnings("ignore")
import visualize


class GoogleStock:
    def __init__(self):
        self.train_data = pd.read_csv('../../dataset/Google_Stock_Price_Train.csv')
        self.test_data = pd.read_csv("../../dataset/Google_Stock_Price_Test.csv")
        self.visualize = visualize.Visualize()

    def stock_price(self):
        # Review Training Data
        print(self.train_data.head(10))

        # Reformat 'Date'
        stock_train = self.train_data[['Date', 'Open']]
        stock_train['Date'] = pd.to_datetime(self.train_data['Date'].apply(lambda x: x.split()[0]))
        stock_train.set_index('Date', drop=True, inplace=True)
        stock_train.head()

        self.visualize.plot_line(stock_train['Open'], 'Open', 'Stock Price for Training data', 'Date', 'Price')
        stock_train.info()
        print(stock_train.shape)

        # Normalization Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(stock_train)
        print(scaled_train)

        # Constructing a sliding window for training set
        x_train = []
        y_train = []
        for i in range(60, 1258):
            x_train.append(scaled_train[i - 60:i, 0])
            y_train.append(scaled_train[i, 0])

        # Numpy array conversion for Keras
        x_train, y_train = np.array(x_train), np.array(y_train)
        print("Shapes of X & Y before Reshaping", x_train.shape, y_train.shape)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        print("Shapes of X after Reshaping", x_train.shape)

        # Construct Sequence Models
        model = Sequential()

        model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=40, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=40))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        model.fit(x_train, y_train, epochs=150, batch_size=64)

        # Error Calculation for Training Data
        y_predicted = model.predict(x_train)
        train_score = math.sqrt(mean_squared_error(y_train, y_predicted))
        print("Train Score (RMSE): ", train_score)

        # Store Model
        try:
            jb.dump(model, '../../models/lstm.pkl')

        except Exception as exc:
            print("! Exception encountered", exc)

        else:
            print("• Model saved successfully", '')
            
        # Prepare Test data
        stock_actual = self.test_data.iloc[:, 1:2].values
        print(stock_actual)

        # Concatenate Train & Test data  
        dataset_total = pd.concat((stock_train["Open"], self.test_data["Open"]), axis=0)
        inputs = dataset_total[len(dataset_total) - len(self.test_data) - 60:].values

        # Transformation for the Test data
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        # Sliding window for the Test set
        x_test = []
        y_test = []
        for i in range(60, 80):
            x_test.append(inputs[i - 60:i, 0])
            y_test.append(inputs[i, 0])

        # Numpy Conversion
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Prediction on the Test data
        stock_predicted = model.predict(x_test)

        # Error Calculation for Test Data
        test_score = math.sqrt(mean_squared_error(y_test, stock_predicted))
        print("Test Score: (RMSE)", test_score)

        # De-scaling
        stock_predicted = scaler.inverse_transform(stock_predicted)
        print(stock_predicted)

        self.visualize.plot_graph(stock_actual, stock_predicted, "Google Stock Price Prediction",
                                  'Time', "Google Stock Price")


if __name__ == "__main__":
    main = GoogleStock()
    main.stock_price()

