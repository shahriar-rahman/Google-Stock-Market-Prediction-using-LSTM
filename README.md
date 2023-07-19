===========================================================================
# Prediction of Google Stock Markets using LSTM
In this research, an LSTM-based Recurrent Neural Network (RNN) is constructed to forecast the Google stock price.

</br>

![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/img/stocks.jpg)

</br>

### ◘ Introduction:
Stock market prices are driven by profit expectations or corporate earnings. If a trader believes that the earnings of their respective company are inflating or will ascend further, they will raise the base price of the stock. One of the most common ways for shareholders is to buy low stocks and sell them at high prices to get a higher return on their investments. As a result, if the company performs poorly and the value of the stock plummets, the shareholder will lose some or all of their investments at the time of sale. Therefore, it is imperative to have an accurate stock price predictive system to infer better information about the future price by analyzing historical prices.

</br>

![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/graphs/stock_price_train_line.png)

</br>

### ◘ Objective
The primary incentive of this research is to:
* Analyze the time series data.
* Create a data structure that is required to cover 60-time stamps, based on which the LSTM would predict the price of the 61st sample.
* Prepare the training data
* Build the Sequential Model
* Fit and evaluate the Training Model
* Evaluate the Test Model
* Make logical comparisons on the actual vs predicted estimations.
* 
</br>

### ◘ Model Summary
![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/img/model_summary.JPG)

</br>

### ◘ Approach
This study is partitioned into x Steps:
1. Make observations on the initial training data
2. Use the MinMax Scaling technique to preserve the shape of the original distribution
3. Constructing a sliding window for the training and the test sets
4. Construct the Sequence Models
5. Reshape the data and feed it to the LSTM Network
6. Calculate the Training Error
7. Store the Model
8. Predict the Test set and calculate the loss
9. De-scale the data and compare the Stock Price predictions

</br></br>

### ◘ LSTM Networks
![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/img/LSTM%20model.png)


### ◘ Model Evaluation based on the data type
Root Mean Squared Error (RMSE)
| **Training Set** | **Test Set** | 
|--|--|
| 0.028 | 0.023 |
</br>
![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/graphs/Google_stock_prediction.png)

<br/><br/>

===========================================================================
