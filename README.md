===========================================================================
# Prediction of Google Stock Markets using LSTM
In this research, an LSTM-based Recurrent Neural Network (RNN) is constructed to forecast the Google stock price.

</br>

![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/img/stocks.jpg)

</br>

### ◘ Introduction 
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

</br></br>

### ◘ Model Summary
![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/img/model_summary.JPG)

</br></br>

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

</br></br>

### ◘ Model Evaluation based on the data type
Root Mean Squared Error (RMSE)
| **Training Set** | **Test Set** | 
|--|--|
| 0.028 | 0.023 |

</br>

![alt text](https://github.com/shahriar-rahman/Google-Stock-Market-Prediction-using-LSTM/blob/main/graphs/Google_stock_prediction.png)

<br/><br/>

### ◘ Project Organization
------------
    ├─-- LICENSE                # MIT License
    |
    ├─-- README.md              # The top-level README for developers using this project
    |
    ├─-- dataset                # Contains different type of dataset (e.g. train and test)
    |    └──  processed        
    |    └──  raw
    |
    |
    ├─-- models                 # Trained and serialized models for future model predictions  
    |    └── lstm.pkl
    |
    |
    ├─ graphs                    # Generated graphics and figures obtained from visualization.py
    |
    |
    ├─-- img                    # Project related files
    |
    ├─-- requirements.txt       # The requirements file for reproducing the analysis environments
    |                         
    |
    ├─-- src                    # Source code for use in this research
    |   └───-- __init__.py    
    |   |
    |   ├─-- models                # Contains py filess for Stock Market Price Prediction          
    |   |   └─── google_stock_lstm.py
    |   |
    |   |
    |   └───-- visualization        # Construct visualizations to identify, evaluate, and compare predicted and actual results
    |       └───-- visualize.py
    |
    ├─
--------   
    
</br></br>

### ◘ Library Installation (using pip)
In order to *install* the required packages on the local machine, Open pip and run the following commands separately:
```
> pip install tensorflow                    

> pip install keras     

> pip install pandas                                                          

> pip install scikit-learn                                      

> pip install matplotlib

> pip install numpy

> pip install joblib                                  
```

<br/><br/>


### ◘ Supplementary Resources
For more details, visit the following links:
* https://pypi.org/project/tensorflow/
* https://pypi.org/project/keras/
* https://pypi.org/project/pandas/
* https://pypi.org/project/scikit-learn/
* https://pypi.org/project/matplotlib/
* https://pypi.org/project/numpy/
* https://pypi.org/project/joblib/

<br/><br/>

### ◘ MIT License
Copyright (c) 2023 Shahriar Rahman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

===========================================================================
