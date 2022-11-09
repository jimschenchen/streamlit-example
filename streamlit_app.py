from sklearn.preprocessing import MinMaxScaler
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


stock_name = "MSFT"
focus_column = "Open"
model_path = os.path.join("models", stock_name)
data_path = os.path.join("data", f"{stock_name}.csv")
retrain = False

"""Load training data set with the "Open" and "High" columns to use in our modeling."""

# url = 'https://raw.githubusercontent.com/jimschenchen/public-storage/main/data/MSFT.csv'
url = data_path
dataset = pd.read_csv(url)
dataset_train = dataset[:-200]
dataset_test = dataset[-200:]

sc = MinMaxScaler(feature_range=(0, 1))

training_set = dataset_train.iloc[:, 1:2].values

"""Import MinMaxScaler from scikit-learn to scale our dataset into numbers between 0 and 1 """

training_set_scaled = sc.fit_transform(training_set)


if os.path.exists(model_path) and not retrain:
    print(f"load model from {model_path}")
    model = keras.models.load_model(model_path)


"""Import the test set for the model to make predictions on """

# url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
# dataset_test = pd.read_csv(url)
real_stock_price = dataset_test.iloc[:, 1:2].values
real_stock_price.size


"""Before predicting future stock prices, we have to manipulate the training set; we merge the training set and the test set on the 0 axis, set the time step to 60, use minmaxscaler, and reshape the dataset as done previously. After making predictions, we use inverse_transform to get back the stock prices in normal readable format.

"""

dataset_total = pd.concat((dataset_train[focus_column], dataset_test[focus_column]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 60 + real_stock_price.size):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

"""Plot our predicted stock prices and the actual stock price"""

plt.plot(real_stock_price, color='black', label='TATA Stock Price')
plt.plot(predicted_stock_price, color='green', label='Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()

"""Next day"""

# dataset_total = dataset_test[focus_column]
# new_data = [228.70] # 11-08
# inputs = dataset_total[len(dataset_test) - 60:].values
# inputs = np.append(inputs, new_data)
# print(inputs.shape)
# inputs = inputs.reshape(-1,1)
# inputs.shape

# inputs = sc.transform(inputs)

# X_test = []
# for i in range(inputs.size, inputs.size + 1):
#     X_test.append(inputs[i-60:i, 0])
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predicted_stock_price = model.predict(X_test)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# predicted_stock_price


with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    st.line_chart(pd.DataFrame({"real_stock_price": real_stock_price,
                  "predicted_stock_price": predicted_stock_price}))
    st.line_chart()

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
                    .mark_circle(color='#0068c9', opacity=0.5)
                    .encode(x='x:Q', y='y:Q'))
