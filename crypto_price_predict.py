import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pandas_datareader as web
import datetime as dt 

from sklearn.preprocessing import MinMaxScaler    # here we're going to scale the financial data in between 0 and 1 ,,, when u are going to work with neural network it's best to have a data that is either scaled from 0 to 1 or from -1 to 1 so the neural network can work with it better
from tensorflow.keras.layers import Dense, Dropout, LSTM   # LSTM--long short term memory 
from tensorflow.keras.models import Sequential

# now get financial data ,,, here we need to know what cryptocurrency we are interested 
crypto_currency = 'BTC'           # BTC- Bitcoin, ETH - Ethereum , XRP- Ripple
against_currency = 'USD'          # 'important' to deaclear Basic currency like Dollars($) or Tk for compare to Bitcoin, for how many $ BTC worth of or any Cryptocurrency
       
# Specify timeframe for the training data       
start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)     # get actual data

# Prepared Data for the neural network 
#print(data.head())
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# 
prediction_days = 90
future_day = 60    # means 60 days er data nibe but predict korbe 90tomo din er 60+30=90

# prepare training data , we need to have an x data and y data , y=result  ,,, supervised learning here,,, showing neural network to 60 days then that will predict 61th days 
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)-future_day):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x+future_day, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create Neural Network for prediction  
model = Sequential()

# add lstm-(recurrent layer its memorized stuff) layer and dropout layer 
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1 )))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# compiled the model 
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Testing the model
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)     # get actual data
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs) 

# prediction using training model  
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# prediction price
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(prediction_prices, color='blue', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

# Predict Next Day  
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
#print(prediction)








