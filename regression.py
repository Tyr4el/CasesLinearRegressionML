import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Opened_Cases_RR_AllTime_Modified.csv', usecols=['Days Since Start Date', 'Day of Week', '# of Cases'])

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x = df.iloc[:, :-1].values
y = df.iloc[:, 2].values.reshape(-1, 1)

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

print(x_scaled)
print("--------------------------------------")
print(y_scaled)
