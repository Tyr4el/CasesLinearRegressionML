import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the CSV
df = pd.read_csv('Opened_Cases_RR_AllTime_Modified.csv', usecols=['Days Since Start Date', 'Day of Week', '# of Cases'])

#########
# x = [[Days Since Start Date, Day of Week]]
# y = [[# of Cases]]
#########
x = df.iloc[:, :-1].values
y = df.iloc[:, 2].values.reshape(-1, 1)

# Create two MinMaxScaler objects for x and y
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Do the actual scaling of x and y (thank you Sklearn)
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# Train Test Split that shit
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Create our Linear Regression object
regressor = LinearRegression()

# Fit the Linear Regression object to our training data
regressor.fit(x_train, y_train)

# Debugging shit
print(len(x_train))
print(len(x_test))
# print(x_scaled)
# print("--------------------------------------")
# print(y_scaled)
