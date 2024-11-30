# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import csv
from sklearn.linear_model import LinearRegression


""" 

We want to predict a car's fuel efficiency in miles per gallon based on how heavy the car is, and we have the following dataset:

view data.csv file
"""


# read data file using csv
with open('data.csv', mode='r') as file:
    # Step 2: Create a CSV reader object
    csv_reader = csv.reader(file)

    next(csv_reader)

    # save data to array
    weight = []
    gallon  = []

    # Step 3: Iterate through the rows
    for row in csv_reader:

        weight.append([float(row[0])]) # use 2d array for the feature - weight of vehicle
        gallon.append(float(row[1]))


# create Linear Regression model
model = LinearRegression()

# train model
model.fit(weight, gallon)


# print gradient 
print(f"Model Coefficient (Slope): {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

# new prediction
new_res = model.predict([[3.22]])

# print predicted value
print(new_res)
