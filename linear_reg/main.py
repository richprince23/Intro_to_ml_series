import matplotlib.pyplot as plt
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


plt.plot(weight, gallon)

# create Linear Regression model
model = LinearRegression()

# train model
model.fit(weight, gallon)


# print gradient 
print(f"Model Coefficient (Slope): {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

# new prediction

# accept use input and predict from it

new_weight =  input("Enter weight of the car: ")

# convert to a 2d array
new_weight = [[float(new_weight)]]

# use user supplied value in prediction
new_res = model.predict(new_weight)

# print predicted value
print(f"Fuel Consupmtion for a car of weight {new_weight[0]}  is {new_res[0]} miles per gallon")
