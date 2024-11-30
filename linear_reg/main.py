# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import csv
from sklearn.linear_model import LinearRegression


with open('data.csv', mode='r') as file:
    # Step 2: Create a CSV reader object
    csv_reader = csv.reader(file)

    next(csv_reader)

    weight = []
    gallon  = []

    # Step 3: Iterate through the rows
    for row in csv_reader:
        weight.append([float( row[0])])
        gallon.append(float( row[1]))

# weight = np.array(weight).reshape(-1, 1)
# gallon = np.array(gallon)


# create Linear Regression model
model = LinearRegression()


# train model
model.fit(weight, gallon)

# new_weight = np.array([3.22]).reshape(1, -1)

print(f"Model Coefficient (Slope): {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

# new prediction

new_res = model.predict([[3.22]])
print(new_res)
