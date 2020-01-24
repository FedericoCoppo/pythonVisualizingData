#
# author: F. Coppo
# description: Data Analysis with Python
#

"""
Model Development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path of data 
file = 'automobileEDA.csv'
df = pd.read_csv(file)
print(df.head())

# Linear Regression [𝑌ℎ𝑎𝑡=𝑎+𝑏𝑋]
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)

# output a prediction
Yhat=lm.predict(X)
print(Y[0:5])
print(Yhat[0:5])   

print(lm.coef_)      # Slope 
print(lm.intercept_) # Intercept

# Multiple Linear Regression [𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4]
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
lm.intercept_
lm.coef_             # array([53.49574423,  4.70770099, 81.53026382, 36.05748882])

"""
Model Evaluation
"""

# import the visualization package: seaborn
import seaborn as sns

# Regression Plot
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
print("correlation table\n:")
print(df[["peak-rpm","highway-mpg","price"]].corr())

# Residual Plot: the difference between the observed value (y) and the predicted value (Yhat)
# residuals on the vertical y-axis and the independent variable on the horizontal x-axis; with high dispersion means non-linear shoud be a better model
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()

# Multiple Linear Regression Plot: a way to look at the fit of the model is by looking at the distribution plot (model Vs actual values distribution)
Y_hat = lm.predict(Z)       #  lets make a prediction
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()

# we can try, for a best fitting, using a polynomial model to the data...
"""
POLYNOMIAL REGRESSION:  particular case of the general linear regression model or multiple linear regression models
"""
# plot function
def PlotPoly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.show()
    plt.close()

# get the variable
x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPoly(p, x, y, 'highway-mpg 3 level')         # plot the function

# try with higher order
f = np.polyfit(x, y, 11)
p = np.poly1d(f)
print(p)
PlotPoly(p, x, y, 'highway-mpg 11 order')         # plot the function

"""
PIPELINE
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# we create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe=Pipeline(Input)  # the list is input of the the pipeline constructor
pipe.fit(Z,y)         # the routine normalize the data, perform a transform and fit the model 
ypipe=pipe.predict(Z) # normalize the data, perform a transform and produce a prediction simultaneously

# model evaluation

# R^2 calculation
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

# predict the output
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

# mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
