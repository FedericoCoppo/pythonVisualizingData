#
# author: F. Coppo
# description: 
#

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

"""
DATA ANALYSIS
"""
filename = 'auto.csv'
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
		  
df = pd.read_csv(filename, names = headers)
df.replace("?", np.nan, inplace = True)										# Convert "?" to NaN
missing_data = df.isnull()

# count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  

"""
DEAL WITH MISSING DATA
"""
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)		# column avg
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)		# replace "NaN" by mean value in "normalized-losses" column
df['num-of-doors'].value_counts()											# see which values are present in a particular column

# convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# list the columns after the conversion
print(df.dtypes)

"""
DATA STANDARDIZATION AND NORMALIZATION
"""
df["highway-mpg"] = 235/df["highway-mpg"]									# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)		# rename column name from "highway-mpg" to "highway-L/100km"
df.head()																	# check your transformed data 

# data normalization example

# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

"""
DATA BINNING
"""
# binning: transforming continuous numerical variables into discrete categorical 'bins')  
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
df["horsepower"]=df["horsepower"].astype(int, copy=True)					# convert data to correct format
# plot the histogram of horsepower
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()															# before binning

# rearrange horsepower values into three ‘bins'
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)			# 3 bins of equal size bandwidth using numpy
group_names = ['Low', 'Medium', 'High']										# set group names
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))						# the function "cut" shows which "horse-power" bin each row belong to .
print(df["horsepower-binned"].value_counts())
pyplot.bar(group_names, df["horsepower-binned"].value_counts())
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")											# plot the distribution of each bin
plt.pyplot.show()

"""
INDICATOR VARIABLE: an indicator variable (or dummy variable) is a numerical variable used to label categories
					for regression analysis.
"""
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)   # change column names

# we now have the value 0 to represent "gas" and 1 to represent "diesel" in the column "fuel-type"
df = pd.concat([df, dummy_variable_1], axis=1)															 # merge data frame "df" and "dummy_variable_1" 
df.drop("fuel-type", axis = 1, inplace=True)															 # drop original column "fuel-type" from "df"
print("Indicatior variable example:")
print(df.head())

"""
ANALYZING CORRELATION BETWEEN COLUMN
"""
# correlation between column
print("correlation:")
print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

import seaborn as sns

# scatterplot of "engine-size" and "price"
ax = sns.regplot(x="engine-size", y="price", data=df)

#  examine the correlation
print(df[["engine-size", "price"]].corr())

"""
DESCRIPTIVE STATISTICAL ANALYSIS
"""
print(df.describe())
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame() 										# convert the series to a Datafram
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
print(drive_wheels_counts)

"""
GROUPING: the "groupby" method groups data by different categories
"""
print(df['drive-wheels'].unique())																		# here are 3 different categories of drive wheels

# calculate the average price for each of the different categories of data
df_group_one = df[['drive-wheels','body-style','price']]		
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()                             # mean of last column!
print(df_group_one)

# you can also group with multiple variables
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

# pivot table: body style Vs Price
import matplotlib.pyplot as plt
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

# heat map: visualize how the price is related to 'drive-wheel' and 'body-style'
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
row_labels = grouped_pivot.columns.levels[1]															# label names
col_labels = grouped_pivot.index
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)										# move ticks and labels to the center
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels, minor=False)																# insert labels
ax.set_yticklabels(col_labels, minor=False)
plt.xticks(rotation=90)																					# rotate label if too long
fig.colorbar(im)
plt.show()

"""
CORRELATION
"""
# Pearson correlation [+1,-1 totola linear correlation; 0 no linear correlation]
from scipy import stats
df = df.dropna()

# Wheel-base vs Price
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Highway-mpg vs Price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 

