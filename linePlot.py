#
# author: F. Coppo
# description: dataset exploring and visualizing tutorial
# 

"""
	DATASET ANALYSYS: it explores some basics dataset features of pandas lib
"""
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library

df_can = pd.read_excel('Canada.xlsx', sheet_name='Canada by Citizenship', skiprows=range(20),skipfooter=2)  # row to skip as input parameters (it depend on excel format)
n = 4
print ('Data read into a pandas dataframe:')
print(df_can.head(n))  	 			# view top n raw
print(df_can.tail(n))				# view the bottom n rows
print(df_can.info())				# database info method
print(df_can.columns.values)		# print the list of column header

# check the default type of dataframe index and columns is NOT list
print(df_can.columns.values)
print(df_can.index.values)
print(type(df_can.columns))
print(type(df_can.index))

# instead to get the index and columns as lists, we can use the tolist() method.
print(df_can.columns.tolist())
print(df_can.index.tolist())
print (type(df_can.columns.tolist()))
print (type(df_can.index.tolist()))

# check the size of dataframe (rows, columns)
print(df_can.shape)    
print(df_can.head(2))
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis='columns', inplace=True) # use pandas drop() method to remove unnecessary columns
print(df_can.head(2))

# rename the columns option
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
print(df_can.columns)

# sum up the total immigrants by country over the entire period 1980 - 2013
df_can['Total'] = df_can.sum(axis=1) # axis=1 -> it sum each row; axis=0 -> it sum each column
print(df_can.columns) # only to verify the "Total" has been added
print("let's view a quick summary of each column in our dataframe using the describe() method")
print(df_can.describe()) 

# Filter on a column name
print(df_can.Country)   		 # returns a series
print(df_can[['Country', 1980]]) # returns a dataframe
df_can.set_index('Country', inplace=True) # set the 'Country' column as the index

# ways to select rows: let's view the number of immigrants from Japan (row 87) 
print(df_can.loc['Japan']) 		 							# The full row data (all columns) 
print(df_can.loc['Italy', ['Continent', 'Region', 2013] ])  # Italian for year 2013

df_can.columns = list(map(str, df_can.columns)) # let's convert the column names into strings

"""
	FILTERING EXAMPLE:
	to filter the dataframe based on a condition, we simply pass the condition as a boolean vector
"""
print("Filtering:")
condition = df_can['Continent'] == 'Asia'  # filter Asian country
print(condition)
df_can_Asia = df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')] # let's filter for AreaName = Asia and RegName = Southern Asia (multiple criteria)
print('Asia(Southern Asia) data dimensions:', df_can_Asia.shape)
print(df_can_Asia)

"""
	VISUALIZING DATA with matplotlib:
	it is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms
"""
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot is Matplotlib's scripting layer
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0 # check if Matplotlib is loaded
print(plt.style.available)
mpl.style.use(['ggplot']) # optional: for ggplot-like style

# plot with pandas (series/dataframe)
years = list(map(str, range(1980, 2014))) # to exclude 'Total column'
haiti = df_can.loc['Haiti', years] # extract the data series for Haiti; years used to exclude the 'Total' column 
print(haiti.head())

# pandas automatically populated the x-axis with the index values (years), and the y-axis with the column values (population). However, notice how the years were not displayed because they are of type string.
haiti.plot(kind='line')              # haiti is a series, no need to transpose because years is already index; other kind of plotare bar, area, pie etc..
plt.title('Immigration from Haiti')
plt.ylabel('# of Immigrants')
plt.xlabel('Years')
plt.text(20, 6000, '2010 Earthquake') # annotate the 2010 Earthquake: since years were stored as type 'string', we would need to specify x as the index position of the year. Eg 20th index is year 2000
plt.show() 

# get the data set for China and India, and display dataframe
df_ChinaIndia = df_can.loc[['India', 'China'], years]
df_ChinaIndia.plot(kind='line')
plt.show()                             

# plot transpose data to have on X years on Y # of immigrants
df_ChinaIndia_T = df_ChinaIndia.transpose() # remember: pandas plots the indices on the x-axis and the columns as individual lines BUT df_ChinaIndia is a dataframe so the country is the index!
df_ChinaIndia_T.index = df_ChinaIndia_T.index.map(int) # let's change the index (years) values of df_ChinaIndia_T to type integer for plotting (it allow X display on graph)
df_ChinaIndia_T.plot(kind='line')
plt.title('China and India immigrants')
plt.ylabel('# of Immigrants')
plt.xlabel('Years')
plt.show()

# Compare the trend of top 5 countries that contributed the most to immigration to Canada
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True) # order
df_top5 = df_can.head(5)											  	
df_top5 = df_top5[years].transpose()
print(df_top5)
df_top5.index = df_top5.index.map(int) # let's change the index values of df_top5 to type integer for plotting
df_top5.plot(kind='line', figsize=(14, 8)) # pass a tuple (x, y) for the size
plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
plt.show()


