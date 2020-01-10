#
# author: F. Coppo
# description: 
# 

"""
	DATASET ANALYSYS: Area Plots, Histograms, and Bar Plots
"""
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library

df_can = pd.read_excel('Canada.xlsx', sheet_name='Canada by Citizenship', skiprows=range(20),skipfooter=2)  # row to skip as input parameters (it depend on excel format)
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis='columns', inplace=True) 							# data clean up
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True) 		# rename the columns option
df_can.columns = list(map(str, df_can.columns))																# ensure that all column labels of type string

df_can.set_index('Country', inplace=True)   																# set country name as index
df_can['Total'] = df_can.sum(axis=1) 																		# add total column
years = list(map(str, range(1980, 2014)))																    # list of years from 1980 - 2013
print ('data dimensions:', df_can.shape)
plotShowEnable = False

"""
	VISUALIZING DATA with matplotlib:
	it is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms
"""
import matplotlib as mpl
import matplotlib.pyplot as plt # pyplot is Matplotlib's scripting layer
mpl.style.use(['ggplot']) # optional: for ggplot-like style

# AREA PLOT (STACKED LINE PLOT): here is used scripting layer method (plt). For advance plot is better use Artist layer (Object oriented method)

df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)  										# keep top 5 countries with the most immigrants (from 1980 to 2013)
df_top5 = df_can.head()
df_top5 = df_top5[years].transpose() 
df_top5.index = df_top5.index.map(int) 																		# let's change the index values of df_top5 to type integer for plotting
df_top5.plot(kind='area', 																			        
             alpha=0.2, # 0-1, default value a= 0.5                                                        	# default transparency parameter
			 stacked=False,																					# unstacked plot area (by default it is stacked)
             figsize=(20, 10), # pass a tuple (x, y) size
             )
plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('# immigrants')
plt.xlabel('Years')
plt.show()

# keep top 5 countries with the least immigrants (from 1980 to 2013)
df_bottom5 = df_can.tail(5)																					
df_bottom5 = df_bottom5[years].transpose() 
df_bottom5.index = df_bottom5.index.map(int) 
df_bottom5.plot(kind='area',alpha=0.3, stacked=False, figsize=(20, 10))
plt.title('Immigration Trend of Least 5 Countries')
plt.ylabel('# immigrants')
plt.xlabel('Years')
plt.show()

# same plot using artist layer
df_top10 = df_can.head(10)
df_top10 = df_top10[years].transpose() 
df_top10.index = df_top10.index.map(int) 
ax = df_top10.plot(kind='area', alpha=0.55, stacked=False, figsize=(20, 10))
ax.set_title('Immigration Trend of 7 Countries with Top Contribution to Immigration')
ax.set_ylabel('# Immigrants')
ax.set_xlabel('Years')
plt.show()


# HISTOGRAM: a histogram is a way of representing the frequency distribution of numeric dataset

# plot frequency distribution of the number (population) of new immigrants from the various countries to Canada in 2013
print(df_can['2013'].head()) 																			# let's quickly view the 2013 data
count, bin_edges = np.histogram(df_can['2013'])															# np.histogram returns 2 values
print(count) 																							# frequency count (# of country contributing to that bin)
print(bin_edges) 																						# bin ranges, default = 10 bins (example -> Bin1 0 to 3412.9 immigrant)
df_can['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)										# graph this distribution
plt.title('Histogram of Immigration from 195 Countries in 2013') 										# add a title to the histogram
plt.ylabel('Number of Countries') 																		# add y-label
plt.xlabel('Number of Immigrants') 																	    # add x-label
plt.show()																										

# Immigration distribution for Denmark, Norway, and Sweden for years 1980 - 1985
df_can.loc[['Denmark', 'Norway', 'Sweden'], years]														# let's quickly view the dataset 
# df_can.loc[['Denmark', 'Norway', 'Sweden'], years].plot.hist()									    # generate histogram ->  population frequency distribution for the `years`
df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()									# transpose dataframe
count, bin_edges = np.histogram(df_t, 15) 																# let's get the x-tick values
df_t.plot(kind ='hist', 																				# un-stacked histogram
          figsize=(10, 6),																				
          bins=15,																						# increase the bin size to 15 by passing in bins parameter
          alpha=0.6,																					# set transparency to 60%
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.show()

# BAR CHART
# let's compare the number of Icelandic immigrants (country = 'Iceland') to Canada from year 1980 to 2013
df_iceland = df_can.loc['Iceland', years]
df_iceland.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Year') 																						# add to x-label to the plot
plt.ylabel('Number of immigrants') 																		# add y-label to the plot
plt.title('Icelandic immigrants to Canada from 1980 to 2013') 											# add title to the plot

# Annotate arrow
plt.annotate('',                      		# s: str. will leave it blank for no text
             xy=(32, 70),             		# place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),         		# place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',         		# will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )

# Annotate Text
plt.annotate('Iceland Financial Crisis', 	 # text to display
             xy=(28, 30),                    # start the text at at point (year 2008 , pop 30)
             rotation=72.5,                  # based on trial and error to match the arrow
             va='bottom',                    # want the text to be vertically 'bottom' aligned
             ha='left',                      # want the text to be horizontally 'left' algned.
            )
plt.show() # years Vs # of imm.


# creating a horizontal bar plot showing the total number of immigrants to Canada from the top 15 countries, for the period 1980 - 2013
df_can.sort_values(by='Total', ascending=True, inplace=True) 										# sort dataframe on 'Total' column 
df_top15 = df_can['Total'].tail(15)																	# keep first 15 elements
df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')										# kind='barh' generating a bar chart with horizontal bars
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')
for index, value in enumerate(df_top15): 
    label = format(int(value), ',') # format int with commas
    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')							# place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
plt.show() # number of immigrant Vs country

