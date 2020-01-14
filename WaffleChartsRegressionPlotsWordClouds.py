#
# author: F. Coppo
# description: Waffle Charts, Regression Plots, Word Cloud
# 

"""
	DATASET ANALYSYS
"""
import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
from PIL import Image # -> python -m pip install Pillow

df_can = pd.read_excel('Canada.xlsx', sheet_name='Canada by Citizenship', skiprows=range(20),skipfooter=2)  # row to skip as input parameters (it depend on excel format)
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis='columns', inplace=True) 							# data clean up
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True) 		# rename the columns option
df_can.columns = list(map(str, df_can.columns))																# ensure that all column labels of type string

df_can.set_index('Country', inplace=True)   																# set country name as index
df_can['Total'] = df_can.sum(axis=1) 																		# add total column
years = list(map(str, range(1980, 2014)))																    # list of years from 1980 - 2013
print ('data dimensions:', df_can.shape)

"""
	VISUALIZING DATA 
"""
import matplotlib as mpl
import matplotlib.pyplot as plt 		# pyplot is Matplotlib's scripting layer
import matplotlib.patches as mpatches 	# needed for waffle Charts
mpl.style.use(['ggplot']) 				# optional: for ggplot-like style

# waffle charts: visualization that is normally created to display progress toward goals
df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]		 # let's create a new dataframe for these three countries 

# unfortunately waffle charts are not built into any of the Python visualization libraries: we will have to create them from scratch
# 1. compute the proportion of each category with respect to the total
total_values = sum(df_dsn['Total'])
category_proportions = [(float(value) / total_values) for value in df_dsn['Total']]

# print out proportions
for i, proportion in enumerate(category_proportions):
    print (df_dsn.index.values[i] + ': ' + str(proportion))

# 2. defining the overall size of the waffle chart
width = 40 # width of chart
height = 10 # height of chart
total_num_tiles = width * height # total number of tiles
print ('Total number of tiles is ', total_num_tiles)

# 3. is using the proportion of each category to determine if respective number of tiles
tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions] # compute the number of tiles for each catagory

for i, tiles in enumerate(tiles_per_category): # print out number of tiles per category
    print (df_dsn.index.values[i] + ': ' + str(tiles))
	
# 4. creating a matrix that resembles the waffle chart and populating it.
# initialize the waffle chart as an empty matrix
waffle_chart = np.zeros((height, width))

# define indices to loop through waffle chart
category_index = 0
tile_index = 0

# populate the waffle chart
for col in range(width):
    for row in range(height):
        tile_index += 1

        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
        if tile_index > sum(tiles_per_category[0:category_index]):
            # ...proceed to the next category
            category_index += 1       
            
        # set the class value to an integer, which increases with class
        waffle_chart[row, col] = category_index
        
print ('Waffle chart populated!')

# 5.  map the waffle chart matrix into a visual
# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

# 6. Prettify the chart

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
ax = plt.gca()	 # get the axis

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)	# add gridlines based on minor ticks
plt.xticks([])
plt.yticks([])

# 7. create a legend and add it to chart

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
ax = plt.gca()					# get the axis

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])

# compute cumulative sum of individual categories to match color schemes between chart and legend
values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]

# create legend
legend_handles = []
for i, category in enumerate(df_dsn.index.values):
    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

# add legend to chart
plt.legend(handles=legend_handles,
           loc='lower center', 
           ncol=len(df_dsn.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )
		  
plt.show()

# Regression Plots: 
# we will explore seaborn and see how efficient it is to create regression lines and fits using this library

import seaborn as sns # import library

# Create a new dataframe that stores that total number of landed immigrants to Canada per year from 1980 to 2013.
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))
df_tot.index = map(float, df_tot.index)				# change the years to type float (useful for regression later on)
df_tot.reset_index(inplace=True)					# reset the index to put in back in as a column in the df_tot dataframe
df_tot.columns = ['year', 'total']					# rename columns

# rename columns
# with seaborn, generating a regression plot is as simple as calling the regplot function
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)						
sns.set_style('whitegrid')								# white background with gridlines
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()

# WORD CLOUD: the more a specific word appears in a source of textual data, the bigger and bolder it appears in the word cloud
from wordcloud import WordCloud, STOPWORDS				# import package and its set of stopwords
alice_novel = open('alisInWonderland.txt', 'r').read()	# open the file and read it into a variable alice_novel
stopwords = set(STOPWORDS)								# remove any redundant stopwords

# Create a word cloud object and generate a word cloud (only the first 2000 words in the novel)
alice_wc = WordCloud(									# instantiate a word cloud object
    background_color='white',
    max_words=2000,
    stopwords=stopwords
)
alice_wc.generate(alice_novel)							# generate the word cloud

# now that the word cloud is created, let's visualize it
fig = plt.figure()
fig.set_figwidth(14) # set width
fig.set_figheight(18) # set height

# display the cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# with the word_cloud package is superimposing the words onto a mask of any shape. Let's use a mask of Alice and her rabbit!
alice_mask = np.array(Image.open('alice_mask.png'))		# save mask to alice_mask

# let's take a look at how the mask looks like
fig = plt.figure()
fig.set_figwidth(14) # set width
fig.set_figheight(18) # set height
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()

# shaping the word cloud according to the mask is straightforward using word_cloud package.
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)	# instantiate a word cloud object
alice_wc.generate(alice_novel)																			# generate the word cloud
fig = plt.figure()		# display the word cloud
fig.set_figwidth(14)    # set width
fig.set_figheight(18)   # set height
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# let's generate sample text data from our immigration dataset, say text data of 90 words
total_immigration = df_can['Total'].sum()
max_words = 90
word_string = ''
for country in df_can.index.values:
    # check if country's name is a single-word name
    if len(country.split(' ')) == 1:
        repeat_num_times = int(df_can.loc[country, 'Total']/float(total_immigration)*max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)
                                     
print(word_string)	# display the generated text

# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

# display the cloud
fig = plt.figure()
fig.set_figwidth(14)
fig.set_figheight(18)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()