#
# author: F. Coppo
# description: visualizing geospazial data with Folium 
# 

import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
import folium		# pip install folium

world_map = folium.Map(location=[56.130, -106.35], zoom_start=8)												# define the world map centered around Canada
world_map.save('world_map_plot_data.html')																		# save the map 

mexico_latitude = 23.6345 
mexico_longitude = -102.5528
mexico_map = folium.Map(location=[mexico_latitude, mexico_longitude], zoom_start=4, tiles='Stamen Toner')		# define the world map centered around Mexico with a lower zoom level
mexico_map.save('mexico_map_plot_data.html')

# CHOROPLETH MAPS: thematic map in which areas are shaded or patterned in proportion to the measurement of the statistical variable being displayed on the map

# read from Canada Immigrant data
df_can = pd.read_excel('Canada.xlsx',
                     sheet_name='Canada by Citizenship',
                     skiprows=range(20),
                     skipfooter=2) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)									# clean up the dataset to remove unnecessary columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)		# let's rename the columns so that they make sense
df_can.columns = list(map(str, df_can.columns))																# for sake of consistency, let's also make all column labels of type string
df_can['Total'] = df_can.sum(axis=1)																		# add total column
years = list(map(str, range(1980, 2014)))																	# years that we will be using in this lesson		
print ('data dimensions:', df_can.shape)

world_geo = r'world_countries.json' 																		# geojson file with wordwide country geometry 
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')								# create a plain world map

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)
							  
threshold_scale = threshold_scale.tolist() 									  								# change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 								  								# make sure that the last value of the list is greater than the maximum immigration
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')  								# let Folium determine the scale

# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)

world_map.save('choropleth_map_plot_data.html')