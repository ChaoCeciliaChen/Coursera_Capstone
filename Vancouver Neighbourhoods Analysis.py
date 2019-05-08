
# coding: utf-8

# # City of Vancouver
#     

# # 1. Prepare data file of the city of Vancouver

# In[1]:


#Import all necessary libraries.
#use the inline backend to generate the plots within the browser
get_ipython().magic(u'matplotlib inline')

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style
import pylab as pl

from sklearn.decomposition import PCA

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0


# In[4]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system(u'conda install -c conda-forge folium=0.5.0 --yes ')
import folium # map rendering library

from pprint import pprint

print('Libraries imported.')


# In[7]:


#import police department incident report from 2013-1-1 to 2018-12-31
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_cafcbf8af8b54038a59eae7ecf54fb9d = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='WLXruEHDFtGd9Tji9WPuGhpBOMxpheLXyb9JNAGblKZs',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_cafcbf8af8b54038a59eae7ecf54fb9d.get_object(Bucket='thebattleofneighbourhoods-donotdelete-pr-ayvxenezx65bqa',Key='crime_csv_all_years.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

crime = pd.read_csv(body)
print(crime.shape)
crime.reset_index(drop=True)
crime.head(5)


# In[8]:


#confirm the date range of crime data
datetime_list = []
for i in range(len(crime)):
    datetime_list.append([crime.YEAR[i],crime.MONTH[i],crime.DAY[i]])
    
print("The data range of crime record of Greater Vancouver is from {} to {}."
      .format(min(datetime_list),max(datetime_list)))

datetime_list[:10]


# In[9]:


#group by neighbourhood and years to get total reported crime number of each year.
crime_num_year = crime.groupby(['NEIGHBOURHOOD','YEAR']).count()
crime_num_year


# In[10]:


#get average crime number of each neighbourhood
crime_ave_year = pd.DataFrame(crime_num_year.groupby(['NEIGHBOURHOOD'])['TYPE'].mean())
crime_ave_year = crime_ave_year.reset_index(col_level=1)
crime_ave_year = crime_ave_year.rename(columns={'NEIGHBOURHOOD':'neighbourhood','TYPE':'crime_avg'})
crime_ave_year


# In[11]:


#display average crime number of each year of each neighbourhood.
crime_ave_year = crime_ave_year.sort_values('crime_avg', ascending=False).reset_index(drop=True)
crime_ave_year.plot(kind='bar', figsize=(20, 10), x='neighbourhood', rot=80)

plt.xlabel('Neighbourhood') # add to x-label to the plot
plt.ylabel('Average Number of Crime') # add y-label to the plot
plt.title('Number of Crime of each neighbourhood in Greater Vancouver 2003-2018') # add title to the plot
plt.legend(['Average Number of Crime'])

plt.show()


# Manually find coordinates by using <a href="https://www.latlong.net/convert-address-to-lat-long.html">LatLong.net</a>

# In[12]:


van_nhood = pd.DataFrame(crime_ave_year, columns=['neighbourhood']) 
van_nhood['latitude'] = ""
van_nhood['longitude'] = ""
van_nhood
lat = ['49.260872','49.592949','49.284131','49.270559','49.209223','49.245331','49.219593','49.300362','49.247632','49.277594','49.247438','49.218416','49.264113','49.269410','49.242024','49.230829','49.263330','49.253460','49.264484','49.246685','49.224274','49.234673','49.230628','49.230629']

lon = ['-123.113953','-125.702560','-123.131795','-123.067942','-123.136150','-123.139664','-123.090239','-123.142593','-123.084207','-123.043920','-123.102966','-123.073287','-123.126835','-123.155267','-123.057679','-123.131134','-123.096589','-123.185044','-123.185433','-123.120915','-123.046250','-123.155389','-123.195379','-123.195381']
for i in range(len(van_nhood)):
    van_nhood['latitude'][i] = lat[i]
    van_nhood['longitude'][i] = lon[i]
van_nhood = van_nhood.sort_values(by=['neighbourhood']).reset_index(drop=True)
van_nhood


# In[13]:


#combine crime_ave_year and van_nhood
van_nhood = van_nhood.sort_values(by=['neighbourhood']).reset_index(drop=True)
crime_nhood_merged = pd.merge(van_nhood, crime_ave_year, on='neighbourhood', how='right')
crime_nhood_merged['latitude'] = pd.to_numeric(crime_nhood_merged['latitude'])
crime_nhood_merged['longitude'] = pd.to_numeric(crime_nhood_merged['longitude'])
print(crime_nhood_merged.dtypes)
crime_nhood_merged


# In[14]:


#map of metro Vancouver
address = 'Metro Vancouver, BC'

geolocator = Nominatim(user_agent="on_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Metro Vancouver are {}, {}.'.format(latitude, longitude))


# In[15]:


# create map of Metro Vancouver using latitude and longitude values
map_vancouver = folium.Map(location=[latitude, longitude], zoom_start=10)
map_vancouver


# In[16]:


# create map of Greater Vancouver using latitude and longitude values
map_vancouver = folium.Map(location=[latitude, longitude], zoom_start=12)

# add markers to map
for lat, lng, neighborhood in zip(crime_nhood_merged['latitude'], 
                                  crime_nhood_merged['longitude'], 
                                  crime_nhood_merged['neighbourhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_vancouver)  


map_vancouver


# In[17]:


#Foursquare API account info
CLIENT_ID = 'SASB2OVN32ZTFVGRD2NJZMZMYX0VJ2LJMYQMCXN5KUGWE4NL' 
CLIENT_SECRET = 'F3B0FLTJ31BFJ5112YBBWSUZFUQVFI15YMXSJK1O5VILFDBY' 
VERSION = '20190507' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[18]:


# get the coordinates of each neighbourhood
neighbourhood_latitude = []
neighbourhood_longitude = []
neighbourhood_name = []

for i in range(len(crime_nhood_merged)):
    neighbourhood_latitude.append(crime_nhood_merged.loc[i, 'latitude']) # neighborhood latitude value
    neighbourhood_longitude.append(crime_nhood_merged.loc[i, 'longitude']) # neighborhood longitude value
    neighbourhood_name.append(crime_nhood_merged.loc[i, 'neighbourhood']) #neighbourhood name value
    print('Latitude and longitude values of {} are {}, {}.'.format(neighbourhood_name[i], 
                                                               neighbourhood_latitude[i], 
                                                               neighbourhood_longitude[i]))


# In[19]:


#Here is to get the top 100 venues that are in The Beaches within a radius of 500 meters.
limit = 100
radius = 500
url = []
for i in range(len(crime_nhood_merged)):
    url.append('https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        CLIENT_ID, 
        CLIENT_SECRET, 
        VERSION, 
        neighbourhood_latitude[i], 
        neighbourhood_longitude[i], 
        radius, 
        limit))
url


# In[20]:


#store venue info(dataframe) of each neighbourhood into a list.
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues = []

for i in range(len(crime_nhood_merged)):
    results = requests.get(url[i]).json()
   #print(results)
    venues = results['response']['groups'][0]['items']
    if len(venues) != 0:
        nearby_venues_temp = json_normalize(venues)
    nearby_venues_temp = nearby_venues_temp.filter(filtered_columns, axis=1)
    nearby_venues.append(nearby_venues_temp)
print(len(nearby_venues))#ensure the size of nearby_venues which is 24(neighouthoods)


# In[21]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
    


# In[22]:


#get category name of each venue and rename the colums' name
for i in range(len(nearby_venues)):
    nearby_venues[i].iloc[:,1] = nearby_venues[i].apply(get_category_type, axis=1)
    nearby_venues[i].columns = [col.split(".")[-1] for col in nearby_venues[i].columns]
    print(nearby_venues[i].head(2))#ensure the format of each dataframe(venue info)


# In[23]:


#add name of neighourhood to each venue info dataframe
for i in range(len(nearby_venues)):
    nearby_venues[i]['neighbourhood'] = crime_nhood_merged.loc[i, 'neighbourhood']
    print(nearby_venues[i].head(2))


# # Explore Venues of Each Neighbourhood

# In[24]:


#count number of venue of each neighbourhood
for i in range(len(nearby_venues)):
    nearby_venues[i]["venue_num"] = len(nearby_venues[i])


# In[25]:


#concatenate venues of each neighbourhood into a new dataframe
df_nearby_venues = pd.concat(nearby_venues).reset_index(drop=True)
print(df_nearby_venues.shape)#for later checking reference
df_nearby_venues.head()


# In[26]:


#prepare a new dataframe that includes neighourhood and number of venue for generating bar plot chart.
df_nei_ven = []
df_nei_ven.append(df_nearby_venues['neighbourhood'].unique())
df_nei_ven.append(df_nearby_venues['venue_num'].unique())
df_nei_ven = pd.DataFrame(df_nei_ven)
df_nei_ven = df_nei_ven.T.sort_values(1, ascending=False).reset_index(drop=True)
df_nei_ven.rename(columns={0:'neighbourhood', 1:'num_venues'}, inplace=True)
df_nei_ven


# In[27]:


#bar plot chart of number of venues of each neighbourhood in Vancouver. There are no venue info collected from Fourquare API of the last five neighbourhood.
df_nei_ven.plot(kind='bar', figsize=(20, 10), x='neighbourhood', rot=80)

plt.xlabel('Neighbourhood') # add to x-label to the plot
plt.ylabel('Number of Venue') # add y-label to the plot
plt.title('Number of Venues of each neighbourhood in Greater Vancouver') # add title to the plot
plt.legend(['Number of Venue'])

plt.show()


# In[29]:


# one hot encoding
vancouver_onehot = pd.get_dummies(df_nearby_venues[['categories']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
vancouver_onehot['neighbourhood'] = df_nearby_venues['neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [vancouver_onehot.columns[-1]] + list(vancouver_onehot.columns[:-1])
vancouver_onehot = vancouver_onehot[fixed_columns]

print(vancouver_onehot.shape)#ensure the size not change
vancouver_onehot.head()


# In[30]:


#group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
vancouver_grouped = vancouver_onehot.groupby('neighbourhood').mean().reset_index()
print(vancouver_grouped.shape)
vancouver_grouped


# In[31]:


#print each neighborhood along with the top 10 most common venues
num_top_venues = 10

for hood in vancouver_grouped['neighbourhood']:
    print("----"+hood+"----")
    temp = vancouver_grouped[vancouver_grouped['neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[32]:


#define a function to sort the venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[33]:


#create the new dataframe and display the top 10 venues for each neighborhood
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['neighbourhood'] = vancouver_grouped['neighbourhood']

for ind in np.arange(vancouver_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(vancouver_grouped.iloc[ind, :], num_top_venues)
neighborhoods_venues_sorted.head()


# # Cluster Neighbourhoods

# In[34]:


# set number of clusters
kclusters = 10

vancouver_grouped_clustering = vancouver_grouped.drop('neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(vancouver_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[35]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

van_nhood_merged = van_nhood

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
van_nhood_merged = van_nhood_merged.join(neighborhoods_venues_sorted.set_index('neighbourhood'), on='neighbourhood')

van_nhood_merged.head() # check the last columns!


# In[36]:


#prepare data for bar plot
van_nhood_merged['latitude'] = pd.to_numeric(van_nhood_merged['latitude'])
van_nhood_merged['longitude'] = pd.to_numeric(van_nhood_merged['longitude'])
van_nhood_merged.dtypes


# In[37]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(van_nhood_merged['latitude'], van_nhood_merged['longitude'], van_nhood_merged['neighbourhood'], van_nhood_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

