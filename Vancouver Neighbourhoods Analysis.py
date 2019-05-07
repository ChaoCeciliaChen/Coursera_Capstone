
# coding: utf-8

# # City of Vancouver
#     

# In[4]:


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


# In[5]:


datetime_list = []
for i in range(len(crime)):
    datetime_list.append([crime.YEAR[i],crime.MONTH[i],crime.DAY[i]])
    
print("The data range of crime record of Greater Vancouver is from {} to {}."
      .format(min(datetime_list),max(datetime_list)))

datetime_list[:10]


# In[7]:


crime_num_year = crime.groupby(['NEIGHBOURHOOD','YEAR']).count()
crime_num_year


# In[8]:


crime_ave_year = pd.DataFrame(crime_num_year.groupby(['NEIGHBOURHOOD'])['TYPE'].mean())
crime_ave_year = crime_ave_year.reset_index(col_level=1)
crime_ave_year = crime_ave_year.rename(columns={'NEIGHBOURHOOD':'neighbourhood','TYPE':'crime_avg'})
crime_ave_year


# Manually find coordinates by using <a href="https://www.latlong.net/convert-address-to-lat-long.html">LatLong.net</a>

# In[9]:


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


# In[10]:


#combine crime_ave_year and van_nhood
van_nhood = van_nhood.sort_values(by=['neighbourhood']).reset_index(drop=True)
crime_nhood_merged = pd.merge(van_nhood, crime_ave_year, on='neighbourhood', how='right')
crime_nhood_merged['latitude'] = pd.to_numeric(crime_nhood_merged['latitude'])
crime_nhood_merged['longitude'] = pd.to_numeric(crime_nhood_merged['longitude'])
print(crime_nhood_merged.dtypes)
crime_nhood_merged


# In[11]:


import numpy as np
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

print('Libraries imported.')


# In[12]:


address = 'Metro Vancouver, BC'

geolocator = Nominatim(user_agent="on_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Metro Vancouver are {}, {}.'.format(latitude, longitude))


# In[13]:


# create map of Metro Vancouver using latitude and longitude values
map_vancouver = folium.Map(location=[latitude, longitude], zoom_start=10)
map_vancouver


# In[14]:


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


# In[20]:


CLIENT_ID = '4FA0LNQQGEFVCVPVTWBWPVHRNEFVCXLVRJA0VJM0EQNVYBB5' 
CLIENT_SECRET = 'BWI0ZXE4AITMQTUW514GZTTMLETWFPEOONU2U4INHGWYN4CF' 
VERSION = '20190507' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[21]:


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


# In[22]:


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


# In[18]:


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
    


# In[19]:


from pprint import pprint
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues = []
for i in range(2):
    results = requests.get(url[i]).json()
    print(results)
    venues = results['response']['groups'][0]['items']
    if len(venues) != 0:
        nearby_venues_temp = json_normalize(venues)
    
    nearby_venues_temp = nearby_venues_temp.filter(filtered_columns, axis=1)
    nearby_venues_temp['venue.categories'] = pd.Series(nearby_venues_temp.apply(get_category_type, axis=1))
    #print(nearby_venues_temp)
          
    #nearby_venues.append(nearby_venues_temp)

#nearby_venues


# In[187]:


nearby_venues = pd.DataFrame(list(map(list, zip(lst1,lst2,lst3))))
nearby_venues


# In[ ]:


len(ven)


# In[128]:


nearby_venues


# In[115]:


json_normalize(venues)


# In[104]:


#nearby_venues = []
#filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']

#for i in range(len(nearby_venues)):
#venues = np.asarray(venues)
test = json_normalize(venues)# flatten JSON
    # filter columns
    #nearby_venues.append(nearby_venues[i].loc[:, filtered_columns])

    # filter the category for each row
    #nearby_venues['venue.categories'].append(nearby_venues[i].apply(get_category_type, axis=1))

    # clean columns
    #nearby_venues.columns.append([col.split(".")[-1] for col in nearby_venues.columns])
    
nearby_venues.head()


# In[9]:


#drop nan

#crime_ave_year.neighbourhood = [x for x in crime_ave_year.neighbourhood if type(x) is not float]
#crime_ave_year.neighbourhood = np.asarray(crime_ave_year.neighbourhood, dtype=object)
#print("length of neighbourhood array: ",len(crime_ave_year.neighbourhood))
#crime_ave_year.neighbourhood


# In[216]:


#nhood = []
#for n in range(0, len(neighbourhood)):
 #   l = len(neighbourhood[n].split('-'))
  #  if l == 1:
   #     nhood.append(neighbourhood[n])
   # elif l > 1:
    #    temp = neighbourhood[n].split('-')
     #   for i in range(0,len(temp)):
      #      nhood.append(temp[i])
            
#nhood = np.asarray(nhood, dtype=object)            
#print("length of neighbourhood array: ",len(nhood))
#nhood


# In[131]:


#Install geocoder package
get_ipython().system(u'conda install -c conda-forge geocoder')


# In[147]:


#!anaconda install -c conda-forge OpenCageGeocode
get_ipython().system(u'conda install -c conda-forge git')


# In[164]:


get_ipython().system(u'git clone https://github.com/OpenCageData/python-opencage-geocoder.git')


# In[173]:


get_ipython().system(u'python setup.py install')


# In[182]:


from opencage.geocoder import OpenCageGeocode
from pprint import pprint

key = 'fc12af447ac84a9faeb6a895ef7a370a'  # get api key from:  https://opencagedata.com
geocoder = OpenCageGeocode(key)
query = 'Metro Vancouver, BC'  
results = geocoder.geocode(query)
pprint (results)


# In[180]:


list_lat = []   # create empty lists

list_long = []

for index, row in df_crime_more_cities.iterrows(): # iterate over rows in dataframe


    City = row['City']
    State = row['State']       
    query = str(City)+','+str(State)

    results = geocoder.geocode(query)   
    lat = results[0]['geometry']['lat']
    long = results[0]['geometry']['lng']

    list_lat.append(lat)
    list_long.append(long)




# In[181]:





# In[207]:


get_ipython().system(u"wget -q -O 'nyu-2451-35688-geojson.json' https://geo.nyu.edu/download/file/nyu-2451-35688-geojson.json")


# In[209]:


with open('nyu-2451-35688-geojson.json') as json_data:
    vancouver_data = json.load(json_data)


# In[210]:


pprint(vancouver_data)


# In[184]:


import geocoder # import geocoder

# initialize your variable to None
lat_lng_coords = None

# loop until you get the coordinates
while(lat_lng_coords is None):
  g = geocoder.google('Vancouver, BC')
  lat_lng_coords = g.latlng

latitude = lat_lng_coords[0]
longitude = lat_lng_coords[1]

