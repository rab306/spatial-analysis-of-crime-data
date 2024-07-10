# Dataset link on kaggle https://www.kaggle.com/datasets/wosaku/crime-in-vancouver

# Loading required libraries
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import plotly.express as px
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# Loading the data
crime_data = pd.read_csv(r'E:\surveying engineering 4th year\1st semester\GIS\project\attribute data\crime.csv')

# Displaying the first few rows and information about the DataFrame
print("First few rows of the DataFrame:")
print(crime_data.head())
print("\nData Shape is")
print(crime_data.shape)
print("\nColumns and data types:")
print(crime_data.dtypes)

# Data preprocessing
# Converting date columns to datetime and dropping unnecessary columns
crime_data['date'] = pd.to_datetime(crime_data[['YEAR', 'MONTH', 'DAY']])
crime_data = crime_data.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'HUNDRED_BLOCK'])


# Handling missing values
print("\nMissing values before dropping:")
print(crime_data.isnull().sum())
crime_data = crime_data.dropna()  # All the NULL values in NEIGHBOURHOOD column, the data size are over 500000 rows so I drpped the NULL values

# Printing the maximum and minimum dates
print("\nMinimum date:", crime_data['date'].min())
print("Maximum date:", crime_data['date'].max())

# Define the date range because the data is large so I can decrease computational time
start_date = '2013-01-01'
end_date = crime_data['date'].max()
start_date = pd.to_datetime(start_date)
crime_data = crime_data[(crime_data['date'] >= start_date) & (crime_data['date'] <= end_date)]

# Another approach is to use Rndom sampling
# sample_size = 50000
# if sample_size < len(crime_data):
#     crime_data = crime_data.sample(n=sample_size, random_state=42)


# Checking unique categories
print("\nUnique neighborhoods:", crime_data['NEIGHBOURHOOD'].nunique())
print("Unique crime types:", crime_data['TYPE'].nunique())
print(crime_data.shape)


# Creating GeoDataFrame with Point geometries
geometry = [Point(xy) for xy in zip(crime_data['Longitude'], crime_data['Latitude'])]
crime_gdf = gpd.GeoDataFrame(crime_data, geometry=geometry, crs='EPSG:4326')

# Plotting crime type distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='TYPE', data=crime_data, order=crime_data['TYPE'].value_counts().index, palette='tab10')
plt.title('Distribution of Crime Types')
plt.xlabel('Count')
plt.ylabel('Crime Type')
plt.tight_layout()
plt.show()


# Plotting crimes over time
plt.figure(figsize=(14, 7))
crime_data['date'].value_counts().sort_index().plot()
plt.title('Crimes Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyzing crimes per year
crime_data['year'] = crime_data['date'].dt.year
crime_yearly = crime_data['year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
crime_yearly.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Crimes Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()


# Analyzing crimes per month
crime_data['month'] = crime_data['date'].dt.month
crime_monthly = crime_data['month'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
crime_monthly.plot(kind='bar', color='green', alpha=0.7)
plt.title('Crimes Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()


# Plotting crime locations by type
plt.figure(figsize=(16, 12))
sns.scatterplot(x='Longitude', y='Latitude', hue='TYPE', data=crime_data, s=5, palette='tab10')
plt.title('Crime Locations by Type')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
plt.grid(True)
plt.tight_layout()
plt.show()


# Plotting crime locations by neighborhood
plt.figure(figsize=(16, 12))
sns.scatterplot(x='Longitude', y='Latitude', hue='NEIGHBOURHOOD', data=crime_data, s=5, palette='tab20')
plt.title('Crime Locations by Neighborhood')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.grid(True)
plt.tight_layout()
plt.show()


# Extracting crime coordinates for plotting hotspots
# I chose to use ptojected system (metric system) so tuning DBSCAN epsilon parameter is easy to understand
# Another approach is to scale the data and use default epsilon value 0.5 and tune around it
crime_gdf = crime_gdf.to_crs(epsg=32610)
coords = crime_gdf['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist()


# Performing DBSCAN clustering
eps = 250
min_samples = 100
dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm= 'kd_tree')
crime_gdf['cluster'] = dbscan.fit_predict(coords)

# Analyzing the clusters
print("\nNumber of clusters:", crime_gdf['cluster'].nunique())
print("Number of crimes in each cluster:")
print(crime_gdf['cluster'].value_counts())


# We can also search for the parameters that give the minimum noise points and reasoaable number of clusters
# Notice that small eps and higher min_samples will result in more noise points 
# Grid search for DBSCAN parameters
# eps_values = [100, 300, 500]
# min_samples_values = [50, 100, 150]
# best_eps = None
# best_min_samples = None
# min_noise_points = float('inf')
#
# print("Grid Search Results:")
# for eps in eps_values:
#     for min_samples in min_samples_values:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree')
#         labels = dbscan.fit_predict(coords)
#         num_noise_points = np.sum(labels == -1)
#         first_class_points = np.sum(labels == 0)
#         print(
#             f"eps={eps}, min_samples={min_samples}: First clas points = {first_class_points}, Min Noise Points = {num_noise_points}")
#
#         if num_noise_points < min_noise_points:
#             min_noise_points = num_noise_points
#             best_eps = eps
#             best_min_samples = min_samples
#
# print("\nBest Combination:")
# print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Min Noise Points: {min_noise_points}")

# # Final DBSCAN with best parameters
# dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, algorithm='kd_tree')
# crime_gdf['cluster'] = dbscan.fit_predict(coords)


# # Estimating bandwidth for Mean Shift
# bandwidth = estimate_bandwidth(coords, quantile=0.2)
# print(bandwidth)

# # Performing Mean Shift clustering
# mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# crime_gdf['cluster'] = mean_shift.fit_predict(coords)

# # Plotting clusters with improved visualization
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# unique_clusters = crime_gdf['cluster'].unique()
# colors = sns.color_palette("tab10", len(unique_clusters))

# for cluster_id, color in zip(unique_clusters, colors):
#     clustered_data = crime_gdf[crime_gdf['cluster'] == cluster_id]
#     ax.scatter(clustered_data['Longitude'], clustered_data['Latitude'], s=5, label=f'Cluster {cluster_id}', color=color)

# plt.title('Crime Hotspots in Vancouver')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend(title='Cluster')
# plt.show()


# Plotting clusters
plt.figure(figsize=(16, 12))
crime_gdf.plot(column='cluster', legend=True, markersize=5, cmap='tab10', alpha=0.5)
plt.title('Crime Hotspots in Vancouver')
plt.xlabel('Easting (UTM)')
plt.ylabel('Northing (UTM)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Interactive visualization using Plotly for crime clusters
fig = px.scatter_mapbox(crime_gdf, lat='Latitude', lon='Longitude', color='cluster', size_max=5, zoom=10,
                        center=dict(lat=crime_data['Latitude'].mean(), lon=crime_data['Longitude'].mean()),
                        mapbox_style="open-street-map", title="Crime Clusters in Vancouver")
fig.show()