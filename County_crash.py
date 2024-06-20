import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def import_clean_data(url: str) -> pd.DataFrame:
    '''
    Imports the clean data as a DataFrame to be used.
    '''
    return pd.read_csv(url)

url = "/Users/donaldyin/Documents/GitHub/ECON481_FinalProject/Car_Crash_Cleaned_AADT.csv"
data = import_clean_data(url)

# Classify collision severity
high_severity = ["Fatal Collision", "Serious Injury Collision"]

# Create a new column 'Collision Severity'
data['Collision Severity'] = data['Injury Severity'].apply(lambda x: 'High' if x in high_severity else 'Low')

# Summarize the data by county
county_summary = data.groupby('County').agg(
    Total_Collisions=('Collision Report Number', 'count'),
    High_Severity_Collisions=('Collision Severity', lambda x: (x == 'High').sum()),
    Low_Severity_Collisions=('Collision Severity', lambda x: (x == 'Low').sum()),
    Avg_AADT=('AADT', 'mean')
).reset_index()

# Calculate the percentage of high severity collisions
county_summary['High_Severity_Percentage'] = (county_summary['High_Severity_Collisions'] / county_summary['Total_Collisions']) * 100

# Sort the counties by percentage of high severity collisions
county_summary = county_summary.sort_values(by='High_Severity_Percentage', ascending=False)
print(county_summary.head())

features = [
    'County', 'Damage Threshold Met', 'Hit and Run', 'Passengers Involved',
    'Commercial Carrier Involved', 'School Bus Involved', 'School Zone',
    'Intersection Related', 'Weather Condition', 'Lighting Condition', 'Collision Severity'
]

# Filter data to only include the relevant features
filtered_data = data[features]

# Correct the summary analysis by providing functions for all columns
feature_summary = filtered_data.groupby(['County', 'Collision Severity']).agg(
    Damage_Threshold_Met=('Damage Threshold Met', 'sum'),
    Hit_and_Run=('Hit and Run', 'sum'),
    Passengers_Involved=('Passengers Involved', 'sum'),
    Commercial_Carrier_Involved=('Commercial Carrier Involved', 'sum'),
    School_Bus_Involved=('School Bus Involved', 'sum'),
    School_Zone=('School Zone', 'sum'),
    Intersection_Related=('Intersection Related', 'sum'),
    Weather_Condition=('Weather Condition', lambda x: x.value_counts().index[0]),  # Most common weather condition
    Lighting_Condition=('Lighting Condition', lambda x: x.value_counts().index[0])  # Most common lighting condition
).reset_index()

# Plots relationship between Avg_AADT and High_Severity_Percentage
plt.figure(figsize=(10, 6))
plt.scatter(county_summary['Avg_AADT'], county_summary['High_Severity_Percentage'], alpha=0.6)
plt.title('High Severity Collisions vs Average Annual Daily Traffic (AADT)')
plt.xlabel('Average Annual Daily Traffic (AADT)')
plt.ylabel('Percentage of High Severity Collisions')
plt.grid(True)
plt.show()

# Bar plot to compare each county based on the percentage of high severity collisions
sorted_county_summary = county_summary.sort_values(by='High_Severity_Percentage', ascending=False)
plt.figure(figsize=(14, 8))
plt.bar(sorted_county_summary['County'], sorted_county_summary['High_Severity_Percentage'], color='skyblue')
plt.title('Percentage of High Severity Collisions by County')
plt.xlabel('County')
plt.ylabel('Percentage of High Severity Collisions')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()




# # POISSON REGRESSION
# # Poisson predictor for counts of car accidents
# # Summarize data by county to get the count of car accidents
county_data = data.groupby('County').agg(
    Total_Collisions=('Collision Report Number', 'count'),
    Avg_AADT=('AADT', 'mean'),
    Damage_Threshold_Met=('Damage Threshold Met', 'sum'),
    Hit_and_Run=('Hit and Run', 'sum'),
    Passengers_Involved=('Passengers Involved', 'sum'),
    Commercial_Carrier_Involved=('Commercial Carrier Involved', 'sum'),
    School_Bus_Involved=('School Bus Involved', 'sum'),
    School_Zone=('School Zone', 'sum'),
    Intersection_Related=('Intersection Related', 'sum')
).reset_index()

# Prepare the data for Poisson regression
X_poisson = county_data.drop(columns=['County', 'Total_Collisions'])
y_poisson = county_data['Total_Collisions']

# Standardize the feature variables
scaler_poisson = StandardScaler()
X_poisson_scaled = scaler_poisson.fit_transform(X_poisson)

# Perform Poisson regression
poisson_reg = PoissonRegressor()
poisson_reg.fit(X_poisson_scaled, y_poisson)

# Get feature importance
poisson_feature_importance = pd.DataFrame({
    'Feature': X_poisson.columns,
    'Importance': poisson_reg.coef_
}).sort_values(by='Importance', ascending=False)

print(poisson_feature_importance)



# Plot the count of car accidents in each county
plt.figure(figsize=(14, 8))
plt.bar(county_data['County'], county_data['Total_Collisions'], color='skyblue', label='Observed Count of Car Accidents')
plt.xlabel('County')
plt.ylabel('Count of Car Accidents')
plt.xticks(rotation=90)
plt.title('Count of Car Accidents in Each County')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

