import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import *
from scipy import stats as st
import statsmodels.api as sm


# DATA CLEANING
def get_AADT_Data() -> pd.DataFrame:
    """
    Get the data for the Traffic Counts
    """
    AADT = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vRXk2ssI2wP32cbxd7gJecku7nS9Mjim7Ed3dorQsgalcBYC6KbxpWKlx0ClBsmsgAcbf5QzQqt2tsy/pub?output=csv") #Data source was uploaded to a google sheets which was published online. This is because the data source is a from a query and it seems to expire after a bit of time.
    return AADT

def clean_crash_summary(file_path):
    """
    Clean the Car Crash Data and the assign yes and no variables to binary 0 and 1
    """
    # Read the CSV file
    crash_summary = pd.read_csv(file_path)

    # Filter and select relevant columns
    crash_summary_clean = crash_summary[(crash_summary['Jurisdiction'] == 'State Road') &
                                        (~crash_summary['Weather Condition'].isna())].drop(columns=['Collision Type'])

    # Create binary columns and convert categorical columns to category type
    crash_summary_clean = crash_summary_clean.assign(
        **{
            'School Zone': crash_summary_clean['School Zone'].apply(lambda x: 1 if x == 'Y' else 0),
            'Intersection Related': crash_summary_clean['Intersection Related'].apply(lambda x: 1 if x == 'Y' else 0),
            'Damage Threshold Met': crash_summary_clean['Damage Threshold Met'].apply(lambda x: 1 if x == 'Y' else 0),
            'Hit and Run': crash_summary_clean['Hit and Run'].apply(lambda x: 1 if x == 'Y' else 0),
            'Passengers Involved': crash_summary_clean['Passengers Involved'].apply(lambda x: 1 if x == 'Y' else 0),
            'Commercial Carrier Involved': crash_summary_clean['Commercial Carrier Involved'].apply(lambda x: 1 if x == 'Y' else 0),
            'School Bus Involved': crash_summary_clean['School Bus Involved'].apply(lambda x: 1 if x == 'Y' else 0),
            'Agency': crash_summary_clean['Agency'].astype('category'),
            'Weather Condition': crash_summary_clean['Weather Condition'].astype('category'),
            'Lighting Condition': crash_summary_clean['Lighting Condition'].astype('category'),
            'Injury Severity': crash_summary_clean['Injury Severity'].astype('category')
        }
    ) 
    crash_summary_clean.reindex(np.arange(len(crash_summary_clean)))
    
    return crash_summary_clean


def Parse_State_Road() -> pd.DataFrame:
    """
    Gets returns the Car Crash Data with the State Roads Parsed out
    """
    Car_Crash = clean_crash_summary("https://docs.google.com/spreadsheets/d/e/2PACX-1vRJhryMDLGWP2PxsaXiDYb5PdBN_vmZxV0aieOFUJNuD5OBBJTR927qUVRnPFBg_5iDbFgxWzDWPvC9/pub?output=csv") #Gets the Car Crash Data cleaned
    AADT = get_AADT_Data() #Get the Traffic Count Data
    validSR = AADT['StateRouteNumber'].unique().tolist() #Lists the Valid State Road Numbers as listed on the Traffic Count data
    Trafficway = ["Primary Trafficway","Secondary Trafficway"]  
    dict = {
      0: [],
      1: [],
    } #Creating a dictionary to append the Trafficway data 
    for i in np.arange(2):
    
        for x in Car_Crash[Trafficway[i]]:
            #Gets the state roads numbers from the primary trafficway. Has to match "[String of nondigit text]integer" 
            if type(x) == str: #Checks if there's a value for the Traffic Way
                if (not re.match("\D+\d+", x) == None): #Checks if there is a state number in the traffic way input
                    if int(re.findall(r'\d+', x)[0]) in validSR: #Checks if the state number gotten is an actual state number
                        dict[i].append(int(re.findall(r'\d+', x)[0]))
                    else:
                        dict[i].append(None)
                else:
                    dict[i].append(None)
            else:
                dict[i].append(None)
    
    State_Road_Num = []
    for x in np.arange(len(dict[0])):
        #From the 2 lists of state road numbers, it will first check the state road number of the primary trafficway and append it to the list if it is available
        #If that is not available, it takes the state road number of the secondary trafficway
        if not dict[0][x] == None:
            State_Road_Num.append(dict[0][x])
        else:
            State_Road_Num.append(dict[1][x])

    #Inserts the gotten road numbers into the data set.
    Car_Crash.insert(7,"Primary Road Number", dict[0])
    Car_Crash.insert(9,"Secondary Road Number", dict[1])
    Car_Crash.insert(10, "Associated State Road Number", State_Road_Num)

    return Car_Crash

def get_Mileposts() -> pd.DataFrame:
    """
    Parses out the mile post number from the Traffic Count Data and return the AADT data with the Mile posts parsed out
    """
    AADT = get_AADT_Data()
    mileposts = []
    for x in AADT["Location"]:
        match = re.search("(MILEPOST) (\d+.\d+)",x) #Gets the mile post number from a specific format (thankfully it is consistent)
        mileposts.append(float(match.group(2)))
    AADT.insert(6, "Mile Posts", mileposts)
    return AADT

def AADT_Assignment() -> pd.DataFrame:
    """
    Assigns each Car Crash with the AADT of the nearest associated mile post based on its road and return the Car Crash data fram with AADT for each collision if applicable
    """
    Car_Crash = Parse_State_Road()
    AADT = get_Mileposts()
    Crash_AADT = []
    for x in np.arange(len(Car_Crash)):
        if not np.isnan(Car_Crash['Associated State Road Number'].iloc[x]): #Checks if Collision has a state road number
            if Car_Crash['Mile Post'].iloc[x] > -1: #Checks if the Collision has a mile post number
                SR = AADT[['Mile Posts', 'AADT']].loc[AADT['StateRouteNumber'] == Car_Crash['Associated State Road Number'].iloc[x]] #Only looks at the AADT data that have the same State Road as our Collision
                SR.index = np.arange(len(SR))
                # Calculates absolute differences between milepost collision and mileposts on state road
                abs_diff = np.abs(SR['Mile Posts'] - Car_Crash['Mile Post'].iloc[x])
                
                # Find the indexs of the minimum absolute difference
                min_index = np.argmin(abs_diff)
    
                #Appends the AADT value of the milepost closest to the Collision
                Crash_AADT.append(SR['AADT'].iloc[min_index])
            else:
                R = AADT['AADT'].loc[AADT['StateRouteNumber'] == int(Car_Crash['Associated State Road Number'].iloc[0])] 
                Crash_AADT.append(R.median()) #If there is no mileposts, it instead using the median AADT of that state road
        else:
            Crash_AADT.append(None) #Appends none if there is no state road
    Car_Crash.insert(Car_Crash.shape[1], "AADT", Crash_AADT)

    return Car_Crash

def CollisionDataFinalize() -> pd.DataFrame:
    """
    Checks how many null values are in the data set, how many values do not have AADT values and removes them.
    """
    Car_Crash = AADT_Assignment()
    print(Car_Crash["AADT"].isna().sum())
    print("Percent of data with AADT value:" + str((float(len(Car_Crash["AADT"])) - Car_Crash["AADT"].isna().sum())/float(len(Car_Crash["AADT"]))))
    Car_Crash.dropna(subset=['AADT'], inplace=True)
    return Car_Crash

def Car_Crash_to_csv() -> None:
    """
    Converts the file of Car_Crash into a csv file for the group to use (so you don't have to keep loading in the data)
    """
    CollisionDataFinalize().to_csv('Car_Crash_Cleaned_AADT.csv', index=False)



# COUNTY/ROAD CRASH ANALYSIS
def county_crash_analysis(data: pd.DataFrame):
    '''
    Runs an analysis on car collisions at the county level in WA state for the year 2022.
    '''

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

    # Plots relationship between Avg_AADT and High_Severity_Percentage
    plt.figure(figsize=(10, 6))
    plt.scatter(county_summary['Avg_AADT'], county_summary['High_Severity_Percentage'], alpha=0.6)
    plt.title('High Severity Collisions vs Average Annual Daily Traffic (AADT)')
    plt.xlabel('Average Annual Daily Traffic (AADT)')
    plt.ylabel('Percentage of High Severity Collisions')
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

    # Plot the count of car accidents in each county
    plt.figure(figsize=(14, 8))
    plt.bar(county_data['County'], county_data['Total_Collisions'], color='skyblue', label='Observed Count of Car Accidents')
    plt.xlabel('County')
    plt.ylabel('Count of Car Accidents')
    plt.xticks(rotation=90)
    plt.title('Count of Car Accidents in Each County')
    plt.legend()
    plt.tight_layout()
    plt.show()

def stateroad_crash_analysis(data: pd.DataFrame):
    '''
    Runs an analysis on car collision severity at the state road level in WA state for the year 2022.
    '''

    # Filter the dataset to include only state roads with more than 100 collisions
    filtered_df = data.groupby('Associated State Road Number').filter(lambda x: len(x) > 100)

    # Take a random sample of 100 collisions for each state road
    sampled_df = filtered_df.groupby('Associated State Road Number').apply(lambda x: x.sample(100)).reset_index(drop=True)

    # Convert the state road numbers to strings for better labeling in the plots
    sampled_df['Associated State Road Number'] = sampled_df['Associated State Road Number'].astype(str)

    # Classify collision severity based on 'Injury Severity'
    high_severity = ["Fatal Collision", "Serious Injury Collision"]
    sampled_df['Collision Severity'] = sampled_df['Injury Severity'].apply(lambda x: 'High' if x in high_severity else 'Low')

    # Summarize the data by state road
    state_road_summary = sampled_df.groupby('Associated State Road Number').agg(
        Total_Collisions=('Collision Report Number', 'count'),
        High_Severity_Collisions=('Collision Severity', lambda x: (x == 'High').sum()),
        Low_Severity_Collisions=('Collision Severity', lambda x: (x == 'Low').sum()),
        Avg_AADT=('AADT', 'mean')
    ).reset_index()

    # Calculate the percentage of high severity collisions
    state_road_summary['High_Severity_Percentage'] = (state_road_summary['High_Severity_Collisions'] / state_road_summary['Total_Collisions']) * 100

    # Sort the state roads by percentage of high severity collisions
    state_road_summary = state_road_summary.sort_values(by='High_Severity_Percentage', ascending=False)

    # Display the summary
    print(state_road_summary.head())

    # Plot the percentage of high severity collisions by state road
    plt.figure(figsize=(14, 8))
    plt.bar(state_road_summary['Associated State Road Number'], state_road_summary['High_Severity_Percentage'], color='skyblue')
    plt.xlabel('Associated State Road Number')
    plt.ylabel('Percentage of High Severity Collisions')
    plt.title('Percentage of High Severity Collisions by State Road')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Plot the relationship between Avg_AADT and High_Severity_Percentage for state roads
    plt.figure(figsize=(14, 8))
    plt.scatter(state_road_summary['Avg_AADT'], state_road_summary['High_Severity_Percentage'], alpha=0.6)
    plt.xlabel('Average Annual Daily Traffic (AADT)')
    plt.ylabel('Percentage of High Severity Collisions')
    plt.title('High Severity Collisions vs Average Annual Daily Traffic (AADT) by State Road')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# KEY FEATURES
def randomForest_function(df1, graph_name):
    """
    Function that generate features importance from random forest model with 100 tree and 5-fold cross-validation
    """
    # Filter if needed
    car_crash_data = df1

    # Dropping specified columns
    columns_to_drop = [
        'Primary Road Number', 'Secondary Trafficway', 
        'Secondary Road Number', 'Block Number', 'Mile Post',
        'Object Struck', 'Associated State Road Number', 'Collision Date'
    ]
    car_crash_data_cleaned = car_crash_data.drop(columns=columns_to_drop)
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    for column in ['Weather Condition', 'Lighting Condition', 'Jurisdiction',
                   'Agency', 'Primary Trafficway', 'City', 'County']:
        car_crash_data_cleaned[column] = label_encoder.fit_transform(car_crash_data_cleaned[column])

    # Features and target variable
    X = car_crash_data_cleaned.drop(columns=['Collision Report Number', 'Injury Severity'])
    y = car_crash_data_cleaned['Injury Severity']
    
    # Encode target variable
    y = label_encoder.fit_transform(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build a Random Forest Classifier with GridSearchCV
    rf_classifier = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    # Get the best estimator
    best_rf_classifier = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_rf_classifier.predict(X_test)
    
    # Evaluation
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    # Get feature importances
    feature_importances = best_rf_classifier.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for the feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Print evaluation metrics
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)
    print("Top 10 Feature Importances:\n", importance_df)

    path_name = 'visualization/' + graph_name
    
    # Visualizing feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(graph_name)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(path_name, bbox_inches='tight')
    plt.show()

def weather_condition(df2):
    # Grouping data by Lighting Condition and calculating summary statistics
    weather_summary = df2.groupby('Weather Condition').agg({
        'Collision Report Number': 'count',
        'Motor Vehicles Involved': 'sum',
        'Passengers Involved': 'sum',
        'Commercial Carrier Involved': 'sum',
        'School Bus Involved': 'sum',
        'Pedestrians Involved': 'sum',
        'Pedalcyclists Involved': 'sum',
        'AADT': 'mean'
    }).reset_index()
    
    # Renaming columns for clarity
    weather_summary.rename(columns={
        'Collision Report Number': 'Total Collisions',
        'Motor Vehicles Involved': 'Total Vehicles Involved',
        'Passengers Involved': 'Total Passengers Involved',
        'Commercial Carrier Involved': 'Total Commercial Carriers Involved',
        'School Bus Involved': 'Total School Buses Involved',
        'Pedestrians Involved': 'Total Pedestrians Involved',
        'Pedalcyclists Involved': 'Total Pedalcyclists Involved',
        'AADT': 'Average AADT'
    }, inplace=True)
    
    # Bar plot for total collisions by lighting condition
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Weather Condition', y='Total Collisions', data=weather_summary, palette='viridis')
    plt.title('Total Collisions by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Total Collisions')
    plt.xticks(rotation=45)
    plt.show()

def lighting_condition(df3):
    # Grouping data by Lighting Condition and calculating summary statistics
    lighting_summary = df3.groupby('Lighting Condition').agg({
        'Collision Report Number': 'count',
        'Motor Vehicles Involved': 'sum',
        'Passengers Involved': 'sum',
        'Commercial Carrier Involved': 'sum',
        'School Bus Involved': 'sum',
        'Pedestrians Involved': 'sum',
        'Pedalcyclists Involved': 'sum',
        'AADT': 'mean'
    }).reset_index()
    
    # Renaming columns for clarity
    lighting_summary.rename(columns={
        'Collision Report Number': 'Total Collisions',
        'Motor Vehicles Involved': 'Total Vehicles Involved',
        'Passengers Involved': 'Total Passengers Involved',
        'Commercial Carrier Involved': 'Total Commercial Carriers Involved',
        'School Bus Involved': 'Total School Buses Involved',
        'Pedestrians Involved': 'Total Pedestrians Involved',
        'Pedalcyclists Involved': 'Total Pedalcyclists Involved',
        'AADT': 'Average AADT'
    }, inplace=True)
    
    # Bar plot for total collisions by lighting condition
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Lighting Condition', y='Total Collisions', data=lighting_summary, palette='viridis')
    plt.title('Total Collisions by Lighting Condition')
    plt.xlabel('Lighting Condition')
    plt.ylabel('Total Collisions')
    plt.xticks(rotation=45)
    plt.show()

def stratified_sample_RF(df4):

    # filter out state road > 100
    filtered_df = df4.groupby('Associated State Road Number').filter(lambda x: len(x) > 100)
    # sample 100 from each state road
    sampled_df = filtered_df.groupby('Associated State Road Number').apply(lambda x: x.sample(100)).reset_index(drop=True)

    # Change fatality
    sampled_df['Injury Severity'] = sampled_df['Injury Severity'].apply(
        lambda x: 1 if x in ['Fatal Collision', 'Serious Injury Collision'] else 0
    )
    # call rf function
    randomForest_function(sampled_df, "Top_10_Feature_Importances_Stratified_sample")



# PREDICTION AND OLS
def PredictionReport(X_values: pd.DataFrame, Y_values: pd.DataFrame, classifier: str = "Gaussian") -> None:
    Observation = X_values.to_numpy() 
    Results = Y_values.to_numpy().ravel() #Converts values into numpy format for the Classifier
    X_train, X_test, y_train, y_test = train_test_split(Observation, Results, test_size=0.2, random_state=0)

    #Allows the choice to use different classifier to test each and see how effective each one is
    #Using Naive Bayesian
    gnb = GaussianNB() 
    if classifier == "Multinomial":
        gnb = MultinomialNB()
    elif classifier == "Complement":
        gnb = ComplementNB()
    elif classifier == "Categorical":
        gnb = CategoricalNB() 
    
    #Trains and fits the data    
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    
    # Getting the confusion matric
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(class_report)
    values_index = Y_values['Injury Severity'].unique().tolist()
    values_index.sort()
    cm_df = pd.DataFrame(conf_matrix,
                     index = values_index, 
                     columns = values_index)
    
    #Plotting the confusion matrix as a visual
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

    #Lists what the predictor is actually predicting
    unique_values, counts = np.unique(y_pred, return_counts=True)
    print("Predicted values:")
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")
    unique_values, counts = np.unique(y_test, return_counts=True)

    #Lists what are the actual values in the test set
    print("\nActual values:")
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")

    return None

def Prepare_Data_for_Prediction(df) -> list:
    LightingCond = {'Dark-Street Lights On':3, 'Dark-No Street Lights':5, 'Daylight':1,
       'Unknown':1, 'Dark-Street Lights Off':4, 'Dawn':0,
       'Dark - Unknown Lighting':5, 'Dusk':2, 'Other':1}

    WeatherCond = {'Clear':0,
              'Partly Cloudy': 1,
              'Overcast': 2,
              'Raining': 5,
              'Snowing': 6,
              'Sleet or Hail or Freezing Rain': 7,
              'Other':0, 'Severe Crosswind':4,
       'Fog or Smog or Smoke':3, 'Blowing Sand or Dirt or Snow':8} 

    #Due to the Classifier and needed numpy, Weather and lighting conditions were reclassifieed as numbers
    #From Naive Bayesian, it should treat them as like outcome and not like numerical effects. Regardless
    #lighting conditions were number with a lower number if it was brighter and higher number if it was darker
    #Weather condition got a higher number based on the presumed level of obscurity brought by the weather conditon
    
    
    Known_Injury = df[(df["Injury Severity"] != "Unknown Injury Collision")] #Taking out values where injury was unknown
    X = Known_Injury[['Associated State Road Number','Mile Post','Intersection Related', 'Weather Condition',
       'Lighting Condition', 'Motor Vehicles Involved',
       'Passengers Involved', 'Commercial Carrier Involved',
       'School Bus Involved', 'Pedestrians Involved', 'Pedalcyclists Involved',
       'AADT']] #Getting the X values
    values = {"Lighting Condition": 'Unknown', "Weather Condition": 'Other', 'Passengers Involved':0.0, 'Commercial Carrier Involved':0.0,
       'School Bus Involved':0.0, 'Pedestrians Involved':0.0, 'Pedalcyclists Involved':0.0}
    X=X.fillna(value=values) #Filling the na values of X
    X=X.dropna() #Dropping any other na values
    Conditions = ['Weather Condition','Lighting Condition']

    #Assigning thwe weather and lighting conditions of X numerical numbers
    for Condition in Conditions:
        use_list = []
        states = []
        if Condition == 'Weather Condition':
            use_dict = WeatherCond
            states = X['Weather Condition'].unique()
        else:
            use_dict = LightingCond
            states = X['Lighting Condition'].unique()
        for i in np.arange(X[Condition].unique().shape[0]):
            X.loc[X[Condition] == states[i], Condition] = use_dict[states[i]]

    #Getting the Y values where there is X values
    Y = df[['Injury Severity']].iloc[X.index.to_list()]

    return {"X": X, "Y":Y}

def AADT_Distribution(df)-> None:
    """
    Showcases the distribution of AADT values for the dataset and shows which AADT is the most common
    """
    print(df['AADT'].value_counts())
    plt.hist(df['AADT'].to_numpy(), bins = 100)
    plt.show
    print(st.mode(df['AADT'].to_numpy()))

def LSData(df: pd.DataFrame, AADT: int = 0):

    """
    Does LS analysis on level of severity (either 0 for low severity or 1 for high severity) and various features
    Allows for an integer input which checks if there is an AADT value corresponding with that number and then
    only regresses on collisions with that same AADT value
    """
    #Only gets values where County, Weather Condition and Lighting Condition have a value
    RData = df[df['County'].notna() & df['Weather Condition'].notna() & df['Lighting Condition'].notna()]
    #Translates County, Weather, and Lighting into dummy variables
    RData = RData.join(pd.get_dummies(RData[['County','Weather Condition','Lighting Condition']], dtype=float))
    #Drops features that we think are not relelavnt 
    RData = RData.drop(['County','Weather Condition','Lighting Condition',
        'Collision Report Number', 'Collision Date', 'City', 'Jurisdiction',
       'Agency', 'Primary Trafficway', 'Primary Road Number',
       'Secondary Trafficway', 'Secondary Road Number', 'Block Number',
                   'Object Struck', 'Mile Post', 'Associated State Road Number'], axis=1)

    #Either drops AADT as a feature if we are looking at a specific AADT value
    if AADT in df['AADT'].unique():
        RData = RData[RData['AADT'] == AADT]
        RData = RData.drop(['AADT'], axis = 1)

    #Replaces injury levels with a 1 or 0 (1 being severe collision and 0 being non-severe collision)
    RData = RData.replace({'Injury Severity': {"No Injury Collision": 0, 
                                   'Minor Injury Collision': 0,
                                                 "Unknown Injury Collision":0,
                                  "Serious Injury Collision": 1, 
                                   'Fatal Collision': 1}})
    RData = RData.dropna()
    RY = RData['Injury Severity']
    RX = RData.drop(['Injury Severity'], axis=1)
    X_model = sm.add_constant(RX)
    model = sm.OLS(RY, X_model)
    results = model.fit(cov_type = "HC1")
    print(results.summary())
    return None



# MAIN METHOD
if __name__ == '__main__':
    # Load the dataset
    CollisionDataFinalize()
    Car_Crash_to_csv()
    car_crash_data = pd.read_csv('Car_Crash_Cleaned_AADT.csv')
    county_crash_analysis(car_crash_data)
    stateroad_crash_analysis(car_crash_data)
    car_crash_data = pd.read_csv('Car_Crash_Cleaned_AADT.csv')
    randomForest_function(car_crash_data, 'Top_10_Feature_Importances')
    weather_condition(car_crash_data)
    lighting_condition(car_crash_data)
    stratified_sample_RF(car_crash_data)
    X, Y = Prepare_Data_for_Prediction(car_crash_data)["X"], Prepare_Data_for_Prediction(car_crash_data)["Y"]
    PredictionReport(X, Y, "Categorical")
    PredictionReport(X, Y, "Complement")
    AADT_Distribution(car_crash_data)
    LSData(car_crash_data)
    LSData(car_crash_data, 5400)