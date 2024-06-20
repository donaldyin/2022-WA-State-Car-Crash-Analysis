import pandas as pd
import re
import numpy as np

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

if __name__ == '__main__':
    # Load the dataset
    CollisionDataFinalize()
    Car_Crash_to_csv()