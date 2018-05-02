# Exploratory Data Analysis of the West Nile Virus problem dataset

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![image title](https://img.shields.io/badge/statsmodels-v0.8.0-blue.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/seaborn-v0.8.1-yellow.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/numpy-1.14.2-green.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg)
<br>

<p align="center">
  <img src="moggie2.png">
</p>                                                                  
<p align="center">
  <a href="#intro"> Introduction </a> •
  <a href="#Dataset"> Dataset </a> 
</p>

<a id = 'intro'></a>

From the [Kaggle](https://www.kaggle.com/c/predict-west-nile-virus) website:

> West Nile virus is most commonly spread to humans through infected mosquitos. Around 20% of people who become infected with the virus develop symptoms ranging from a persistent fever, to serious neurological illnesses that can result in death. 

> By 2004 the City of Chicago and the Chicago Department of Public Health (CDPH) had established a comprehensive surveillance and control program that is still in effect today. Every week from late spring through the fall, mosquitos in traps across the city are tested for the virus. The results of these tests influence when and where the city will spray airborne pesticides to control adult mosquito populations.

In this notebook I will perform a detailed exploratory data analysis of the Kaggle West Nile virus dataset.

## Imports

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # see the value of multiple statements at once.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
import seaborn as sns
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 

## Importing the data

weather = pd.read_csv('weather.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
spray = pd.read_csv('spray.csv')

## Data dictionary

#### train.csv and test.csv

- Id: the id of the record
- Date: date that the WNV test is performed
- Address: approximate address of the location of trap. This is used to send to the GeoCoder. 
- Species: the species of mosquitos
- Block: block number of address
- Street: street name
- Trap: Id of the trap
- AddressNumberAndStreet: approximate address returned from GeoCoder
- Latitude, Longitude: Latitude and Longitude returned from GeoCoder
- AddressAccuracy: accuracy returned from GeoCoder
- NumMosquitos: number of mosquitoes caught in this trap
- WnvPresent: whether West Nile Virus was present in these mosquitos. 1 means WNV is present, and 0 means not present. 

#### spray.csv 

- Date, Time: the date and time of the spray
- Latitude, Longitude: the Latitude and Longitude of the spray


#### weather.csv 
- Column descriptions in noaa_weather_qclcd_documentation.pdf. 

## Function to perform some steps of the EDA:

def eda(df):
    print("1) Are there missing values:")
    if df.isnull().any().unique().shape[0] == 2:
        if df.isnull().any().unique()[0] == False and df.isnull().any().unique()[1] == False:
            print('No\n')
        else:
            print("Yes|Percentage of missing values in each column:\n",df.isnull().sum()/df.shape[0],'\n')
    elif df.isnull().any().unique().shape[0] == 1:
        if df.isnull().any().unique() == False:
            print('No\n')
        else:
            print("Yes|Percentage of missing values in each column:\n",df.isnull().sum()/df.shape[0],'\n')

    print("2) Which are the data types:\n")
    print(df.dtypes,'\n')
    print("3) Dataframe shape:",df.shape)
    print("4) Unique values per columm")
    for col in df.columns.tolist():
        print (col,":",df[col].nunique())  
    return

## Looking at `train` `DataFrame`:

train = pd.read_csv('train.csv')
train.head()

## Using `pandas-profiling` to elimite highly correlated features

import pandas_profiling
profile = pandas_profiling.ProfileReport(train)
profile

rejected_variables = profile.get_rejected_variables(threshold=0.9)
rejected_variables

## Feature to keep
- Address features are redundant and some of them can be removed
- `NumMosquitos` and `WnvPresent` are not in the test set. I will remove the first since the number of mosquitos is less relevant than whether West Nile Virus was present in these mosquitos.

cols_to_keep = ['Date', 'Species', 'Trap','Latitude', 'Longitude', 'WnvPresent']
train = train[cols_to_keep]
train.head()

## Using the argument `df` equal to the training set `train`

eda(train)

## Comments
- Only `Species` can be transformed into dummies. The others have too many unique values.
- We should examine categorical columns to see if they are unbalanced. Using `value_counts` we find that the `WnvPresent` column is highly unbalanced with $\approx$ 95$\%$ of zeros.

import auxiliar_v2 as aux
s = round(100*train['WnvPresent'].value_counts()/train.shape[0],0)
aux.s_to_df(s,'WnvPresent','Yes/No')

train.to_csv('train_new.csv')

## Creating dummies from `Species` and tranforming dates to `Datetime`

train = pd.read_csv('train_new.csv',index_col=0)
train['Date'] = pd.to_datetime(train['Date'])
train = pd.concat([train,pd.get_dummies(train['Species'], drop_first = True)], axis = 1)
train.drop('Species', inplace=True, axis=1)
train.head()
train.dtypes

## Build a `DataFrame` with the dates broken into pieces

train2 = train.copy()
train2['Year']= train2.Date.dt.year
train2['DayofYear']= train2.Date.dt.dayofyear
train2.drop('Date', inplace=True, axis=1)  
train2.head()

## Exporting `train` and `train2` after EDA

The `DataFrame` with the full `Date` is kept because it may be useful for merging different dataframes. Hence:
- `'train_after_eda.csv'` has a `Date` column
- `'train_after_eda_without_date.csv'` has no `Date` column but columns `Year` and `DayofYear`

train.to_csv('train_after_eda.csv')
train2.to_csv('train_after_eda_without_date.csv')

## Applying similar changes to the test data

cols_to_keep_test = ['Date', 'Species','Trap', 'Latitude', 'Longitude']

test = pd.read_csv('test.csv')
test['Date'] = pd.to_datetime(test['Date'])
test = test[cols_to_keep_test]
test.head()

test = pd.concat([test,pd.get_dummies(test['Species'], drop_first = True)], axis = 1)
test.drop('Species', inplace=True, axis=1)
test.head()

### Build a `DataFrame` with the dates broken into pieces

test2 = test.copy()
test2['Year']= test2.Date.dt.year
test2['DayofYear']= test2.Date.dt.dayofyear
test2.drop('Date', inplace=True, axis=1)  
test2.head()

### Exporting `test` and `test2` after EDA

test.to_csv('test_after_eda.csv')
test2.to_csv('test_after_eda_without_date.csv')

## Now, we look at the `spray` data and perform similar steps:

spray = pd.read_csv('spray.csv')
spray['Date'] = pd.to_datetime(spray['Date'])
spray.head()
spray.dtypes

## Printing out the `DataFrame` we see that there are several `NaNs` but the percentage is low. 

aux.s_to_df(spray.isnull().sum(),'NaNs','Features')
spray[spray['Time'].isnull()].head()
print('% of NaNs in the `Time` column:',round(spray[spray['Time'].isnull()].shape[0]/spray.shape[0],2))

## Remove `Time` column altogether. 

spray.drop('Time', inplace=True, axis=1)
spray.head()

aux.s_to_df(spray.isnull().sum(),'NaNs','Features')

spray2 = spray.copy()
spray2['Year']= spray2.Date.dt.year
spray2['DayofYear']= spray2.Date.dt.dayofyear
spray2.drop('Date', inplace=True, axis=1)  
spray2.head()

spray.to_csv('spray_after_eda.csv')
spray2.to_csv('spray_after_eda_without_date.csv')

## Looking at the `weather` `DataFrame`:

weather = pd.read_csv('weather.csv')

## The `Water1` column has just 1 value namely  `M` and the latter means missing. We remove this column.

weather['Water1'].value_counts()
weather['Water1'].nunique()
weather['Water1'].unique()
weather.drop('Water1', inplace=True, axis=1)

## The `Depth` column has just two values namely 0 and `M` and the latter means missing. We remove this column.

aux.s_to_df(weather['Depth'].value_counts(),'value_counts','features')
weather['Depth'].nunique()
weather['Depth'].unique()
weather.drop('Depth', inplace=True, axis=1)

## Converting dates into `datetime`:

weather['Date'] = pd.to_datetime(weather['Date'])

weather.to_csv('weather_new.csv')

## Concerning stations 1 and 2

- As we saw above, there are two types of `Station`, namely, 1 and 2.

### From Kaggle's Website Weather Data:

- Hot and dry conditions are more favorable for West Nile virus than cold and wet. 
- We provide you with the dataset from NOAA of the weather conditions of 2007 to 2014, during the months of the tests. 

    - Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
    - Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level
    
- Each date had 2 records, 1 for each `Station=1` and other for `Station=2`. However as we shall see most missing values are in the latter which we will drop.

weather['Station'].value_counts()
weather['Station'].unique()

### The `for` below searches each column for data that cannot be converted to numbers:

cols_to_keep = ['Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', \
                'Cool', 'Sunrise', 'Sunset', 'SnowFall', \
                'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']

print('Columns with non-convertibles:\n')
for station in [1,2]:
    print('Station',station,'\n')
    weather_station = weather[weather['Station']==station]
    for col in weather_station[cols_to_keep]:
        for x in sorted(weather_station[col].unique()):
            try:
                x = float(x)
            except:
                print(col,'| Non-convertibles, their frequency and their station:',\
                      (x,weather_station[weather_station[col] == x][col].count()))
    print("")

weather.to_csv('weather_new_2.csv')

## Indeed, as stated above, most missing values are in the station 2. We will there drop rows with `Station=2`

weather = weather[weather['Station'] == 1]
weather.dtypes
weather['Station'].unique()
del weather['Station']

## Only for station 1 we have:

print('Columns with non-convertibles:\n')
for col in weather[cols_to_keep]:
    for x in sorted(weather[col].unique()):
        try:
            x = float(x)
        except:
            print(col,'| Non-convertibles, their frequency and their station:',
                  (x,weather[weather[col] == x][col].count()))

## The strings 'T' and 'M' stand for trace and missing data. Traces are defined to be smaller that 0.05. Following cells take care of that:

cols_with_M = ['WetBulb', 'StnPressure', 'SeaLevel']
for col in cols_with_M:
    weather[col] = weather[col].str.strip()
    weather[col] = weather[col].str.replace('M','0.0').astype(float)
    
cols_with_T = ['SnowFall', 'PrecipTotal']
for col in cols_with_T:
    weather[col] = weather[col].str.replace('  T','0.05').astype(float)
    
for col in cols_to_keep:
    weather[col] = weather[col].astype(float)

weather.to_csv('weather_new_4.csv')

## There are many zeros in the data 

In particular in the columns:

        cols_zeros = ['Heat','Cool','SnowFall']
        
there is a substantial quantity of zeros. We will drop these.

weather = pd.read_csv('weather_new_4.csv',index_col=0)
weather['Date'] = pd.to_datetime(weather['Date'])

cols_zeros = ['Heat','Cool','SnowFall']
for col in cols_zeros:
    print('{}'.format(col),weather[weather[col] == 0.0][col].value_counts()/weather.shape[0]);

for col in cols_zeros:
    weather.drop(col, inplace=True, axis=1)
weather.head()

weather.to_csv('weather_new_5.csv')

weather = pd.read_csv('weather_new_5.csv',index_col=0)
weather['Date'] = pd.to_datetime(weather['Date'])

## `CodeSum`

If `CodeSum` entries are letters, they indicate some significant weather event. We can dummify it.

Let us use regex. We use `'^\w'` to match a string consisting of a single character where that character is alphanumeric (the '\w' means "any word character"), an underscore or an asterisk.

weather['CodeSum'].str.strip()  # strips empty spaces
weather['CodeSum'][weather['CodeSum'].str.contains('^\w')] = '1'
weather['CodeSum'][weather['CodeSum'] !='1'] = '0'

weather['CodeSum']= weather['CodeSum'].astype(int)
weather.dtypes

## Sunset and sunrise are obviously correlated

weather.drop('Sunrise', inplace=True, axis=1)

## 5) Quick stop:  `DataFrames` now

train.isnull().any()
test.isnull().any()
spray.isnull().any()
weather.isnull().any()

## Correlations and feature engineering

### Train data

for df in [train,test,spray,weather]:
    df.corr()

### The temperatures are highly correlated and other features as well. Let's remove the extra baggage.

weather.drop('Tmax', inplace=True, axis=1)
weather.drop('Tmin', inplace=True, axis=1)
weather.corr()

weather.drop('WetBulb', inplace=True, axis=1)
weather.drop('DewPoint', inplace=True, axis=1)
weather.corr()

sns.heatmap(weather.corr())

sns.heatmap(test.corr())

sns.heatmap(spray.corr())

sns.heatmap(weather.corr())


