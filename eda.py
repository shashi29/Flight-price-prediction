import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import math

train = pd.read_excel("Data_Train.xlsx",parse_dates=[1])
test = pd.read_excel("Test_set.xlsx",parse_dates=[1])


#Pre-processing pipeline 
#Removing repeating rows
def pre_process(df):
    
    df.drop_duplicates(keep=False,inplace=True)
    df['Destination'].value_counts()
    cleanup = {'New Delhi':'Delhi'}
    df['Source'] = df['Source'].replace(cleanup,regex=True)
    df['Destination'] = df['Destination'].replace(cleanup,regex=True)

    return df

train = pre_process(train)
test = pre_process(test)
#Convert duration into minutes
def duration_min(df):
        
    new = df["Duration"].str.split(" ", n = 1, expand = True)
    new[0] = new[0].str.replace(r'[^\d.]+', '')
    new[0] = pd.to_numeric(new[0])
    new[0] = new[0]*60
    new[1] = new[1].str.replace(r'[^\d.]+', '')
    #new[1] = new[1].replace('None',0)
    print(new[0])
    new[1] = pd.to_numeric(new[1])
    new[1] = new[1].fillna(0)
    return (new[0] + new[1])
    

#Check it should not contain emply rows
#train['Duration_min'].isna().sum()
train['Duration_min'] = duration_min(train)
test['Duration_min'] = duration_min(test)

JourneyDate = train['Date_of_Journey']

#Feature engineering pipeline
#Feature engineering
#Feature engineering for Journeydate
def add_feature(df):
    column_1 = df['Date_of_Journey']
    temp = pd.DataFrame({"year": column_1.dt.year,
                  "month": column_1.dt.month,
                  "day": column_1.dt.day,
                  "dayofyear": column_1.dt.dayofyear,
                  "week": column_1.dt.week,
                  "quarter": column_1.dt.quarter,
                 })
    
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df,temp],axis=1)
    
    return df

#add feature for train and test data set
train = add_feature(train)
test = add_feature(test)

#Pre-process for Dept_Time
#As we need to convert it into categorical variable
def category_arrival_departure(df):
    new = df["Arrival_Time"].str.split(":", n = 1, expand = True)
    new[0] = pd.to_numeric(new[0])
    #Convert it into category
    df['Arrival_cat'] = pd.cut(new[0],
                         bins=[0,6,12,18,24],
                         labels=["Early morning", "Morning", "Evening", "Night"])
    new = df["Dep_Time"].str.split(":", n = 1, expand = True)
    new[0] = pd.to_numeric(new[0])
    #Convert it into category
    df['Dep_cat'] = pd.cut(new[0],
                     bins=[0,6,12,18,24],
                     labels=["Early morning", "Morning", "Evening", "Night"])

    return df

#Category for departure and arrival 
train = category_arrival_departure(train)
test = category_arrival_departure(test)

#target variable 
price = train['Price']
train.drop(columns=['Price'],axis=1,inplace=True)

#Delete irrelevant columns
def del_column(df):
    df.drop(columns=['Date_of_Journey','Dep_Time','Arrival_Time','Duration'],axis=1,inplace=True)
    return df

train = del_column(train)
test = del_column(test)
    
#First build a model for the traininn and test dataset
X_train , X_test , y_train,y_test = train_test_split(train,price,test_size=0.2,random_state=29)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
test = pd.get_dummies(test)
#Pre-process for submission test set
missing_cols = set( X_train.columns ) - set( test.columns )
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[X_train.columns]


#Deal with missing columns 
missing_cols = set( X_train.columns ) - set( X_test.columns )
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]


#Regression :2.XGboost regression
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
#xgb.fit(X_train,y_train)
eval_set = [(X_test, y_test)]
xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_set=eval_set, verbose=True)

predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))

from xgboost.sklearn import XGBRegressor  
import scipy.stats as st

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}


xgbreg = XGBRegressor(nthreads=-1)  

from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(xgbreg, params, n_jobs=-1)  
gs.fit(X_train, y_train,verbose=True)  

predictions = gs.predict(X_test)
print(explained_variance_score(predictions,y_test))

test_result = gs.predict(test)
test_result = pd.DataFrame(test_result)
test_result.columns = ['Price']
test_result.to_csv('submission_1.csv',index=False)
