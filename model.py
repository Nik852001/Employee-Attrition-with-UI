
#import libraries
import numpy as np 
import pandas as pd
import pickle

data=pd.read_csv('HR_Engagement_Sat_Sales.csv')

### Handling missing values-:

for i in data[['EMP_Sat_OnPrem_1','EMP_Sat_OnPrem_2','EMP_Sat_OnPrem_3','EMP_Sat_OnPrem_4','EMP_Sat_OnPrem_5']]:
  data[i]=data[i].fillna(data[i].mode()[0])

### Feature Engineering-:

data['EMP_Sat_OnPrem']=data[['EMP_Sat_OnPrem_1', 'EMP_Sat_OnPrem_2', 'EMP_Sat_OnPrem_3', 'EMP_Sat_OnPrem_4', 'EMP_Sat_OnPrem_5']].mean(axis=1)

"""We created a new column 'EMP_Sat_OnPrem' by taking the mean of the 5 columns having data of survey that was sent to employees. On prem (On premise) means that the employee maintains a high percentage of work on the corporation’s physical work locations. Scale (1-10)."""

data['EMP_Sat_Remote']=data[['EMP_Sat_Remote_1', 'EMP_Sat_Remote_2', 'EMP_Sat_Remote_3', 'EMP_Sat_Remote_4', 'EMP_Sat_Remote_5']].mean(axis=1)

"""We created a new column 'EMP_Sat_Remote' by taking the mean of the 5 columns having data of survey that was sent to employees. Remote (distance employee) means that the employee does a high percentage of work away from the corporation’s physical work locations. Scale (1-10)"""

data['EMP_Engagement']=data[['EMP_Engagement_1', 'EMP_Engagement_2', 'EMP_Engagement_3','EMP_Engagement_4', 'EMP_Engagement_5']].mean(axis=1)

"""We created a new column 'EMP_Engagement' by taking the mean of the 5 columns having data of survey that was sent to employees. Engagement represents the employee's feeling about how they feel about being engaged in company activities. Scale(1-5)"""

data['Emp_Work_Status']=data[['Emp_Work_Status2','Emp_Work_Status_3', 'Emp_Work_Status_4', 'Emp_Work_Status_5']].mean(axis=1)

"""We created a new column 'Emp_Work_Status' by taking the mean of the 4 columns having data of survey that was sent to employees. Status represents how strongly employee feels about their status level in the organization. Scale (1-10)"""

data['Emp_Competitive']=data[['Emp_Competitive_1', 'Emp_Competitive_2', 'Emp_Competitive_3', 'Emp_Competitive_4', 'Emp_Competitive_5']].mean(axis=1)

"""We created a new column 'Emp_Competitive' by taking the mean of the 5 columns having data of survey that was sent to employees. It shows how the employee feels about the competitive nature of work in the organization. Scale (1-10)"""

data['Emp_Collaborative']=data[['Emp_Collaborative_1', 'Emp_Collaborative_2', 'Emp_Collaborative_3','Emp_Collaborative_4', 'Emp_Collaborative_5']].mean(axis=1)

"""We created a new column 'Emp_Collaborative' by taking the mean of the 5 columns having data of survey that was sent to employees. It indicates how employee feels about the collaborative nature of work in the organization.Scale (1-10)

### Encoding-:
"""

#label encoding
from sklearn.preprocessing import LabelEncoder
label_en=LabelEncoder()
for i in data[['Department', 'GEO', 'Role', 'sales', 'salary', 'Gender']]:
    data[i]=label_en.fit_transform(data[i])

"""We have encoded our categorical variables required for modelling using label encoding.

### Feature Reduction-:
"""

data.drop(['ID', 'Name','Sensor_StepCount','Sensor_Heartbeat(Average/Min)','Sensor_Proximity(1-highest/10-lowest)'], axis=1, inplace=True)

"""We drop these columns since they are irrelevant while modelling."""

data.drop(['Rising_Star','Critical','CSR Factor','Women_Leave','Men_Leave'], axis=1, inplace=True)

"""We drop these columns since they have huge amount of missing values."""

data.drop(['EMP_Sat_OnPrem_1',
       'EMP_Sat_OnPrem_2', 'EMP_Sat_OnPrem_3', 'EMP_Sat_OnPrem_4',
       'EMP_Sat_OnPrem_5', 'EMP_Sat_Remote_1', 'EMP_Sat_Remote_2',
       'EMP_Sat_Remote_3', 'EMP_Sat_Remote_4', 'EMP_Sat_Remote_5',
       'EMP_Engagement_1', 'EMP_Engagement_2', 'EMP_Engagement_3',
       'EMP_Engagement_4', 'EMP_Engagement_5','Emp_Work_Status2',
       'Emp_Work_Status_3', 'Emp_Work_Status_4', 'Emp_Work_Status_5','Emp_Competitive_1', 'Emp_Competitive_2',
       'Emp_Competitive_3', 'Emp_Competitive_4', 'Emp_Competitive_5','Emp_Collaborative_1', 'Emp_Collaborative_2',
        'Emp_Collaborative_3','Emp_Collaborative_4', 'Emp_Collaborative_5'], axis=1, inplace=True)

"""We drop these columns because we have generated new columns using the data in these columns."""


data.drop(['Trending Perf', 'Talent_Level', 'Validated_Talent_Level','last_evaluation'], axis=1, inplace=True)

"""### Standardisation-:

First we will split the dataset into target and features.
"""

x=data.drop('left_Company', axis=1)
y=data['left_Company']


"""## 3.Predictive Modelling and Fine Tuning-:"""


### Feature Importance-:
    
x.drop(['EMP_Sat_OnPrem','sales','Emp_Collaborative','Emp_Work_Status','GEO','Role','Department','Will_Relocate','Gender','promotion_last_5years','Emp_Competitive','Work_accident','salary','Emp_Position','Emp_Title', 'Emp_Identity'], axis=1, inplace=True)

#split the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42, test_size=0.3)

#import random forest classifier
from sklearn.ensemble import RandomForestClassifier
#create the instance of the model
rf=RandomForestClassifier()
#train the data
rf.fit(x_train,y_train)

pickle.dump(rf ,open('model.pkl','wb'))
