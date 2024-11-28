# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas
2.Import Decision tree classifier
3.Fit the data in the model
4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:Eesha Ranka
RegisterNumber:24900107 
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print(data.head())
x=data[["Position","Level"]]
print(x.head())
y=data["Salary"]
print(y.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("R2 Score:",r2)
print(dt.predict([[5,6]]))

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
```
 Position  Level  Salary
0   Business Analyst      1   45000
1  Junior Consultant      2   50000
2  Senior Consultant      3   60000
3            Manager      4   80000
4    Country Manager      5  110000

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10 entries, 0 to 9
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Position  10 non-null     object
 1   Level     10 non-null     int64 
 2   Salary    10 non-null     int64 
dtypes: int64(2), object(1)
memory usage: 368.0+ bytes
None

Position    0
Level       0
Salary      0
dtype: int64

 Position  Level  Salary
0         0      1   45000
1         4      2   50000
2         8      3   60000
3         5      4   80000
4         3      5  110000

 Position  Level
0         0      1
1         4      2
2         8      3
3         5      4
4         3      5

0     45000
1     50000
2     60000
3     80000
4    110000
Name: Salary, dtype: int64

[80000. 45000.]

R2 Score: 0.48611111111111116

[200000.]

```



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
