from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#Read Data
data=pd.DataFrame({"ML":[10,10,9,8,7],"Problem Solving":[10,10,3,5,8],"Behavioral":["Good","Bad","Average","Excellent","Good"],"Ratings":[10,9,6,4,8]})

#Feature and Target attribute
feature=["ML","Problem Solving","Behavioral"]
x=data[feature]
y=data.Ratings

#Column classification
categoral_cols=[col for col in x if x[col].dtype=="object"]
numerical_cols=[col for col in x if x[col].dtype in["int64","float64"]]
cols=categoral_cols+numerical_cols

#Preprocess
numerical_preprocess=SimpleImputer(strategy="mean")
categoral_preprocess=Pipeline(steps=[("impute",SimpleImputer(strategy=("most_frequent"))),("encoder",OneHotEncoder(handle_unknown="ignore"))])
preprocessor=ColumnTransformer(transformers=[("num",numerical_preprocess,numerical_cols),("categorical",categoral_preprocess,categoral_cols)])

#model
model_1=XGBRegressor(n_estimators=10,learning_rate=0.5,random_state=0)
model_2=RandomForestRegressor(n_estimators=10,random_state=0)
model_3=DecisionTreeRegressor(random_state=0)

#Bundle
pipeLine=Pipeline(steps=[("preprocess",preprocessor),("model",model_1)])

#Split data
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
score=1*cross_val_score(pipeLine,x,y,cv=2,scoring="neg_mean_absolute_error")

print("scores",score)

#Train
pipeLine.fit(x_train,y_train)
pipeline.fit(x_train,y_train,early_stopping_rounds=5,eval_set=[(x_valid,y_valid)]) #xgboost

#Predict
predictions=pipeLine.predict(x_test)

#Error
mae=mean_absolute_error(y_test,predictions)

print(predictions,mae)

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(data=data,x="X",y="y",hue="X")
scatterplot,regplot,lmplot,kdeplot,jointplot,swarmplot