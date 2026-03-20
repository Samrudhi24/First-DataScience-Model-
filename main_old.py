import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1 Load the Data;
housing=pd.read_csv("housing.csv")

# 2  Create Stratified test set 
housing["income_cat"]=pd.cut(housing["median_income"],
                          bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                          labels=[1, 2, 3, 4, 5]
)
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1) # we will work on this data
    strat_test_set=housing.loc[test_index].drop("income_cat",axis=1) # Test Data 20%

# 3 We will work on copy of training set Data
housing=strat_train_set.copy()

# 4 List the features and labels
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)

print(housing,housing_labels)

# 5 Seperate numerical and categorical values
num_attributes=housing.drop("ocean_proximity",axis=1) .columns.tolist()
cat_attributes=["ocean_proximity"]

# 6 Lets Build Pipeline 
#for Numerical value
num_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())

])
#for categorical value
cat_pipeline=Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))

])
#Construt the full pipeline
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attributes),
    ("cat",cat_pipeline,cat_attributes)
])

# 7 Transforn the Data
housing_prepared=full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 8 Train the model
#LinearRegression Module
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred=lin_reg.predict(housing_prepared)
lin_rmse=root_mean_squared_error(housing_labels,lin_pred)
print(f"The rrot mean squared error for LinearRegression is {lin_rmse}")




#DecisionTree Module
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_pred=dec_reg.predict(housing_prepared)
#dec_rmse=root_mean_squared_error(housing_labels,dec_pred)
dec_rmses=-cross_val_score(dec_reg,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10) 
#print(f"The root mean squared error for DecisionTreeRegressor is {dec_rmse}")
print(pd.Series(dec_rmses).describe())




#RandomForestRegressor Module # its good than other
random_forest_reg=RandomForestRegressor()
random_forest_reg.fit(housing_prepared,housing_labels)
random_forest_pred=random_forest_reg.predict(housing_prepared)
random_forest_rmse=root_mean_squared_error(housing_labels,random_forest_pred)
print(f"The root mean squared error for randomforest is {random_forest_rmse}")



