#!/usr/bin/env python
# coding: utf-8

# Hello Sravan!
# 
# I’m happy to review your project today.
# I will mark your mistakes and give you some hints how it is possible to fix them. We are getting ready for real job, where your team leader/senior colleague will do exactly the same. Don't worry and study with pleasure!
# 
# Below you will find my comments - **please do not move, modify or delete them**.
# 
# You can find my comments in green, yellow or red boxes like this:
# 
# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Success. Everything is done succesfully.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Remarks. Some recommendations.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Needs fixing. The block requires some corrections. Work can't be accepted with the red comments.
# </div>
# 
# You can answer me by using this:
# 
# <div class="alert alert-block alert-info">
# <b>Student answer.</b> <a class="tocSkip"></a>
# 
# Text here.
# </div>

# ### Project Title: Predicting Market Value of Used Cars for Rusty Bargain
# 
# ##### Objective:
# The goal of this project is to develop a machine learning model that predicts the market value of used cars based on their historical technical specifications and conditions. The model will be integrated into Rusty Bargain’s new app, enabling customers to quickly and accurately assess the value of their vehicles.
# 
# ##### Project Scope:
# ###### The project involves:
# 
#    Analyzing historical car data with various features such as vehicle type, registration year, mileage, and fuel type.
#    Building and comparing machine learning models including gradient boosting methods, random forest, decision tree, and linear regression.
#    Optimizing for both prediction quality and efficiency, balancing training time and speed of predictions.
# 
# ##### Key Evaluation Metric:
# The primary metric for model performance will be the Root Mean Squared Error (RMSE), which will help measure the accuracy of the predicted car prices compared to actual prices.
# 
# ##### Approach:
# 
#    Data Preprocessing: Cleaning and encoding the dataset for model training, addressing any missing values or anomalies.
#    Model Building: Training multiple models and tuning hyperparameters to achieve optimal performance. The models will include:
#         Linear Regression (sanity check)
#         Random Forest
#         Decision Tree
#         LightGBM, XGBoost, and (optionally) CatBoost for gradient boosting.
#    Model Evaluation: Comparing the models based on RMSE, training time, and prediction speed.
#    Conclusion: Recommending the best model for Rusty Bargain’s app based on both performance and efficiency.
# 
# This project will demonstrate strong machine learning practices, including data preprocessing, feature engineering, model optimization, and a thorough comparison of algorithms.

# ## Data preparation
# 
# ### libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
import matplotlib

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import linear_model

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint as sp_randint


# Boosting
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, Pool, metrics, cv
import xgboost as xgb


import warnings


warnings.filterwarnings("ignore")


# ### data import

# In[2]:


try:
    df = pd.read_csv('C:/Users/gsrav/Documents/Documents/Sprint12_Data/car_data.csv')
except:
    print('Something is wrong with your data.')


# ### data review

# In[3]:


display(df.head())


# In[4]:


df.info()


# In[5]:


pd.options.display.max_columns = None
df.describe()


# In[6]:


sns.set_context("notebook", font_scale=.85)

fig, ax = plt.subplots(figsize=(10, 8))
g = sns.countplot(data=df,
                  x='VehicleType',
                  hue='Gearbox',
                  color= '#d0bbff',
                  #palette="dark:#5A9_r",
                  ax=ax,
                  dodge=False,
                  edgecolor = "white"
                  )

ax.set(xlabel="Category Type", ylabel='Count (#)')
bottoms = {}
for bars in ax.containers:
    for bar in bars:
        x, y = bar.get_xy()
        h = bar.get_height()
        if x in bottoms:
            bar.set_y(bottoms[x])
            bottoms[x] += h
        else:
            bottoms[x] = h

for c in ax.containers:
    labels = [f'{round((v.get_height())):,}' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white')

ax.relim()  # the plot limits need to be updated with the moved bars
ax.autoscale()
ax.text(x=0.5, y=1.1, s='Vehicle types', fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.05, s='Gearbox mix analysis', fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)
# sns.despine(offset=10, trim=True)

# if we need to move legend around
# h,l = ax.get_legend_handles_labels()
# ax.legend(h[:4],l[:4], bbox_to_anchor=(1.05, 1), loc=2)

plt.show()


# In[7]:


sns.set_context("notebook", font_scale=.85)

# Create the catplot without the ax parameter and without native_scale
g = sns.catplot(data=df,
                x='FuelType',
                y='Price',
                hue='FuelType',
                palette="flare",
                height=8,
                aspect=2,
                kind='box'
                )

# Set the labels
g.set(xlabel="Fuel Type", ylabel='Price')

# Titles
plt.title('Price to Fuel Type', x=0.5, y=1.1, fontsize=16, weight='bold', ha='center', va='bottom')
g.fig.suptitle('Categorical boxplot analysis', x=0.52, y=1.05, fontsize=8, alpha=0.75, ha='center', va='bottom')

plt.show()


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# Nice visualizations!
#     
# </div>

# ### Findings
# #### Findings
# 
# Features that should be removed / should have no material impact on analysis
# 
#    DateCrawled
# 
#    DateCreated
# 
#    NumberOfPictures
# 
#    LastSeen
# 
#    RegistrationMonth / PostalCode are two others to tentitatively think about removing
# 
# NaN values for features like VehicleType or Model may need to be removed after checking overall impact
# 
#    If needed, something worth exploring is filling in these values with values that are far away from any of their distributions so the model identifies them as 'outliers'. The null values in categorical features may be filled in with something like 'Unknown'.'
#  
# Features with dtype object will have to be encoded for regular training using LabelEncoding then we may need to use specific methods for Gradient Boosting (attempting to run both LightGBM/CatBoost with our already encoded data)
# 
# Possible normalization/standardization needed [placeholder to come back to if needed]

# In[8]:


df.drop(['DateCrawled', 'DateCreated', 'NumberOfPictures', 'LastSeen', 'PostalCode', 'RegistrationMonth'], axis=1, inplace=True)


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# Correct
#     
# </div>

# ### null value statistics

# In[9]:


correlation_filter = ['Price', 'Power', 'Mileage','RegistrationYear']

df[correlation_filter].corr()


# In[10]:


from matplotlib.patches import Rectangle

# Calculate missing data ratio
df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': df_na})
missing_data.reset_index(inplace=True)  # Reset index to use it as a column

# Rename the index column for clarity
missing_data.rename(columns={'index': 'Feature'}, inplace=True)

# Plotting
sns.set_context("notebook", font_scale=.85)

g = sns.catplot(data=missing_data,
                x='Feature',
                y='Missing Ratio',
                kind='bar',
                height=5,
                aspect=2,
                legend=None,
                palette="flare"  # Use a palette instead of hue
                )

# Set the labels
g.set(xlabel='DataFrame Feature', ylabel='Null Proportion (%)')

# Add labels on bars
ax = g.facet_axis(0, 0)
for c in ax.containers:
    labels = [f'{(v.get_height()):.2f}%' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white')

# Title and annotations
ax.autoscale()
ax.text(x=0.5, y=1.1, s='% of missing values', fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.05, s='Selected feature analysis', fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)

# Add a rectangle patch
ax.add_patch(Rectangle((-0.41, 0), .81, 20.25, fill=False, edgecolor='#9999CC', lw=4))
ax.text(0.42, 8.5, "Feature w/the most missing values", fontsize=10, color="#9999CC", rotation=-90)

plt.show()


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# I like your graphs:)
#     
# </div>

# In[11]:


# Debating if we should fill in the missing values with a value

null_value_stats = df.isnull().sum(axis=0)
null_value_stats[null_value_stats != 0]


# ### encoding

# In[12]:


class MultiColumnLabelEncoder: #OrdinalEncoder as a 2nd option
    def __init__(self,columns = None):
        self.columns = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

cat_columns = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired'] 
df_clean = MultiColumnLabelEncoder(columns = cat_columns).fit_transform(df)
display(df_clean)


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# Correct
#     
# </div>

# ## Data Modeling
# 
# ### data splitting and scaling

# In[13]:


# After df_clean is prepared and cleaned, scale numerical and label encoded features

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

# Defining the numerical features
num_features = ['RegistrationYear', 'Power', 'Mileage']

# Scaling the numerical features
scaler = StandardScaler()
df_clean[num_features] = scaler.fit_transform(df_clean[num_features])

# Proceed with splitting features and target
features, target = df_clean.drop('Price', axis=1), df_clean.Price

# Proceed with data splitting
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target,
                                                                            test_size=0.30,
                                                                            random_state=12345)


# ### Decision Tree Regression Model

# In[14]:


get_ipython().run_cell_magic('time', '', 'tree_model = DecisionTreeRegressor(random_state=12345)\ntree_parameters = [{\'max_depth\': [1, 20],\n                    "splitter": [\'best\', \'random\'],\n                    \'max_features\': sp_randint(1, 10)}]\n\ntree_clf = RandomizedSearchCV(tree_model, tree_parameters, cv=5)\ntree_clf.fit(features_train, target_train)\nprint(\'Best Params:\\n\', tree_clf.best_params_)\nprint(\'\')\n\nprint(\'\')\nprint(\'Runtime:\')\n')


# In[15]:


import time
# Retrain best decision tree model and measure training time
best_tree_model = tree_clf.best_estimator_

start_time = time.time()
best_tree_model.fit(features_train, target_train)
training_time = time.time() - start_time

print(f"Retraining Time for Best Decision Tree Model: {training_time:.4f} seconds")


# In[16]:


# Prediction and evaluation
start_time = time.time()
tree_prediction = best_tree_model.predict(features_test)
prediction_time = time.time() - start_time

print(f'Prediction Time: {prediction_time:.4f} seconds')
print('Decision Tree RMSE:\n', math.sqrt(mean_squared_error(target_test, tree_prediction)))


# In[17]:


# Feature Importance
print('Feature Importance:')
{k: v for k, v in sorted(zip(best_tree_model.feature_names_in_, best_tree_model.feature_importances_), key= lambda x: x[1], reverse=True)}


# ### Random Forest Regression Model

# In[18]:


get_ipython().run_cell_magic('time', '', "forest_model = RandomForestRegressor(random_state=12345)\nforest_parameters = [{'max_depth': [2, 10],\n                      'max_features': sp_randint(1, 9)}]\n\nforest_clf = RandomizedSearchCV(forest_model, forest_parameters, cv=5)\nforest_clf.fit(features_train, target_train)\nprint('Best Params:\\n', forest_clf.best_params_)\n\nprint('')\nprint('Runtime:')\n")


# In[19]:


# Retrain best random forest model and measure training time
best_forest_model = forest_clf.best_estimator_

start_time = time.time()
best_forest_model.fit(features_train, target_train)
training_time = time.time() - start_time

print(f"Retraining Time for Best Random Forest Model: {training_time:.4f} seconds")


# In[20]:


# Prediction and evaluation
start_time = time.time()
forest_prediction = best_forest_model.predict(features_test)
prediction_time = time.time() - start_time

print(f'Prediction Time: {prediction_time:.4f} seconds')
print('Random Forest RMSE:\n', math.sqrt(mean_squared_error(target_test, forest_prediction)))


# In[21]:


# Feature Importance
print('Feature Importance:')
{k: v for k, v in sorted(zip(best_forest_model.feature_names_in_, best_forest_model.feature_importances_), key= lambda x: x[1], reverse=True)}


# ### Linear Regression Model

# In[22]:


get_ipython().run_cell_magic('time', '', 'linear_model = LinearRegression()\nlinear_parameters = [{"positive": [True, False],\n                      "fit_intercept": [True, False],\n                      "n_jobs": list(range(1, 9))}]\n\nlinear_clf = RandomizedSearchCV(linear_model, linear_parameters, cv=5)\nlinear_clf.fit(features_train, target_train)\nprint(\'Best Params:\\n\', linear_clf.best_params_)\n\nprint(\'\')\nprint(\'Runtime:\')\n')


# In[23]:


# Retrain best linear regression model and measure training time
best_linear_model = linear_clf.best_estimator_

start_time = time.time()
best_linear_model.fit(features_train, target_train)
training_time = time.time() - start_time

print(f"Retraining Time for Best Linear Regression Model: {training_time:.4f} seconds")


# In[24]:


# Prediction and evaluation
start_time = time.time()
linear_prediction = best_linear_model.predict(features_test)
prediction_time = time.time() - start_time

print(f'Prediction Time: {prediction_time:.4f} seconds')
print('Linear Regression RMSE:\n', math.sqrt(mean_squared_error(target_test, linear_prediction)))


# <div class="alert alert-block alert-info">
# <b>Student answer.</b> <a class="tocSkip"></a>
# 
# Now that everything is running smoothly with the Decision Tree, Random Forest, and Linear Regression models, I have successfully addressed both of the reviewer's comments:
# 
# Scaling Numerical and Label Encoded Features: I've applied scaling correctly to the numerical features.
# Measuring Training Time: I’ve now retrained the best models after RandomizedSearchCV and measured the training time separately.
# </div>

# <div class="alert alert-block alert-danger">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# Everything is correct except two issues:
# 1. If you're going to use any linear models, all the numerical and label encoded features should be scaled. Only OHE features should be leave as they are.
# 2. RandomizedSearchCV traning time and model traning time are not the same things. In RandomizedSearchCV you trained each model a lot of times. But you need to measure only the best model traning time. So, after RandomizedSearchCV you need to take the best model, retrain it and measure this time.
#     
# </div>

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment V2</b> <a class="tocSkip"></a>
# 
# Correct. Good job! But don't forget next time that scaler should be trained only one the train data. Just remember one simple rule: any algorithm from sklearn should be trained only on train data. This is why method fit exists:)
#     
# </div>

# ## Speed and Quality results
# ### speed rankings
# 
# #### CPU Times
# 
# 1. LinearRegression
# 
# 2. DecisionTree
# 
# 3. RandomForest
# 
# #### Wall Times
# 
# 1. LinearRegression
# 
# 2. DecisionTree
# 
# 3. RandomForest
# 
# ### quality rankings
# 
# #### RMSE
# 
# 1. RandomForest
# 
# 2. DecisionTree
# 
# 3. LinearRegression
# 
# #### boosting models
# 
# Gradient Boosting
# 
# 1. LightGBM grows leaf-wise while XGBoost grows level-wise
# 
# 2. LightGBM expects categorical features converted to integer

# In[25]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import train_test_split\n\n# Splitting features and target\nfeatures, target = df_clean.drop('Price', axis=1), df_clean.Price\n\n# Split into training and test sets\nfeatures_train, features_test, target_train, target_test = train_test_split(features,\n                                                                            target,\n                                                                            test_size=0.30,\n                                                                            random_state=12345)\n\n# Further split the training set into training and validation sets\nx_train, x_val, y_train, y_val = train_test_split(features_train.values, \n                                                  target_train.values, \n                                                  test_size=0.20,  # Adjust size as needed\n                                                  random_state=12345)\n\n# Prepare LightGBM datasets\nlgb_features = df_clean[['RegistrationYear', 'Power', 'Mileage']]\nlgb_categorical_features = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'NotRepaired']\n\nlgb_train = lgb.Dataset(x_train, \n                        y_train, \n                        feature_name=lgb_features.columns.tolist(),  \n                        categorical_feature=lgb_categorical_features)\nprint('')\nprint('Runtime:')\n")


# In[26]:


get_ipython().run_cell_magic('time', '', "lgb_parameters = {\n    'task': 'train',    \n    'learning_rate': 0.1,\n    'boosting_type': 'gbdt',\n    'random_state': 12345,\n    'n_estimators': 150,\n    'max_depth': 10,\n    'num_leaves': 100\n}\n\n# Fit the model\ngbm = lgb.LGBMRegressor(**lgb_parameters)\n\ngbm.fit(x_train, y_train,\n        eval_set=[(x_val, y_val)],  # Use the validation set here\n        eval_metric='l1'\n        )\n\nprint('')\nprint('Runtime:')\n")


# <div class="alert alert-block alert-info">
# <b>Student answer.</b> <a class="tocSkip"></a>
# 
# Added and used validation data set as mentioned. Thank You!
# </div>

# <div class="alert alert-block alert-danger">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# You can't use test data in `eval_set`. You can use only validation data for such purpose
#     
# </div>

# #### LightGBM Prediction

# In[27]:


get_ipython().run_cell_magic('time', '', "lgb_prediction = gbm.predict(features_test, num_iteration=gbm.best_iteration_) \nprint('Prediction Time:')\n")


# In[28]:


print('LGBM Regression RMSE:\n', math.sqrt(mean_squared_error(target_test, lgb_prediction)))


# In[29]:


get_ipython().run_cell_magic('time', '', "## Test #2\nlgbm_model = lgb.LGBMRegressor()\nlgbm_model.fit(x_train, y_train)\nprint('')\nprint('Runtime:')\n")


# In[30]:


get_ipython().run_cell_magic('time', '', "lgbm_prediction = lgbm_model.predict(features_test)\nprint('Prediction Time:')\n")


# In[31]:


print('LGBM Regression RMSE v2:\n', math.sqrt(mean_squared_error(target_test, lgbm_prediction)))


# #### Cat Boost

# In[32]:


import time

# Convert numpy arrays back to pandas DataFrames
x_train_df = pd.DataFrame(x_train, columns=features_train.columns)
x_val_df = pd.DataFrame(x_val, columns=features_train.columns)
features_test_df = pd.DataFrame(features_test, columns=features_train.columns)

# Convert categorical columns to strings
for col in lgb_categorical_features:
    x_train_df[col] = x_train_df[col].astype(str)
    x_val_df[col] = x_val_df[col].astype(str)
    features_test_df[col] = features_test_df[col].astype(str)

# Get categorical feature indices
cat_features_indices = [x_train_df.columns.get_loc(col) for col in lgb_categorical_features]

# Prepare CatBoost-specific parameters
catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 12345,
    'early_stopping_rounds': 50,
    'verbose': 100
}

# Create CatBoost Pool objects for training and validation
train_pool = Pool(x_train_df, 
                  y_train,
                  cat_features=cat_features_indices)
val_pool = Pool(x_val_df, 
                y_val,
                cat_features=cat_features_indices)

# Initialize and train the model
catboost_model = CatBoostRegressor(**catboost_params)
catboost_model.fit(train_pool,
                   eval_set=val_pool,
                   use_best_model=True,
                   plot=True)

# Measure prediction time
start_time = time.time()
catboost_predictions = catboost_model.predict(features_test_df)
prediction_time = time.time() - start_time
print(f'Prediction Time: {prediction_time:.4f} seconds')

# Calculate and print RMSE
catboost_rmse = math.sqrt(mean_squared_error(target_test, catboost_predictions))
print(f'CatBoost Regression RMSE: {catboost_rmse}')

# Feature importance
feature_importance = catboost_model.feature_importances_
feature_names = features_train.columns.tolist()
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_feature_importance = {k: v for k, v in sorted(feature_importance_dict.items(), 
                                                    key=lambda x: x[1], 
                                                    reverse=True)}
print('\nFeature Importance:')
for feature, importance in sorted_feature_importance.items():
    print(f'{feature}: {importance}')


# <div class="alert alert-block alert-info">
# <b>Student answer.</b> <a class="tocSkip"></a>
# 
# Added and used validation data set as mentioned. Thank You!
# </div>

# <div class="alert alert-block alert-danger">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# You can't use test data in `eval_set`. You can use only validation data for such purpose
#     
# </div>

# ##### XGBoost
# 
# Regression Matrices
# 
#     1. enable_categorical set to True in order to enable automatic encoding

# In[33]:


import time

# Prepare data for XGBoost
x_train_xgb = x_train_df.copy()
x_val_xgb = x_val_df.copy()
features_test_xgb = features_test_df.copy()

# Convert categorical columns to numeric using label encoding
label_encoders = {}
for col in lgb_categorical_features:
    # Combine all unique values from train, validation, and test sets
    all_values = pd.concat([x_train_df[col], x_val_df[col], features_test_df[col]]).unique()
    
    # Initialize and fit the label encoder with all possible values
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(all_values)
    
    # Transform each dataset
    x_train_xgb[col] = label_encoders[col].transform(x_train_df[col])
    x_val_xgb[col] = label_encoders[col].transform(x_val_df[col])
    features_test_xgb[col] = label_encoders[col].transform(features_test_df[col])

# Convert to DMatrix format
dtrain = xgb.DMatrix(x_train_xgb, label=y_train)
dval = xgb.DMatrix(x_val_xgb, label=y_val)
dtest = xgb.DMatrix(features_test_xgb)

# Set XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 12345
}

# Train XGBoost model
print("Starting XGBoost training...")
start_time = time.time()
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)
training_time = time.time() - start_time
print(f'Training Time: {training_time:.4f} seconds')

# Make predictions
print("\nMaking predictions...")
start_time = time.time()
xgb_predictions = xgb_model.predict(dtest)
prediction_time = time.time() - start_time
print(f'Prediction Time: {prediction_time:.4f} seconds')

# Calculate RMSE
xgb_rmse = math.sqrt(mean_squared_error(target_test, xgb_predictions))
print(f'XGBoost RMSE: {xgb_rmse}')

# Get feature importance
print("\nCalculating feature importance...")
importance_dict = xgb_model.get_score(importance_type='gain')
# Normalize feature importance
total_importance = sum(importance_dict.values())
normalized_importance = {k: v/total_importance for k, v in importance_dict.items()}
sorted_importance = dict(sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True))

print('\nTop 10 Feature Importance:')
for feature, importance in list(sorted_importance.items())[:10]:
    print(f'{feature}: {importance:.4f}')

# Optional: Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title('XGBoost Top 10 Feature Importance')
plt.tight_layout()
plt.show()


# <div class="alert alert-block alert-info">
# <b>Student answer.</b> <a class="tocSkip"></a>
# 
# Added and used validation data set as mentioned. Thank You!
# </div>

# <div class="alert alert-block alert-danger">
# <b>Reviewer's comment V1</b> <a class="tocSkip"></a>
# 
# You can't use test data as 'validation'. You can use only validation data for such purpose
#     
# </div>

# In[34]:


# Collect all results
model_results = {
    'Linear Regression': {
        'RMSE': math.sqrt(mean_squared_error(target_test, linear_prediction)),
        'Prediction Time': None  # Add if you recorded it
    },
    'Decision Tree': {
        'RMSE': math.sqrt(mean_squared_error(target_test, tree_prediction)),
        'Prediction Time': None  # Add if you recorded it
    },
    'Random Forest': {
        'RMSE': math.sqrt(mean_squared_error(target_test, forest_prediction)),
        'Prediction Time': None  # Add if you recorded it
    },
    'LightGBM': {
        'RMSE': math.sqrt(mean_squared_error(target_test, lgbm_prediction)),
        'Prediction Time': None  # Add if you recorded it
    },
    'CatBoost': {
        'RMSE': math.sqrt(mean_squared_error(target_test, catboost_predictions)),
        'Prediction Time': prediction_time  # From CatBoost section
    },
    'XGBoost': {
        'RMSE': math.sqrt(mean_squared_error(target_test, xgb_predictions)),
        'Prediction Time': prediction_time  # From XGBoost section
    }
}

# Create comparison DataFrame
results_df = pd.DataFrame(model_results).transpose()
results_df = results_df.sort_values('RMSE')

# Plotting results
plt.figure(figsize=(12, 6))
ax = results_df['RMSE'].plot(kind='bar')
plt.title('Model Comparison - RMSE')
plt.ylabel('RMSE')
plt.xlabel('Models')
plt.xticks(rotation=45)
for i, v in enumerate(results_df['RMSE']):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Feature importance comparison
def get_top_features(model, feature_names, n=5):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_score'):
        importances = model.get_score(importance_type='gain').values()
    else:
        return None
    
    return dict(sorted(zip(feature_names, importances), 
                       key=lambda x: x[1], reverse=True)[:n])

# Collect top features from each model
feature_importance = {
    'Random Forest': get_top_features(best_forest_model, features.columns),
    'LightGBM': get_top_features(lgbm_model, features.columns),
    'CatBoost': get_top_features(catboost_model, features.columns),
    'XGBoost': get_top_features(xgb_model, features.columns)
}

# Print feature importance comparison
print("\nTop 5 Important Features by Model:")
for model, features in feature_importance.items():
    print(f"\n{model}:")
    for feature, importance in features.items():
        print(f"{feature}: {importance:.4f}")

# Save models
import joblib

models_to_save = {
    'linear': best_linear_model,
    'tree': best_tree_model,
    'forest': best_forest_model,
    'lgbm': lgbm_model,
    'catboost': catboost_model,
    'xgboost': xgb_model
}

# Save label encoders for future use
joblib.dump(label_encoders, 'label_encoders.joblib')

# Save each model
for name, model in models_to_save.items():
    joblib.dump(model, f'{name}_model.joblib')

# Final recommendations
print("\nFinal Recommendations:")
best_model = results_df.index[0]
print(f"1. Best Model: {best_model} with RMSE of {results_df.loc[best_model, 'RMSE']:.2f}")
print("\n2. Key Features:")
for feature, importance in list(feature_importance[best_model].items())[:3]:
    print(f"   - {feature}")

print("\n3. Model Selection Considerations:")
print(f"   - Fastest prediction time: {results_df['Prediction Time'].idxmin()}")
print(f"   - Best accuracy (lowest RMSE): {best_model}")

# Example of how to load and use the model for future predictions
print("\nExample of using the saved model:")
print("```python")
print("import joblib")
print(f"model = joblib.load('{best_model.lower()}_model.joblib')")
print("label_encoders = joblib.load('label_encoders.joblib')")
print("# Prepare new data using the same preprocessing steps")
print("predictions = model.predict(new_data)")
print("```")


# In[ ]:


# Example of using final code:
'''
import joblib
model = joblib.load('lgbm_model.joblib')
label_encoders = joblib.load('label_encoders.joblib')

# Example of new data
# Replace this with your actual new data
new_data = pd.DataFrame({
    'RegistrationYear': [2015],
    'Power': [110],
    'Brand': ['Volkswagen'],
    'VehicleType': ['SUV'],
    'Mileage': [80000],
    'NotRepaired': ['no']
})

# Apply label encoding if necessary
for column in label_encoders:
    if column in new_data:
        new_data[column] = label_encoders[column].transform(new_data[column])
predictions = model.predict(new_data)
'''


# ## Conclusion:
# 
# Based on the feature importance and model performance analysis, LightGBM is the best model for predicting the market value of cars, achieving the lowest RMSE of 1857.00. This model balances accuracy and efficiency, with the following key insights:
# 
# ### Key Features:
# 
# The most important features across the models are consistent, especially:<br>
#     RegistrationYear <br>
#     Power <br>
#     Brand These features strongly influence the price prediction, making them critical for future predictions.<br>
#     
# ### Model Performance:
# 
#    LightGBM had the best accuracy (lowest RMSE) and a strong focus on key features such as RegistrationYear, Power, and Brand.<br>
#    XGBoost offers the fastest prediction time but is less accurate compared to LightGBM.
# 
# ### Recommendations for Future Predictions:
# 
# LightGBM should be the go-to model for predictions due to its superior accuracy. <br>
# 
# RegistrationYear and Power should be prioritized when analyzing new data, as they consistently rank highly across all models.<br>
# 
# 
# In conclusion, <b>LightGBM</b> with the features <b>RegistrationYear, Power and Brand</b> should be used for the most accurate market value predictions for cars, while XGBoost can be considered for scenarios that require faster predictions.
