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

# In[1]:


pip install --user plotly --upgrade


# In[2]:


pip install --user --upgrade scikit-learn


# # Project description
# 
# Sweet Lift Taxi company has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the amount of taxi orders for the next hour. Build a model for such a prediction.
# 
# The RMSE metric on the test set should not be more than 48.
# 
# ## Project instructions
# 
# 1. Download the data and resample it by one hour.
# 2. Analyze the data.
# 3. Train different models with different hyperparameters. The test sample should be 10% of the initial dataset. 
# 4. Test the data using the test sample and provide a conclusion.
# 
# ## Data description
# 
# The data is stored in file `taxi.csv`. The number of orders is in the '*num_orders*' column.

# ## Initialization
# 
# ### libraries

# In[3]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[4]:


import pandas as pd
import numpy as np
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import randint as sp_randint
from numpy import unique

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.express as px
# px.defaults.template = "ggplot2"
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ### data import

# In[5]:


try:
    data = pd.read_csv('/datasets/taxi.csv', index_col=[0], parse_dates=[0])
except:
    data = pd.read_csv('/datasets/taxi.csv', index_col=[0], parse_dates=[0])


# ### data review

# In[6]:


data.shape


# In[7]:


data.info() # no perceived NaN values


# In[8]:


data.isna().sum()


# In[9]:


data.describe()


# In[10]:


display(data)


# ### data sorting

# In[11]:


data.sort_index(inplace=True)


# ### data resampling - hourly

# In[12]:


## rs = resampled

data = data.resample('1H').sum()

df_2 = data.copy()
monthly_data = df_2.resample('M').sum()

df_3 = data.copy()
daily_data = df_3.resample('D').sum()


# In[13]:


display(data)
data.info()


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Correct
# 
# </div>

# ## Analysis
# 
# ### data plotting

# In[14]:


fig = px.line(data,
             title="Number of Airport Taxi Orders<br><sup>6 month horizon</sup>",
              labels={
                     "datetime": "Month",
                     "value": "Taxi Orders"
                 })

fig.update_layout(showlegend=False, autosize=False)

fig.show()


# ### average number of orders per month

# In[15]:


grouped_data_mean = data.groupby(by=[data.index.month]).mean()
display(grouped_data_mean)

grouped_data_sum = data.groupby(by=[data.index.month]).sum()


# In[16]:


fig = px.bar(grouped_data_mean, 
              title="Average Taxi Orders per Month",
              labels={
                     "datetime": "Month (#)",
                     "value": "Average Taxi Orders"
                 }, text_auto=True)

fig.update_layout(showlegend=False, autosize=False)
fig.update_layout(yaxis= dict(title='Taxi Orders', showticklabels=False),
                  xaxis=dict(title='Month',
                  # gridcolor='grey',
                  #tickmode= 'array',
                  #tickmode= 'linear',
                  #tick0= 3,
                  #dtick=2,
                  tickvals= [3, 4, 5, 6, 7, 8],
                  ticktext = ["March", "April", "May", "June", "July", "August"]))

fig.show()


# ### Findings
# 
# Taxi Orders - Orders per Month
# 
#     Total and average taxi order increases as we head into the Fall, August volume is double of that of March possibly indicating (without looking at any additional data or series horizon) a strong relationship between taxi orders and the holiday season (e.g., summer vacations).

# ## Trends and Seasonality

# In[17]:


decomposed = seasonal_decompose(data, extrapolate_trend='freq')

trend_data = decomposed.trend
seasonal_data = decomposed.seasonal
residual_data = decomposed.resid


# In[18]:


fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig.append_trace(go.Scatter(
    x = trend_data.index, y= trend_data
), row=1, col=1)

fig.append_trace(go.Scatter(
    x = seasonal_data.index, y= seasonal_data
), row=2, col=1) ## assuming this is shown this way given the short time frame (not enough for seasonality)

fig.append_trace(go.Scatter(
    x = residual_data.index, y= residual_data
), row=3, col=1)


fig.update_layout(height=800, width=1000, title_text="Trends and Seasonality<br><sup>Hourly basis</sup>")
fig.update_yaxes(title_text="Trends", row=1, col=1)
fig.update_yaxes(title_text="Seasonality <br><sup>no seasonality due to short horizon</sup>", row=2, col=1)
fig.update_yaxes(title_text="Residuals", row=3, col=1)

fig.update_layout(showlegend=False)


fig.show()


# ### Findings
# 
# Taxi Orders - Hourly Trends and Seasonality
# 
#     From our 2018 dataset, which was resampled to an hourly frequency, we see an upwards trend in taxi orders as we head into the later months of the year. While we see no seasonality due to the short series horizon we are given, if the data were to be similar each year then we could safely assume there is/will be predictability with airport taxi orders as orders contract during the winter/spring and expand the following seasons (summer/fall).

# In[19]:


decomposed_resampled = seasonal_decompose(monthly_data, extrapolate_trend='freq', period=2)
# .asfreq('MS')

sliced_trend_data = decomposed_resampled.trend
sliced_seasonal_data = decomposed_resampled.seasonal
sliced_residual_data = decomposed_resampled.resid

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig.append_trace(go.Scatter(
    x = sliced_trend_data.index, y= sliced_trend_data #['2018-05':'2018-08']
), row=1, col=1)

fig.append_trace(go.Scatter(
    x = sliced_seasonal_data.index, y= sliced_seasonal_data #['2018-05':'2018-08']
), row=2, col=1) ## assuming this is shown this way given the short time frame (not enough for seasonality)

fig.append_trace(go.Scatter(
    x = sliced_residual_data.index, y= sliced_residual_data #['2018-05':'2018-08']
), row=3, col=1)


fig.update_layout(height=800, width=1000, title_text="Monthly Trends and Seasonality<br><sup>Monthly basis</sup>")
fig.update_yaxes(title_text="Trends", row=1, col=1)
fig.update_yaxes(title_text="Seasonality", row=2, col=1)
fig.update_yaxes(title_text="Residuals", row=3, col=1)

fig.update_layout(showlegend=False)


fig.show()


# ### data validation - decomposed data vs original dataset

# In[20]:


print(decomposed.trend[1:6] + decomposed.seasonal[1:6] + decomposed.resid[1:6])
print()
print(data.num_orders[1:6])


# ### time series differencing

# In[21]:


ts_difference = data - data.shift()

fig = px.line(grouped_data_sum, 
              title="Original Dataset",
              labels={
                     "datetime": "Month (#)",
                     "value": "Taxi Orders"
                 },#text_auto=True
                 )
fig.update_layout(showlegend=False, autosize=False)
fig.update_layout(yaxis= dict(title='Total Taxi Orders', showticklabels=False),
                  xaxis=dict(title='Month',
                  # gridcolor='grey',
                  #tickmode= 'array',
                  #tickmode= 'linear',
                  #tick0= 3,
                  #dtick=2,
                  tickvals= [3, 4, 5, 6, 7, 8],
                  ticktext = ["March", "April", "May", "June", "July", "August"]))

fig.show()

fig = px.line(ts_difference, 
              title="Time Series Differencing",
              labels={
                     "datetime": "Month",
                     "value": "Taxi Orders"
                 })

fig.update_layout(showlegend=False, autosize=False, yaxis= dict(title='Total Taxi Orders'))

fig.show()


# ### stationary series

# In[22]:


ts_difference['mean'] = ts_difference['num_orders'].rolling(15).mean()
ts_difference['std'] = ts_difference['num_orders'].rolling(15).std()

fig = px.line(ts_difference, 
              title="Orders / Mean / Standard Deviation",
              labels={
                     "datetime": "Month",
                     "value": "Taxi Orders",
                 })

fig.update_layout(legend_title_text='', autosize=False)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1,
    bgcolor="white",
    bordercolor="lightgray",
    borderwidth=1,
    ), 
)
newnames = {'num_orders':'Orders', 'mean': 'Mean', 'std': 'Standard Deviation'}
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                      )
                                     )

fig.show()


# ### Findings
# 
# Taxi Orders - Sum, Mean and Standard Deviation
# 
#     Through the use of time series differencing, we create a more stationary series which which makes it easier to model and forecast. We stabilize the mean and reduce trend/seasonality. We then quickly compare to the mean and standard deviation to verify this stabilization.

# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Amazing EDA. Well done!
# 
# </div>

# ## Data Modeling
# 
# ### features function

# In[23]:


def create_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek

    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)

    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()

create_features(data, 11, 8)


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Correct
# 
# </div>

# ### data splitting

# In[24]:


train, test = train_test_split(data, shuffle=False, test_size=0.091) #splitting without reshuffling
train = train.dropna()

print(train.shape)
print(test.shape)

print(test.shape[0] / train.shape[0] * 100) ## 10% of train dataset


# In[25]:


print(train.index.min(), train.index.max())
print(test.index.min(), test.index.max())


# In[26]:


print('Median daily taxi orders:', test['num_orders'].median())

pred_previous = test.shift()
#print(pred_previous) # shifting to find differences
pred_previous.iloc[0] = train.iloc[-1:]
#print(pred_previous.iloc[0]) # previous value
print('RMSE:', math.sqrt(mean_squared_error(test, pred_previous))) # using previous value as a baseline prediction 


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Good job!
# 
# </div>

# ### features & targets

# In[27]:


features_train = train.drop(['num_orders'], axis=1)
target_train = train['num_orders']
features_test = test.drop(['num_orders'], axis=1)
target_test = test['num_orders']


# ### feature importance

# In[28]:


def feature_imp(model):
    feature_name = model.best_estimator_.feature_names_in_
    feature_importance = model.best_estimator_.feature_importances_
    return feature_name, feature_importance


# ### model training

# In[29]:


tscv = TimeSeriesSplit(n_splits=5)


# ### regression and tree models

# In[30]:


get_ipython().run_cell_magic('time', '', 'linear_model = LinearRegression()\nlinear_parameters = [{"positive": [True, False],\n                      "fit_intercept": [True, False],\n                      "n_jobs": list(range(1,250))}]\n\nlinear_clf = RandomizedSearchCV(linear_model, linear_parameters, scoring=\'neg_root_mean_squared_error\', cv=tscv)\nlinear_clf.fit(features_train, target_train)\nlinear_prediction = linear_clf.predict(features_test)\n\nprint(\'Best Params:\\n\', linear_clf.best_params_)\nprint()\nprint(\'Runtime:\')\n')


# In[31]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x = target_test, y = linear_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="m")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('Linear Regression Model', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

g.fig.tight_layout()
plt.show()


# In[33]:


get_ipython().run_cell_magic('time', '', 'ridge_model = Ridge(random_state=12345)\nridge_parameters = [{"alpha": list(range(0,1000)), \n                     "fit_intercept": [True, False], \n                     "solver": [\'svd\', \'cholesky\', \'lsqr\', \'sparse_cg\', \'sag\', \'saga\']}]\n\nridge_clf = RandomizedSearchCV(ridge_model, ridge_parameters, scoring=\'neg_root_mean_squared_error\', cv=tscv)\nridge_clf.fit(features_train, target_train)\nridge_prediction = ridge_clf.predict(features_test)\n\nprint(\'Best Params:\\n\', ridge_clf.best_params_)\nprint()\nprint(\'Runtime:\')\n')


# In[34]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x=target_test, y=ridge_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="m")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('Ridge Regression Model', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

# Use g.ax_joint to get the joint axis and customize it
g.ax_joint.tick_params(color='m', labelcolor='m')
for spine in g.ax_joint.spines.values():
    spine.set_edgecolor('m')

g.fig.tight_layout()
plt.show()


# In[35]:


get_ipython().run_cell_magic('time', '', 'sgd_model = SGDRegressor(random_state=12345)\nsgd_parameters = [{  "alpha": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0010],\n                     "fit_intercept": [True, False], \n                     "learning_rate": [\'constant\', \'optimal\', \'invscaling\', \'adaptive\'],\n                     "penalty": [\'l2\', \'l1\', \'elasticnet\', None],\n                      "average": [True, False],\n                      "shuffle": [True, False],\n                      "early_stopping": [True, False],\n                       "max_iter": [1000, 6000]}]\n\nsgd_clf = RandomizedSearchCV(sgd_model, sgd_parameters, scoring=\'neg_root_mean_squared_error\', cv=tscv)\nsgd_clf.fit(features_train, target_train)\nsgd_prediction = sgd_clf.predict(features_test)\n\nprint(\'Best Params:\\n\', sgd_clf.best_params_)\nprint()\nprint(\'Runtime:\')\n')


# In[36]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x = target_test, y = sgd_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="m")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('SGDRegressor Model', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

g.fig.tight_layout()
plt.show()


# In[37]:


get_ipython().run_cell_magic('time', '', 'tree_model = DecisionTreeRegressor(random_state=12345)\ntree_parameters = [{\'max_depth\': [1, 10],\n                    "splitter": [\'best\', \'random\'],\n                    \'max_features\': sp_randint(1, 12)}]\n\ntree_clf = RandomizedSearchCV(tree_model, tree_parameters, scoring=\'neg_root_mean_squared_error\', cv=tscv)\ntree_clf.fit(features_train, target_train)\ntree_prediction = tree_clf.predict(features_test)\n\nprint(\'Best Params:\\n\', tree_clf.best_params_)\nprint()\nprint(\'Runtime:\')\n')


# In[38]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x = target_test, y = tree_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="b")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('Decision Tree Model', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

g.fig.tight_layout()
plt.show()


# In[39]:


print('Feature Importance:')
{k: v for k, v in sorted(zip(tree_clf.best_estimator_.feature_names_in_, tree_clf.best_estimator_.feature_importances_), key= lambda x: x[1], reverse=True)}


# In[43]:


get_ipython().run_cell_magic('time', '', 'forest_model = RandomForestRegressor(random_state=12345)\nforest_parameters = [{\'max_depth\': [2, 30],\n                      \'max_features\': sp_randint(1, 20),\n                     \'min_samples_split\': [2, 12],\n                     "criterion": [\'squared_error\', \'absolute_error\', \'friedman_mse\', \'poisson\'],\n                     "warm_start": [True, False],\n                     \'n_estimators\': [300]}]\n\nforest_clf = RandomizedSearchCV(forest_model, forest_parameters, scoring=\'neg_root_mean_squared_error\', cv=tscv)\nforest_clf.fit(features_train, target_train)\nforest_prediction = forest_clf.predict(features_test)\n\nprint(\'Best Params:\\n\', forest_clf.best_params_)\nprint()\nprint(\'Runtime:\')\n')


# In[44]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x = target_test, y = forest_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="b")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('Random Forest Model', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

g.fig.tight_layout()
plt.show()


# In[45]:


print('Feature Importance:')
{k: v for k, v in sorted(zip(forest_clf.best_estimator_.feature_names_in_, forest_clf.best_estimator_.feature_importances_), key= lambda x: x[1], reverse=True)}


# In[46]:


get_ipython().run_cell_magic('time', '', "extra_model = ExtraTreesRegressor(random_state=12345)\nextra_parameters = [{'max_depth': [2, 20],\n                      'max_features': sp_randint(1, 22), \n                     'n_estimators': [150]}]\n\nextra_clf = RandomizedSearchCV(extra_model, extra_parameters, scoring='neg_root_mean_squared_error', cv=tscv)\nextra_clf.fit(features_train, target_train)\nextra_prediction = extra_clf.predict(features_test)\n\nprint('Best Params:\\n', extra_clf.best_params_)\nprint()\nprint('Runtime:')\n")


# In[47]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x = target_test, y = extra_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="b")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('Extra Trees Model', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

g.fig.tight_layout()
plt.show()


# In[48]:


print('Feature Importance:')
{k: v for k, v in sorted(zip(extra_clf.best_estimator_.feature_names_in_, extra_clf.best_estimator_.feature_importances_), key= lambda x: x[1], reverse=True)}


# ### gradient boosting models

# In[49]:


get_ipython().run_cell_magic('time', '', "\nlgb_reg = lgb.LGBMRegressor(random_state=12345)\n# **lgb_parameters\n\nlgb_reg_parameters = {  \n'learning_rate': [0.01, 0.05, 0.08, 0.09, 0.1, 0.11, 0.15, 0.2]\n, 'boosting_type': ['gbdt', 'dart']\n, 'max_depth': [1, 20]\n, 'num_leaves': [31, 175]\n, 'n_estimators': [20, 300]\n, 'class_weight': ['balanced', None]\n, 'n_jobs': [1, 30]\n, 'importance_type': ['split', 'gain']\n, 'min_child_samples': [20,55]\n}\n\nlgb_clf = RandomizedSearchCV(lgb_reg, lgb_reg_parameters, scoring='neg_root_mean_squared_error', cv=tscv)\nlgb_clf.fit(features_train, target_train)\nlgb_prediction = lgb_clf.predict(features_test)\n\nprint('')\nprint('Runtime:')\n")


# In[50]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x = target_test, y = lgb_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="g")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('LightGBM', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

g.fig.tight_layout()
plt.show()


# In[51]:


boost = lgb_reg.fit(features_train, target_train).booster_
# print('Feature names',boost.feature_name())

print('Feature Importance:')
{k: v for k, v in sorted(zip(boost.feature_name(), lgb_clf.best_estimator_.feature_importances_), key= lambda x: x[1], reverse=True)}


# In[52]:


get_ipython().run_cell_magic('time', '', 'xgb_param = {"booster": ["gblinear", \'dart\', \'gbtree\'],\n             "validate_parameters": [True, False],\n             "max_depth": [1, 10],\n             "subsample": [0.5, 1],\n             "sampling_method": ["uniform", "gradient_based", "subsample"],\n             "tree_method": ["auto", "exact", "approx"],\n             "max_leaves": [2, 10],\n             "feature_selector": ["shuffle", "cyclic", "random"],\n             "updater": ["coord_descent", "shotgun"],\n             "eta": [0.1, 0.2, 0.3]\n        }\n\n\nxgb_reg = XGBRegressor(random_state=12345)\nxgb_clf = RandomizedSearchCV(xgb_reg, xgb_param, scoring=\'neg_root_mean_squared_error\', cv=tscv)\nxgb_clf.fit(features_train, target_train)\nxgb_prediction = xgb_clf.predict(features_test)\n\nprint(\'\')\nprint(\'Runtime:\')\n')


# In[53]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

g = sns.jointplot(x = target_test, y = xgb_prediction, kind='reg', palette='mako', height=8, ratio=3, marginal_ticks=True, color="g")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

plt.title('XGB Regressor', fontsize=10)
g.set_axis_labels('Target Test', 'Prediction', fontsize=10)

g.fig.tight_layout()
plt.show()


# ## Model Comparison

# In[54]:


def best_score(model_name, model_type):
    print(f'Best Score for the {model_name} model:\n', model_type.best_score_)


# In[55]:


best_score('LinearRegression', linear_clf)
print()
best_score('RidgeRegression', ridge_clf)
print()
best_score('TreeRegressor', tree_clf)
print() 
best_score('SGDRegressor', sgd_clf)
print()
best_score('RandomForestRegressor', forest_clf)
print()
best_score('ExtraRegressor', extra_clf)
print()
best_score('LGBRegressor', lgb_clf)
print()
best_score('XGBRegressor', xgb_clf)


# ### Findings
# 
# Model Training - Scoring comparison
# 
#     Through the use of RandomizedSearchCV, we compared scores for all of our regression models including those leveraging boosting methods using scoring='neg_root_mean_squared_error'. Those closest to zero were the Extra Tree Regressor as well as the Light Gradient Regressor.

# ## Final Evaluation

# In[56]:


def eval_regressor(model, y_true, y_pred):
    
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f'{model} RMSE: {rmse:.2f}')

eval_regressor('LGB Regressor', target_test, lgb_prediction) # final evaluation on test set


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# You did great job! Everything is correct. I'd say everything is perfect:)
# 
# </div>

# ## Conclusion
# 
# ### Data
# 
#     Based on the available data provided by Sweet Lift, we see increases in the number of taxi orders (per hour) largely starting from the summer towards the fall seasons. This can be due to 1) holidays, 2) decreases in airline prices (allowing for more people to travel), 3) weather (rain/cold leading to more taxi orders for example). Average taxi orders double from March to August of 2018.
# 
#     While further seasonality analysis would be helpful, due to the limited time horizon given to us there isn't much to decipher from this aside from making very general assumptions (like increases due to holidays).
# 
#     The analysis would benefit, in my opinion, from 1) a much wider scope, 2) airline price information, 3) weather information, 4) taxi price data (as well as competitor pricing), 5) any marketing/promotion information (by Sweet Lift), 6) booking information (calling vs app usage), 7) vehicle (taxi) information (model, type, etc...).
# 
# ### Models
# 
#     Model training begain with feature creation, adding features to the data set based on datetime like: year, month, day, lag and rolling mean. This was then split with a test size target of 10% of the training set, along with all NaN values being dropped from the training set.
# 
#     Once complete, we ran simple forecasting (checking the time horizon) and tested for accuracy to make sure there weren't any issues as we dive deeper into model training.
# 
#     Using RandomizedSearchCV we trained various regression models and tuned their hyperparameters. Various of these models weren't providing the desired training results (see SGDRegressor for example) or were becoming overfitted. Gradient Boosting methods were then deployed as the desired RMSE result was hard to achieve.
# 
#     Once we honed in on an optimal selection, being the Light Gradient Boosting Method, we performed evaluations on the test dataset. While LGBM training takes longer and it's more sensitive to overfitting, this was the selection due to the overall advantages/flexibility of LGBM parameter tuning (e.g., boosting) and it should be a good starting base model if we were to introduce more features/pattern complexity (like some of the ones mentioned above).
# 
#     Evaluating the test dataset, we achieved an RMSE value of ~44. This was partially achieved by tuning the max_lag and rolling_mean parameters as we were creating the features. These parameters seem to have played a significant role (compared to some of the hyperparameters) in getting the RMSE near the target value.
