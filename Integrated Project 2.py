#!/usr/bin/env python
# coding: utf-8

# **Review**
# 
# Hello Sravan!
# 
# I'm happy to review your project today.
#   
# You can find my comments in colored markdown cells:
#   
# <div class="alert alert-success">
#   If everything is done successfully.
# </div>
#   
# <div class="alert alert-warning">
#   If I have some (optional) suggestions, or questions to think about, or general comments.
# </div>
#   
# <div class="alert alert-danger">
#   If a section requires some corrections. Work can't be accepted with red comments.
# </div>
#   
# Please don't remove my comments, as it will make further review iterations much harder for me.
#   
# Feel free to reply to my comments or ask questions using the following template:
#   
# <div class="alert alert-info">
#   Thank you so much for your feedbacks. I've split the cells into multiple so it's easier. Hopefully i got it right this time. Thank you!
# </div>
#   
# First of all, thank you for turning in the project! You did a great job overall, but there are some small problems that need to be fixed before the project will be accepted. Let me know if you have any questions!
# 

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment V2</b> <a class="tocSkip"></a>
# 
# Please, do not remove my comments next time. I have no idea what was incorrect previous time. But now everything looks correct. Well done! 
#   
# </div>

# <b> Project Description </b>
# 
# The project involves analyzing and modeling gold recovery data from three datasets:
# 
#    gold_recovery_train.csv: Training dataset<br>
#    gold_recovery_test.csv: Test dataset<br>
#    gold_recovery_full.csv: Source dataset with all features
# 
# <b> Objective: </b> Prepare the data, analyze it, and build a predictive model for gold recovery.
# 
# <b> Instructions: </b>
#     
# Prepare the Data:
#      Open and inspect the datasets.<br>
#      Verify the accuracy of recovery calculations in the training set.<br>
#      Analyze the features not available in the test set, including their types.<br>
#      Perform necessary data preprocessing.
# 
# <b> Analyze the Data: </b><br>
#      Observe how metal concentrations change at different purification stages.<br>
#      Compare feed particle size distributions between the training and test sets.<br>
#      Evaluate the total concentrations at different stages for anomalies and address any issues.
# 
# <b> Build the Model: </b><br>
#      Create a function to calculate the final sMAPE (symmetric Mean Absolute Percentage Error) value.<br>
#      Train and evaluate various models using cross-validation. Select the best model and test it on the test sample.
# 
# <b> Evaluation Metrics: </b> Follow the provided formulas for calculating evaluation metrics.

# In[1]:


import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.stats import iqr
from itertools import islice

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MaxAbsScaler, RobustScaler
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor


# <b> Explore Data </b>

# In[3]:


try:
    df_full = pd.read_csv('/datasets/gold_recovery_full.csv')
    df_train = pd.read_csv('/datasets/gold_recovery_train.csv')
    df_test = pd.read_csv('/datasets/gold_recovery_test.csv')

except:
    print('Something is wrong with your data')


# In[4]:


df_full.info()
df_full.shape


# In[5]:


df_train.info()
df_train.shape
df_train.drop(columns='date', index=1, inplace=True)


# In[6]:


df_test.info()
df_test.shape
# df_test.drop(columns='date', index=1, inplace=True)


# In[7]:


print((df_full.duplicated().sum(), df_train.duplicated().sum(), df_test.duplicated().sum()))


# In[8]:


df_full_nan = pd.DataFrame(data=[df_full.isna().sum().tolist(), ['{:.2f}'.format(i)+'%' \
           for i in (df_full.isna().sum()/df_full.shape[0]*100).tolist()]], 
           columns=df_full.columns, index=['NaN Count', 'NaN Percent']).transpose()

df_full_nan.style.background_gradient(cmap='Blues', subset=['NaN Count'])


# In[9]:


df_train_nan = pd.DataFrame(data=[df_train.isna().sum().tolist(), ['{:.2f}'.format(i)+'%' \
           for i in (df_train.isna().sum()/df_train.shape[0]*100).tolist()]], 
           columns=df_train.columns, index=['NaN Count', 'NaN Percent']).transpose()

df_train_nan.style.background_gradient(cmap='Oranges', subset=['NaN Count'])


# <b> Training set cleaning </b>

# In[10]:


# NaN removal and Inf replacement in order to be able to calculate our recovery calculation and sMAPE correctly
## Data was already placed into Training and Test sets, dropping NaNs here should be okay and not spill over into data leakage territory
df_train.dropna(inplace=True)
df_train.replace([np.inf, -np.inf], 0, inplace=True)


# In[11]:


df_test_nan = pd.DataFrame(data=[df_test.isna().sum().tolist(), ['{:.2f}'.format(i)+'%' \
           for i in (df_test.isna().sum()/df_test.shape[0]*100).tolist()]], 
           columns=df_test.columns, index=['NaN Count', 'NaN Percent']).transpose()

df_test_nan.style.background_gradient(cmap='Greens', subset=['NaN Count'])


# <b> Descriptive statistics </b>

# In[12]:


df_full.describe()


# In[13]:


df_train.describe()


# In[14]:


df_test.describe()


# <b> Recovery Calculation </b>
# 
# <b> C: Share of Gold in Concentrate </b>

# In[15]:


C = df_train['rougher.output.concentrate_au']
print('Null values:', C.isnull().sum())


# <b> F: Share of Gold in Feed </b>

# In[16]:


F = df_train['rougher.input.feed_au']
print('Null values:', F.isnull().sum())


# <b> T: Share of Gold in Rougher Tails </b>

# In[17]:


T = df_train['rougher.output.tail_au']
print('Null values:', T.isnull().sum())


# <b> Visualizing our Recovery Metrics </b>

# In[18]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize=(14, 8))
for r in [C, F, T]:
    sns.histplot(r, bins=25, ax=ax, kde=True).set_title('Au Distribution - Recovery Estimation')
    ax.set_xlabel('Recovery Metrics')
    ax.legend(labels=['C: Share of Gold in Concentrate','F: Share of Gold in Feed','T: Share of Gold in Rougher Tails'])


# <b> Rougher Output Recovery </b>

# In[19]:


recovery_calc = lambda df, C, F, T: ((df[C] * (df[F] - df[T])) \
                               / (df[F] * (df[C] - df[T])))  * 100

estimated_recovery = recovery_calc(df_train.dropna(subset=['rougher.output.recovery']), 
                                'rougher.output.concentrate_au',
                                'rougher.input.feed_au', 
                                'rougher.output.tail_au')

estimated_recovery = estimated_recovery.fillna(0)
estimated_recovery.replace([np.inf, -np.inf], 0, inplace=True)

#display(estimated_recovery.sort_values())


# <b> Final Output Recovery </b>

# In[20]:


final_estimated_recovery = recovery_calc(df_train.dropna(subset=['final.output.recovery']), 
                                'final.output.concentrate_au',
                                'rougher.input.feed_au',
                                'final.output.tail_au')

final_estimated_recovery = final_estimated_recovery.fillna(0) # filling in NaN with zero as final_error below has a mismatch after dropping NaN
final_estimated_recovery.replace([np.inf, -np.inf], 0, inplace=True)
#display(final_estimated_recovery.sort_values())


# <b> Mean Absolute Error - Rougher </b>

# In[21]:


recovery_feature = df_train['rougher.output.recovery']
print('Null values:', recovery_feature.isna().sum())


# In[22]:


error = mean_absolute_error(recovery_feature, estimated_recovery)

print("Mean absolute error:", error)
print(f"MAE with supression: {error:.17f}")


# <b> Mean Absolute Error - Final </b>

# In[23]:


final_recovery_feature = df_train['final.output.recovery']
print('Null values:', final_recovery_feature.isna().sum())
#print('Null values:', final_recovery_feature.mean())


# In[24]:


final_error = mean_absolute_error(final_recovery_feature, final_estimated_recovery)

print("Mean absolute error : ", final_error)
print(f"MAE with supression: {final_error:.17f}")


# <b> Findings </b>
# 
# After removing NaN values, the average error between the predictions and actuals in this feature comparison (rougher recovery) is ~0.00000000000000946, which is a good value considering the average feature value is ~82.74 For final recovery, following the same process as prior and replacing infinity values, we see a slightly lower absolute error of ~0.00000000000000819 with the average final feature value of about 66.8 -- not a crazy MAE given the average value.

# <b> Missing Features from Test dataset </b>

# In[25]:


column_difference = df_train.columns.difference(df_test.columns)
display(pd.Series(column_difference))


# <b> Findings </b>
# 
# The above is a list of features between the train and test datasets where the 34 columns displayed are all missing from the test dataset (all float types). These will be dropped from the Train dataset once we evaluate the chosen model and leverage the Test dataset.
# 
# Parameters include (with their respective types):
# 
# concentrate (outputs)
# 
# tail (outputs)
# 
# pb_ratio (calculation)
# 
# floatbank10_sulfate_to_au_feed & floatbank11_sulfate_to_au_feed (calculation)
# 
# sulfate_to_au_concentrate (calculation)
# 
# recovery (outputs)

# <b> Concentration of Metals </b>
# 
# <b> Au </b>

# In[26]:


# Base Concentrate
au_metal = df_train['rougher.output.concentrate_au']
# First Purification Stage (Concentrate)
first_purif_au = df_train['primary_cleaner.output.concentrate_au']
# Second Purification Stage (Tail - Residues)
second_purif_au = df_train['secondary_cleaner.output.tail_au']
# Final Output
final_output_au = df_train['final.output.concentrate_au']
# summary
dict_au = ['rougher.output.concentrate_au', 'primary_cleaner.output.concentrate_au',
           'secondary_cleaner.output.tail_au', 'final.output.concentrate_au']
df_train[dict_au].agg(['mean', 'median', 'var', 'std'])


# In[27]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize=(14, 8))
au_values = [au_metal, first_purif_au,final_output_au]
for r in au_values:
    sns.histplot(r, bins=25, ax=ax, kde=True).set_title('Au Concentrate Distribution')
    ax.legend(labels=['rougher.output.concentrate_au','primary_cleaner.output.concentrate_au','final.output.concentrate_au'])


# <b> Ag </b>

# In[28]:


# Base Concentrate
ag_metal = df_train['rougher.output.concentrate_ag']
# First Purification Stage (Concentrate)
first_purif_ag = df_train['primary_cleaner.output.concentrate_ag']
# Second Purification Stage (Tail - Residues)
second_purif_ag = df_train['secondary_cleaner.output.tail_ag']
# Final Output
final_output_ag = df_train['final.output.concentrate_ag']
# summary
dict_ag = ['rougher.output.concentrate_ag', 'primary_cleaner.output.concentrate_ag',
           'secondary_cleaner.output.tail_ag', 'final.output.concentrate_ag']
df_train[dict_ag].agg(['mean', 'median', 'var', 'std'])


# In[29]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize=(14, 8))
ag_values = [ag_metal, first_purif_ag,final_output_ag]
for r in ag_values:
    sns.histplot(r, bins=25, ax=ax, kde=True).set_title('Ag Concentrate Distribution')
    ax.legend(labels=['rougher.output.concentrate_ag','primary_cleaner.output.concentrate_ag','final.output.concentrate_ag'])


# <b> Pb </b>

# In[30]:


# Base Concentrate
pb_metal = df_train['rougher.output.concentrate_pb']
# First Purification Stage (Concentrate)
first_purif_pb = df_train['primary_cleaner.output.concentrate_pb']
# Second Purification Stage (Tail - Residues)
second_purif_pb = df_train['secondary_cleaner.output.tail_pb']
# Final Output
final_output_pb = df_train['final.output.concentrate_pb']
# summary
dict_pb = ['rougher.output.concentrate_pb', 'primary_cleaner.output.concentrate_pb',
           'secondary_cleaner.output.tail_pb', 'final.output.concentrate_pb']
df_train[dict_pb].agg(['mean', 'median', 'var', 'std'])


# In[31]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize=(14, 8))
pb_values = [pb_metal, first_purif_pb,final_output_pb]
for r in pb_values:
    sns.histplot(r, bins=25, ax=ax, kde=True).set_title('Pb Concentrate Distribution')
    ax.legend(labels=['rougher.output.concentrate_pb','primary_cleaner.output.concentrate_pb','final.output.concentrate_pb'])


# In[32]:


# summary
print('Train dataset: \n',df_train['rougher.input.feed_size'].agg(['mean', 'median', 'var', 'std']))
print('')
print('Test dataset: \n',df_test['rougher.input.feed_size'].agg(['mean', 'median', 'var', 'std']))


# In[33]:


train_feed = df_train['rougher.input.feed_size']
test_feed = df_test['rougher.input.feed_size']

sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize=(14, 8))
feed_values = [train_feed, test_feed]
for r in feed_values:
    sns.histplot(r, bins=25, ax=ax, kde=True).set_title('Particle Feed Distribution')
    ax.legend(labels=['rougher.input.feed_size','rougher.input.feed_size'])


# In[34]:


# summary
raw_feed = df_train[['rougher.input.feed_au','rougher.input.feed_ag','rougher.input.feed_pb']].sum(axis=1)
rougher_concentrate = df_train[['rougher.output.concentrate_au','rougher.output.concentrate_ag','rougher.output.concentrate_pb']].sum(axis=1)
final_concentrate = df_train[['final.output.concentrate_au','final.output.concentrate_ag','final.output.concentrate_pb']].sum(axis=1).dropna()

print('\033[1mRaw Feed:\033[0m \n', raw_feed.agg(['mean', 'median', 'var', 'std']))
print('')
print('\033[1mRougher Concentrate:\033[0m \n', rougher_concentrate.agg(['mean', 'median', 'var', 'std']))
print('')
print('\033[1mFinal Concentrate:\033[0m \n', final_concentrate.agg(['mean', 'median', 'var', 'std']))


# In[35]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize=(14, 8))
feed_values = [raw_feed, rougher_concentrate, final_concentrate]
for r in feed_values:
    sns.histplot(r, bins=25, ax=ax, kde=True).set_title('Concentrate Distribution')
    ax.legend(labels=['Raw Feed','Rougher Concentrate','Final Concentrate'])


# <b> Findings </b>
# 
# <b> Individual Concentrates </b>
# 
# The concentrate distribution varies across Au, Ag, and Pb.
# 
# Au (gold) has a higher frequency of instances overall and its values are larger with Base concentration in the 15-25 range, followed by First Purification in the 30-40 range then 40+ under the Final Output stage. The feature variance increases as the material is purified and converted into the Final Output (very acute compression at that stage) which in turn is showing us more final output concentration out of this metal.
# 
# Ag (silver) concentrate shows a path reversal along with much smaller values overall (and less instances). As this material goes through its process towards the Final Output, we see a shift towards the left (less frequency in both values and instances compared to the First Purification and Base stages). Telling us that there is more difficulty refining/extracting this metal compared to gold.
# 
# Pb's (lead) concentration is much more centered compared to the other two, where we see Base concentrate spead over a 'longer' path as it passes through the First Purification stage then it 'shrinks' as it goes through its Final Output stage. Sitting in the middle of the pack and giving us a sense that this metal does just okay as it passes through the process.
# 
# Total Concentrates - Raw Feed, Rougher and Final
# 
# Overall concentrate distribution follows a positive path towards its final output (good extraction overall from feed, all the way to the final output).
# 
# As raw feed is introduced into the flotation process, we see a slightly higher rougher concentrate count as the metal is stabilized and 'concentrated'.
# 
# Once the raw feed is stabilized and is now a rougher concentrate, it enters the purification stages (two) where we see the final output being about double the amount from rougher to final.
# 
# Particle Feed
# 
# Feed distribution across the Training and Test datasets are similar.
# 
# We see higher frequency of values in the Test dataset which is expected given the preprocessing we did with the Train dataset (removal of NaNs, replacement, etc...).
# 
# Both follow the same path, positive skews. Mean is higher than the median in both instances.
# 
# Anomalies: performed in earlier stages for Training dataset (NaN removal, 0 fills and Infinity value replacement)
# 
# The concentrate distribution across Au, Ag, and Pb originally showed 'outliers' where values ranging from 0 to 1, NaN or Inf values took a decent 'bite' out of the dataset, after corrections on these, all the charts above have diminished values closely 'glued' to the y-axis without affecting the overall analysis.
# 
# Removing these 'anomalies' helps the modeling process so the model itself can 'focus' on the more recurring and significant values across the metals/stages.
# 

# <b> Final sMAPE Calculation </b>

# In[36]:


rougher_target = pd.Series(df_train['rougher.output.recovery'])
final_target = pd.Series(df_train['final.output.recovery'])

rougher_predict = pd.Series(estimated_recovery, name='estimated_recovery')
final_predict = pd.Series(final_estimated_recovery, name='final_estimated_recovery')

target = pd.concat([rougher_target, final_target], axis=1).to_numpy()
prediction = pd.concat([rougher_predict, final_predict], axis=1).to_numpy()


# In[37]:


def sMAPE_final_calc(target, prediction):
    """Function that follows our sMAPE calculation.
    Similar to MAE, but is expressed in relative values instead of absolute ones.
    It equally takes into account the scale of both the target and the prediction.
    """
    target = np.array(target)
    prediction = np.array(prediction)
    
    RT, FT = target[:, 0], target[:, 1]
    RP, FP = prediction[:, 0], prediction[:, 1]
    
    # creating the calculations for both the rougher and final recovery values
    rougher = 100/len(RT) * np.nansum(2 * np.abs(RP - RT) / (np.abs(RT) + np.abs(RP))) # + np.finfo(float).eps
    final = 100/len(FT) * np.nansum(2 * np.abs(FP - FT) / (np.abs(FT) + np.abs(FP))) # + np.finfo(float).eps
    final_sMAPE =  .25 * rougher + .75 * final
    
    return final_sMAPE

sMAPE_scorer = make_scorer(sMAPE_final_calc, greater_is_better=False)


# In[38]:


np.seterr(invalid='ignore')
result = sMAPE_final_calc(target, prediction)
print('Final sMAPE:', result)


# <b> Modeling </b>
# 
# <b> Reproducibility </b>

# In[39]:


seed = 12345


# In[40]:


cv = RepeatedKFold(n_repeats=3, n_splits=3, random_state=seed)


# <b> Training Data </b>

# In[41]:


# creating the features and target variables from our Training set
target = df_train[['rougher.output.recovery','final.output.recovery']] # extracting target
features = df_train.drop(['rougher.output.recovery','final.output.recovery'], axis=1) # feature extraction
features_net = df_train.drop(column_difference, axis=1) # net differences between Train and Test set

# constant model for in case there is a need for comparison
constant = pd.Series(target.mean(), index = target.index)

# scaled Training set to scale and take into account outliers in our data
scaler = RobustScaler(unit_variance=True)
features_net_scaled = scaler.fit_transform(features_net)


# In[42]:


# initializing the models we will be iterating through
dm = DummyRegressor()
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=seed)
ridge = Ridge()
lasso = Lasso()
rf = RandomForestRegressor(random_state=seed)
multi_out = MultiOutputRegressor(rf)

regressors = [('DummyRegressor', dm),('LinearRegression', lr),
               ('Ridge', ridge), ('Lasso', lasso),
              ('DecisionTreeRegressor', dt), ('RandomForestRegressor', rf),
              ('MultiOutputRegressor', multi_out)]


# In[43]:


for clf_name, clf in regressors:
    """Iterates through the regressors we initialized, takes into account the Name and Model we have provided.
    Fits each model and provides scoring metrics through cross_val_score and our sMAPE_scorer.
    The iteration is meant to provide us with baseline comparisons in order to select a model and then further tune via hyperparameter tuning/GridSearch.
    """
    # fit
    clf.fit(features_net_scaled, target)

    # label prediction
    # y_pred = clf.predict()

    # evaluations
    scoring = cross_val_score(clf, features_net_scaled, target, scoring=sMAPE_scorer, cv=cv)
    print('')
    print('{:s} Avg. sMAPE Cross-Val Score: {:.3f}'.format(clf_name, np.abs(scoring.mean())))


# <b> Model Selection </b>

# In[44]:


get_ipython().run_cell_magic('time', '', 'model = RandomForestRegressor(random_state=seed)\nmodel_params = [{\'max_depth\': list(range(5, 11)), \'max_features\': list(range(3,7)), \'n_estimators\':[100, 150, 300]}]\n\ncv_search = GridSearchCV(model, model_params, cv=cv, scoring=sMAPE_scorer)\ncv_search.fit(features_net_scaled, target)\n"""\nfeatures_net takes into account the differences in columns between our train and test sets\ntarget is our target from the df_train set\n"""\nbest_parameters = cv_search.best_params_\nbest_score = cv_search.best_score_\n')


# In[45]:


print('Best Params: {},\n Best sMAPE Score: {:.3f}'.format(best_parameters, np.abs(best_score)))


# In[ ]:


cv_results = pd.DataFrame(cv_search.cv_results_)

print('Average Fold Score:', np.mean(abs(cv_results['split0_test_score']) + abs(cv_results['split1_test_score'] +
                              abs(cv_results['split2_test_score'] + abs(cv_results['split3_test_score'])))))


# In[ ]:


"""Our target is not included in our Test set (known discrepancy).
Pulling the target from the Full set.
"""
# we start with matching through our date columns
match_list = df_test['date'].to_list()
full_match_list = df_full['date'].to_list()
mask = df_full['date'].isin(match_list)
df_full.drop(columns='date', index=1, inplace=True)
matching_rows = df_full[mask].fillna(0)

# from our 'filter' we extract our target(s) and as a reminder, we are predicting two values in this project
test_target = matching_rows[['rougher.output.recovery','final.output.recovery']]
test_target = test_target[:-1] # kept getting a mismatch of one row, removed the last one

# further DF clean-up by removing our date feature as well as filling in any NaNs or Inf values
df_test.drop(columns='date', index=1, inplace=True)
df_test = df_test.fillna(0)
df_test.replace([np.inf, -np.inf], 0, inplace=True)

# features for our Test set is simply the Test set DF
features_test_scaled = scaler.transform(df_test)
# features_test = df_test

# since we have our features, we begin the prediction process on test data
y_pred_final = cv_search.predict(features_test_scaled)

# we utilize mean squared error metric in order to analyze performance
mse = mean_squared_error(test_target, y_pred_final)

# results, including our MSE and our RMSE in order to get a value that makes more sense 
print("RandomForestRegression Test dataset MSE:", mse)
print("RandomForestRegression Test dataset RMSE:", mse ** 0.5)
print('')

dummy_model = DummyRegressor().fit(features_net_scaled, target)
y_pred_dummy = dummy_model.predict(features_test_scaled)
dummy_mse = mean_squared_error(test_target, y_pred_dummy)

print("DummyRegression Test dataset MSE:", dummy_mse)
print("DummyRegression Test dataset RMSE:", dummy_mse ** 0.5)


# In[ ]:


print('DummyRegressor Final sMAPE score (Test set) \n', sMAPE_final_calc(test_target,y_pred_dummy))
print('')
print('RandomForestRegressor Final sMAPE score (Test set) \n', sMAPE_final_calc(test_target,y_pred_final))
print('')

# sMAPE score function score based off of the Train set
y_train_pred_final = cv_search.predict(features_net_scaled)
print('RandomForestRegressor Final sMAPE score (Train set) \n', sMAPE_final_calc(target,y_train_pred_final))
print('')


# <b> Conclusions </b>
# 
# We being our project by performing various checks on variability across the datasets as they were already split into a Full, Train and Test set.
# 
# Our analysis provides us with clues on missing values and potential outliers that will impact regression model performance. Variability is expected as the features include values from different points of the extraction process.
# 
# Clean-up is performed on the Train set to start (NaN removal), in order to get a truer sense on the various metals and their processes as missing values hinder our recovery/sMAPE calculations. Note: we created imputation and outlier removal cells in case these methods are needed and if they are truly impactful to our analysis/modeling. Imputing on NaN values negatively affects our modeling but the option is there.
# 
# Once we get a sense of distributions across metals, concentrates and feeds we head into modeling our data via utilization of 5+ regression models. This includes a DummyRegressor model for comparability and make sure there aren't any initial issues. RobustScaling is implemented to take into account outlier values (scaling that resists the pull of outliers).
# 
# We iterate through our regression models (without tuning) in order to cross-compare our cross_val scores which leverages our sMAPE_scorer function under the scoring parameter. This lets us have a baseline comparison to see how each model performs on the Train set before going further with hyperparameter tuning.
# Our analysis shows us that the RandomForestRegressor is the better performing model across the regressors but comes at the cost of lower speed. Both LinerRegression and Ridge follow closely with slightly worse scores but at a much higher computation speed. Our base scores from cross_val were as follows:
# 
#     - Dummy: 9.88
#     - Linear: 7.83
#     - Ridge: 7.83
#     - Lasso: 8.04
#     - DecisionTree: 7.86
#     - RandomForest: 5.91
#     - MultiOutput: 5.95
# Further analyzing the most optimal model, in this case the RandomForestRegressor, we perform final evaluations on our Test set and deploy hyperparameter tuning through GridSearchCV. The evaluation yields the scores below which perform better than our DummyRegressor.
# 
#     - MSE: 460
#     - RMSE: 21
# 
# Our optimal model suggestion is based on our understanding that the company is placing more weight at the accuracy of the model compared to time/speed. RandomForestRegressor also gives added flexibility through more hyperparameter tuning than Linear or Ridge Regression (carefully calling out possible over-fitting nonetheless). If time/speed is desired, the recommendation is to use the Linear or Ridge regressors.
# 
# Based on the sMAPE_final_calc comparison between the RandomForestRegressor and the DummyRegressor model, we conclude that the RandomForestRegressor gives us a % error of ~25% when leveraging the Test set (compared to a nearly identical but slightly better result to our DummyRegressor model). This % error result from our RandomForestRegressor is still a more optimal result compared to all other models deployed.
# 
# When going one step further and analyzing the final sMAPE value on our train dataset, we find a ~6% error. Meaning our final sMAPE result on our Test set introduces a +19% delta to our overall results/predictions.

# In[ ]:




