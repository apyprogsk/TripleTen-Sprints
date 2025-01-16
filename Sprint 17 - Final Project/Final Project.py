#!/usr/bin/env python
# coding: utf-8

# ### Part 1: Project Workplan
# The aim of our project is to predict churn rates* for our customer, Interconnect.
# 
# From our results/final report, the company will be better prepared when it comes to forecasting which clients are in a higher likelihood of disconneting their services (which more than likely is to a competitor with more enticing perks).
# 
# Interconnect will then plan on offering a larger scope of benefits to those clients in danger of turning over. This may also prove particularly useful in its market share battle against its competitors as the telecomm industry sprints to expand their perks each year as new cell phones come out (a constant cycle).

# #### Step 1 - Initialization
# 
# Import all required and prospective libraries that will be leveraged in future stages of the project.
# 
# Download the data and briefly inspect each DataFrame structure.
# 
# #### Step 2 - Preprocessing & EDA
# 
# Determine the necessary process to clean and massage the data.
# 
# Create short, concise summaries on each DataFrame along with visualizations (plots) for thought organization and guidance.
# 
# Perform EDA, including but not limited to: dtype and naming revisions, class balancing, value scaling, feature engineering, encoding and merging.
# 
# Fill in any gaps due to changes made to the DataFrame(s).
# 
# Deploy time series tools/analysis to get a sense of trends and seasonality to paint a more complete picture.
# 
# #### Step 3 - Model Selection, Training and Fine-Tuning
# 
# Select various models based on the goal, binary classification. Compare initial scoring performance across the model selections, incorporate a dummy model benchmark.
# 
# Fine-tune models using hyperparameters and gridsearches. Include gradient boosting techniques.
# 
# #### Step 4 - Model Evaluation
# 
# Decide on the optimal model based on cross comparisons and boosting techniques. Perform model evaluation based on a new set of data (test dataset) and document results.
# 
# If statisfactory, record why the model was selected, its results (along with speed and accuracy insights) and what needs to be done in order to monitor/manage the model in a go-forward basis.
# 
# #### Step 5 - Comprehensive Report
# 
# Create an extensive report on the process/steps taken, the results and the overall recommendations.
# Illustrate findings/results incorporating 'quick hits' or 'highlights' so the customer has a better handle on the report and can easily share internally.

# **The churn rate measures a company's loss in subscribers for a given period of time. The cost of acquiring new customers is much higher than it is to retain current customers.*

# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# This is a good project with a good workplan. The only part to be judged in this phase was the workplan, you can re-submit to 
# </div>

# ### Part 2: Solution Code
# #### Initialization

# In[1]:


def warn(*args, **kwargs): # attempt at removing warnings
    pass
import warnings
warnings.warn = warn

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# common libraries
import pandas as pd
import numpy as np
from functools import reduce
from numpy import unique

# other libraries
# from scipy.stats import randint as 

# viz libraries
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
px.defaults.template = "plotly_white"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
# import pygwalker as pyg # leveraging once DFs are merged

# sklearn
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, accuracy_score, ConfusionMatrixDisplay, auc, roc_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.utils import shuffle, resample
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.neighbors import KNeighborsClassifier

# gradient boosting
import lightgbm as lgb
import xgboost as xgb

# other
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'


# In[3]:


contract_data = pd.read_csv('C:/Users/gsrav/Desktop/Sprint 17/final_provider/contract.csv') # index_col=[0], parse_dates=[0]
personal_data = pd.read_csv('C:/Users/gsrav/Desktop/Sprint 17/final_provider/personal.csv') 
internet_data = pd.read_csv('C:/Users/gsrav/Desktop/Sprint 17/final_provider/internet.csv') 
phone_data = pd.read_csv('C:/Users/gsrav/Desktop/Sprint 17/final_provider/phone.csv')


# #### Exploratory Data Analysis

# In[4]:


encoder = LabelEncoder() 


# In[5]:


# to bypass limitations and encode multiple columns
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # column names to encode

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
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


# In[6]:


# function to classify features
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []

    for column in df.columns:
        if df[column].dtype in ['object', 'bool', 'category']: 
            if df[column].nunique() < 15:
                categorical_features.append(column)
            else: 
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 10:
                discrete_features.append(column)
            else: 
                continuous_features.append(column)
    return categorical_features, non_categorical_features, discrete_features, continuous_features


# #### Contract data

# In[7]:


contract_data.info()


# In[8]:


categorical, non_categorical, discrete, continuous = classify_features(contract_data)

print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)


# In[9]:


contract_data.describe()


# In[10]:


contract_data.isna().sum()


# In[11]:


display(contract_data)


# #### Preprocessing

# In[12]:


contract_df = contract_data.copy()
# column renaming
contract_df = contract_df.rename(columns={"customerID": "customer_id", "BeginDate": "begin_date", "EndDate": "end_date", "Type": "contract_type",
                           "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method", "MonthlyCharges": "monthly_charges",
                           "TotalCharges": "total_charges"})

# datatype conversions
contract_df['begin_date'] = pd.to_datetime(contract_df['begin_date'])


# display(contract_df.iloc[488]) # first error callout, turns out there are 11 rows without a 'total_charges' value
# display(contract_df['total_charges'].isnull().sum()) 
problem_cells = [ ]

for value in contract_df['total_charges']:
    try:
        pd.to_numeric(value)
    except:
        problem_cells.append(value)
        
display(problem_cells)
display()

contract_df['total_charges'].replace(" ", np.nan, inplace=True) # replacing empty strings so we can drop the rows
contract_df['total_charges'] = pd.to_numeric(contract_df['total_charges']) # conversion to match 'monthly_charges'
contract_df = contract_df.dropna(subset=['total_charges']) # 11 rows should not make that big of a dent given it's a tiny percentage of total

contract_df.info()


# In[13]:


# target handling
contract_df.query('end_date == " "') # no empty cells in target column

contract_df['churn_target'] = np.where(contract_df['end_date'] == 'No', 1, 0) # 1 = no churn, 0 = churn
contract_df = contract_df.drop(['end_date'], axis=1)

display(contract_df)


# In[14]:


payment_grp = contract_df.groupby(['contract_type','payment_method']).size().reset_index().groupby('payment_method')[[0]].max()
# display(payment_grp)
fig = px.bar(payment_grp,title="Customer Payment Method",text_auto = True)
fig.update_layout(showlegend=False,autosize=False)
fig.show()

contract_type_grp = contract_df.groupby(['contract_type','payment_method']).size().reset_index().groupby('contract_type')[[0]].max()
# display(contract_type_grp)
fig = px.bar(contract_type_grp,title="Customer Contract Type",text_auto = True)
fig.update_layout(showlegend=False,autosize=False)
fig.show()


# In[15]:


crosstab = pd.crosstab(contract_df['payment_method'], contract_df['contract_type'])
fig, ax = plt.subplots(figsize=(13, 6))
g = sns.heatmap(crosstab, cbar=False, cmap="BuGn", linewidths=0.3, annot=True, fmt='d', ax=ax)

g.set_ylabel('Payment Method')
g.set_xlabel('Type')

ax.text(x=0.5, y=1.1, s='Contract Data', fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.05, s='Heatmap Analysis', fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)

plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()


# In[16]:


# label encoding
contract_df = MultiColumnLabelEncoder(columns = ['contract_type','paperless_billing', 'payment_method']).fit_transform(contract_df)

display(contract_df)


# ##### Summary
# 
# Peaking into our contract data, we see issues with column naming, and column datatypes. We attempt amend those two to start and find that one of the features has empty strings in it's value range (through an iterative examination).
# 
# Once we find the specific culprits, we replace with NaN values so we can then drop the rows themselves. Thse are small in numbers compared to the entire DF and so are more comfortable with the removal and do not expect much, if any, impact down the line.

# #### Personal data

# In[17]:


personal_data.info()


# In[18]:


categorical, non_categorical, discrete, continuous = classify_features(personal_data)

print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)


# In[19]:


for i in categorical:
    #print(i, ':')
    print(personal_data[i].value_counts())
    fig = px.bar(personal_data[i].value_counts(), labels={'value':'Count (#)'}, text_auto=True).update_layout(showlegend=False,autosize=False).update_traces(marker_color='darkgreen')
    print()
    fig.show()
    fig.show()


# In[20]:


personal_data.describe()


# In[21]:


personal_data.isna().sum()


# In[22]:


display(personal_data)


# #### Preprocessing

# In[23]:


personal_df = personal_data.copy()
# column renaming
personal_df = personal_df.rename(columns={"customerID": "customer_id", "SeniorCitizen": "senior_citizen",
                                          "Partner": "partner", "Dependents": "dependents"})


# In[24]:


personal_grp = personal_df.groupby(['partner','gender']).size().reset_index().groupby('gender')[[0]].max()

labels = ['Male', 'Female']
fig = px.pie(personal_grp, names=labels, title='Gender Mix')
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
),autosize=False)

fig.show()


# In[25]:


crosstab = pd.crosstab(personal_df['partner'], personal_df['gender'])
fig, ax = plt.subplots(figsize=(10, 6))
g = sns.heatmap(crosstab, cbar=False, cmap="Blues", linewidths=0.3, annot=True, fmt='d', ax=ax)

g.set_ylabel('Has Partner')
g.set_xlabel('Gender')

ax.text(x=0.5, y=1.1, s='Personal Data', fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.05, s='Heatmap Analysis', fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)

plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()


# In[26]:


# label encoding
personal_df = MultiColumnLabelEncoder(columns = ['gender','senior_citizen', 'partner', 'dependents']).fit_transform(personal_df)

display(personal_df)


# #### Summary
# 
# Similar instances of column revisions for personal_data but we preemtively being feature encoding through a function that handles multiple column encoding. These values then become inputs the eventual models can handle and make sense of.

# ### Internet data

# In[27]:


internet_data.info()


# In[28]:


categorical, non_categorical, discrete, continuous = classify_features(internet_data)

print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)


# In[29]:


for i in categorical:
    #print(i, ':')
    print(internet_data[i].value_counts())
    fig = px.bar(internet_data[i].value_counts(), labels={'value':'Count (#)'}, text_auto=True).update_layout(showlegend=False,autosize=False).update_traces(marker_color='darkblue')
    print()
    fig.show()
    fig.show()


# In[30]:


internet_data.describe()


# In[31]:


internet_data.isna().sum()


# In[32]:


display(internet_data)


# #### Preprocessing

# In[33]:


internet_df = internet_data.copy()
# column renaming
internet_df = internet_df.rename(columns={"customerID": "customer_id", "InternetService": "internet_service", "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
                                         "DeviceProtection": "device_protection", "TechSupport": "tech_support", "StreamingTV": "streaming_tv",
                                         "StreamingMovies": "streaming_movies"})

display(internet_df)


# In[34]:


internet_grp = internet_df.groupby(['online_security','internet_service']).size().reset_index().groupby('internet_service')[[0]].max()

fig = px.bar(internet_grp, title="Customer Internet Service Type",text_auto = True)
fig.update_layout(showlegend=False, autosize=False)
fig.show()


# In[35]:


crosstab = pd.crosstab(internet_df['online_security'], internet_df['internet_service'])
fig, ax = plt.subplots(figsize=(10, 6))
g = sns.heatmap(crosstab, cbar=False, cmap="Grays", linewidths=0.0029, annot=True, fmt='d', linecolor='lightgray', ax=ax)

g.set_ylabel('Selected Online Security')
g.set_xlabel('Service Type')

ax.text(x=0.5, y=1.1, s='Internet Data', fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.05, s='Heatmap Analysis', fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)

plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()


# In[36]:


# label encoding
internet_df = MultiColumnLabelEncoder(columns = ['internet_service','online_security', 'online_backup', 'device_protection','tech_support', 'streaming_tv', 'streaming_movies']).fit_transform(internet_df)

display(internet_df)


# #### Summary
# 
# Looking into our internet_data, we see a smaller number of rows and therefore have a smaller number of customer_ids (compared to all the other sets).
# 
# We perform some renaming and label encoding to the columns. After doing so, we are making the assumption that those customer_ids that are not included in this dataset did not sign up for the internet services therefore we will also make the assumption that where we find NaNs in our final, merged set we can replace those with 'No' or 'Not subscribed'.

# ### Phone data

# In[37]:


phone_data.info()


# In[38]:


categorical, non_categorical, discrete, continuous = classify_features(phone_data)

print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)


# In[39]:


phone_data.describe()


# In[40]:


phone_data.isna().sum()


# In[41]:


display(phone_data)


# #### Preprocessing

# In[42]:


phone_df = phone_data.copy()
# column renaming
phone_df = phone_df.rename(columns={"customerID": "customer_id", "MultipleLines": "multiple_lines"})

display(phone_df)


# In[43]:


# label encoding
phone_df.multiple_lines = encoder.fit_transform(phone_df.multiple_lines.values)
display(phone_df)


# #### Summary
# 
# Our last DF in question, phone_data, has a smaller subset of data and one feature we are insterested in called 'multiple_lines'. We fix the column naming convention to mirror the edits we made to the other DFs and we encode our selected feature using label encoding.

# ### Merging

# In[45]:


# merge data based off of customer_id, main DF should be contract_df
# contract, personal, phone merge to start
# null values from phone_df will need to be monitored with options to 1) remove (if small impact), 2) replace (average,std) or, 3) keep in place

# merged_df = contract_df.merge(personal_df,
#                              on='customer_id',
#                              ).merge(phone_df, on='customer_id', how='left')

# display(merged_df)

# display(merged_df.query('multiple_lines.isna()')) # attempting to find commonality for NaN values under 'multiple_lines' -- nothing in common 

data_frames = [contract_df, personal_df, internet_df, phone_df]

merged_df = reduce(lambda  left, right: pd.merge(left, right, on=['customer_id'],
                                            how='left'), data_frames)

display(merged_df) # should include most of our 'customer_id' instances


# In[46]:


# dropping columns that don't impact analysis 
merged_df = merged_df.drop('customer_id', axis=1)


# In[47]:


# dropping rows with NaNs under 'multiple_lines' as a test, reverting back to this and filling in if there are score issues with models
# lines_filter = merged_df.query('multiple_lines.isna()')
# merged_df.drop(merged_df[merged_df['multiple_lines'].isna()].index, inplace=True)

# dropping rows with NaNs under 'internet_service' as a test, reverting back to this and filling in if there are score issues with models
# service_filter = merged_df.query('internet_service.isna()') #1520 rows with NaNs, removal might be required 

# we are making the assumption that NaN under multiple_lines == No and NaN under the internet_df means they did not sign up, so == No as well
merged_df = merged_df.fillna(0)

display(merged_df)


# In[48]:


fig, ax = plt.subplots(figsize=(13, 8))
g = sns.regplot(data=merged_df, x=merged_df['begin_date'].dt.year,y='total_charges', order=2,
                x_jitter=.05
                #x_estimator=np.mean
                )

g.set_ylabel('Total Charges')
g.set_xlabel('Beginning Year of Service')

sns.despine()

ax.text(x=0.5, y=1.1, s='Merged Data', fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
ax.text(x=0.5, y=1.05, s='Regression Analysis', fontsize=8, alpha=0.75, ha='center', va='bottom', transform=ax.transAxes)

plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()
plt.show()


# #### Summary
# 
# We merged all but one of our DFs in our project with our 'contract_data' being the lead DF given the substantial information included. We decide to keep all the matches on the DFs which, knowingly, gives us some NaNs based off of our 'phone_data' -- we keep in our merged DF for now to see how our models react to seeing such values.
# 
# If issues arise, we will revisit this section and run through a few options for our NaNs. This includes removing the rows or replacing them.

# ### Target Frequency

# In[49]:


class_frequency = merged_df['churn_target'].value_counts()
print('Checking frequency:')
print(class_frequency)

#class_frequency.plot.pie(autopct='%.2f',textprops={'fontsize':12}, colors=['cyan', 'teal']) # calls for upsampling 'rare' or churn target values (26.58%)

labels = ['No Churn', 'Churn']
fig = px.pie(class_frequency, values='count', names=labels, title='Class Frequency')
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
), autosize=False)

fig.show()


# ### Model Preparation
# 
# #### Fixed Parameter

# In[50]:


# random_state parameter for all models
random_state = 12345


# #### CV
# 
# Rearranging the data so as to ensure that each fold is a good representative of the whole

# In[51]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=random_state)


# #### Feature Engineering

# In[52]:


# extracting date features from datetime column
def make_features(data, col):
    data['begin_year'] = data[col].dt.year
    data['begin_month'] = data[col].dt.month
    # data['begin_day'] = data[col].dt.day # no impact to model outputs
    data['begin_dayofweek'] = data[col].dt.dayofweek
    
#     for lag in range(1, max_lag + 1):
#         data['lag_{}'.format(lag)] = data['PJME_MW'].shift(lag)

#     data['rolling_mean'] = data['PJME_MW'].shift().rolling(rolling_mean_size).mean()
#     #data['rolling_mean'] = data['PJME_MW'].rolling(rolling_mean_size).mean()

make_features(merged_df, 'begin_date')
merged_df = merged_df.drop('begin_date', axis = 1) # removing as this has been replaced by the newly created features

display(merged_df)


# #### Features and Target

# In[53]:


features = merged_df.drop('churn_target', axis=1)
target = merged_df['churn_target']


# #### Data Splitting

# In[54]:


# splitting the data into a training, validation and test dataset
features_train, features_test, target_train, target_test = train_test_split(features, target, 
                                                                            test_size=0.2, random_state=12345, stratify=target)

features_train, features_valid, target_train, target_valid = train_test_split(features_train, target_train, 
                                                                          test_size=0.25, random_state=12345, stratify=target_train)

print(features_train.shape)
print(target_train.shape)
print(features_valid.shape)
print(features_test.shape)


# In[55]:


# target_train summary
classes = unique(target_train)
total = len(target_train)
for c in classes:
    n_examples = len(target_train[target_train==c])
    percent = n_examples / total * 100
    print('> Class = %d : %d/%d (%.1f%%)' % (c, n_examples, total, percent))


# In[56]:


# summarize by set
train_0, train_1 = len(target_train[target_train==0]), len(target_train[target_train==1])
valid_0, valid_1 = len(target_valid[target_valid==0]), len(target_valid[target_valid==1])
test_0, test_1 = len(target_test[target_test==0]), len(target_test[target_test==1])
print('Train: 0=%d, 1=%d, \nValid: 0=%d, 1=%d, \nTest: 0=%d, 1=%d' % (train_0, train_1, valid_0, valid_1 , test_0, test_1))


# #### Scaling

# In[57]:


numeric = ['monthly_charges', 'total_charges', 'begin_year','begin_month', 'begin_dayofweek']

def scaling(x_train, x_valid, x_test):
    scaler = MinMaxScaler()
    scaler.fit(x_train[numeric])
    x_train[numeric] = scaler.transform(x_train[numeric])
    x_valid[numeric] = scaler.transform(x_valid[numeric])
    x_test[numeric] = scaler.transform(x_test[numeric])
    return x_train, x_valid, x_test

scaling(features_train, features_valid, features_test)


# #### Summary
# 
# After taking into account class imbalancing, we've split the data into Train, Validation and Test datasets where each has been scaled in order to take value magnitute into account and create good inputs for our model training.

# ### Dummy Model

# In[58]:


dummy = DummyClassifier(random_state=random_state,strategy="most_frequent")
dummy.fit(features_train, target_train)
DummyClassifier(strategy='most_frequent')
dummy.predict(features_valid)
print('Dummy Model Score:', dummy.score(features_valid, target_valid))


# ### KNeighbors

# In[59]:


get_ipython().run_cell_magic('time', '', "train_accuracies = {}\nvalidation_accuracies = {}\nneighbors = np.arange(1, 15)\n\nfor neighbor in neighbors: \n    knn = KNeighborsClassifier(n_neighbors=neighbor) # non-RandomizedSearchCV\n    knn.fit(features_train, target_train)\n    train_accuracies[neighbor] = knn.score(features_train, target_train)\n    validation_accuracies[neighbor] = knn.score(features_valid, target_valid)\n\nprint('Runtime:')\n")


# In[60]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
plt.figure(figsize=(10, 5))

plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, validation_accuracies.values(), label="Validation Accuracy")

plt.legend(fontsize=8)
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

plt.show()


# In[61]:


get_ipython().run_cell_magic('time', '', 'knn = KNeighborsClassifier()\nknn_params = [{"n_neighbors": np.arange(1,15), \n                    "metric": [\'cosine\', \'euclidean\', \'manhattan\', \'minkowski\'],\n                     "algorithm": [\'auto\', \'ball_tree\', \'kd_tree\'],\n                     "weights": [\'uniform\', \'distance\'],\n                   }]\nknn_clf = RandomizedSearchCV(knn, knn_params, scoring=\'roc_auc\', n_jobs=-1, cv=cv)\nknn_clf.fit(features_train, target_train)\nprint(\'Runtime:\')\n')


# In[62]:


get_ipython().run_cell_magic('time', '', "knn_scores = cross_val_score(knn_clf, features_train, target_train, cv=cv, scoring='roc_auc')\nprint('Cross Validation Scores: {}'.format(knn_scores))\nprint('Runtime:')\n")


# ### ROC AUC Curve - KNeighbors

# In[63]:


target_score = knn_clf.predict_proba(features_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, target_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()

precision, recall, thresholds = precision_recall_curve(target_valid, target_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()


# In[64]:


# summary
print('Best hyperparameters:',  knn_clf.best_params_)
print()
print('Best score:', knn_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(knn_scores.mean()))
print()
print('ROC AUC Score - Validation Dataset:',  roc_auc_score(target_valid, knn_clf.predict_proba(features_valid)[:, 1]))


# ### Random Forest

# In[65]:


get_ipython().run_cell_magic('time', '', 'forest_model = RandomForestClassifier(random_state=random_state)\nforest_parameters = [{\'max_depth\': [2,6,12,18,30],\n                     \'min_samples_split\': [2,6,12],\n                     "criterion": [\'gini\', \'entropy\', \'log_loss\'],\n                     "warm_start": [True, False],\n                     \'n_estimators\': [50,100,200]}]\n\nforest_clf = RandomizedSearchCV(forest_model, forest_parameters, scoring=\'roc_auc\', n_jobs=-1, cv=cv)\nforest_clf.fit(features_train, target_train)\n# create a variable for the best model\nbest_for = forest_clf.best_estimator_\nfor_pred = best_for.predict(features_valid)\nprint(\'Runtime:\')\n')


# In[66]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

fig, ax = plt.subplots(figsize = (10,5))
disp = CalibrationDisplay.from_estimator(forest_clf, features_valid, target_valid, ax=ax)
plt.title('Calibration Chart - Random Forest Classifier', fontsize=10)
ax.set_xlabel('Mean predicted probability (Positive class: 1)', fontsize=10)
ax.set_ylabel('Fraction of positives (Positive class: 1)',fontsize=10)

ax.tick_params(color='gray', labelcolor='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')


fig.tight_layout()
plt.legend(fontsize=8)
plt.show()


# In[67]:


forest_scores = cross_val_score(forest_model, features_train, target_train, cv=cv, scoring='roc_auc')
print('Cross Validation Scores: {}'.format(forest_scores))


# In[68]:


# summary
print('Best hyperparameters:',  forest_clf.best_params_)
print()
print('Best score:', forest_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(forest_scores.mean()))
print()
print('ROC AUC Score - Validation Dataset:',  roc_auc_score(target_valid, forest_clf.predict_proba(features_valid)[:, 1]))


# ### ROC AUC Curve - RandomForestClassifier

# In[69]:


target_score = forest_clf.predict_proba(features_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, target_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()

precision, recall, thresholds = precision_recall_curve(target_valid, target_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()


# ### Decision Tree

# In[70]:


get_ipython().run_cell_magic('time', '', 'tree_model = DecisionTreeClassifier(random_state=random_state)\ntree_parameters = [{\'max_depth\': [2,6,12,18,22,35],\n                     \'min_samples_split\': [2,6,12],\n                     "criterion": [\'gini\', \'entropy\', \'log_loss\'],\n                     "splitter": [\'best\', \'random\'],\n                   }]\n\ntree_clf = RandomizedSearchCV(tree_model, tree_parameters, scoring=\'roc_auc\', n_jobs=-1, cv=cv)\ntree_clf.fit(features_train, target_train)\n# create a variable for the best model\nbest_tree = tree_clf.best_estimator_\ntree_pred = best_tree.predict(features_valid)\nprint(\'Runtime:\')\n')


# In[71]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

fig, ax = plt.subplots(figsize = (10,5))
disp = CalibrationDisplay.from_estimator(tree_clf, features_valid, target_valid, ax=ax)
plt.title('Calibration Chart - Decision Tree Classifier', fontsize=10)
ax.set_xlabel('Mean predicted probability (Positive class: 1)', fontsize=10)
ax.set_ylabel('Fraction of positives (Positive class: 1)',fontsize=10)

ax.tick_params(color='gray', labelcolor='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')


fig.tight_layout()
plt.legend(fontsize=8)
plt.show()


# In[72]:


tree_scores = cross_val_score(tree_model, features_train, target_train, cv=cv, scoring='roc_auc')
print('Cross Validation Scores: {}'.format(tree_scores))


# In[73]:


# summary
print('Best hyperparameters:',  tree_clf.best_params_)
print()
print('Best score:',  tree_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(tree_scores.mean()))
print()
print('ROC AUC Score - Validation Dataset:',  roc_auc_score(target_valid, tree_clf.predict_proba(features_valid)[:, 1]))


# ### ROC AUC Curve - DecisionTreeClassifier

# In[74]:


target_score = tree_clf.predict_proba(features_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, target_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()

precision, recall, thresholds = precision_recall_curve(target_valid, target_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()


# ### Extra Trees

# In[75]:


get_ipython().run_cell_magic('time', '', 'extra_trees_model = ExtraTreesClassifier(random_state=random_state)\nextra_trees_parameters = [{\'max_depth\': [2,6,8,12,18,30],\n                     \'min_samples_split\': [2,6,8,12],\n                     "criterion": [\'gini\', \'entropy\', \'log_loss\'],\n                     "warm_start": [True, False],\n                     \'n_estimators\': [50,100,200]}]\n\nextra_trees_clf = RandomizedSearchCV(extra_trees_model, extra_trees_parameters, scoring=\'roc_auc\', n_jobs=-1, cv=cv)\nextra_trees_clf.fit(features_train, target_train)\n# create a variable for the best model\nbest_ext = extra_trees_clf.best_estimator_\next_pred = best_ext.predict(features_valid)\nprint(\'Runtime:\')\n')


# In[76]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

fig, ax = plt.subplots(figsize = (10,5))
disp = CalibrationDisplay.from_estimator(extra_trees_clf, features_valid, target_valid, ax=ax)
plt.title('Calibration Chart - Extra Trees Classifier', fontsize=10)
ax.set_xlabel('Mean predicted probability (Positive class: 1)', fontsize=10)
ax.set_ylabel('Fraction of positives (Positive class: 1)',fontsize=10)

ax.tick_params(color='gray', labelcolor='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')


fig.tight_layout()
plt.legend(fontsize=8)
plt.show()


# In[77]:


extra_trees_scores = cross_val_score(extra_trees_model, features_train, target_train, cv=cv, scoring='roc_auc')
print('Cross Validation Scores: {}'.format(extra_trees_scores))


# In[78]:


# summary
print('Best hyperparameters:',  extra_trees_clf.best_params_)
print()
print('Best score:',  extra_trees_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(extra_trees_scores.mean()))
print()
print('ROC AUC Score - Validation Dataset:',  roc_auc_score(target_valid, extra_trees_clf.predict_proba(features_valid)[:, 1]))


# ### ROC AUC Curve - ExtraTreesClassifier

# In[79]:


target_score = extra_trees_clf.predict_proba(features_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, target_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()

precision, recall, thresholds = precision_recall_curve(target_valid, target_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()


# ### Logistic Regression

# In[80]:


get_ipython().run_cell_magic('time', '', 'log_model = LogisticRegression()\nlog_parameters = [{"solver": [\'lbfgs\', \'liblinear\', \'newton-cg\', \'newton-cholesky\', \'sag\', \'saga\'],\n                      "fit_intercept": [True, False],\n                       "penalty": [\'l1\', \'l2\', \'elasticnet\'],\n                      "n_jobs": list(range(1,200))}]\n\nlog_clf = RandomizedSearchCV(log_model, log_parameters, scoring=\'roc_auc\', n_jobs=-1, cv=cv)\nlog_clf.fit(features_train, target_train)\n# create a variable for the best model\nbest_log = log_clf.best_estimator_\nlog_pred = best_log.predict(features_valid)\nprint(\'Runtime:\')\n')


# In[81]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

fig, ax = plt.subplots(figsize = (10,5))
disp = CalibrationDisplay.from_estimator(log_clf, features_valid, target_valid, ax=ax)
plt.title('Calibration Chart - Logistic Regression', fontsize=10)
ax.set_xlabel('Mean predicted probability (Positive class: 1)', fontsize=10)
ax.set_ylabel('Fraction of positives (Positive class: 1)',fontsize=10)

ax.tick_params(color='gray', labelcolor='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')


fig.tight_layout()
plt.legend(fontsize=8)
plt.show()


# In[82]:


log_scores = cross_val_score(log_model, features_train, target_train, cv=cv, scoring='roc_auc')
print('Cross Validation Scores: {}'.format(log_scores))


# In[83]:


# summary
print('Best hyperparameters:',  log_clf.best_params_)
print()
print('Best score:',  log_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(log_scores.mean()))
print()
print('ROC AUC Score - Validation Dataset:',  roc_auc_score(target_valid, log_clf.predict_proba(features_valid)[:, 1]))


# ### ROC AUC Curve - LogisticRegression

# In[84]:


target_score = log_clf.predict_proba(features_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, target_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()

precision, recall, thresholds = precision_recall_curve(target_valid, target_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()


# ### Ridge Classification

# In[85]:


ridge_model = RidgeClassifier(random_state=random_state)
ridge_parameters = [{"alpha": list(range(0,1000)), 
                     "fit_intercept": [True, False], 
                     "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}]

ridge_clf = RandomizedSearchCV(ridge_model, ridge_parameters, scoring='roc_auc', n_jobs=-1, cv=cv)
ridge_clf.fit(features_train, target_train)
# create a variable for the best model
best_ridge = ridge_clf.best_estimator_
ridge_pred = best_ridge.predict(features_valid)
print('Runtime:')


# In[86]:


ridge_scores = cross_val_score(ridge_model, features_train, target_train, cv=cv, scoring='roc_auc')
print('Cross Validation Scores: {}'.format(ridge_scores))


# In[87]:


# summary
print('Best hyperparameters:',  ridge_clf.best_params_)
print()
print('Best score:',  ridge_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(ridge_scores.mean()))


# ### LightGBM

# In[88]:


get_ipython().run_cell_magic('time', '', "\nlgb_model = lgb.LGBMClassifier(random_state=random_state)\n\nlgb_params = {\n#'objective': ['binary'],\n'boosting_type': ['gbdt', 'dart', 'rf'],\n'num_leaves': [1,6,8,12,22,26,28,31,35,40],\n'learning_rate': [0.001,0.01, 0.05, 0.08, 0.09, 0.1, 0.11, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1],\n'feature_fraction': [0.1, 0.2, 0.5, 0.6, 0.8, 0.9, 1],\n'max_depth': [1,6,8,12,15,18,22,25,30],\n'min_data_in_leaf': [20,25,30],\n'bagging_fraction': [0.1,0.3,0.5,0.7,1],\n#'num_iterations': [1,6,8,12,20,22,30,35]\n}\n\nlgb_clf = RandomizedSearchCV(lgb_model, lgb_params, scoring='roc_auc', n_jobs=10, cv=cv)\nlgb_clf.fit(features_train, target_train)\n# create a variable for the best model\nbest_lgb = lgb_clf.best_estimator_\nlgb_pred = best_lgb.predict(features_valid)\nprint('Runtime:')\n")


# In[89]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

fig, ax = plt.subplots(figsize = (10,5))
disp = CalibrationDisplay.from_estimator(lgb_clf, features_valid, target_valid, ax=ax)
plt.title('Calibration Chart - LGBM Classifier', fontsize=10)
ax.set_xlabel('Mean predicted probability (Positive class: 1)', fontsize=10)
ax.set_ylabel('Fraction of positives (Positive class: 1)',fontsize=10)

ax.tick_params(color='gray', labelcolor='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')


fig.tight_layout()
plt.legend(fontsize=8)
plt.show()


# In[90]:


lgb_scores = cross_val_score(lgb_model, features_train, target_train, cv=cv, scoring='roc_auc')
print('Cross Validation Scores: {}'.format(lgb_scores))


# In[91]:


# summary
print('Best hyperparameters:',  lgb_clf.best_params_)
print()
print('Best score:',  lgb_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(lgb_scores.mean()))
print()
print('ROC AUC Score - Validation Dataset:',  roc_auc_score(target_valid, lgb_clf.predict_proba(features_valid)[:, 1]))


# ### ROC AUC Curve - LightGBM

# In[92]:


target_score = lgb_clf.predict_proba(features_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, target_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()

precision, recall, thresholds = precision_recall_curve(target_valid, target_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()


# ### XGBoost

# In[93]:


get_ipython().run_cell_magic('time', '', "\nxgb_model = xgb.XGBClassifier(random_state=random_state) # fix encoding if issues arise\n\nxgb_params = {\n#'min_child_weight': [1, 5, 10],\n#'gamma': [0.5, 1, 1.5, 2, 5],\n#'subsample': [0.6, 0.8, 1.0],\n#'colsample_bytree': [0.6, 0.8, 1.0],\n'booster': ['dart', 'gblinear', 'gbtree'],\n#'max_leaves': [0,6,12,16,25,35],\n'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],\n#'max_depth': [1,,6,8,12,15,18,20],\n'eval_metric': ['auc'],\n}\n\nxgb_clf = RandomizedSearchCV(xgb_model, xgb_params, scoring='roc_auc', n_jobs=-1, cv=cv)\nxgb_clf.fit(features_train, target_train)\n# create a variable for the best model\nbest_xgb = xgb_clf.best_estimator_\nxgb_pred = best_xgb.predict(features_valid)\nprint('Runtime:')\n")


# In[94]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1

fig, ax = plt.subplots(figsize = (10,5))
disp = CalibrationDisplay.from_estimator(xgb_clf, features_valid, target_valid, ax=ax)
plt.title('Calibration Chart - XGB Classifier', fontsize=10)
ax.set_xlabel('Mean predicted probability (Positive class: 1)', fontsize=10)
ax.set_ylabel('Fraction of positives (Positive class: 1)',fontsize=10)

ax.tick_params(color='gray', labelcolor='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')


fig.tight_layout()
plt.legend(fontsize=8)
plt.show()


# In[95]:


xgb_scores = cross_val_score(xgb_model, features_train, target_train, cv=cv, scoring='roc_auc')
print('Cross Validation Scores: {}'.format(xgb_scores))


# In[96]:


# summary
print('Best hyperparameters:',  xgb_clf.best_params_)
print()
print('Best score:',  xgb_clf.best_score_)
print()
print('Average Cross Validation Score: {}'.format(xgb_scores.mean()))
print()
print('ROC AUC Score - Validation Dataset:',  roc_auc_score(target_valid, xgb_clf.predict_proba(features_valid)[:, 1]))


# ### ROC AUC Curve - XGBoost

# In[97]:


target_score = xgb_clf.predict_proba(features_valid)[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, target_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()

precision, recall, thresholds = precision_recall_curve(target_valid, target_score)

fig = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=800, height=600
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.update_layout(showlegend=False)
fig.show()


# ### Final Evaluation

# #### Model Selection - Average Score Comparison
# 
# #### Note
# 
# We've selected LightGBM as our optimal model based on score comparisons, it also performs much better compared to other models when it comes to scoring and speed. XGBClassifier is similar to our LightGBM model but takes considerably longer to compute.

# In[98]:


print('Average Cross Validation Scores:')
print('RandomForestClassifier: {}'.format(forest_scores.mean()))
print('DecisionTreeClassifier: {}'.format(tree_scores.mean()))
print('ExtraTreesClassifier: {}'.format(extra_trees_scores.mean()))
print()
print('LogisticRegression: {}'.format(log_scores.mean()))
print('RidgeClassifier: {}'.format(ridge_scores.mean()))
print()
print('LightGBM: {}'.format(lgb_scores.mean()))
print('XGBoost: {}'.format(xgb_scores.mean()))


# #### Model Selection - Feature Importances

# In[99]:


boost = lgb_model.fit(features_train, target_train).booster_
# print('Feature names',boost.feature_name())

print('Feature Importance:')
{k: v for k, v in sorted(zip(boost.feature_name(), lgb_clf.best_estimator_.feature_importances_), key= lambda x: x[1], reverse=True)}


# In[100]:


r = {k: v for k, v in sorted(zip(boost.feature_name(), lgb_clf.best_estimator_.feature_importances_), key= lambda x: x[1], reverse=True)}


# In[101]:


fig = go.Figure(data=go.Scatterpolar(r = list(r.values())[:10],
               theta= list(r.keys())[:10]))

fig.update_traces(fill='toself')
fig.update_layout(polar=dict(radialaxis_angle=-45, angularaxis= dict(direction='clockwise',period=6)))

fig.update_layout(title_text='<b>Top 10 Most Important Features</b><br><sup>LightGBM</sup>',autosize=False)

fig.show()


# #### Model Selection - Confusion Matrix

# In[102]:


import plotly.figure_factory as ff

# heatmap

cm = confusion_matrix(target_valid, lgb_pred)

labels = ['Class: 0', 'Class: 1']
heatmap = go.Heatmap(z=cm, x=labels, y=labels, colorscale='Blues')

# data labels
data_labels = [[y for y in x] for x in cm]

# figure
fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, annotation_text=data_labels, colorscale='Blues')

# title
fig.update_layout(title_text='<b>Confusion matrix</b><br><sup>LightGBM</sup>'
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.15,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# margin update
fig.update_layout(height=500, width=500)

# colorbar
fig['data'][0]['showscale'] = True


fig.show()


# #### Model Selection - Final Evaluation
# 
# #### Note
# 
# After further review, given the ROC curve is calculated by taking each possible probability I've replaced my final ROC AUC Score with predictions using .predict_proba() which yields better results (91.56% vs 80.15% in the previously reviewed solution code).

# In[103]:


print('Final evaluation on test set:')
print()
# model
final_model = lgb.LGBMClassifier(random_state=random_state)
# train
final_model.fit(features_train, target_train)
# predict
final_predictions_test = final_model.predict(features_test)
final_predictions_test_proba = final_model.predict_proba(features_test)[:, 1]
# accuracy check
# final_model_accuracy = accuracy_score(target_test, final_predictions_test)
# final_model_f1 = f1_score(target_test, final_predictions_test)
final_model_roc_auc = roc_auc_score(target_test, final_predictions_test_proba)
#result
print('ROC AUC Score:',   round((final_model_roc_auc * 100), 2),'%') 


# ### Part 3: Solution Report
# 
# #### Steps Performed and Skipped:
# We followed most of the steps outlined in Part 1, except for: <br>
# 
# **Time Series Analysis:** This was ultimately deemed unnecessary as the general exploratory data analysis (EDA) and preprocessing provided sufficient insights. However, it could have been useful for visualizing client sign-up trends.  <br>
# 
# #### Difficulties Encountered and Solutions:
# **Scaling and Class Imbalance Issues:** <br>
# Initially, there were challenges in properly scaling the data and addressing class imbalance in the models. These were resolved by: <br>
# 
# Reading documentation to better understand best practices. <br>
# Adjusting the process to scale the training data only, ensuring no data leakage. <br>
# Using RepeatedStratifiedKFold() in RandomizedSearchCV() to ensure balanced sampling in cross-validation. <br>
# 
# **Missing Data Challenges:** <br>
# Some missing data was subtle and not immediately noticeable. To address this: <br>
# 
# We implemented an iterative function to detect discrepancies. <br>
# Chose appropriate methods for handling missing data, such as imputation, replacement, or removal, based on the context of each case. <br>
# 
# #### Key Steps in Solving the Task:
# **Defining Objectives:** Starting with a clear understanding of the business goals and aligning the project plan accordingly. <br>
# **Choosing Metrics:** Selecting a relevant scoring metric, such as ROC AUC, rather than absolute measures like accuracy, to focus on the relative performance across classes. <br>
# **Data Understanding:** Gaining a thorough understanding of the dataset, particularly addressing class imbalance and its impact on modeling. <br>
# **EDA and Preprocessing Decisions:** Documenting and maintaining flexibility in preprocessing steps to accommodate potential roadblocks. <br>
# **Benchmarking and Model Comparison:** <br>
#     - Building a benchmark model (e.g., dummy classifier) for initial comparison. <br>
#     - Performing grid searches to optimize model parameters. <br>
#     - Balancing model performance with computational efficiency to ensure practical applicability. <br>
# 
# #### Final Model and Quality Score:
# The chosen final model is LightGBM, which achieved a strong balance between performance and computational efficiency. Initial evaluations yielded an AUC ROC score of 80.15% using .predict(). However, after switching to .predict_proba() for score calculation, the AUC ROC improved to 91.56% on an imbalanced test dataset. <br>
# 
# #### Recommendation:
# We recommend Interconnect adopt LightGBM for predicting customer churn due to its high performance, speed, and flexibility. The model’s adaptability ensures it can handle evolving data scenarios and changes effectively.
