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
#   For your comments and questions. 
#     Thank you so much for feedback and encouraging words. It is certainly challenging to do. I copied the LaTeX from discord ai bot. It is still not in right format. Thanks! 
# </div>
#   
# First of all, thank you for turning in the project! You did a great job overall, but there are some small problems that need to be fixed before the project will be accepted. Let me know if you have any questions!
# 

# ## Project Title: 
# ### Customer Insights and Benefit Prediction for Sure Tomorrow Insurance

# Project Overview:
# 
# We will explore machine learning techniques to solve key business problems for the Sure Tomorrow insurance company. The project will address the following:
# 
# Customer Similarity: We'll develop a method to find customers who are similar to a given customer, helping agents target marketing campaigns effectively.
# 
# Benefit Prediction: We'll create a predictive model to determine whether a new customer is likely to receive an insurance benefit, comparing its performance against a baseline (dummy) model.
# 
# Insurance Benefits Prediction: A linear regression model will predict the number of benefits a customer might receive, based on features like age, salary, and family members.
# 
# Data Protection: We'll implement data obfuscation techniques to protect customers' personal information while ensuring that model performance is not compromised.
# 
# The dataset includes features like gender, age, salary, and family members, with the target variable being the number of benefits received in the last five years.

# # Data Preprocessing & Exploration
# 
# ## Initialization

# In[1]:


import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sklearn.metrics
import sklearn.neighbors

from plotly.subplots import make_subplots
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from scipy.spatial import distance
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from IPython.display import display


# ## Load Data

# Load data and conduct a basic check that it's free from obvious issues.

# In[2]:


df = pd.read_csv('/datasets/insurance_us.csv')

display(df)


# We rename the colums to make the code look more consistent with its style.

# In[3]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})


# In[4]:


df.sample(10)


# In[5]:


df.info()

df.shape


# In[6]:


# we may want to fix the age type (from float to int) though this is not critical

# write your conversion here if you choose:
df['age'] = df['age'].astype('string').str.split('.').str[0]
df['age'] = df['age'].apply(np.int64)


# In[7]:


# check to see that the conversion was successful
df.info()


# In[8]:


# now have a look at the data's descriptive statistics. 
df.describe().T
# Does everything look okay?


# <div class="alert alert-success">
# <b>Reviewer's comment V1</b>
# 
# Correct
# 
# </div>

# <b> Looking at the descriptive statistics, here’s an overview of what the data reveals: </b>
# 
# <b> gender: </b><br>
#     Mean is close to 0.5, suggesting a nearly equal distribution between the two gender categories (assuming 0 = one gender, 1 = another).<br>
#     Data appears binary, which is expected for a gender column.<br>
#     No apparent issues here.
# 
# <b> age: </b><br>
#     The minimum age is 18, and the maximum is 65, which seems reasonable for an insurance dataset.<br>
#     The mean age is ~30.95, and the distribution between the 25th and 75th percentiles (24 to 37 years) shows a reasonable spread.<br>
#     No extreme values or missing data are present.
# 
# <b> income: </b><br>
#     The income values range from 5,300 to 79,000 with a mean of ~39,916.<br>
#     The range looks plausible, but we should investigate the lower-income tail further (e.g., incomes close to 5,300).
# 
# <b> family_members: </b><br>
#     The number of family members ranges from 0 to 6, with an average of 1.19.<br>
#     No glaring issues, although we might need to confirm the legitimacy of cases where there are 0 family members.
# 
# <b> insurance_benefits: </b><br>
#     This column (the target variable) has a mean of 0.148, indicating that most customers receive no benefits.<br>
#     The maximum value is 5, which might represent outliers and could require closer examination.
# 
# <b> Conclusion: </b><br>
# 
# Overall, the data looks clean at first glance, with no immediate issues like missing or extreme values. However, it might be worth conducting further checks for outliers or unusual patterns in income and family_members.

# ## EDA

# Let's quickly check whether there are certain groups of customers by looking at the pair plot.

# In[9]:


sns.set_theme()
g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# In[10]:


df.corr()


# In[11]:


df.nunique()


# In[12]:


df.duplicated().sum()


# In[13]:


df.isnull().sum()


# Lasso for feature selection

# In[14]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize = (10,5))

features_lasso = df.drop('insurance_benefits', axis=1)
target_lasso = df.insurance_benefits
names = df.drop('insurance_benefits', axis=1).columns

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(features_lasso, target_lasso).coef_

sns.barplot(x=names, y=lasso_coef, hue=names, ax=ax)

plt.title('Feature Selection Based on Lasso - Full Dataset', fontsize=10)
ax.set_xlabel('Feature', fontsize=10)
ax.set_ylabel('Importance',fontsize=10)

ax.tick_params(color='gray', labelcolor='gray')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')


plt.xticks(rotation=0)
plt.show()


# Ok, it is a bit difficult to spot obvious groups (clusters) as it is difficult to combine several variables simultaneously (to analyze multivariate distributions). That's where LA and ML can be quite handy.

# # Task 1. Similar Customers

# In the language of ML, it is necessary to develop a procedure that returns k nearest neighbors (objects) for a given object based on the distance between the objects.
# 
# You may want to review the following lessons (chapter -> lesson)
# - Distance Between Vectors -> Euclidean Distance
# - Distance Between Vectors -> Manhattan Distance
# 
# To solve the task, we can try different distance metrics.

# Write a function that returns k nearest neighbors for an $n^{th}$ object based on a specified distance metric. The number of received insurance benefits should not be taken into account for this task. 
# 
# You can use a ready implementation of the kNN algorithm from scikit-learn (check [the link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) or use your own.
# 
# Test it for four combination of two cases
# - Scaling
#   - the data is not scaled
#   - the data is scaled with the [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) scaler
# - Distance Metrics
#   - Euclidean
#   - Manhattan
# 
# Answer these questions:
# - Does the data being not scaled affect the kNN algorithm? If so, how does that appear?
# - How similar are the results using the Manhattan distance metric (regardless of the scaling)?

# In[15]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[16]:


def get_knn(df, n, k, metric):
    
    """
    Returns k nearest neighbors

    :param df: pandas DataFrame used to find similar objects within
    :param n: object no for which the nearest neighbours are looked for
    :param k: the number of the nearest neighbours to return
    :param metric: name of distance metric
    """

    nbrs = NearestNeighbors(metric=metric).fit(df)
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res


# Scaling the data.

# In[17]:


feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[18]:


df_scaled.sample(5)
df_scaled = df_scaled.drop('insurance_benefits', axis=1)


# In[19]:


df_not_scaled = df.copy()
df_not_scaled.loc[:, feature_names] = df[feature_names].to_numpy()
df_not_scaled = df_not_scaled.drop('insurance_benefits', axis=1)


# Now, let's get similar records for a given one for every combination

# In[20]:


print('Scaled dataset - Euclidean Distance')
display(get_knn(df_scaled, 10, 100, 'euclidean'))


# In[21]:


print('No scaling dataset - Euclidean Distance')
display(get_knn(df_not_scaled, 10, 100, 'euclidean'))


# In[22]:


print('Scaled dataset - Manhattan Distance')
display(get_knn(df_scaled, 10, 20, 'manhattan'))


# In[23]:


print('No scaling dataset - Manhattan Distance')
display(get_knn(df_not_scaled, 10, 20, 'manhattan'))


# Answers to the questions

# **Does the data being not scaled affect the kNN algorithm? If so, how does that appear?** 
# 
# It influences the results. In the case of non-scaled data, the distance measurements are excessively 'dispersed' due to the differing magnitudes in both the age and income columns (this becomes particularly evident when increasing k to +100 and observing a broader range of the table). Additionally, we observe varying neighboring customers because the data is not scaled and/or the features do not accurately represent each other.

# **How similar are the results using the Manhattan distance metric (regardless of the scaling)?** 
# 
# When examining the two tables that utilize Manhattan distance, we observe some similarities in the distance measurements. However, the neighboring lists vary between them. For instance, in the scaled data, there is a greater prevalence of gender=1 near our n object, which is 10, compared to the evaluation without scaling. This disparity becomes more pronounced as k increases.

# <div class="alert alert-success">
# <b>Reviewer's comment V1</b>
# 
# Good job!
#  
# </div>

# # Task 2. Is Customer Likely to Receive Insurance Benefit?

# In terms of machine learning we can look at this like a binary classification task.

# With `insurance_benefits` being more than zero as the target, evaluate whether the kNN classification approach can do better than a dummy model.
# 
# Instructions:
# - Build a KNN-based classifier and measure its quality with the F1 metric for k=1..10 for both the original data and the scaled one. That'd be interesting to see how k may influece the evaluation metric, and whether scaling the data makes any difference. You can use a ready implemention of the kNN classification algorithm from scikit-learn (check [the link](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) or use your own.
# - Build the dummy model which is just random for this case. It should return "1" with some probability. Let's test the model with four probability values: 0, the probability of paying any insurance benefit, 0.5, 1.
# 
# The probability of paying any insurance benefit can be defined as
# 
# $$
# P\{\text{insurance benefit received}\}=\frac{\text{number of clients received any insurance benefit}}{\text{total number of clients}}.
# $$
# 
# Split the whole data in the 70:30 proportion for the training/testing parts.

# In[24]:


# calculate the target

df['insurance_benefits_received'] = df['insurance_benefits'] > 0
#print(df['insurance_benefits_received'])

insurance_benefits = df['insurance_benefits_received'].values
#print(insurance_benefits)

insurance_benefits_all = df.query("insurance_benefits_received > 0").values
#print(insurance_benefits_all)


# In[25]:


# check for the class imbalance with value_counts()

sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize = (10,5))

df['insurance_benefits_received'].value_counts().plot(kind='bar', title='count (target)')


# In[26]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# if you have an issue with the following line, restart the kernel and run the notebook again
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Confusion Matrix')
    print(cm)


# In[27]:


# generating output of a random model

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[28]:


for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'The probability: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, len(insurance_benefits), seed=42) 
        
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    
    print()


# In[29]:


# orig. data

y_orig = df['insurance_benefits'].to_numpy()
X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits_received'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=12345)

for n in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    prediction_test = knn.predict(X_test)
    eval_classifier(y_test, prediction_test)
    print('')


# In[30]:


# scaled data (df_scaled)

features_train, features_test, target_train, target_test = train_test_split(X,
                                                                            y, 
                                                                            test_size=0.3, 
                                                                            random_state=12345)
scaler = MaxAbsScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

for n in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(features_train, target_train)
    prediction_test = knn.predict(features_test)
    eval_classifier(target_test, prediction_test)
    print('')


# 
# 
# Construct a classifier utilizing the K-nearest neighbors (KNN) algorithm and assess its performance using the F1 metric for values of k ranging from 1 to 10, applied to both the original dataset and the scaled version. It would be intriguing to observe how varying k affects the evaluation metric, as well as to determine if scaling the data produces any significant impact.
# 
# The effect of scaling is significant in this context. When examining scaled data and as the value of K rises, the F1 score experiences a slight decline; however, it begins from a considerably higher initial point compared to the unscaled or original dataset, where the F1 score initiates at 0.61 and swiftly approaches zero. Overall, scaled data demonstrates consistently superior performance across all iterations of K.

# <div class="alert alert-success">
# <b>Reviewer's comment V1</b>
# 
# Well done!
#  
# </div>

# # Task 3. Regression (with Linear Regression)

# With `insurance_benefits` as the target, evaluate what RMSE would be for a Linear Regression model.

# Build your own implementation of LR. For that, recall how the linear regression task's solution is formulated in terms of LA. Check RMSE for both the original data and the scaled one. Can you see any difference in RMSE between these two cases?
# 
# Let's denote
# - $X$ — feature matrix, each row is a case, each column is a feature, the first column consists of unities
# - $y$ — target (a vector)
# - $\hat{y}$ — estimated tagret (a vector)
# - $w$ — weight vector
# 
# The task of linear regression in the language of matrices can be formulated as
# 
# $$
# y = Xw
# $$
# 
# The training objective then is to find such $w$ that it would minimize the L2-distance (MSE) between $Xw$ and $y$:
# 
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
# 
# It appears that there is analytical solution for the above:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$
# 
# The formula above can be used to find the weights $w$ and the latter can be used to calculate predicted values
# 
# $$
# \hat{y} = X_{val}w
# $$

# Split the whole data in the 70:30 proportion for the training/validation parts. Use the RMSE metric for the model evaluation.

# In[31]:


class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)

    def predict(self, X):
        
        # adding the unities
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X2.dot(self.weights)
        
        return y_pred


# In[32]:


def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    


# Initial/Base Data

# In[33]:


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(lr.weights)

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# Scaled Data - MaxAbsScaler()

# In[34]:


scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test = train_test_split(df_scaled, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(scaled_X_train, scaled_y_train)
print('lr weights:', lr.weights)
print('')
scaled_y_test_pred = lr.predict(scaled_X_test)
eval_regressor(scaled_y_test, scaled_y_test_pred)


# <b> Can you see any difference in RMSE between these two cases? </b>
# 
# Based on our observations, the RMSE values remain consistent regardless of variations in weights. However, scaling could influence linear regression outcomes when utilizing L1/L2 regularization or when the model is trained through stochastic gradient descent.

# <div class="alert alert-success">
# <b>Reviewer's comment V1</b>
# 
# You're absolutely right:)
#  
# </div>

# # Task 4. Obfuscating Data

# It best to obfuscate data by multiplying the numerical features (remember, they can be seen as the matrix $X$) by an invertible matrix $P$. 
# 
# $$
# X' = X \times P
# $$
# 
# Try to do that and check how the features' values will look like after the transformation. By the way, the intertible property is important here so make sure that $P$ is indeed invertible.
# 
# You may want to review the 'Matrices and Matrix Operations -> Matrix Multiplication' lesson to recall the rule of matrix multiplication and its implementation with NumPy.

# In[35]:


personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]
display(df_pn)


# In[36]:


X = df_pn.to_numpy()
print(X)


# Generating a random matrix $P$.

# In[37]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))


# Checking the matrix $P$ is invertible

# In[38]:


print('X @ P - Transformed Matrix\n')
transformation = X @ P
print(transformation)

print('')
print('P Inverse\n')
P_inverse = np.linalg.inv(P)
print(P)
print('')


# <b> Can you guess the customers' ages or income after the transformation? </b>

# It is impossible to accurately determine a customer's age or income based solely on the information available. Nevertheless, one can observe some trends and general patterns within those two specific columns. These trends tend to align with the structure and characteristics of the initial dataset, exhibiting behavior that is quite comparable to that of the original data. This suggests that while exact figures remain elusive, there are still discernible indicators that reflect similar tendencies as found in the primary dataset.

# In[39]:


display(df_pn)
display(pd.DataFrame(transformation, columns=personal_info_column_list))


# <b> Can you recover the original data from $X'$ if you know $P$? Try to check that with calculations by moving $P$ from the right side of the formula above to the left one. The rules of matrix multiplcation are really helpful here. </b>

# Assuming that this data recovery is accomplished by multiplying the altered data with the inverse of P.

# In[40]:


# 𝑋′ is the transpose matrix

print('Recovered\n')
recovered = transformation.dot(P_inverse)
print(recovered)
print('')


# Print all three cases for a few customers
# - The original data
# - The transformed one
# - The reversed (recovered) one

# In[41]:


print('The original data:')
display(pd.DataFrame(X, columns=personal_info_column_list).head(3))

print('')
print('The transformed one:')
display(pd.DataFrame(transformation, columns=personal_info_column_list).head(3))

print('')
print('The reversed (recovered) one:')
display(pd.DataFrame(recovered, columns=personal_info_column_list).head(3))
print('')


# <b> You can probably see that some values are not exactly the same as they are in the original data. What might be the reason for that? </b>

# One might infer that it results from the general procedure of obscuring the data, such as using floating-point numbers during the conversion from integers.

# <div class="alert alert-success">
# <b>Reviewer's comment V1</b>
# 
# Correct
#  
# </div>

# ## Proof That Data Obfuscation Can Work with LR

# The regression task has been solved with linear regression in this project. Your next task is to prove _analytically_ that the given obfuscation method won't affect linear regression in terms of predicted values i.e. their values will remain the same. Can you believe that? Well, you don't have to, you should prove it!

# So, the data is obfuscated and there is $X \times P$ instead of just $X$ now. Consequently, there are other weights $w_P$ as
# $$
# w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
# $$
# 
# How would $w$ and $w_P$ be linked if you simplify the formula for $w_P$ above? 
# 
# What would be predicted values with $w_P$? 
# 
# What does that mean for the quality of linear regression if you measure it with RMSE?
# 
# Check Appendix B Properties of Matrices in the end of the notebook. There are useful formulas in there!
# 
# No code is necessary in this section, only analytical explanation!

# **Answer**

# <b> How would 𝑤 and 𝑤𝑃 be linked if you simplify the formula for 𝑤𝑃 above? </b>
# 
# Based off of the condensed/manipulated result: $$ w_P = P^{-1} .w $$
# 
# What would be predicted values with 𝑤𝑃?
# 
# Same as the values before the obsfucation. We can get back to the original formula using the multiplicative identity property:
# 
# <table>
# <tr>
# <td>replacing 𝑤 with 𝑤𝑃:</td><td>$\hat{y_P} = X_{val}P \cdot w_P$</td>
# </tr>    
# <tr>
# <td>Multiplicative identity:</td><td>$\hat{y_P} = X_{val}P \cdot P^{-1}w$</td>
# </tr>    
# <tr>
# <td>Multiplicative identity:</td><td>$\hat{y_P} = X_{val} P \cdot I \cdot w$</td>
# </tr>    
# <tr>
# <td>Multiplicative identity:</td><td>$\hat{y_P} = X_{val}w$</td>
# </tr>    
# <tr>
# <td>$\hat{y_P} = \hat{y}$</td>
# </tr> 
# </table>	
# 
# <b> What does that mean for the quality of linear regression if you measure it with RMSE? </b>
# 
# RMSE is not expected to change given equality.
# 

# **Analytical proof**

# Formula is condesed as things get removed or cancelled out (one by one; manupulation)
# Masking does not affect linear regression itself
# 
# <table>
# <tr>
# <td>$$w_P = [(XP)^T XP]^{-1} (XP)^T y$$</td>
# </tr>    
# <tr>
# <td>Reversivity:</td><td>$$w_P = [(P^TX^T XP]^{-1} \cdot P^TX^T \cdot y$$</td>
# </tr>
# <tr>
# <td>Associative property:</td><td>$$w_P = [(P^T(X^TX)P]^{-1} P^TX^T y$$</td>
# </tr>
# <tr>
# <td>Multiplicative identity:</td><td>$$w_P = P^{-1}(X^TX)^{-1}(P^{-1}) P^TX^T y$$</td>
# </tr>
# <tr>
# <td>Associative property:</td><td>$$w_P = P^{-1}(X^TX)^{-1}[(P^{-1}) \cdot P^T]X^T y$$</td>
# </tr>
# <tr>
# <td>Multiplicative identity:</td><td>$$w_P = P^{-1}(X^TX)^{-1}I \cdot X^T y$$</td>
# </tr>
# <tr>
# <td>$$w_P = P^{-1}(X^TX)^{-1}X^T \cdot y$$</td>
# </tr>
# <tr>
# <td>$$w_P = P^{-1}\cdot w$$</td>
# </tr>
# </table>

# <div class="alert alert-success">
# <b>Reviewer's comment V1</b>
# 
# Everything is correct as usual:)
#  
# </div>

# ## Test Linear Regression With Data Obfuscation

# Now, let's prove Linear Regression can work computationally with the chosen obfuscation transformation.
# 
# Build a procedure or a class that runs Linear Regression optionally with the obfuscation. You can use either a ready implementation of Linear Regression from sciki-learn or your own.
# 
# Run Linear Regression for the original data and the obfuscated one, compare the predicted values and the RMSE, $R^2$ metric values. Is there any difference?

# **Procedure**
# 
# - Create a square matrix $P$ of random numbers.
# - Check that it is invertible. If not, repeat the first point until we get an invertible matrix.
# - <! your comment here !>
# - Use $XP$ as the new feature matrix

# In[42]:


# original data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
y_test_pred = lr.predict(X_test) # metrics were similar leveraging the training dataset
eval_regressor(y_test, y_test_pred) # metrics were similar leveraging the training dataset


# Checking feature importance

# In[43]:


sns.set(style="white", color_codes=True)
plt.rcParams['axes.linewidth'] = 0.1
fig, ax = plt.subplots(figsize = (10,5))

# calling the model with the best parameter
lasso1 = Lasso(alpha=0.1)
lasso1.fit(X_train, y_train)

# Using np.abs() to make coefficients positive.  
lasso1_coef = np.abs(lasso1.coef_)

# plotting the Column Names and Importance of Columns. 
sns.barplot(x=names, y=lasso1_coef, hue=names)
plt.xticks(rotation=0)
plt.grid()
plt.title("Feature Selection Based on Lasso - Split Dataset")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.ylim(0, 0.15)
plt.show()


# In[44]:


# obfuscation transformation

X_train, X_test, y_train, y_test = train_test_split(transformation, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
y_test_pred = lr.predict(X_test) # metrics were similar leveraging the training dataset
eval_regressor(y_test, y_test_pred) # metrics were similar leveraging the training dataset


# <div class="alert alert-success">
# <b>Reviewer's comment V1</b>
# 
# You did a really great job. Well done!
#  
# </div>

# # Conclusions

# <b> "Adjusting and correcting disparities - enhancing output following the normalization or standardization of features." </b>
# 
# The analysis reveals the impact of unscaled data on distance value outcomes. When features exhibit greater variability or magnitude, the resulting distance outputs become increasingly dispersed, a trend that is more pronounced when we increase k beyond 100. Additionally, by applying the same distance measurement to both scaled and unscaled data, we can observe the actual differences in output and gain insight into the extent of variability present.
# 
# Scaling significantly impacts scores. In the case of scaled data, as K rises, we observe declines in the F1 score compared to the non-scaled or original dataset, which begins at a higher level overall. With our non-scaled data, the F1 score starts at 0.61 and swiftly drops to zero by the 10th iteration of K. In contrast, the scaled data remains relatively stable across the same number of K iterations, consistently around the 0.90 mark after 10 iterations.
# 
# <b> "Data masking - demonstrating that concealing our data can be effective with linear regression models without impacting specific results." </b>
# 
# During our transformation process, we determined that there is sufficient evidence (both qualitative and quantitative) indicating that data masking does not influence our RMSE metrics within this particular modeling scenario and specific parameters. However, if our transformation matrix is accessible, it is probable that the original data could be partially reconstructed or retrieved, although it will never be possible to completely ascertain the original dataset.

# # Checklist

# Type 'x' to check. Then press Shift+Enter.

# - [x]  Jupyter Notebook is open
# - [ ]  Code is error free
# - [ ]  The cells are arranged in order of logic and execution
# - [ ]  Task 1 has been performed
#     - [ ]  There is the procedure that can return k similar customers for a given one
#     - [ ]  The procedure is tested for all four proposed combinations
#     - [ ]  The questions re the scaling/distances are answered
# - [ ]  Task 2 has been performed
#     - [ ]  The random classification model is built and tested for all for probability levels
#     - [ ]  The kNN classification model is built and tested for both the original data and the scaled one, the F1 metric is calculated.
# - [ ]  Task 3 has been performed
#     - [ ]  The linear tegression solution is implemented with matrix operations.
#     - [ ]  RMSE is calculated for the implemented solution.
# - [ ]  Task 4 has been performed
#     - [ ]  The data is obfuscated with a random and invertible matrix P
#     - [ ]  The obfuscated data is recoved, few examples are printed out
#     - [ ]  The analytical proof that the transformation does not affect RMSE is provided 
#     - [ ]  The computational proof that the transformation does not affect RMSE is provided
# - [ ]  Conclusions have been made

# # Appendices 
# 
# ## Appendix A: Writing Formulas in Jupyter Notebooks

# You can write formulas in your Jupyter Notebook in a markup language provided by a high-quality publishing system called $\LaTeX$ (pronounced "Lah-tech"), and they will look like formulas in textbooks.
# 
# To put a formula in a text, put the dollar sign (\\$) before and after the formula's text e.g. $\frac{1}{2} \times \frac{3}{2} = \frac{3}{4}$ or $y = x^2, x \ge 1$.
# 
# If a formula should be in its own paragraph, put the double dollar sign (\\$\\$) before and after the formula text e.g.
# 
# $$
# \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
# $$
# 
# The markup language of [LaTeX](https://en.wikipedia.org/wiki/LaTeX) is very popular among people who use formulas in their articles, books and texts. It can be complex but its basics are easy. Check this two page [cheatsheet](http://tug.ctan.org/info/undergradmath/undergradmath.pdf) for learning how to compose the most common formulas.

# ## Appendix B: Properties of Matrices

# Matrices have many properties in Linear Algebra. A few of them are listed here which can help with the analytical proof in this project.

# <table>
# <tr>
# <td>Distributivity</td><td>$A(B+C)=AB+AC$</td>
# </tr>
# <tr>
# <td>Non-commutativity</td><td>$AB \neq BA$</td>
# </tr>
# <tr>
# <td>Associative property of multiplication</td><td>$(AB)C = A(BC)$</td>
# </tr>
# <tr>
# <td>Multiplicative identity property</td><td>$IA = AI = A$</td>
# </tr>
# <tr>
# <td></td><td>$A^{-1}A = AA^{-1} = I$
# </td>
# </tr>    
# <tr>
# <td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
# </tr>    
# <tr>
# <td>Reversivity of the transpose of a product of matrices,</td><td>$(AB)^T = B^TA^T$</td>
# </tr>    
# </table>
