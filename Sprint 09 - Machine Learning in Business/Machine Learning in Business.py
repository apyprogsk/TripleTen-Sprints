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
# Thank you so much for the feedback, I appreacaite it! I should have double checked before submitting. Thanks! 
# </div>
# 

# <b> Project Overview: OilyGiant Mining Company Well Location Analysis </b>
# 
# <b> Objective: </b>
# 
# The goal of this project is to identify the most profitable region for developing new oil wells for OilyGiant mining company. We are tasked with analyzing oil well data from three different regions to determine which location offers the best combination of high returns and low risk.
# 
# <b> Data: </b>
# 
# We have geological exploration data for three regions, each containing information on 100,000 oil wells. The data includes features of each well (f0, f1, f2) and the volume of reserves (product) in thousands of barrels.
# 
# <b> Methodology: </b>
# 
# Data Preparation: Load and examine the data from all three regions. <br>
# Model Development: Create a linear regression model for each region to predict oil reserves based on well features. <br>
# Profit Calculation: Determine the break-even point and calculate potential profits for the top 200 wells in each region. <br>
# Risk Assessment: Use bootstrapping to estimate the distribution of profits and assess the risk of losses for each region. <br>
# Region Selection: Choose the best region based on the highest average profit with a risk of loss below 2.5%.
# 
# 
# <b> KEY CONDITIONS: </b>
# 
# <b> Budget: </b> 100 million dollars for developing 200 oil wells <br>
# <b> Revenue: </b> 4,500 dollars per thousand barrels of oil <br>
# <b> Model Constraint: </b> Only linear regression is suitable for predictions <br>
# <b> Well Selection: </b> 200 best wells are chosen out of 500 explored in each region <br>
# <b> Risk Threshold: </b> Regions with over 2.5% risk of loss are disqualified
# 
# 
# <b> Expected Outcome: </b>
# By the end of this analysis, we will recommend the most suitable region for OilyGiant to develop new oil wells, balancing the potential for high profits with an acceptable level of risk.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load the data
region_0 = pd.read_csv('/datasets/geo_data_0.csv')
region_1 = pd.read_csv('/datasets/geo_data_1.csv')
region_2 = pd.read_csv('/datasets/geo_data_2.csv')

# Check the data
for i, region in enumerate([region_0, region_1, region_2]):
    print(f"Region {i} shape: {region.shape}")
    display(region.head())
    print(region.info())
    print("\n")


# <b> Analysis </b>
# 
# <b> Data structure: </b>
# 
# All three regions have the same structure: 100,000 entries with 5 columns each.
# The columns are: 'id', 'f0', 'f1', 'f2', and 'product'.
# 
# 
# <b> Data types: </b>
# 
# 'id' is an object (string) type, likely used as a unique identifier for each well.
# 'f0', 'f1', 'f2', and 'product' are all float64 type, which is appropriate for numerical data.
# 
# 
# <b> Missing data: </b>
# 
# There are no null values in any of the datasets, which is good for our analysis.
# 
# 
# <b> Feature ranges: </b>
# 
# Region 0: Features seem to be mostly within a smaller range, roughly -1 to 5. <br>
# Region 1: Features have a wider range, roughly -15 to 15. <br>
# Region 2: Features seem to have a range similar to Region 0.
# 
# 
# <b> Target variable ('product'): </b>
# 
# Region 0: The 'product' values seem to be higher, ranging from about 70 to 170. <br>
# Region 1: The 'product' values have a wider range, from very low (3.17) to higher values (137.94). <br>
# Region 2: The 'product' values seem to be in between, ranging from about 27 to 150.
# 
# 
# 
# These observations suggest that the regions have different characteristics, which might lead to different predictions and profitability estimates.
# <br>
# <br>
# <br>
# <b> Step 2: </b> <br>Train and test the model for each region.

# In[2]:


def check_data_quality(region_data, region_name):
    print(f"Data Quality Check for {region_name}:")
    
    # Check for missing values
    missing_values = region_data.isnull().sum()
    print("Missing values:")
    print(missing_values)
    
    # Check for duplicates
    duplicates = region_data.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    print("\n")

# Run the check for each region
for i, region in enumerate([region_0, region_1, region_2]):
    check_data_quality(region, f"Region {i}")


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Good job
# 
# </div>

# <b> For all three regions: </b>
# 
# <b> Missing Values: </b> There are no missing values in any of the columns. This is excellent as it means we don't need to handle missing data.
# 
# <b> Duplicate Rows: </b> There are no duplicate rows in any of the regions. This is good because it means we don't have any completely identical entries.

# In[3]:


def train_test_model(region_data):
    X = region_data[['f0', 'f1', 'f2']]
    y = region_data['product']
    
    # 2.1 Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 2.2 Train the model and make predictions
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    # 2.3 Save predictions and correct answers
    validation_results = pd.DataFrame({'actual': y_val, 'predicted': y_pred})
    
    # 2.4 Print average volume and RMSE
    avg_predicted = np.mean(y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"Average predicted volume: {avg_predicted:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    return model, validation_results, avg_predicted, rmse

# Run for each region
for i, region in enumerate([region_0, region_1, region_2]):
    print(f"Region {i}:")
    model, val_results, avg_pred, rmse = train_test_model(region)
    print("\n")


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Correct
# 
# </div>

# <b> Analysis: </b>
# 
# <b> Average Predicted Volume: </b>
# 
# Region 2 has the highest average predicted volume (94.77), followed closely by Region 0 (92.40). <br>
# Region 1 has a significantly lower average predicted volume (68.71).
# 
# 
# <b> Root Mean Square Error (RMSE): </b>
# 
# Region 1 has a remarkably low RMSE (0.89), indicating that the model's predictions are very close to the actual values for this region. <br>
# Regions 0 and 2 have much higher RMSE values (37.76 and 40.15 respectively), suggesting that the predictions for these regions are less accurate.
# 
# 
# 
# <b> Interpretation: </b>
# 
# <b> Region 1: </b>
# 
# The model performs exceptionally well for this region, with very low prediction error. <br>
# However, it predicts the lowest average volume of oil.
# 
# 
# <b> Regions 0 and 2: </b>
# 
# These regions show promise with higher predicted volumes. <br>
# However, the high RMSE values indicate that there's more uncertainty in these predictions.
# 
# 
# <b> Model Performance: </b>
# 
# The linear regression model seems to fit Region 1 data very well, but struggles with Regions 0 and 2. <br>
# This could be due to more complex relationships between features and oil volume in Regions 0 and 2, which the linear model can't capture fully.
# 
# 
# <b> Potential Trade-offs: </b>
# 
# We have a trade-off between prediction accuracy (favoring Region 1) and potential high yields (favoring Regions 0 and 2).
# 
# 
# 
# <b> Next Steps: </b>
# For the profit calculation, we'll need to consider both the predicted volumes and the uncertainty in our predictions. The high RMSE values for Regions 0 and 2 suggest that we should be cautious about relying too heavily on their higher predicted volumes.
# <br>
# <br>
# <br>
# <b> Step 3: </b> <br>Preparing for profit calculation.

# In[4]:


# 3.1 Key values
budget = 100_000_000  # $100 million
price_per_barrel = 4500  # $4500 per 1000 barrels
num_wells = 200

# 3.2 Calculate break-even volume
break_even_volume = budget / (price_per_barrel * num_wells)

print(f"Break-even volume per well: {break_even_volume:.2f} thousand barrels")

# Compare with average volume in each region
for i, region in enumerate([region_0, region_1, region_2]):
    avg_volume = region['product'].mean()
    print(f"Region {i} average volume: {avg_volume:.2f} thousand barrels")


# <div class="alert alert-success">
# <b>Reviewer's comment</b>
# 
# Correct
# 
# </div>

# <b> Analysis: </b>
# 
# <b> Break-even volume: </b>
# 
# The break-even volume per well is 111.11 thousand barrels. This means that each well needs to produce at least this much oil to cover the costs.
# 
# 
# <b> Average volumes: </b>
# 
# Region 0: 92.50 thousand barrels <br>
# Region 1: 68.83 thousand barrels <br>
# Region 2: 95.00 thousand barrels <br>
# 
# 
# <b> Analysis: </b>
# 
# <b> Profitability Threshold: </b>
# 
# None of the regions have an average volume that exceeds the break-even point of 111.11 thousand barrels. <br>
# This suggests that, on average, wells in all regions would operate at a loss if we were to develop all wells.
# 
# 
# <b> Comparison between regions: </b>
# 
# Region 2 has the highest average volume (95.00), closest to the break-even point. <br>
# Region 0 is second (92.50), not far behind Region 2. <br>
# Region 1 has a significantly lower average volume (68.83), far below the break-even point.
# 
# 
# <b> Potential Strategies: </b>
# 
# Since we're only developing 200 wells out of the 100,000 in each dataset, we should focus on selecting the most promising wells rather than relying on averages. <br>
# We'll need to identify the wells with the highest predicted volumes in each region.
# 
# 
# <b> Risk Considerations: </b>
# 
# Remember that Regions 0 and 2 had high RMSE values, indicating more uncertainty in predictions. <br>
# While Region 1 has lower average volume, its low RMSE suggests more reliable predictions.
# 
# 
# 
# <b> Conclusions: </b>
# 
# The project appears risky as average volumes are below the break-even point. <br>
# Success will depend on our ability to accurately identify the most productive wells. <br>
# We'll need to balance the potential for high volumes (Regions 0 and 2) against prediction reliability (Region 1).
# <br>
# <br>
# <br>
# <b> Step 4: </b> <br>We will calculate the potential profit for each region by selecting the top 200 wells based on our model predictions. This will give us a more realistic picture of the potential profitability of each region.

# In[5]:


def calculate_profit(predictions, targets, n_wells=200, price_per_barrel=4500, budget=100_000_000):
    # 4.1 Select top 200 wells based on predictions
    top_indices = np.argsort(predictions)[-n_wells:]
    
    # 4.2 Calculate total volume using corresponding actual values (targets)
    top_actual_volumes = targets[top_indices]
    total_volume = np.sum(top_actual_volumes)
    
    # 4.3 Calculate profit
    revenue = total_volume * price_per_barrel
    profit = revenue - budget
    
    return profit, total_volume

# Calculate profit for each region
for i, region in enumerate([region_0, region_1, region_2]):
    # Split the data
    X = region[['f0', 'f1', 'f2']]
    y = region['product']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the entire dataset
    predictions = model.predict(X)
    
    # Calculate profit
    profit, volume = calculate_profit(predictions, y)
    
    print(f"Region {i}:")
    print(f"Total volume: {volume:.2f} thousand barrels")
    print(f"Profit: ${profit:,.2f}")
    print("\n")


# <b> Analysis: </b>
# 
# <b> Profitability Ranking: </b>
# 
# <b> Region 0: </b> Highest profit at about USD 35.25 million <br>
# <b> Region 2: </b> Second-highest profit at about USD 25.96 million <br>
# <b> Region 1: </b> Lowest profit at about USD 24.15 million <br>
# 
# 
# <b> Volume Comparison: </b>
# 
# The ranking by total volume matches the ranking by profit, which is expected given the fixed price per barrel. <br>
# Region 0 has significantly higher volume than the other two regions.
# 
# 
# <b> Comparison to Previous Results: </b>
# 
# The profits are lower across all regions compared to our earlier calculations. This is because we're now using actual values (targets) to calculate the profit, rather than predicted values. <br>
# The ranking of regions has changed slightly, with Region 2 now showing higher profit than Region 1.
# 
# 
# <b> Profitability: </b>
# 
# All three regions still show positive profits, which is encouraging. <br>
# However, the margins are smaller than in our previous analysis, reflecting a more realistic assessment based on actual oil volumes.
# 
# 
# <b> Model Performance Implications: </b>
# 
# The difference between these results and our previous ones suggests that our model's predictions were somewhat optimistic, especially for Regions 0 and 2. <br>
# This aligns with the high RMSE values we observed earlier for these regions.
# 
# 
# 
# <b> Conclusions: </b>
# 
# Region 0 remains the most profitable option, with a significant lead over the other two regions. <br>
# The gap between Region 1 and Region 2 has narrowed, with Region 2 now slightly outperforming Region 1. <br>
# All regions are still profitable, but the profits are more modest than our initial estimates suggested.
# 
# <b> Step 5: </b>
# 
# Let's proceed with the risk assessment (bootstrapping) using this corrected profit calculation method. This will give us a more accurate picture of the potential outcomes and risks for each region. <br>
# Given the significant change in our profit estimates, we may want to revisit our break-even analysis to ensure that these lower profits still meet the company's investment criteria. <br>
# It would be valuable to analyze the discrepancies between our model's predictions and the actual values, especially for Regions 0 and 2, to understand where our model might be overestimating oil volumes. <br>
# Consider if there are any additional features or data that could improve our model's accuracy, particularly for Regions 0 and 2.

# <div class="alert alert-danger">
# <b>Reviewer's comment</b>
# 
# Unfortunately, this function is incorrect. 
# 1. You should calculate profit using both predictions and targets. You need to pick the top wells using predictions but then you need to use corresponding targets to calculate the profit.
# 2. The function should take predictions and targets and calculate profit based on it. So, predictions should be made outside the function.
#     
# P.S. In the lesson you have very similar example about lessons and students.
# 
# </div>

# <div class="alert alert-block alert-info">
# <b>Student answer.</b> <a class="tocSkip"></a>
# 
# Thank you for the corrections mentioned. I have updated the code accordingly.  
# </div>
# 

# <div class="alert alert-danger">
# <b>Reviewer's comment</b>
# 
# The results are incorrect due two reasons:
# 1. The function calculate_profit is not correct. And yes, you should use the function calculate_profit inside the boostrap loop. This is the reason why you need to write this function:)
# 2. According to the project description in the bootstrap you should sample with size=500
#   
# 
# </div>

# In[6]:


def calculate_profit(predictions, targets, n_wells=200, price_per_barrel=4500, budget=100_000_000):
    top_indices = np.argsort(predictions)[-n_wells:]
    top_actual_volumes = targets[top_indices]
    total_volume = np.sum(top_actual_volumes)
    revenue = total_volume * price_per_barrel
    profit = revenue - budget
    return profit

def bootstrap_profit(region_data, n_samples=1000, sample_size=500):
    X = region_data[['f0', 'f1', 'f2']].values  # Convert to numpy array
    y = region_data['product'].values  # Convert to numpy array
    profits = []
    
    for _ in range(n_samples):
        sample_indices = np.random.choice(len(X), size=sample_size, replace=True)
        X_sample, y_sample = X[sample_indices], y[sample_indices]  # No .iloc[] needed
        
        model = LinearRegression().fit(X_sample, y_sample)
        predictions = model.predict(X_sample)
        
        profit = calculate_profit(predictions, y_sample)
        profits.append(profit)
    
    return profits

def analyze_profits(profits):
    avg_profit = np.mean(profits)
    ci_lower, ci_upper = np.percentile(profits, [2.5, 97.5])
    risk_of_loss = (np.array(profits) < 0).mean() * 100
    return avg_profit, ci_lower, ci_upper, risk_of_loss

# Perform bootstrapping for each region
for i, region in enumerate([region_0, region_1, region_2]):
    profits = bootstrap_profit(region)
    avg_profit, ci_lower, ci_upper, risk_of_loss = analyze_profits(profits)
    
    print(f"Region {i}:")
    print(f"Average profit: ${avg_profit:,.2f}")
    print(f"95% Confidence Interval: (${ci_lower:,.2f}, ${ci_upper:,.2f})")
    print(f"Risk of loss: {risk_of_loss:.2f}%")
    print("\n")

# Select the best region
valid_regions = [i for i in range(3) if analyze_profits(bootstrap_profit([region_0, region_1, region_2][i]))[3] < 2.5]

if valid_regions:
    best_region = max(valid_regions, key=lambda i: analyze_profits(bootstrap_profit([region_0, region_1, region_2][i]))[0])
    print(f"The best region for development is Region {best_region}")
else:
    print("No region satisfies the risk criteria.")


# <div class="alert alert-success">
# <b>Reviewer's comment V2</b>
# 
# Everything is correct now. Well done!
#   
# 
# </div>

# <b> Analysis: </b>
# 
# <b> Region 1: </b>
# 
# Average profit: USD 4,416,336.53 <br>
# 95% Confidence Interval: (USD 523,280.70, USD 8,439,282.41) <br>
# Risk of loss: 1.50% <br>
# 
# Region 1 stands out as the best option for several reasons: <br>
# 
# Highest average profit among all regions <br>
# Lowest risk of loss at only 1.50% <br>
# The confidence interval is entirely positive, meaning even in the worst-case scenario within this interval, there's still a profit <br>
# 
# <b> Region 0: </b>
# 
# Average profit: USD 4,463,955.21 <br>
# 95% Confidence Interval: (USD -618,091.05, USD 9,458,595.77) <br>
# Risk of loss: 4.60% <br>
# 
# Region 0 has the second-highest average profit, but: <br>
# 
# The lower bound of the confidence interval is negative <br>
# Higher risk of loss compared to Region 1 <br>
# 
# <b> Region 2: </b>
# 
# Average profit: USD 3,757,110.55 <br>
# 95% Confidence Interval: (USD -1,637,759.81, USD 9,016,179.78) <br>
# Risk of loss: 8.10% <br>
# 
# Region 2 is the least attractive option: <br>
# 
# Lowest average profit <br>
# Highest risk of loss <br>
# Widest confidence interval with the most negative lower bound <br>
# 
# 
# 
# <b> Conclusions: </b>
# 
# Region 1 is the best choice for development. This conclusion is supported by the following key factors: <br>
# 
# Highest average profit: Region 1 has the highest expected profit at USD 4,416,336.53, which is superior to both Region 0 (USD 4,463,955.21) and Region 2 (USD 3,757,110.55). <br>
# Lowest risk: Region 1 has the lowest risk of loss at 1.50%, compared to 4.60% for Region 0 and 8.10% for Region 2. <br>
# Positive confidence interval: Region 1 is the only region with a fully positive 95% confidence interval (USD 523,280.70 to USD 8,439,282.41), indicating that even in less favorable scenarios, it's likely to remain profitable. <br>
# 
# These factors combine to make Region 1 the most attractive option for development, offering the best balance of high potential returns and lower risk compared to the alternatives. The project should therefore proceed with development plans focused on Region 1 to maximize the likelihood of success and profitability.
