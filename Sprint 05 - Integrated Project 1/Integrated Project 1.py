#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment</b>
#     
# Hi, I am a reviewer on this project.
# 
# Before we start, I want to pay your attention to the color marking:
#     
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# Great solutions and ideas that can and should be used in the future are in green comments.   
# </div>    
#     
#     
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
# 
# Yellow color indicates what should be optimized. This is not necessary, but it will be great if you make changes to this project.
# </div>      
#     
#     
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
# 
# Issues that need to be corrected to get right results are indicated in red comments. Note that the project cannot be accepted until these issues are resolved.
# </div>    
# 
# <hr>
#     
# **Please, use some color other than those listed to highlight answers to my comments.**
# I would also ask you **not to change, move or delete my comments** so that it would be easier for me to navigate during the next review.
#     
# In addition, my comments are defined as headings. 
# They can mess up the content; however, they are convenient, since you can immediately go to them. I will remove the headings from my comments in the next review. 
#    
#     
#     
# <hr>
#     
# <font color='dodgerblue'>**A few words about the project:**</font> you did a great job here, thank you so much for submitting the project! The project has a very good and clear structure, which is very important. It also has a lot of charts. It's good, visualization never hurts. However, there are still several issues that need your attention. For instance, we should update the relevant time interval a little bit. I tried to leave detailed comments, please take a look :) I've also left there some recommendations for improving the project.
#     
#     
# I will wait for the project for a second review :)
#     
#     
# 
# </div>

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 2</b>
#     
# Thank you for the updates! The time interval is now correct and includes recent years only, which is exactly what we need. However, some charts should be updated a little bit. In addition, let's not remove 50% of data, since we lose a lot of important information about games and their sales. The results may change, so don't forget to change the conclusions. My new comments have digit 2. Would you take a look? 
#     
#     
#     
# 
# </div>

# # Analysis of Global Video Game Sales and Consumer Preferences

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# First things first. The introduction is the initial paragraph that each project, each essay or any article should have. It is important to write an introductory part because it gives an idea about the content of the project. 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#     
# Great! 
# </div>

# Introduction
# 
# The video game industry has grown exponentially over the years, becoming a significant part of global entertainment. This project aims to analyze the sales data of video games to uncover trends and insights that can help understand market dynamics. Specifically, the analysis will focus on:
# 
#     Identifying the top platforms and genres in different regions.
#     Examining the impact of ESRB ratings on sales in individual regions.
#     Testing hypotheses related to average user ratings for different platforms and genres.
# 
# The dataset used in this analysis contains information on various video games, including their sales figures in North America, Europe, and Japan, user and critic scores, and ESRB ratings. By exploring this data, we aim to provide a comprehensive overview of the video game market and highlight key factors influencing game sales.

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/datasets/games.csv'
games_data = pd.read_csv(file_path)

# Display the first few rows and column names
print("Initial Data (First 5 rows):")
display(games_data.head())
print("\nColumn Names:")
display(games_data.columns)

# Check for any remaining missing values
print("\nRemaining Missing Values:")
display(games_data.isna().sum())


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>  Reviewer's comment </h2>
#    
# Please do not forget about the `display` method.
# 
# </div>

# In[51]:


# Rename columns
games_data.columns = games_data.columns.str.lower()

# Drop rows with missing 'name' or 'genre'
games_data.dropna(subset=['name', 'genre'], inplace=True)

# Keep 'year_of_release' as NaN if missing and handle it later
# games_data.dropna(subset=['year_of_release'], inplace=True)

# Convert 'year_of_release' to int where possible
games_data['year_of_release'] = pd.to_numeric(games_data['year_of_release'], errors='coerce')

# Replace 'TBD' with NaN in 'user_score'
games_data['user_score'].replace('tbd', np.nan, inplace=True)

# Convert 'user_score' to float
games_data['user_score'] = games_data['user_score'].astype(float)

# Keep 'critic_score' and 'user_score' as NaN if missing
# games_data.dropna(subset=['critic_score', 'user_score'], inplace=True)

# Convert other columns to appropriate data types
games_data['na_sales'] = games_data['na_sales'].astype(float)
games_data['eu_sales'] = games_data['eu_sales'].astype(float)
games_data['jp_sales'] = games_data['jp_sales'].astype(float)
games_data['other_sales'] = games_data['other_sales'].astype(float)
games_data['critic_score'] = games_data['critic_score'].astype(float)

# Calculate total sales
games_data['total_sales'] = games_data[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)

# Check for duplicates
duplicate_rows = games_data[games_data.duplicated()]
print("\nDuplicate Rows (First 5):")
display(duplicate_rows.head())

# Check for name-year-platform duplicates
duplicate_name_year_platform = games_data[games_data.duplicated(subset=['name', 'year_of_release', 'platform'])]
print("\nName-Year-Platform Duplicates (First 5):")
display(duplicate_name_year_platform.head())

# Drop duplicates
games_data.drop_duplicates(inplace=True)
games_data.drop_duplicates(subset=['name', 'year_of_release', 'platform'], inplace=True)

# Display data after preprocessing
print("\nData After Preprocessing (First 5 rows):")
display(games_data.head())

# Verify data types
print("\nData Types:")
display(games_data.dtypes)

# Check for any remaining missing values
print("\nRemaining Missing Values:")
display(games_data.isna().sum())

# Intermediate Conclusion
print("\nIntermediate Conclusion:")
print("""
Data Preprocessing Steps Completed:
1. Renamed columns to lowercase for consistency.
2. Dropped rows with missing 'name' or 'genre' as these are essential for analysis.
3. Kept 'year_of_release' as NaN and converted to numeric where possible.
4. Replaced 'TBD' with NaN in 'user_score' and converted it to float.
5. Kept 'critic_score' and 'user_score' as NaN where missing.
6. Converted sales and score columns to appropriate data types.
7. Calculated total sales for each game.
8. Checked for and removed duplicate rows, including duplicates based on 'name', 'year_of_release', and 'platform'.

Hypotheses about the Data:
1. Average user ratings of the Xbox One and PC platforms are the same.
2. Average user ratings for the Action and Sports genres are different.

Next Steps:
1. Analyze the data to identify top platforms and genres by region.
2. Test the formulated hypotheses using statistical tests.
3. Draw conclusions based on the analysis and tests performed.
""")


# Comment on Filling 'year_of_release' with the Median
# 
# Filling the 'year_of_release' with the median year may not be appropriate because the release year of games is critical and platform-specific. Therefore, we should keep the original approach of dropping rows with missing 'year_of_release' values to maintain the integrity of our dataset.
# Comment on Filling 'critic_score' and 'user_score' with the Median
# 
# Filling 'critic_score' and 'user_score' with the median value might artificially inflate or deflate the scores, especially if a large portion of the data is missing. It's better to leave these values as NaN and handle them during analysis to avoid biasing the results.
# Comment on Not Grouping Data by Name
# 
# Grouping data by 'name' might lead to the loss of important information about platform-specific variations in game ratings and sales. We should maintain platform-specific data for more accurate analysis.
# Revised Approach
# 
# We will:
# 
#     Keep missing 'year_of_release' values as NaN and handle them later.
#     Keep missing 'critic_score' and 'user_score' values as NaN.
#     Maintain platform-specific data without grouping by 'name'.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>  Reviewer's comment 2</h2>
#    
# - Please do not delete 50% of data. 
# 
#     
# - Try not to use mode to replace the missing values :) Just ignore them. It is ok to have missing values. </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>  Reviewer's comment 2</h2>
#    
# Since we are working in Jupyter notebook, try to use Markdown cells for text.</div>

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b>  Reviewer's comment </b>
#    
# > Fill missing values in 'year_of_release' with the median year
# 
# 
# Is it a safe approach? It is possible that a platform (and a game) was released later compared to the median year.
# 
# 
# > Fill missing values in 'critic_score' and 'user_score' with median
# 
#     
# Same question. Can we replace missing values with just a median taken among all games? What is more important here is that we are trying to replace 50% of values, which means that the half of the column is going to be artificial. It may play an important role in hypotheses testing.  
# 
# 
# What do you think about it? Shouldn't we just ignore the missing values in critics' and users' ratings? I think we should also not group data by name as well because different games can have disparate ratings on several platforms. One can really have a different perception playing some game on different platforms, since platforms have their own characteristics. It's hard to figure out a decent replacement for these Nan-s.</div>

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#  
# - In your opinion, how these missing values could occur? From the task: `Why do you think the values are missing? Give possible reasons.`
#     
#     
# - Please do not forget to check data for the duplicates. 
# 
# 
# - Try to check for the name-year-platform duplicates as well.     
#     
#     
#     
# - Please, add an intermediate conclusion about this introductory part. What have been done, what hypotheses about the data we have and what we are going to do next. 
# 
# </div>
# 

# In[52]:


# Convert 'year_of_release' to numeric, keeping NaN for non-convertible values
games_data['year_of_release'] = pd.to_numeric(games_data['year_of_release'], errors='coerce')

# Replace 'TBD' with NaN in 'user_score' and convert to float
games_data['user_score'].replace('tbd', np.nan, inplace=True)
games_data['user_score'] = games_data['user_score'].astype(float)

# Convert sales columns to float
sales_columns = ['na_sales', 'eu_sales', 'jp_sales', 'other_sales']
games_data[sales_columns] = games_data[sales_columns].astype(float)

# Convert 'critic_score' to float
games_data['critic_score'] = games_data['critic_score'].astype(float)

# Calculate total sales
games_data['total_sales'] = games_data[sales_columns].sum(axis=1)

# Drop duplicates based on 'name', 'year_of_release', and 'platform'
games_data.drop_duplicates(subset=['name', 'year_of_release', 'platform'], inplace=True)

# Number of games released per year
games_per_year = games_data['year_of_release'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(games_per_year.index, games_per_year.values, marker='o')
plt.title('Number of Games Released Per Year')
plt.xlabel('Year of Release')
plt.ylabel('Number of Games')
plt.grid(True)
plt.show()

# Sales variation by platform
platform_sales = games_data.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
top_platforms = platform_sales.head(10).index

# Sales distribution over years for top platforms
plt.figure(figsize=(14, 8))
for platform in top_platforms:
    platform_data = games_data[games_data['platform'] == platform]
    platform_sales_per_year = platform_data.groupby('year_of_release')['total_sales'].sum()
    plt.plot(platform_sales_per_year.index, platform_sales_per_year.values, marker='o', label=platform)

plt.title('Sales Distribution Over Years for Top Platforms')
plt.xlabel('Year of Release')
plt.ylabel('Total Sales')
plt.legend()
plt.grid(True)
plt.show()

# Identify platforms that were once popular but now have zero sales
recent_year = games_data['year_of_release'].max()
# Fill NaN values with 0 for total_sales before checking for inactive platforms
games_data['total_sales'].fillna(0, inplace=True)
# Correctly identify inactive platforms
inactive_platforms = platform_sales[platform_sales == 0].index
print(f"Inactive Platforms as of {recent_year}: {inactive_platforms}")

# Calculate the lifecycle of platforms
platform_lifecycles = {}
for platform in games_data['platform'].unique():
    platform_data = games_data[games_data['platform'] == platform]
    first_year = platform_data['year_of_release'].min()
    last_year = platform_data['year_of_release'].max()
    lifecycle = last_year - first_year
    platform_lifecycles[platform] = lifecycle

average_lifecycle = sum(platform_lifecycles.values()) / len(platform_lifecycles)
print(f"Average Lifecycle of Platforms: {average_lifecycle:.2f} years")


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>  Reviewer's comment 2</h2>
#    
# > platform_sales[platform_sales == 0]
# 
# 
# Good idea! However, the approach is not quite correct, since they may have Nan instead of zero. That is why you get an empty list. </div>

# Platform Lifecycle:
# 
#     Analyzing platform sales over time, we found that there are no platforms with zero sales as of 2016.
#     The average lifecycle of platforms is approximately 8.94 years, indicating the typical duration a platform remains active and popular in the market.
#     
# Chart 1: Number of Games Released Per Year
# 
#     Trend Analysis:
#         The number of games released per year shows a significant increase starting around the mid-1990s.
#         There is a notable peak around 2008-2009, with over 1400 games released.
#         Post-2009, there is a decline in the number of games released per year, stabilizing around 2012 onwards.
# 
# Chart 2: Sales Distribution Over Years for Top Platforms
# 
#     Platform Popularity Over Time:
#         Platforms like PS2, PS3, X360, and Wii show clear cycles of popularity with peaks followed by declines.
#         PS2 peaked around 2002-2003 and then declined sharply after 2006.
#         PS3 and X360 peaked around 2010-2011, with declines starting around 2013.
#         Newer platforms like PS4 show an upward trend starting from 2013.
# 
#     Lifecycle of Platforms:
#         The average lifecycle of platforms, approximately 9 years, is evident as platforms like PS2, X360, and Wii have their entire lifecycle captured within the chart.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Agreed.
# </div>
# 

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#  
# 
# Before we choose relevant time interval, we have to finish these two tasks: 
#     
#     
# - Look at how many games were released in different years. Is the data for every period significant?
#     
#     
#     
# - Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?
# </div>
# 

# In[53]:


# Filter data from the last 10 years (relevant period)
current_year = 2017
start_year = current_year - 5
relevant_games_data = games_data[games_data['year_of_release'] >= start_year]

print(f"Data filtered from {start_year} to {current_year}")


# Conclusion
# 
# Market Dynamics:
#     The video game market is dynamic, with shifts in platform popularity and the number of games released each year.
#     Historical data shows that older platforms fade out as newer ones are introduced.
# 
# Relevance of Recent Data:
#     Considering the dynamic nature of the market, focusing on the most recent data is crucial for accurate forecasting.
#     Given that the market stabilized around 2012 after the peak in 2008-2009, using data from 2012 onwards provides a fresh and relevant perspective on current trends.
# 
# Selection of Relevant Time Period:
#     Based on the analysis, it is more appropriate to use data from the last 5 years rather than 10 years to capture the current market dynamics and user preferences.
#     This approach ensures that the insights are reflective of recent trends and are not skewed by outdated information.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Excellent! </div>
# 

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# - Please explain your reasoning. 
#     
#     
#     
# - Solving this task, I also took about 10 years first. But then I realized that the market is dynamic, everything is changing. Some popular genre may not remain popular 10 years later. And vice versa. During this time, several platform generations will change :) The industry is evolving, the games are getting better, the graphics is getting better and the users are getting more demanding. In such tasks, we need fresh information, fresh estimation. If we forecast for 2017, then who cares what happened 10 years ago?  Potentially profitable platforms can be easily selected with charts and pivot tables, but user portraits may change. If you look at the sales chart, you will see in which year the market stabilized after huge volume sales. We can also take the half of a lifetime period.
# 
# </div>

# # Analysis of Relevant Data

# In[54]:


# 1. Number of Games Released per Year
games_per_year = relevant_games_data['year_of_release'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
games_per_year.plot(kind='bar')
plt.title('Number of Games Released per Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.show()


# The chart displays the number of games released each year from 2012 to 2016. The bars are colored in blue and are of varying heights, indicating the number of games released in each year. The years are labeled on the horizontal axis, and the vertical axis represents the number of games released, ranging from 0 to 700. The chart provides a visual representation of the trend in the number of games released over the given period.

# In[55]:


# 2. Sales Variation by Platform
platform_sales = relevant_games_data.groupby('platform')['total_sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
platform_sales.plot(kind='bar')
plt.title('Total Sales of Leading Platforms (Last 10 Years)')
plt.xlabel('Platform')
plt.ylabel('Total Sales (in millions)')
plt.show()


# It displays the sales figures in millions for various gaming platforms. The platforms listed are PS4, PS3, X360, 3DS, Xome, WiiU, PSV, and PSP. The sales figures for each platform are as follows: PS4 has the highest sales with 300 million, followed by PS3 with 250 million. X360 has 200 million, 3DS has 150 million, Xome has 100 million, WiiU has 50 million, PSV has 250 million, and PSP has 50 million.

# 

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment </b>
#     
# Be careful. Here you took platforms that have most total sales. For all years. Of course, PS2 is an old platfrom that had enough time to sell a lot of games. But the next generations, XOne and PS4 didn't have enough time for it. In order to choose promising platforms, please look at how their sales change over time. 
# 
# </div>

# In[56]:


# Group by platform and calculate yearly sales totals
platform_sales_by_year = games_data.groupby(['platform', 'year_of_release'])['total_sales'].sum().unstack(fill_value=0)

# Verify the structure of the DataFrame
print("Platform Sales by Year DataFrame:")
print(platform_sales_by_year.head())

# Plot sales trends over time with years on X-axis
plt.figure(figsize=(16, 10))
for platform in platform_sales_by_year.index:
    plt.plot(platform_sales_by_year.columns, platform_sales_by_year.loc[platform], marker='o', label=platform)

plt.title('Sales Trends Over Time for Different Platforms')
plt.xlabel('Year of Release')
plt.ylabel('Total Sales (in millions)')
plt.legend(title='Platforms', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate X-axis labels for better readability
plt.tight_layout()
plt.show()


# Breakdown of the Code:
# 
#     Loading and Preparing Data:
#         Convert column names to lowercase for consistency.
#         Convert 'year_of_release' to numeric, handle 'TBD' values, and ensure sales and score columns are numeric.
# 
#     Data Reshaping:
#         Group the data by platform and year_of_release, then unstack it to have years as columns and platforms as rows.
# 
#     Plotting:
#         Loop through each platform to plot its sales over time.
#         Set years as the X-axis and platforms as the Y-axis.
#         Adjust the legend to fit outside the plot area for clarity.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# I think we should change the chart a little bit :) Try to display years on X axis. Then lines will indicate platforms. 
# 
# </div>
# 

# In[57]:


import seaborn as sns

# Prepare data for the box plot
# We will use 'platform' as the category and 'total_sales' as the values
plt.figure(figsize=(14, 8))
sns.boxplot(x='platform', y='total_sales', data=games_data, palette='viridis')

# Set plot title and labels
plt.title('Global Sales by Platform')
plt.xlabel('Platform')
plt.ylabel('Total Sales in Millions')

# Set y-axis limit
plt.ylim(0, games_data['total_sales'].max() / 1e6 + 10)  # Set the ylim to a reasonable range

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# The box plot titled "Global Sales by Platform" displays the distribution of total sales (in millions) for different gaming platforms. Each vertical box represents the interquartile range (IQR) of sales for a particular platform, with the horizontal line inside the box indicating the median sales value. The whiskers extending from the boxes represent the minimum and maximum values, excluding outliers (which are shown as individual dots).
# 
# The plot reveals a wide range of sales performance across platforms. Some platforms, like the Wii (WI), Nintendo DS (NDS), and Game Boy (GB), exhibit higher median sales and a more compact distribution, indicating consistent commercial success. Other platforms, such as the PlayStation 3 (PS3), Xbox 360 (X360), and PlayStation 2 (PS2), have larger IQRs, suggesting a wider variability in sales performance among games released on those platforms.
# 
# Outliers, represented by individual dots, are present for many platforms, indicating the presence of exceptionally high-selling games that deviate significantly from the typical sales range for that platform.
# 
# Overall, the box plot provides a concise visual summary of the sales distribution for each platform, allowing for easy comparison and identification of platforms with consistently high sales, as well as those with more variability or the presence of outlier games.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# 
# 
# - It is difficult to compare the boxes. Please, try to use the **ylim**, it will scale the graph. 
#     
#     
# - There will be some outliers there, don't drop them. What do you think outliers can tell us? Just write any suggestions you have.</div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# This comment is still relevant. 
# 
# </div>
# 

# In[58]:


# 6. Effect of User and Professional Reviews on Sales
ps4_data = relevant_games_data[relevant_games_data['platform'] == 'PS4']
plt.figure(figsize=(12, 6))
plt.scatter(ps4_data['critic_score'], ps4_data['total_sales'])
plt.title('Critic Score vs. Total Sales (PS4)')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales (in millions)')
plt.show()


# The graph displays a scatter plot that illustrates the relationship between critic scores and total sales for games on the PlayStation 4 (PS4) platform. The x-axis represents the critic score, ranging from approximately 20 to 100, while the y-axis represents the total sales in millions of units.
# 
# The data points are scattered across the plot, showing no clear linear relationship between critic scores and total sales. There are games with high critic scores that have achieved both high and low sales figures, as well as games with low critic scores that have varying sales numbers.
# 
# The distribution of points suggests that while critic scores may influence sales to some extent, there are likely other factors that play a significant role in determining a game's commercial success on the PS4 platform. Some games with lower critic scores have managed to achieve high sales figures, indicating that critical reception alone is not the sole determinant of a game's commercial performance.
# 
# Overall, the scatter plot highlights the complex nature of the relationship between critic reviews and sales, implying that factors beyond just critical acclaim, such as marketing, genre popularity, and player preferences, can significantly impact a game's commercial success on the PS4 platform.

# In[59]:


plt.figure(figsize=(12, 6))
plt.scatter(ps4_data['user_score'], ps4_data['total_sales'])
plt.title('User Score vs. Total Sales (PS4)')
plt.xlabel('User Score')
plt.ylabel('Total Sales (in millions)')
plt.show()

critic_corr = ps4_data['critic_score'].corr(ps4_data['total_sales'])
user_corr = ps4_data['user_score'].corr(ps4_data['total_sales'])
print(f'Correlation between Critic Score and Total Sales (PS4): {critic_corr}')
print(f'Correlation between User Score and Total Sales (PS4): {user_corr}')


# The scatter plot of User Score vs. Total Sales for PS4 games shows no clear linear relationship, with data points scattered without a discernible pattern. The correlation value of -0.063 indicates a negligible weak negative correlation, suggesting that user scores are not reliable predictors of total sales. Other factors such as marketing, genre popularity, pricing, or franchise recognition likely have a more substantial impact on sales. While user scores may not directly correlate with sales, they can still influence word-of-mouth and community feedback, which may affect sales indirectly. Thus, user scores alone are not a strong indicator of a game’s commercial success on PS4.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# 
# Yes, there's a very-very weak correlation between sales and critics' scores.
# 
# </div>
# 
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
#     
# According to the task, we have to choose 1 platform, analyze how sales depend on critics' and users' scores, display scatter plots and calculate the correlation coefficients. Then we have to choose  2 or more other platforms, repeat the correlation analysis for them and compare the results. Would you add it?  
# 
# 
# The wording in this task is ambiguous, but since we are comparing platforms, it is reasonable to take scores for all games on the platform within a chosen period. In other words, you do not need to check each game in the dataset. 
# 
# All you need here is to take 2 or more other platforms and repeat the analysis. Then compare the results. </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# I see the updates below, well done!  </div>
# 

# In[60]:


import pandas as pd
import matplotlib.pyplot as plt

# Ensure user_score and critic_score are numeric and within valid ranges
games_data['user_score'] = pd.to_numeric(games_data['user_score'], errors='coerce')
games_data['critic_score'] = pd.to_numeric(games_data['critic_score'], errors='coerce')

# Clean data by removing rows with non-numeric scores or total sales
games_data_clean = games_data[(games_data['user_score'].notna()) &
                              (games_data['critic_score'].notna()) &
                              (games_data['total_sales'].notna())]

# Check if there are still any issues with the cleaned data
print("Data types after conversion:")
print(games_data_clean[['user_score', 'critic_score', 'total_sales']].dtypes)
print("\nSummary statistics for cleaned data:")
print(games_data_clean[['user_score', 'critic_score', 'total_sales']].describe())

# Define platforms for analysis
primary_platform = 'PS4'
comparison_platforms = ['PC', 'X360']

# Function to analyze and plot data for a given platform
def analyze_platform(data, platform):
    platform_data = data[data['platform'] == platform]
    
    # Scatter plot for User Score vs. Total Sales
    plt.figure(figsize=(10, 6))
    plt.scatter(platform_data['user_score'], platform_data['total_sales'], alpha=0.5)
    plt.title(f'User Score vs. Total Sales for {platform}')
    plt.xlabel('User Score')
    plt.ylabel('Total Sales (in millions)')
    plt.grid(True)
    plt.show()
    
    # Scatter plot for Critic Score vs. Total Sales
    plt.figure(figsize=(10, 6))
    plt.scatter(platform_data['critic_score'], platform_data['total_sales'], alpha=0.5)
    plt.title(f'Critic Score vs. Total Sales for {platform}')
    plt.xlabel('Critic Score')
    plt.ylabel('Total Sales (in millions)')
    plt.grid(True)
    plt.show()
    
    # Calculate correlation coefficients
    user_score_corr = platform_data['user_score'].corr(platform_data['total_sales'])
    critic_score_corr = platform_data['critic_score'].corr(platform_data['total_sales'])
    
    return user_score_corr, critic_score_corr

# Analyze the primary platform
primary_user_corr, primary_critic_corr = analyze_platform(games_data_clean, primary_platform)

# Analyze comparison platforms
comparison_results = {}
for platform in comparison_platforms:
    comparison_results[platform] = analyze_platform(games_data_clean, platform)

# Display correlation results
print(f"Correlation for {primary_platform}:")
print(f"User Score vs. Total Sales: {primary_user_corr:.3f}")
print(f"Critic Score vs. Total Sales: {primary_critic_corr:.3f}\n")

for platform, (user_corr, critic_corr) in comparison_results.items():
    print(f"Correlation for {platform}:")
    print(f"User Score vs. Total Sales: {user_corr:.3f}")
    print(f"Critic Score vs. Total Sales: {critic_corr:.3f}\n")

# Summary of findings
print("Summary:")
print("""
- Compare the correlation coefficients for user and critic scores across different platforms.
- Discuss which platform shows stronger or weaker correlations and possible reasons for the differences.
""")


# The provided information summarizes the analysis of correlations between user scores, critic scores, and total sales for games across different platforms: PlayStation 4 (PS4), PC, and Xbox 360 (X360).
# The key points are:
# 
# Critic Score Correlation:
# 
# PS4: Critic score has a moderately positive correlation (0.406) with total sales.
# PC: Critic score has a weaker but still positive correlation (0.256) with total sales.
# X360: Critic score has a moderately positive correlation (0.390) with total sales, similar to PS4.
# 
# 
# User Score Correlation:
# 
# PS4: User score has a very weak negative correlation (-0.034) with total sales.
# PC: User score has an almost negligible correlation (0.001) with total sales.
# X360: User score has a weak positive correlation (0.110) with total sales.
# 
# 
# 
# The analysis suggests that critic scores tend to have a stronger positive correlation with total sales compared to user scores across all three platforms. This implies that games with higher critic scores generally tend to achieve higher sales figures.
# However, the strength of the correlation varies between platforms. The PS4 and X360 exhibit stronger correlations between critic scores and sales, while the correlation is weaker for PC games. This could be due to differences in the gaming communities, marketing strategies, or the influence of critical reviews on purchasing decisions for each platform.
# On the other hand, user scores show weak or negligible correlations with total sales, suggesting that user ratings may not be as strongly associated with commercial success as critic scores.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Very good. 
# 
# </div>
# 

# In[61]:


# 7. Compare Sales of the Same Games on Other Platforms
top_ps4_games = ps4_data.nlargest(10, 'total_sales')
top_games_all_platforms = relevant_games_data[relevant_games_data['name'].isin(top_ps4_games['name'])]
sales_comparison = top_games_all_platforms.pivot_table(index='name', columns='platform', values='total_sales', aggfunc='sum', fill_value=0)
print(sales_comparison)


# In[62]:


# 8. Distribution of Games by Genre
genre_distribution = relevant_games_data['genre'].value_counts()
plt.figure(figsize=(12, 6))
genre_distribution.plot(kind='bar')
plt.title('Distribution of Games by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Games')
plt.show()


# The bar graph illustrates the distribution of game genres.
# 
# Key Observations:
# Most Common Genre: Action games are the most prevalent, with over 2,000 titles.
# Next Popular Genres: Miscellaneous, Sports, and Adventure genres each have between 1,000 and 1,500 games.
# Less Common Genres: The least common genres include Strategy, Puzzle, and Platform, each with fewer than 500 titles.

# In[63]:


genre_sales = relevant_games_data.groupby('genre')['total_sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
genre_sales.plot(kind='bar')
plt.title('Total Sales by Genre')
plt.xlabel('Genre')
plt.ylabel('Total Sales (in millions)')
plt.show()

print(genre_sales)


# Key Observations:
# Highest Sales: Action games significantly lead in total sales, surpassing 1,000 million units.
# Next Top Sellers: Shooter and Sports genres also have high total sales, each exceeding 600 million units.
# Moderate Sales: Miscellaneous, Role-Playing, and Racing genres have moderate sales, ranging from approximately 400 to 500 million units.
# Lower Sales: Genres like Puzzle and Strategy have the lowest total sales, under 100 million units.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
#     
# Is it possible that Action has made such volumes only due to the number of games? Maybe other genres have more sales per game unit. Please, do the same, but with median or average sales.
# 
# </div>

# In[64]:


# Calculate the average sales per game for each genre
genre_sales_per_game = relevant_games_data.groupby('genre')['total_sales'].mean().sort_values(ascending=False)

# Alternatively, you can use median sales per game
# genre_sales_per_game = relevant_games_data.groupby('genre')['total_sales'].median().sort_values(ascending=False)

# Plot the results
plt.figure(figsize=(12, 6))
genre_sales_per_game.plot(kind='bar')
plt.title('Average Sales per Game by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Sales per Game (in millions)')
plt.xticks(rotation=45)
plt.show()

print(genre_sales_per_game)


# The results show the average sales per game for each genre. Here are some observations based on the output:
# 
# Genres with Higher Average Sales per Game:
#     Shooter : $1.297 million per game 
#     Platform: $0.718 million per game
#     Sports  : $0.678 million per game
# These genres have higher average sales per game, suggesting that individual games in these genres tend to perform better on average in terms of sales.
# 
# Genres with Lower Average Sales per Game:
#     Adventure: $0.097 million per game
#     Strategy : $0.188 million per game
#     Puzzle   : $0.175 million per game
# These genres have much lower average sales per game, indicating that games in these genres typically sell less on average.
# 
# Comparison with Total Sales:
#     While the Action genre had high total sales, its average sales per game ($0.428 million) are not the highest among all genres. This indicates that the high total sales for Action may be due to a large number of games rather than exceptionally high sales per game.
# 
# Further Analysis:
#     To understand the genre dynamics better, you might consider comparing the number of games per genre to see if the number of titles significantly influences the average sales per game.
#     You can also explore median sales per game, as the median can provide insights into the central tendency, especially if the sales data is skewed.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Correct! 
# </div>
# 

# # Step 4: Create a User Profile for Each Region

# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# 
# - The distributions may change after we use the correct time interval, so don't forget to update the conclusions :) 
# 
# 
# 
# - Please try to use Markdown cells for headings. 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Looks good! 
# </div>
# 

# In[65]:


#1. Top Five Platforms by Region

def top_platforms_by_region(region_sales_column):
    top_platforms = relevant_games_data.groupby('platform')[region_sales_column].sum().sort_values(ascending=False).head(5)
    return top_platforms

# Get top platforms for each region
top_platforms_na = top_platforms_by_region('na_sales')
top_platforms_eu = top_platforms_by_region('eu_sales')
top_platforms_jp = top_platforms_by_region('jp_sales')

# Display the top platforms
print("Top Platforms in North America:")
print(top_platforms_na)

print("\nTop Platforms in Europe:")
print(top_platforms_eu)

print("\nTop Platforms in Japan:")
print(top_platforms_jp)

# Plot the top platforms for each region
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
top_platforms_na.plot(kind='bar', color='skyblue')
plt.title('Top Platforms in North America')
plt.xlabel('Platform')
plt.ylabel('Total Sales (in millions)')

plt.subplot(1, 3, 2)
top_platforms_eu.plot(kind='bar', color='lightgreen')
plt.title('Top Platforms in Europe')
plt.xlabel('Platform')
plt.ylabel('Total Sales (in millions)')

plt.subplot(1, 3, 3)
top_platforms_jp.plot(kind='bar', color='lightcoral')
plt.title('Top Platforms in Japan')
plt.xlabel('Platform')
plt.ylabel('Total Sales (in millions)')

plt.tight_layout()
plt.show()


# Analysis and Observations:
# 
# In North America, the Xbox 360 (X360) emerges as the top-selling platform, followed by the PlayStation 4 (PS4), PlayStation 3 (PS3), Xbox One (XOne), and Nintendo 3DS (3DS). This indicates a preference for Microsoft's and Sony's consoles in the North American market.
# 
# In Europe, the PlayStation 4 (PS4) takes the lead as the top-selling platform, followed by the PlayStation 3 (PS3), Xbox 360 (X360), Xbox One (XOne), and Nintendo 3DS (3DS). Similar to North America, Sony and Microsoft platforms dominate the European market.
# 
# However, in Japan, the Nintendo 3DS (3DS) stands out as the top-selling platform, followed by the PlayStation 3 (PS3), PlayStation Vita (PSV), PlayStation 4 (PS4), and Wii U (WiiU). This highlights the strong preference for handheld and Nintendo consoles in the Japanese market.
# Based on these observations, we can draw the following conclusions:
# 
# 
# In North America and Europe, Sony and Microsoft's home consoles, such as the PlayStation and Xbox series, have been the most successful platforms, catering to the preferences of gamers in these regions.
# 
# In Japan, Nintendo's handheld consoles, particularly the 3DS, have been the top performers, reflecting the Japanese market's affinity for portable and family-friendly gaming experiences.
# The PlayStation 3 (PS3) has had significant success across all three regions, indicating its widespread popularity and appeal to a diverse gaming audience.
# 
# The more recent Xbox One (XOne) and PlayStation 4 (PS4) consoles have gained substantial traction in North America and Europe but have yet to achieve the same level of success in Japan, where Nintendo and handheld devices dominate.
# The Wii U console from Nintendo has struggled to gain significant market share in North America and Europe but has found a more receptive audience in its home market of Japan.
# 
# 
# These regional preferences and market dynamics can provide valuable insights for game developers, publishers, and console manufacturers when strategizing product releases, marketing campaigns, and resource allocation for different markets.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Exactly. 
# </div>
# 

# In[66]:


#2. Top Five Genres by Region

# Define a function to get the top five genres by region
def top_genres_by_region(region_sales_column):
    top_genres = relevant_games_data.groupby('genre')[region_sales_column].sum().sort_values(ascending=False).head(5)
    return top_genres

# Get top genres for each region
top_genres_na = top_genres_by_region('na_sales')
top_genres_eu = top_genres_by_region('eu_sales')
top_genres_jp = top_genres_by_region('jp_sales')

# Display the top genres
print("Top Genres in North America:")
print(top_genres_na)

print("\nTop Genres in Europe:")
print(top_genres_eu)

print("\nTop Genres in Japan:")
print(top_genres_jp)

# Plot the top genres for each region
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
top_genres_na.plot(kind='bar', color='skyblue')
plt.title('Top Genres in North America')
plt.xlabel('Genre')
plt.ylabel('Total Sales (in millions)')

plt.subplot(1, 3, 2)
top_genres_eu.plot(kind='bar', color='lightgreen')
plt.title('Top Genres in Europe')
plt.xlabel('Genre')
plt.ylabel('Total Sales (in millions)')

plt.subplot(1, 3, 3)
top_genres_jp.plot(kind='bar', color='lightcoral')
plt.title('Top Genres in Japan')
plt.xlabel('Genre')
plt.ylabel('Total Sales (in millions)')

plt.tight_layout()
plt.show()


# Summary of Top Genres by Region
# 
# North America:
# 
# Top Genres: Action and Shooter are the most popular, followed by Role-Playing and Sports.
# Insights: A strong preference for fast-paced and action-oriented games, with significant interest in sports simulations.
# 
# Europe:
# 
# Top Genres: Action leads, with Shooter and Sports following. Role-Playing also performs well.
# Insights: Similar to North America, with a strong preference for action games, though Racing games have less impact.
# 
# Japan:
# 
# Top Genres: Role-Playing dominates, followed by Action and Miscellaneous genres like Simulation and Fighting.
# Insights: A clear preference for games with deep storylines and character development, with niche popularity for simulation and fighting games.
# 
# Cross-Regional Insights:
# 
# Consistent Popularity: Action and Shooter genres are popular in North America and Europe.
# Cultural Variations: Japan’s preference for Role-Playing and Simulation highlights cultural differences, emphasizing narrative depth and character interaction over action and competition.
# 
# These insights suggest that game developers should tailor their offerings to regional preferences to enhance appeal and sales potential. Further analysis of specific game titles within these genres could refine understanding of consumer behaviors.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Correct. 
# </div>
# 

# In[67]:


# 3. Effect of ESRB Ratings on Sales in Individual Regions

# Define a function to calculate sales by ESRB rating for each region
def sales_by_esrb_rating(region_sales_column):
    esrb_sales = relevant_games_data.groupby('rating')[region_sales_column].sum().sort_values(ascending=False)
    return esrb_sales

# Get ESRB sales for each region
esrb_sales_na = sales_by_esrb_rating('na_sales')
esrb_sales_eu = sales_by_esrb_rating('eu_sales')
esrb_sales_jp = sales_by_esrb_rating('jp_sales')

# Display the ESRB sales
print("ESRB Sales in North America:")
print(esrb_sales_na)

print("\nESRB Sales in Europe:")
print(esrb_sales_eu)

print("\nESRB Sales in Japan:")
print(esrb_sales_jp)

# Plot the ESRB sales for each region
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
esrb_sales_na.plot(kind='bar', color='skyblue')
plt.title('ESRB Ratings Sales in North America')
plt.xlabel('ESRB Rating')
plt.ylabel('Total Sales (in millions)')

plt.subplot(1, 3, 2)
esrb_sales_eu.plot(kind='bar', color='lightgreen')
plt.title('ESRB Ratings Sales in Europe')
plt.xlabel('ESRB Rating')
plt.ylabel('Total Sales (in millions)')

plt.subplot(1, 3, 3)
esrb_sales_jp.plot(kind='bar', color='lightcoral')
plt.title('ESRB Ratings Sales in Japan')
plt.xlabel('ESRB Rating')
plt.ylabel('Total Sales (in millions)')

plt.tight_layout()
plt.show()


# In North America, the highest sales are observed for games rated "M" (Mature), followed by "E" (Everyone), "E10+" (Everyone 10+), and "T" (Teen). This suggests that mature-rated games have the highest commercial success in the North American market.
# 
# In Europe, a similar trend is observed, with "M" (Mature) rated games having the highest sales, followed by "E" (Everyone), "E10+" (Everyone 10+), and "T" (Teen). This aligns with the North American market preferences.
# 
# However, in Japan, the trend is different. Games rated "E" (Everyone) have the highest sales, followed by "T" (Teen), "M" (Mature), and "E10+" (Everyone 10+). This indicates that in the Japanese market, games catered to a broader audience, including family-friendly titles, tend to perform better than mature-rated games.
# 
# 
# Based on these observations, we can conclude that:
# 
# Mature-rated games ("M") have the highest commercial success in North America and Europe, suggesting a preference for more mature content and themes in these regions.
# 
# In Japan, games rated "E" (Everyone) perform the best, indicating a stronger demand for family-friendly and accessible gaming experiences in the Japanese market.
# 
# The "T" (Teen) rating generally has lower sales compared to "M" (Mature) and "E" (Everyone) ratings across all three regions, suggesting that games targeted at the teenage audience may have a smaller market share.
# The "E10+" (Everyone 10+) rating falls somewhere in between, with moderate sales figures in all regions, catering to a slightly older audience than the "E" (Everyone) rating.
# 
# These findings can provide valuable insights for game developers and publishers when planning their content strategy and targeting specific regional markets. 
# Cultural preferences, demographics, and audience interests appear to play a significant role in determining the commercial success of games based on their ESRB ratings.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment </h2>
# 
#     
# Nice job here.  
# 
# </div>   
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment </b>
#     
#     
# Don't forget to update the conclusions. 
# </div>

# Conclusion:
# 
# Based on the data, we can draw the following conclusions about the video game market across different regions:
# 
# Platform Preferences:
# 
# In North America and Europe, consoles from Sony (PlayStation) and Microsoft (Xbox) dominate the market, with the Xbox 360 being the top-selling platform in North America and the PlayStation 4 leading in Europe.
# In Japan, handheld consoles from Nintendo, particularly the 3DS, are the most popular platforms, reflecting the Japanese market's preference for portable and family-friendly gaming experiences.
# 
# 
# Genre Preferences:
# 
# Action and Shooter games are highly popular in both North America and Europe, indicating a strong demand for action-oriented and competitive gaming experiences in these regions.
# Role-Playing games (RPGs) are particularly successful in Japan, aligning with the country's rich history and cultural appreciation for narrative-driven and immersive gaming experiences.
# Sports games also have a significant presence in North America and Europe, catering to the regions' interest in sports-related entertainment.
# 
# 
# ESRB Rating Preferences:
# 
# Mature-rated games (M) have the highest sales in North America and Europe, suggesting a preference for more mature content and themes among gamers in these regions.
# In Japan, games rated "E" (Everyone) perform the best, reflecting a stronger demand for family-friendly and accessible gaming experiences suitable for a wider audience.
# 
# 
# Regional Differences:
# 
# While there are some common preferences across regions, such as the popularity of Action and Shooter games in North America and Europe, there are also distinct regional differences in terms of platform, genre, and content preferences.
# Japan stands out as a unique market, with a stronger affinity for handheld consoles, RPGs, and family-friendly content, aligning with the country's cultural and gaming traditions.
# 
# 
# 
# These insights highlight the importance of understanding regional preferences and tailoring game development, marketing, and distribution strategies accordingly. Game developers and publishers should consider these regional nuances to maximize their chances of success in different markets. Localization, culturally relevant content, and understanding the unique gaming landscape of each region are crucial for achieving commercial success in the global video game industry.

# Step 5. Test the following hypotheses:
# 
# To analyze the hypotheses regarding user ratings for the Xbox One and PC platforms and for Action and Sports genres, we will follow these steps:
# 
# Filter Data: Extract the relevant user ratings for the Xbox One, PC, Action, and Sports genres. Perform Statistical Tests: Conduct a two-sample t-test for each hypothesis.  
# Interpret Results: Determine whether to reject or fail to reject the null hypothesis based on the p-value.

# In[68]:


#Step 1: Filter Data
# Filter data for Xbox One and PC platforms
xbox_one_data = games_data[games_data['platform'] == 'XOne']
pc_data = games_data[games_data['platform'] == 'PC']

# Filter data for Action and Sports genres
action_data = games_data[games_data['genre'] == 'Action']
sports_data = games_data[games_data['genre'] == 'Sports']


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment </b>
#     
#     
# - Please, add a wording for both hypotheses.  H0 and H1 for the 1st hypotheses pair and H0 and H1 for the second one.  Be sure to clarify, why did you decide to use a specific statistical test and you did you choose a specific significance level. 
# 
# 
# 
# 
# - Don't forget about the relevant time interval. </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# I see them below, very good.
# </div>
# 
# 

# In[69]:


# Import necessary libraries
from scipy.stats import ttest_ind

# Define the null and alternative hypotheses
# For Hypothesis 1: Comparing Xbox One and PC user ratings
# Null Hypothesis (H0): The means of user ratings for Xbox One and PC are equal.
# Alternative Hypothesis (H1): The means of user ratings for Xbox One and PC are not equal.

# Justification for the t-test:
# We use the independent two-sample t-test because we are comparing the means of two independent groups (Xbox One and PC user ratings).
# We chose a significance level of 0.05, which is commonly used in hypothesis testing to determine statistical significance.

# Filter data for Xbox One and PC platforms within the 5-year interval (2012-2016)
xbox_one_data = games_data[(games_data['platform'] == 'XOne') & (games_data['year_of_release'] >= 2012) & (games_data['year_of_release'] <= 2016)]
pc_data = games_data[(games_data['platform'] == 'PC') & (games_data['year_of_release'] >= 2012) & (games_data['year_of_release'] <= 2016)]

# Extract user ratings
xbox_one_ratings = xbox_one_data['user_score'].dropna()
pc_ratings = pc_data['user_score'].dropna()

# Perform t-test
t_stat_xbox_pc, p_value_xbox_pc = ttest_ind(xbox_one_ratings, pc_ratings)

# Print results
print(f"Null Hypothesis (H0): The means of user ratings for Xbox One and PC are equal.")
print(f"Alternative Hypothesis (H1): The means of user ratings for Xbox One and PC are not equal.")
print(f"Significance Level: 0.05")
print(f"T-statistic for Xbox One vs. PC: {t_stat_xbox_pc}")
print(f"P-value for Xbox One vs. PC: {p_value_xbox_pc}")

# Interpretation of the results:
# If the p-value is less than 0.05, we reject the null hypothesis, indicating a significant difference between the user ratings of Xbox One and PC.
# If the p-value is greater than or equal to 0.05, we fail to reject the null hypothesis, suggesting no significant difference between the user ratings of Xbox One and PC.


# This information describes the results of a hypothesis test comparing the user ratings of Xbox One and PC. Here’s a breakdown of each component and its implications:
# 
#     Null Hypothesis (H0):
#         Statement: The means of user ratings for Xbox One and PC are equal.
#         Interpretation: There is no significant difference between the average user ratings of Xbox One and PC. Any observed difference is due to random chance.
# 
#     Alternative Hypothesis (H1):
#         Statement: The means of user ratings for Xbox One and PC are not equal.
#         Interpretation: There is a significant difference between the average user ratings of Xbox One and PC. This suggests that the user ratings for the two platforms are different and not due to random variation.
# 
#     Significance Level (α):
#         Value: 0.05
#         Interpretation: This is the threshold for determining statistical significance. If the p-value is less than or equal to 0.05, we reject the null hypothesis in favor of the alternative hypothesis.
# 
#     T-statistic:
#         Value: 0.5930304646661255
#         Interpretation: The t-statistic is a measure of the difference between the sample means, scaled by the variability of the data. A t-statistic of 0.593 indicates that the difference in means is relatively small when compared to the variability in the data.
# 
#     P-value:
#         Value: 0.5535080500643661
#         Interpretation: The p-value represents the probability of obtaining a test statistic at least as extreme as the observed value under the null hypothesis. A p-value of 0.5535 is greater than the significance level of 0.05, indicating that the observed difference in means is not statistically significant.
# 
# Conclusion:
# 
#     Since the p-value (0.5535) is greater than the significance level (0.05), we fail to reject the null hypothesis.
#     Interpretation: There is not enough evidence to conclude that the user ratings for Xbox One and PC are different. The means of user ratings for these two platforms are likely equal, and any observed difference is probably due to random variation rather than a true difference in user ratings.

# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
#     
# 
# The fact that we reject the null hypothesis does not entail that the 2nd hypothesis is true. We just reject it at a particular significance level. I would change the wording in the conclusion. 
#     
# </div>

# In[70]:


from scipy.stats import ttest_ind

# Define the null and alternative hypotheses
# Hypothesis 2: Action vs. Sports Genre User Ratings
# Null Hypothesis (H0): The means of user ratings for Action and Sports genres are equal.
# Alternative Hypothesis (H1): The means of user ratings for Action and Sports genres are not equal.
# Significance Level: 0.05 (commonly used threshold for statistical significance)

# Filter data for Action and Sports genres within the 5-year interval (2012-2016)
action_data = games_data[(games_data['genre'] == 'Action') & (games_data['year_of_release'] >= 2012) & (games_data['year_of_release'] <= 2016)]
sports_data = games_data[(games_data['genre'] == 'Sports') & (games_data['year_of_release'] >= 2012) & (games_data['year_of_release'] <= 2016)]

# Extract user ratings
action_ratings = action_data['user_score'].dropna()
sports_ratings = sports_data['user_score'].dropna()

# Perform t-test
t_stat_action_sports, p_value_action_sports = ttest_ind(action_ratings, sports_ratings)

print(f"Null Hypothesis (H0): The means of user ratings for Xbox One and PC are equal.")
print(f"Alternative Hypothesis (H1): The means of user ratings for Xbox One and PC are not equal.")
print(f"Significance Level: 0.05")
print(f"T-statistic for Action vs. Sports: {t_stat_action_sports:.3f}")
print(f"P-value for Action vs. Sports: {p_value_action_sports:.3f}")

# Interpretation based on the p-value and significance level
if p_value_action_sports < 0.05:
    print("We reject the null hypothesis. There is a statistically significant difference in user ratings between Action and Sports genres.")
else:
    print("We fail to reject the null hypothesis. There is no statistically significant difference in user ratings between Action and Sports genres.")


# This outlines the results of a hypothesis test comparing the user ratings for the Action and Sports genres. Here’s a detailed description of each component and its implications:
# 
#     Null Hypothesis (H0):
#         Statement: The means of user ratings for the Action and Sports genres are equal.
#         Interpretation: There is no significant difference between the average user ratings for Action and Sports genres. Any observed difference in ratings is attributed to random chance.
# 
#     Alternative Hypothesis (H1):
#         Statement: The means of user ratings for the Action and Sports genres are not equal.
#         Interpretation: There is a significant difference between the average user ratings for Action and Sports genres. This suggests that the user ratings for these two genres differ in a meaningful way.
# 
#     Significance Level (α):
#         Value: 0.05
#         Interpretation: This is the threshold for determining statistical significance. If the p-value is less than or equal to 0.05, we reject the null hypothesis in favor of the alternative hypothesis.
# 
#     T-statistic:
#         Value: 10.984
#         Interpretation: The t-statistic measures the size of the difference between the sample means relative to the variability in the data. A t-statistic of 10.984 is quite large, indicating a substantial difference between the means of the two genres.
# 
#     P-value:
#         Value: 0.000
#         Interpretation: The p-value indicates the probability of observing a test statistic as extreme as, or more extreme than, the observed value under the null hypothesis. A p-value of 0.000 is extremely small and is much less than the significance level of 0.05.
# 
# Conclusion:
# 
#     Since the p-value (0.000) is significantly less than the significance level (0.05), we reject the null hypothesis.
#     Interpretation: There is strong evidence to suggest that the user ratings for Action and Sports genres are different. The means of user ratings for these genres are statistically significantly different, indicating that the rating preferences for these genres are not the same.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Correct. 
# </div>
# 

# Conclusion:
# 
# Xbox One vs. PC:
#     The user ratings for Xbox One and PC are statistically similar based on the given data, indicating that the platform does not significantly impact user ratings during the 2012-2016 period.
# 
# Action vs. Sports Genres:
#     The user ratings for Action and Sports genres are significantly different. This suggests that users may have distinct preferences, with Action games generally receiving higher ratings compared to Sports games during the specified period.
# 
# These findings highlight the importance of considering genre and platform differences when analyzing user preferences and ratings in the gaming industry.

# In[71]:


# Step 6: General Conclusion


# General Conclusion
# 
# The analysis of gaming data from 2012 to 2016 across different regions has revealed distinct patterns in user preferences, platform popularity, genre popularity, and ESRB ratings. Here’s a summary of the key findings and additional insights:
# 
# Top Platforms by Region:
# 
#     North America: The top platforms include Xbox 360, PS4, PS3, Xbox One, and 3DS. Xbox 360 and PS4 lead in sales, indicating strong popularity in this region.
#     Europe: PS4, PS3, Xbox 360, Xbox One, and 3DS are the top platforms. PS4 stands out with the highest sales, showing strong market presence in Europe.
#     Japan: The 3DS, PS3, PSV, PS4, and WiiU are the leading platforms. The 3DS is particularly dominant, reflecting a strong handheld gaming market in Japan.
# 
# Top Genres by Region:
# 
#     North America: Action, Shooter, Sports, and Role-Playing are the top genres. The dominance of Action and Shooter genres indicates a preference for fast-paced and competitive games.
#     Europe: Action, Shooter, Sports, and Role-Playing also lead. The popularity of Action and Shooter genres aligns with the North American trends, but Sports is slightly more prominent.
#     Japan: Role-Playing, Action, and Miscellaneous genres are top performers. The high ranking of Role-Playing games highlights a cultural preference for narrative-driven and immersive experiences.
# 
# Top ESRB Ratings by Region:
# 
#     North America: Mature (M) rated games dominate, followed by Everyone (E) and Everyone 10+ (E10+). This suggests a significant market for mature content.
#     Europe: Mature (M) and Everyone (E) ratings are top, with a strong presence of E10+ and Teen (T) ratings. The market shows a diverse age range preference.
#     Japan: The market favors Everyone (E) and Teen (T) ratings, with fewer mature games. This aligns with a preference for family-friendly and non-violent content.
# 
# Hypothesis Testing Results:
# 
#     Xbox One vs. PC User Ratings:
#         The test result (p-value = 0.5535) indicated no significant difference in user ratings between Xbox One and PC. This means the user ratings are likely similar for both platforms.
#     Action vs. Sports User Ratings:
#         The test result (p-value = 0.000) showed a significant difference in user ratings between Action and Sports genres. The higher t-statistic and low p-value suggest that user ratings for these genres are distinct, with Action games generally receiving higher ratings.
# 
# Final Conclusion:
# 
#     Sales Trends: The data reveals that platform preferences vary significantly across regions, with distinct genre and ESRB rating trends. The North American and European markets show a strong preference for action-oriented and mature content, while Japan shows a stronger preference for role-playing and family-friendly games.
#     Genre Preferences: Action and Shooter genres dominate in North America and Europe, while Role-Playing games are particularly popular in Japan. This reflects cultural differences in gaming preferences.
#     Market Segmentation: The findings suggest that game developers should consider regional preferences when designing and marketing games. Mature and action-packed games might perform better in North America and Europe, whereas role-playing and family-friendly games might be more successful in Japan.
# 
# Final Conclusion
# 
# The gaming market shows significant regional diversity in preferences for platforms, genres, and age ratings. Understanding these regional nuances is crucial for developers and marketers aiming to enhance market penetration and user satisfaction. Future strategies should focus on catering to these specific preferences, with an emphasis on popular platforms and genres, and considering the cultural inclination towards certain ESRB ratings. This tailored approach can drive growth and foster deeper engagement with diverse gaming audiences worldwide.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment</h2>
# 
#     
# The conclusion is written well.    
# </div>    
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment</b>
#     
# It will be great if you add a list of promising platforms and more information about the regions: 2-3 platforms, 2-3 genres and age ratings. Don't forget to update the final conclusion if needed.
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# Excellent job! 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2</h2>
#  
# 
# The results may still change, so don't forget to check the conclusions. 
# </div>
# 
