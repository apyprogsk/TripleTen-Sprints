#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Great! You've done a great job on all the comments and now your project has been accepted.
#     
# Thank you for your work and I wish you success in the following projects!

# <div style="border:solid blue 2px; padding: 20px">
#   
# **Hello**
# 
# My name is Dima, and I will be reviewing your project. 
# 
# You will find my comments in coloured cells marked as 'Reviewer's comment'. The cell colour will vary based on the contents - I am explaining it further below. 
# 
# **Note:** Please do not remove or change my comments - they will help me in my future reviews and will make the process smoother for both of us. 
# 
# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment</b> 
#     
# Such comment will mark efficient solutions and good ideas that can be used in other projects.
# </div>
# 
# <div class="alert alert-warning"; style="border-left: 7px solid gold">
# <b>⚠️ Reviewer's comment</b> 
#     
# The parts marked with yellow comments indicate that there is room for optimisation. Though the correction is not necessary it is good if you implement it.
# </div>
# 
# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment</b> 
#     
# If you see such a comment, it means that there is a problem that needs to be fixed. Please note that I won't be able to accept your project until the issue is resolved.
# </div>
# 
# You are also very welcome to leave your comments / describe the corrections you've done / ask me questions, marking them with a different colour. You can use the example below: 
# 
# <div class="alert alert-info"; style="border-left: 7px solid blue">
# <b>Student's comment</b>

# ## Basic Python - Project <a id='intro'></a>

# <div style="border:solid green 2px; padding: 20px">
#     
# <div class="alert alert-success">
# <b>Review summary</b> 
#     
# Thanks for submitting the project. You've done a very good job and I enjoyed reviewing it.
#     
# - You completed all the tasks.
# - Your code was optimal and easy to read. 
# - You wrote your own functions.
#     
# There are only a few critical comments that need to be corrected. You will find them in the red-colored cells in relevant sections. If you have any questions please write them when you return your project. 
#     
# I'll be looking forward to getting your updated notebook.

# ## Introduction <a id='intro'></a>
# In this project, you will work with data from the entertainment industry. You will study a dataset with records on movies and shows. The research will focus on the "Golden Age" of television, which began in 1999 with the release of *The Sopranos* and is still ongoing.
# 
# The aim of this project is to investigate how the number of votes a title receives impacts its ratings. The assumption is that highly-rated shows (we will focus on TV shows, ignoring movies) released during the "Golden Age" of television also have the most votes.
# 
# ### Stages 
# Data on movies and shows is stored in the `/datasets/movies_and_shows.csv` file. There is no information about the quality of the data, so you will need to explore it before doing the analysis.
# 
# First, you'll evaluate the quality of the data and see whether its issues are significant. Then, during data preprocessing, you will try to account for the most critical problems.
#  
# Your project will consist of three stages:
#  1. Data overview
#  2. Data preprocessing
#  3. Data analysis

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Title and introduction are essential parts of the project. Make sure you do not forget to include it in your further projects. 
#     
# It is optimal if introduction consists of:
#     
# - brief description of the situation;
# - goal of the project;
# - description of the data we are going to use.
# </div>
# 

# ## Stage 1. Data overview <a id='data_review'></a>
# 
# Open and explore the data.

# You'll need `pandas`, so import it.

# In[2]:


# importing pandas
import pandas as pd


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> Needed library has been added </div>

# Read the `movies_and_shows.csv` file from the `datasets` folder and save it in the `df` variable:

# In[3]:


# reading the files and storing them to df
df = pd.read_csv('/datasets/movies_and_shows.csv')


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> The correct path to the file is specified: the slash at the beginning of the path is very important, as it indicates that you need to search for the file in the root folder. </div>

# Print the first 10 table rows:

# In[4]:


# obtaining the first 10 rows from the df table
# hint: you can use head() and tail() in Jupyter Notebook without wrapping them into print()
print(df.head(10))


# Obtain the general information about the table with one command:

# In[5]:


# obtaining general information about the data in df
print(df.info())


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Great - you've used a comprehensive set of methods to have a first look at the data.
#     

# The table contains nine columns. The majority store the same data type: object. The only exceptions are `'release Year'` (int64 type), `'imdb sc0re'` (float64 type) and `'imdb v0tes'` (float64 type). Scores and votes will be used in our analysis, so it's important to verify that they are present in the dataframe in the appropriate numeric format. Three columns (`'TITLE'`, `'imdb sc0re'` and `'imdb v0tes'`) have missing values.
# 
# According to the documentation:
# - `'name'` — actor/director's name and last name
# - `'Character'` — character played (for actors)
# - `'r0le '` — the person's contribution to the title (it can be in the capacity of either actor or director)
# - `'TITLE '` — title of the movie (show)
# - `'  Type'` — show or movie
# - `'release Year'` — year when movie (show) was released
# - `'genres'` — list of genres under which the movie (show) falls
# - `'imdb sc0re'` — score on IMDb
# - `'imdb v0tes'` — votes on IMDb
# 
# We can see three issues with the column names:
# 1. Some names are uppercase, while others are lowercase.
# 2. There are names containing whitespace.
# 3. A few column names have digit '0' instead of letter 'o'. 
# 

# ### Conclusions <a id='data_review_conclusions'></a> 
# 
# Each row in the table stores data about a movie or show. The columns can be divided into two categories: the first is about the roles held by different people who worked on the movie or show (role, name of the actor or director, and character if the row is about an actor); the second category is information about the movie or show itself (title, release year, genre, imdb figures).
# 
# It's clear that there is sufficient data to do the analysis and evaluate our assumption. However, to move forward, we need to preprocess the data.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Please note that it is highly recommended to add a conclusion / summary after each section and describe briefly your observations and / or major outcomes of the analysis.

# ## Stage 2. Data preprocessing <a id='data_preprocessing'></a>
# Correct the formatting in the column headers and deal with the missing values. Then, check whether there are duplicates in the data.

# In[6]:


# the list of column names in the df table
column_names = df.columns.tolist()
print(column_names)


# Change the column names according to the rules of good style:
# * If the name has several words, use snake_case
# * All characters must be lowercase
# * Remove whitespace
# * Replace zero with letter 'o'

# In[7]:


# renaming columns
column_names = [column.strip().lower().replace(' ', '_').replace('0', 'o') for column in column_names]
df.columns = column_names


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# This is a good way to rename the columns.

# Check the result. Print the names of the columns once more:

# In[8]:


# checking result: the list of column names
print(column_names)


# ### Missing values <a id='missing_values'></a>
# First, find the number of missing values in the table. To do so, combine two `pandas` methods:

# In[9]:


# calculating missing values
print(df.isna().sum())


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# The isna() method is selected to find the missing values, it's great!

# Not all missing values affect the research: the single missing value in `'title'` is not critical. The missing values in columns `'imdb_score'` and `'imdb_votes'` represent around 6% of all records (4,609 and 4,726, respectively, of the total 85,579). This could potentially affect our research. To avoid this issue, we will drop rows with missing values in the `'imdb_score'` and `'imdb_votes'` columns.

# In[10]:


# dropping rows where columns with title, scores and votes have missing values
df = df.dropna(axis='rows')


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Perfect!

# Make sure the table doesn't contain any more missing values. Count the missing values again.

# In[11]:


# counting missing values
print(df.isna().sum())


# ### Duplicates <a id='duplicates'></a>
# Find the number of duplicate rows in the table using one command:

# In[12]:


# counting duplicate rows
print(df.duplicated().sum())


# Review the duplicate rows to determine if removing them would distort our dataset.

# In[13]:


# Produce table with duplicates (with original rows included) and review last 5 rows
df = df.sort_values(by='name')
duplicates_with_originals = df[df.duplicated(keep=False)]
print(duplicates_with_originals.tail(5))


# There are two clear duplicates in the printed rows. We can safely remove them.
# Call the `pandas` method for getting rid of duplicate rows:

# In[20]:


# removing duplicate rows
df = df.drop_duplicates()
print(df.tail(5))


# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment, v. 1</b> 
#     
# Not quite right. Please note, that now we have **only duplicates** in our dataframe. We should rewrite our dataframe just like:
#     
#     df = df.drop_duplicates()

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
# 
# Well done!

# Check for duplicate rows once more to make sure you have removed all of them:

# In[21]:


# checking for duplicates
print(df.duplicated().sum())


# Now get rid of implicit duplicates in the `'type'` column. For example, the string `'SHOW'` can be written in different ways. These kinds of errors will also affect the result.

# Print a list of unique `'type'` names, sorted in alphabetical order. To do so:
# * Retrieve the intended dataframe column 
# * Apply a sorting method to it
# * For the sorted column, call the method that will return all unique column values

# In[22]:


# viewing unique type names
display(sorted(df['type'].unique()))


# Look through the list to find implicit duplicates of `'show'` (`'movie'` duplicates will be ignored since the assumption is about shows). These could be names written incorrectly or alternative names of the same genre.
# 
# You will see the following implicit duplicates:
# * `'shows'`
# * `'SHOW'`
# * `'tv show'`
# * `'tv shows'`
# * `'tv series'`
# * `'tv'`
# 
# To get rid of them, declare the function `replace_wrong_show()` with two parameters: 
# * `wrong_shows_list=` — the list of duplicates
# * `correct_show=` — the string with the correct value
# 
# The function should correct the names in the `'type'` column from the `df` table (i.e., replace each value from the `wrong_shows_list` list with the value in `correct_show`).

# In[26]:


# function for replacing implicit duplicates
def replace_wrong_show(wrong_shows_list, correct_show):
    df['type'].replace(wrong_shows_list, correct_show)


# Call `replace_wrong_show()` and pass it arguments so that it clears implicit duplicates and replaces them with `SHOW`:

# In[32]:


# removing implicit duplicates
def replace_wrong_show(wrong_shows_list, correct_show):
    for show in wrong_shows_list:
        df['type'] = df['type'].replace(show, correct_show)
    return df

replace_wrong_show(['shows', 'tv show', 'tv shows', 'tv series', 'tv'], 'SHOW')


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Yes, this is what was needed!

# Make sure the duplicate names are removed. Print the list of unique values from the `'type'` column:

# In[34]:


# viewing unique genre names
display(sorted(df['type'].unique()))


# ### Conclusions <a id='data_preprocessing_conclusions'></a>
# We detected three issues with the data:
# 
# - Incorrect header styles
# - Missing values
# - Duplicate rows and implicit duplicates
# 
# The headers have been cleaned up to make processing the table simpler.
# 
# All rows with missing values have been removed. 
# 
# The absence of duplicates will make the results more precise and easier to understand.
# 
# Now we can move on to our analysis of the prepared data.

# ## Stage 3. Data analysis <a id='hypotheses'></a>

# Based on the previous project stages, you can now define how the assumption will be checked. Calculate the average amount of votes for each score (this data is available in the `imdb_score` and `imdb_votes` columns), and then check how these averages relate to each other. If the averages for shows with the highest scores are bigger than those for shows with lower scores, the assumption appears to be true.
# 
# Based on this, complete the following steps:
# 
# - Filter the dataframe to only include shows released in 1999 or later.
# - Group scores into buckets by rounding the values of the appropriate column (a set of 1-10 integers will help us make the outcome of our calculations more evident without damaging the quality of our research).
# - Identify outliers among scores based on their number of votes, and exclude scores with few votes.
# - Calculate the average votes for each score and check whether the assumption matches the results.

# To filter the dataframe and only include shows released in 1999 or later, you will take two steps. First, keep only titles published in 1999 or later in our dataframe. Then, filter the table to only contain shows (movies will be removed).

# In[35]:


# using conditional indexing modify df so it has only titles released after 1999 (with 1999 included)
# give the slice of dataframe new name
shows_1999_or_later = df[df['release_year'] >= 1999]


# In[36]:


# repeat conditional indexing so df has only shows (movies are removed as result)
only_shows = shows_1999_or_later[shows_1999_or_later['type'] == 'SHOW']


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# All the transformations were performed absolutely correctly

# The scores that are to be grouped should be rounded. For instance, titles with scores like 7.8, 8.1, and 8.3 will all be placed in the same bucket with a score of 8.

# In[37]:


# rounding column with scores
only_shows['imdb_score'] = only_shows['imdb_score'].round().astype(int)


# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment, v. 1</b> 
#     
# Please round 'imdb_score' in `only_shows` instead of `df`

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Now it's perfect!

# It is now time to identify outliers based on the number of votes.

# In[40]:


# Use groupby() for scores and count all unique values in each group, print the result
only_shows.groupby('imdb_score').agg({'name': 'nunique'})


# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment, v. 1</b> 
#     
# Please use `only_shows` instead of `df` in the group by section

# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment, v. 1</b> 
#     
# The result here will be different after the correct processing of the dataframe in the section with duplicates

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Everything is correct in the grouping now

# Based on the aggregation performed, it is evident that scores 2 (24 voted shows), 3 (27 voted shows), and 10 (only 8 voted shows) are outliers. There isn't enough data for these scores for the average number of votes to be meaningful.

# To obtain the mean numbers of votes for the selected scores (we identified a range of 4-9 as acceptable), use conditional filtering and grouping.

# In[41]:


# filter dataframe using two conditions (scores to be in the range 4-9)
filtered_scores = only_shows[(only_shows['imdb_score'] >= 4) & (only_shows['imdb_score'] <= 9)]

# group scores and corresponding average number of votes, reset index and print the result
avg_votes_per_score = filtered_scores.groupby('imdb_score')['imdb_votes'].mean().reset_index()
avg_votes_per_score


# <div class="alert alert-danger"; style="border-left: 7px solid red">
# <b>⛔️ Reviewer's comment, v. 1</b> 
#     
# Please use `only_shows` instead of `df` in the filtering

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Everything is done flawlessly!

# Now for the final step! Round the column with the averages, rename both columns, and print the dataframe in descending order.

# In[42]:


# round column with averages
avg_votes_per_score['imdb_votes'] = avg_votes_per_score['imdb_votes'].round()
# rename columns
avg_votes = avg_votes_per_score.rename(columns= {'imdb_score': 'Score', 'imdb_votes': 'Avg Votes'})
# print dataframe in descending order
avg_votes = avg_votes[['Score', 'Avg Votes']].sort_values(by='Score', ascending=False)
avg_votes


# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 2</b> 
#     
# Great! Also correct rounding and grouping in this section

# The assumption macthes the analysis: the shows with the top 3 scores have the most amounts of votes.

# ## Conclusion <a id='hypotheses'></a>

# The research done confirms that highly-rated shows released during the "Golden Age" of television also have the most votes. While shows with score 4 have more votes than ones with scores 5 and 6, the top three (scores 7-9) have the largest number. The data studied represents around 94% of the original set, so we can be confident in our findings.

# <div class="alert alert-success"; style="border-left: 7px solid green">
# <b>✅ Reviewer's comment, v. 1</b> 
#     
# Overall conclusion is an important part, where we should include the summary of the outcomes of the project.
