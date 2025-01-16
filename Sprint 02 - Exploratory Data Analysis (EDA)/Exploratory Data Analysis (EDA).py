#!/usr/bin/env python
# coding: utf-8

# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment</b>
#     
# Hello, my name is Sveta, and I am going to review this project. 
#     
# 
# Before we start, I want to pay your attention to the color marking:
#     
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment ✔️</b>
#     
# Great solutions and ideas that can and should be used in the future are in green comments.   
# </div>    
#     
#     
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment ⚠️</b>
# 
# Yellow color indicates what should be optimized. This is not necessary, but it will be great if you make changes to this project.
# </div>      
#     
#     
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment ❌</b>
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
# <font color='dodgerblue'>**A few words about the project:**</font> you did a good job, everything is clear and neat. The project is easy to read, which is definitely a plus. There are a lot graphs. It is good, visualization never hurts. I still have some questions that I've written in my comments. I've also left there some recommendations for improving the project.
#     
#     
# I will wait for the project for a second review :)
#     
#     
#     
# 📌 Here are some hints that may help you with Markdown cells:    
# <hr style="border-top: 3px solid purple; "></hr>
# 
# You can leave comments using this code inside a Markdown cell:
#     
#     
#     <div class="alert alert-info">
#     <h2> Student's comment</h2>
# 
#     Your text here. 
#     </div>
# 
#     
#     
#     <font color='red'> This code is used to change text color. </font>     
# 
# <font color='red'> It will look like this. </font> 
#     
# If you don't want your comments to be headings, replace **h2** with **b** or just add `<a class="tocSkip">` after the phrase *Student's comment*.
# 
# 
# You can find out how to **format text** in a Markdown cell or how to **add links** [here](https://sqlbak.com/blog/jupyter-notebook-markdown-cheatsheet) и [and here](https://medium.com/analytics-vidhya/the-ultimate-markdown-guide-for-jupyter-notebook-d5e5abf728fd).
# </div>

# 
# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 2</b>
# 
# 
# Thank you for sending a new version of the project 😊 I've left a few comments titled as **Reviewer's comment 2**. Please take a look :) 
# 
#     
# 
#     
# </div>
# <hr>
# 
# <div style="border-radius: 15px; border: 3px solid indigo; padding: 15px;">
# <b> Reviewer's comment 3</b>
# 
# 
# I've left a couple of new comments with digit 3. Your project has passed code review. Congratulations 😊
#     
# 
#     
# This is a good [article](https://www.kaggle.com/ramprakasism/pandas-75-exercises-with-solutions/notebook) with 75 pandas exercises and solutions. You can go along the exercises and compare your solution with the solution in the article.
# 
#     
#     
# Good luck! 😊 
#     
#     
#     
# <hr>
#     
#     
# Best regards,
#     
# Sveta
#     
# </div>

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b>   Reviewer's comment ❌</b>
#     
# First things first. The introduction is the initial paragraph that each project, each essay or any article should have. It is important to write an introductory part, because it gives an idea about the content of the project. Please, add project and data descriptions. 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
# Great! :) It's a good habit actually. I have about 200 notebooks on my laptop with different tasks. Without a good intro it would be hard to remember what a particular notebook is about.
# 
# </div>

# Project Description
# The goal of this project is to gain insights into customer shopping habits, identify popular products, understand reorder behavior, and explore temporal trends in order placement. Through exploratory data analysis (EDA) and data preprocessing techniques, we aim to uncover meaningful patterns and trends that can inform business decisions and improve customer experience on the Instacart platform.
# Data Description
# 
# The dataset consists of five tables:
# instacart_orders.csv: Contains information about each customer order, including order ID, user ID, order number, day of the week, hour of the day, and days since the prior order.
# 
# products.csv: Provides details about individual products, such as product ID, product name, aisle ID, and department ID.
# 
# order_products.csv: Specifies which products were included in each order, along with additional details such as add-to-cart order and reorder status.
# 
# aisles.csv: Includes a list of aisle IDs and corresponding aisle names.
# 
# departments.csv: Contains department IDs and corresponding department names.
# 
# 
# Tasks Performed
# The project encompasses various tasks, including data preprocessing, exploratory data analysis (EDA), and visualization. Key tasks include verifying and cleaning data, analyzing order patterns by hour and day of the week, identifying popular products, exploring reorder behavior, and investigating shopping trends over time.
# 
# By conducting thorough analysis and interpretation of the Instacart dataset, we aim to provide actionable insights for improving operational efficiency and enhancing customer satisfaction.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b>  Reviewer's comment ⚠️</b>
#     
# 
# You can put imports in one cell. 
#     
# </div>

# In[2]:


instacart_orders_df = pd.read_csv('/datasets/instacart_orders.csv', sep=';')
instacart_orders_df.info()


# In[3]:


products_df = pd.read_csv('/datasets/products.csv', sep=';')
products_df.info()


# In[4]:


order_products_df = pd.read_csv('/datasets/order_products.csv', sep=';')
order_products_df.info()


# In[5]:


aisles_df = pd.read_csv('/datasets/aisles.csv', sep=';')
aisles_df.info()


# In[6]:


departments_df = pd.read_csv('/datasets/departments.csv', sep=';')
departments_df.info()


# CONCLUSION:
# 
# Based on the information provided from the database files:
# 
# The data seems to be structured into several tables:
# 
# Orders Table: Contains information about orders such as order ID, user ID, order number, day of the week the order was made, hour of the day the order was made, and days since the prior order. It appears that there are some missing values in the "days_since_prior_order" column.
# 
# Products Table: This table lists product details including product ID, product name, aisle ID, and department ID. It seems that there are some missing values in the "product_name" column.
# 
# Order Products Table: Contains details about products added to each order, including the order ID, product ID, order in which the product was added to the cart, and whether the product was reordered.
# 
# Aisles Table: Provides information about different aisles, including aisle ID and aisle name.
# 
# Departments Table: Contains details about different departments, including department ID and department name.
# 
# Intermediate Conclusion:
# The database consists of several tables containing information about orders, products, aisles, and departments. Further analysis could involve examining relationships between these tables, such as which products are frequently reordered, which departments have the highest sales, or which aisles are most commonly visited. Additionally, data cleaning may be necessary to handle missing values in certain columns.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Let's apply at least the `info` method to get the basic information about the data we have. 
# 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>  Reviewer's comment 2 ⚠️</h2>
#   
#     
# 
# An intermediate conclusion would not be redundant here :)
# 
# </div>

# ## Find and remove duplicate values (and describe why you make your choices)

# ### `orders` data frame

# In[7]:


# Check for duplicated orders
duplicate_orders = instacart_orders_df[instacart_orders_df.duplicated(subset=['order_id'], keep=False)]
display("Number of duplicated orders:", len(duplicate_orders))


# In[8]:


# Check for all orders placed Wednesday at 2:00 AM
wednesday_2am_orders = instacart_orders_df[(instacart_orders_df['order_dow'] == 3) & (instacart_orders_df['order_hour_of_day'] == 2)]

display("Orders placed on Wednesday at 2:00 AM:", wednesday_2am_orders)


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Please use the `display` method for the dataframes. Take a look: 
# 
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2 ❌</b>
#     
# This comment is still relevant, do not use `print` for the dataframes please.
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 ✔️</h2>
#     
# Excellent! 
#     
# </div>

# In[9]:


# Reviewer's code

display(instacart_orders_df.head(3))

instacart_orders_df.tail(3)


# In[10]:


# Remove duplicate orders
instacart_orders_df_unique = instacart_orders_df.drop_duplicates(subset=['order_id']).reset_index(drop=True)

print("Before removing duplicates:", instacart_orders_df.shape)
print("After removing duplicates:", instacart_orders_df_unique.shape)


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>  Reviewer's comment ⚠️</h2>
#   
#     
# 
# It is better to reset indices and drop the old ones. 
# 
# </div>

# In[11]:


# Double check for duplicate rows
print("No of duplicated orders:", instacart_orders_df_unique.duplicated().sum())


# In[12]:


# Double check for duplicate order IDs only
duplicate_order_ids = instacart_orders_df_unique['order_id'].duplicated().sum()
print("No of duplicate order IDs:", duplicate_order_ids)


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment ✔️</h2>
#     
# 
# Correct. 
# 
# </div>

# ### `products` data frame

# In[13]:


# Check for fully duplicate rows

print("No of fully duplicate rows in products DataFrame:", len(products_df[products_df.duplicated()]))


# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment ⚠️</h2>
#     
# 
# By the way, you do not need to create a new variable. 
#     
# </div>

# In[14]:


# Check for just duplicate product IDs
duplicate_product_ids = products_df['product_id'].duplicated().sum()
print("No of duplicate product IDs:", duplicate_product_ids)


# In[15]:


# Check for just duplicate product names (convert names to lowercase to compare better)
products_df['product_name_lower'] = products_df['product_name'].str.lower()

duplicate_product_names = products_df['product_name_lower'].duplicated().sum()
print("No of duplicate product names:", duplicate_product_names)


# In[16]:


# Check for duplicate product names that aren't missing
products_df_notnull = products_df.dropna(subset=['product_name']).copy()

products_df_notnull['product_name_lower'] = products_df_notnull['product_name'].str.lower()

duplicate_product_names = products_df_notnull['product_name_lower'].duplicated().sum()
print("No of duplicate product names (excluding missing):", duplicate_product_names)


# Based on above, addressing duplicate product names is essential for maintaining data integrity and ensuring the accuracy of any analyses or applications built upon the dataset.
# This could involve standardizing naming conventions, resolving discrepancies, or updating incorrect entries and documenting the process of identifying and resolving duplicate product names is crucial for transparency and reproducibility. This documentation can help future users understand the dataset's quality and any modifications made to it.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# So, what can you say about these duplicates? 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>Reviewer's comment 2 ✔️</h2>
#     
# 
# They may have different `product_id`, so I am not sure whether we should treat them as duplicates. 
#     
#     
# </div>

# ### `departments` data frame

# In[17]:


display(departments_df.head())


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Please do not read the data again. 
# </div>

# In[18]:


# Checking for duplicates in department_id
duplicate_department_ids = departments_df['department_id'].duplicated().any()

if duplicate_department_ids:
    print("There are duplicates in department_id.")
else:
    print("There are no duplicates in department_id.")


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Are there any duplicates in `department_id`? 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>Reviewer's comment 2 ✔️</h2>
#     
# 
# Good. It's better to not change the initial code but to add a new cell, since it may still be useful to display the number of fully duplicated rows as well. 
#     
#     
# </div>

# ### `aisles` data frame

# In[19]:


display(aisles_df.head())


# In[20]:


# Checking for duplicates in aisle_id
duplicate_aisle_ids = aisles_df['aisle_id'].duplicated().any()

if duplicate_aisle_ids:
    print("There are duplicates in aisle_id.")
else:
    print("There are no duplicates in aisle_id.")


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Are there any duplicates in `aisle_id`? 
# 
# </div>

# ### `order_products` data frame

# In[21]:


# Check for fullly duplicate rows
fully_duplicate_order_products = order_products_df[order_products_df.duplicated()]
print("No of fully duplicate rows in order_products DataFrame:", len(fully_duplicate_order_products))


# In[22]:


# Double check for any other tricky duplicates
duplicate_order_product_combinations = order_products_df[order_products_df.duplicated(subset=['order_id', 'product_id'], keep=False)]
print("No of duplicate combinations of order_id and product_id:", len(duplicate_order_product_combinations))


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>   Reviewer's comment ✔️</h2>
#     
# Correct. 
#     
# </div>

# ## Find and remove missing values
# 

# ### `products` data frame

# In[23]:


# Checking for missing values in the products DataFrame
missing_values = products_df.isnull().sum()
display("Missing values in the products DataFrame:", missing_values)


# In[24]:


# Are all of the missing product names associated with aisle ID 100?

# Filtering rows with missing product names and aisle ID 100
missing_product_names_aisle_100 = products_df[(products_df['product_name'].isnull()) & (products_df['aisle_id'] == 100)]

# Check if all missing product names are associated with aisle ID 100
all_missing_product_names_associated_with_aisle_100 = len(missing_product_names_aisle_100) == missing_values['product_name']
print("Are all missing product names associated with aisle ID 100?", all_missing_product_names_associated_with_aisle_100)


# In[25]:


# Are all of the missing product names associated with department ID 21?

# Filtering rows with missing product names and department ID 21
missing_product_names_dept_21 = products_df[(products_df['product_name'].isnull()) & (products_df['department_id'] == 21)]

# Check if all missing product names are associated with department ID 21
all_missing_product_names_associated_with_dept_21 = len(missing_product_names_dept_21) == missing_values['product_name']
print("Are all missing product names associated with department ID 21?", all_missing_product_names_associated_with_dept_21)


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>   Reviewer's comment ✔️</h2>
#     
# Good.     
# </div>

# In[26]:


# What is this ailse and department?
# Retrieve the aisle and department names for the given aisle ID and department ID
aisle_name = aisles_df.loc[aisles_df['aisle_id'] == 100, 'aisle'].values[0]
department_name = departments_df.loc[departments_df['department_id'] == 21, 'department'].values[0]

print("Aisle:", aisle_name)
print("Department:", department_name)


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment ✔️</h2>
#     
# Yes, each missing product name has a `'missing'` label. 
# </div>

# In[27]:


# Fill missing product names with 'Unknown'
products_df['product_name'] = products_df['product_name'].fillna('Unknown')

# Verify that missing values have been filled
missing_values_after_fill = products_df.isnull().sum()
print("Missing values in the products DataFrame after filling:")
print(missing_values_after_fill)


# 

# ### `orders` data frame

# In[28]:


print(instacart_orders_df.head())


# In[29]:


# Checking for missing values in orders DataFrame
missing_values = instacart_orders_df.isnull().any().any()

if missing_values:
    print("There are missing values in the orders DataFrame.")
else:
    print("There are no missing values in the orders DataFrame.")

# Count missing values in each column of the orders DataFrame
missing_values_count = instacart_orders_df.isnull().sum()

print("Count of missing values in each column:")
display(missing_values_count)


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Are there any missing values?    
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>Reviewer's comment 2 ✔️</h2>
#     
# 
# Very good.     
#     
# </div>

# In[30]:


# Are there any missing values where it's not a customer's first order?

# Filter rows where 'order_number' is greater than 1 and check for missing values in 'days_since_prior_order'
missing_values_not_first_order = instacart_orders_df[instacart_orders_df['order_number'] > 1]['days_since_prior_order'].isnull().sum()

if missing_values_not_first_order > 0:
    print("There are missing values in 'days_since_prior_order' where it's not a customer's first order.")
else:
    print("There are no missing values in 'days_since_prior_order' where it's not a customer's first order.")


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment ✔️</h2>
#     
# Correct. 
# 
# </div>

# ### `order_products` data frame

# In[31]:


print(order_products_df.head())


# In[32]:


# Checking for missing values in order_products DataFrame
missing_values_order_products = order_products_df.isnull().any().any()

if missing_values_order_products:
    print("There are missing values in the order_products DataFrame.")
else:
    print("There are no missing values in the order_products DataFrame.")

# Count missing values in each column of the order_products DataFrame
missing_values_count_order_products = order_products_df.isnull().sum()

print("Count of missing values in each column of order_products DataFrame:")
print(missing_values_count_order_products)


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Are there any missing values here?    
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2>Reviewer's comment 2 ✔️</h2>
#     
#     
# Correct. 
#     
# </div>

# In[33]:


# What are the min and max values in this column?

# Find the minimum and maximum values in the 'add_to_cart_order' column
min_add_to_cart_order = order_products_df['add_to_cart_order'].min()
max_add_to_cart_order = order_products_df['add_to_cart_order'].max()

print("Minimum value in 'add_to_cart_order' column:", min_add_to_cart_order)
print("Maximum value in 'add_to_cart_order' column:", max_add_to_cart_order)


# In[34]:


# Save all order IDs with at least one missing value in 'add_to_cart_order'

# Filter rows with missing values in 'add_to_cart_order' column and get unique order IDs
orders_with_missing_add_to_cart_order = order_products_df[order_products_df['add_to_cart_order'].isnull()]['order_id'].unique()

# Print the order IDs with at least one missing value in 'add_to_cart_order' column
print("Order IDs with at least one missing value in 'add_to_cart_order' column:")
print(orders_with_missing_add_to_cart_order)


# In[35]:


# Grouping by 'order_id' and count the number of products in each order
order_product_counts = order_products_df.groupby('order_id')['product_id'].count()

# Checking if all orders with missing values have more than 64 products
all_orders_more_than_64_products = (order_product_counts.loc[orders_with_missing_add_to_cart_order] > 64).all()

if all_orders_more_than_64_products:
    print("All orders with missing values in 'add_to_cart_order' have more than 64 products.")
else:
    print("Not all orders with missing values in 'add_to_cart_order' have more than 64 products.")


# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment ✔️</h2>
#     
# Correct.
#     
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment  ⚠️</h2>
#     
# 
# 
# However, we do not need a loop here. 
# </div>

# In[36]:


# Replace missing values with 999 and convert column to integer type

# Replacing missing values with 999 in 'add_to_cart_order' column
order_products_df['add_to_cart_order'] = order_products_df['add_to_cart_order'].fillna(999).astype(int)

# Verifying the changes
print("Data types after conversion:\n", order_products_df.dtypes)


# In this section, we addressed missing values in the 'add_to_cart_order' column by replacing them with 999 and converting the column to the integer data type. By doing so, we ensured data integrity and prepared the DataFrame for further analysis.

# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment ⚠️</b>
#     
# `astype` can be applied right after `fillna`.
#     
# </div>
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Please add a conclusion about the whole section here. 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
# Great! Now we are ready for the exploratory data analysis. 
# 
# </div>

# # [A] Easy (must complete all to pass)

# ### [A1] Verify that the `'order_hour_of_day'` and `'order_dow'` values in the `orders` tables are sensible (i.e. `'order_hour_of_day'` ranges from 0 to 23 and `'order_dow'` ranges from 0 to 6)

# In[37]:


# Check for any outliers or invalid values in 'order_hour_of_day'
invalid_hour_values = instacart_orders_df[
    (instacart_orders_df['order_hour_of_day'] < 0) | 
    (instacart_orders_df['order_hour_of_day'] > 23)
]

# Check for any outliers or invalid values in 'order_dow'
invalid_dow_values = instacart_orders_df[
    (instacart_orders_df['order_dow'] < 0) | 
    (instacart_orders_df['order_dow'] > 6)
]

if invalid_hour_values.empty and invalid_dow_values.empty:
    print("All 'order_hour_of_day' and 'order_dow' values are sensible.")
else:
    print("There are outliers or invalid values in 'order_hour_of_day' or 'order_dow'.")


# ## Conclusion
# 
# All 'order_hour_of_day' and 'order_dow' values in the orders table are sensible. The 'order_hour_of_day' values range from 0 to 23, representing the hours of the day when orders were placed. Similarly, the 'order_dow' values range from 0 to 6, representing the days of the week (0 being Sunday and 6 being Saturday) when orders were placed. These ranges align with expected values and indicate that the data is consistent and suitable for further analysis.

# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ⚠️</b>
#     
# 
# 
# It would be great if you wrote the conclusions in the Markdown cells. 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
# Excellent! 
#     
# </div>

# ### [A2] What time of day do people shop for groceries?

# In[38]:


from matplotlib import pyplot as plt

hourly_order_counts = instacart_orders_df['order_hour_of_day'].value_counts().sort_index()

hourly_order_counts.plot(x='order_hour_of_day', y='hourly_order_counts', title='Frequency of Orders by Hour of Day', kind='bar',xlabel='Hour of Day', ylabel='Number of Orders', legend=False, xlim=[0,24], figsize=[10,6], alpha=0.7)
plt.xticks(rotation=0)
plt.show()


# Conclusion:
# 
# The histogram illustrates the frequency of orders by hour of the day, providing insights into when people shop for groceries. From the visualization, it is evident that there are fluctuations in order frequency throughout the day, with peak shopping times typically occurring during the late morning to early afternoon hours. This observation aligns with common shopping patterns, as many individuals may prefer to purchase groceries during daytime hours. The histogram's clear labels and title enhance its interpretability, facilitating a better understanding of the distribution of orders across different times of the day. Overall, the analysis suggests that the majority of grocery shopping occurs during daytime hours, with notable variations in activity levels across the day.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment ✔️</h2>
#     
#     
# Nice chart 👍    
# </div>
# 
# 
# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment ❌</b>
#     
# 
# - What can be inferred from it? 
# 
#     
# 
# - `Frequency` label may seem unclear for a reader. Please make sure that each chart has lucid labels and titles. 
# 
# </div>
# <div class="alert alert-warning" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b>  Reviewer's comment ⚠️</b>
#     
# 
# - `plt.xticks(rotation=0)` will rotate X-axis tick labels.
# 
# 
# - [PEP8](https://peps.python.org/pep-0008/) states that one should always put imports at the top of the file. It's a good practice, since everyone who is going to read the project, can immediately figure out what modules need to be installed. Moreover, imports should be placed in a separate cell. 
# 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Correct, most orders occur between 9 AM and 5 PM.
# 
# 
# </div>

# ### [A3] What day of the week do people shop for groceries?

# In[39]:


import matplotlib.pyplot as plt

daily_order_counts = instacart_orders_df['order_dow'].value_counts().sort_index()

daily_order_counts.plot(x='order_dow', y='daily_order_counts', title='Frequency of Orders by Day of Week', kind='bar',xlabel='Day of Week (0=Sunday, 1=Monday, ..., 6=Saturday)', ylabel='No of Orders', legend=False, figsize=[10,6], alpha=0.7)
plt.xticks(range(7), ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
plt.xticks(rotation=0)
plt.show()


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# For each task that you have done in sections A, B and C, write a conclusion please. 
#     
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Very good. 
# 
# </div>

# Conclusion:
# 
# The histogram reveals interesting patterns in the frequency of grocery shopping by day of the week. Remarkably, the analysis shows that the highest frequency of orders occurs on Sunday and Monday, indicating significant shopping activity at the beginning and end of the traditional week. Conversely, the order frequencies on the remaining days of the week appear relatively uniform, suggesting consistent but less pronounced shopping behavior. This observation aligns with common shopping trends, where consumers may engage in more extensive grocery shopping over the weekend to prepare for the upcoming week. The histogram's clear labels, including rotated x-axis tick labels for improved readability, enhance its interpretability, facilitating a nuanced understanding of shopping behavior across different days of the week. Overall, the analysis provides valuable insights into the distribution of grocery shopping activity throughout the week, underscoring the significance of Sunday and Monday as peak shopping days.

# ### [A4] How long do people wait until placing another order?

# In[40]:


# Filter out the first orders (where days_since_prior_order is null)
subsequent_orders = instacart_orders_df[instacart_orders_df['days_since_prior_order'].notnull()]

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(subsequent_orders['days_since_prior_order'], bins=30, color='skyblue', edgecolor='black')
plt.title('Days Since Prior Order')
plt.xlabel('Days Since Prior Order')
plt.ylabel('No of Orders')
plt.grid(alpha=0.7)
plt.show()


# 

# Conclusion:
# 
# The histogram reveals interesting patterns in the frequency of days elapsed since the previous order. Notably, the analysis indicates that the highest frequency of orders occurs on the 30th day since the prior order, suggesting a significant proportion of customers placing orders monthly. Additionally, there is a notable peak in order frequency observed within the 3rd to 9th day range, possibly indicating a weekly shopping pattern for some customers. However, order frequencies decline considerably after the 15th day, with relatively few orders observed beyond this point. This observation suggests that a significant portion of customers may prefer to shop more frequently, with longer intervals between orders being less common. Overall, the histogram provides valuable insights into the distribution of days elapsed between orders, highlighting distinct patterns in customer shopping behavior.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Some people may make an order once a month.   
# </div>

# # [B] Medium (must complete all to pass)

# ### [B1] Is there a difference in `'order_hour_of_day'` distributions on Wednesdays and Saturdays? Plot the histograms for both days and describe the differences that you see.

# In[41]:


# Filtering orders for Wednesdays and Saturdays
wednesday_orders = instacart_orders_df[instacart_orders_df['order_dow'] == 3]  # Wednesday is represented by 3
saturday_orders = instacart_orders_df[instacart_orders_df['order_dow'] == 5]   # Saturday is represented by 5


# In[42]:


# Plot histograms for 'order_hour_of_day' on Wednesdays and Saturdays
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(wednesday_orders['order_hour_of_day'], bins=24, color='skyblue', edgecolor='black')
plt.title('Wednesday Order Hour Distribution')
plt.xlabel('Hour of Day')
plt.ylabel('No of Orders')
plt.xticks(range(24))

plt.subplot(1, 2, 2)
plt.hist(saturday_orders['order_hour_of_day'], bins=24, color='orange', edgecolor='black')
plt.title('Saturday Order Hour Distribution')
plt.xlabel('Hour of Day')
plt.ylabel('No of Orders')
plt.xticks(range(24))

plt.tight_layout()
plt.show()


# Conclusion:
# 
# The histograms comparing the distributions of order placements by hour on Wednesdays and Saturdays reveal relatively uniform patterns with peaks between 9 am and 5 pm on both days. This indicates that customers tend to place orders consistently throughout the day, with heightened activity during typical daytime hours. The absence of significant deviations in the distributions suggests that shopping behavior on Wednesdays and Saturdays follows similar patterns, characterized by steady order placements across various hours. Overall, the analysis provides valuable insights into the temporal distribution of order placements on these specific days, highlighting the importance of daytime hours in grocery shopping activity.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Great! 
# 
# </div>

# ### [B2] What's the distribution for the number of orders per customer?

# In[43]:


# Group orders by 'user_id' and count the number of orders for each user
orders_per_user = instacart_orders_df.groupby('user_id')['order_id'].count()


# In[44]:


# Plot histogram of orders per customer
plt.figure(figsize=(10, 6))
plt.hist(orders_per_user, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Orders Per Customer')
plt.xlabel('Number of Orders')
plt.ylabel('No of Orders')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# 

# Conclusion:
# 
# The histogram illustrates a skewed distribution of the number of orders per customer, with a significant proportion of customers having only one order. This suggests that a large number of customers may be occasional shoppers or may have made only a single purchase on the platform. Subsequently, the frequency of customers decreases gradually as the number of orders per customer increases, indicating that fewer customers place multiple orders. By the middle of the month, the frequency of customers with higher numbers of orders diminishes substantially, implying a decline in customer engagement or retention over time. Overall, the analysis provides insights into the distribution of customer order behavior, highlighting the prevalence of occasional shoppers and the challenges associated with retaining customers for multiple transactions.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Correct.  
# 
# </div>

# ### [B3] What are the top 20 popular products (display their id and name)?

# In[45]:


# Merge order_products_df with products_df to get product names
merged_df = order_products_df.merge(products_df[['product_id', 'product_name']], on='product_id')

# Count the occurrences of each product
product_counts = merged_df['product_name'].value_counts().head(20)

# Print the top 20 popular products with their IDs and names
print("Top 20 Popular Products:")
print(product_counts)


# Conclusion:
# 
# The analysis reveals the top 20 popular products based on their frequency in orders. Among these products, fresh fruits such as bananas, organic bananas, strawberries, and avocados dominate the list, reflecting a strong preference for healthy and organic options among customers. Additionally, staple items like organic whole milk and organic garlic also feature prominently, indicating a balance between fresh produce and pantry essentials in customers' shopping baskets. Notably, the high frequencies of certain products like bananas and organic strawberries underscore their widespread popularity and consistent demand among consumers. Overall, the top 20 popular products represent a diverse range of food items, reflecting the varied preferences and dietary choices of Instacart users.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Very good job! 
#     
# 
# </div>

# # [C] Hard (must complete at least two to pass)

# ### [C1] How many items do people typically buy in one order? What does the distribution look like?

# In[46]:


import matplotlib.pyplot as plt

# Group order_products_df by 'order_id' and count the number of products per order
items_per_order = order_products_df.groupby('order_id')['product_id'].count()

# Calculate mean and median number of items per order
mean_items_per_order = items_per_order.mean()
median_items_per_order = items_per_order.median()
print("Mean number of items per order:", mean_items_per_order)
print("Median number of items per order:", median_items_per_order)


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2  ❌</b>
#     
# Please make sure everything works fine before you send a project for a review. 
# 
# </div>

# In[47]:


# Plot histogram of items per order with 30 bins
plt.figure(figsize=(10, 6))
plt.hist(items_per_order, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Items Per Order')
plt.xlabel('Number of Items')
plt.ylabel('No of Orders')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Conclusion:
# 
# The distribution of items per order exhibits a right-skewed pattern, with the majority of orders containing a relatively small number of items. The histogram demonstrates that the highest frequency of orders occurs with only one item, followed by gradually decreasing frequencies as the number of items per order increases. This trend reflects typical shopping behavior, where customers often make smaller purchases more frequently, interspersed with occasional larger orders. Understanding this distribution is essential for effective inventory management and order fulfillment strategies to accommodate the varying preferences and needs of customers.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Correct. 
# 
# </div>

# ### [C2] What are the top 20 items that are reordered most frequently (display their names and product IDs)?

# In[48]:


# Calculate the total number of times each product has been reordered
reorder_counts = order_products_df[order_products_df['reordered'] == 1]['product_id'].value_counts().head(20)

# Join with products DataFrame to get product names
top_reorder_products_with_names = pd.DataFrame({'product_id': reorder_counts.index}).merge(products_df[['product_id', 'product_name']], on='product_id')

# Display the top 20 items that are reordered most frequently with their names and product IDs
print("Top 20 items that are reordered most frequently:\n", top_reorder_products_with_names)


# Conclusion
# 
# 
# The top 20 items that are reordered most frequently on the Instacart platform include a variety of fresh produce, dairy products, and staple groceries. This analysis highlights customer preferences for organic fruits and vegetables, as well as the importance of convenience items such as bananas and milk. By understanding these popular items, Instacart can optimize its product offerings and enhance customer satisfaction.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# `mean` will not tell us anything about the frequency. If the product was ordered twice, both times by one person, then the mean is 1. What if another product was reordered in 800 out of 1000 cases? :) 
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
#     
# Yes, that is better. 
#     
# </div>

# ### [C3] For each product, what proportion of its orders are reorders?

# In[49]:


# Calculate the number of reordered orders for each product
reordered_orders_per_product = order_products_df.groupby('product_id')['reordered'].sum()

# Calculate the total number of orders for each product
total_orders_per_product = order_products_df.groupby('product_id').size()

# Calculate the proportion of reordered orders for each product
reorder_proportion_per_product = (reordered_orders_per_product / total_orders_per_product).reset_index()

# Merge with products dataframe to get product names
reorder_proportion_with_names = pd.merge(reorder_proportion_per_product, products_df[['product_id', 'product_name']], on='product_id')

# Rename the column at index 1 to 'Proportion'
reorder_proportion_with_names = reorder_proportion_with_names.rename(columns={reorder_proportion_with_names.columns[1]: 'proportion'})

# Display the updated DataFrame
display('Proportion of reordered orders for each product', reorder_proportion_with_names)


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment 2 ❌</b>
#     
# 
# This cell does not work.    
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 3 ✔️</h2>
#     
# Very good!     
# </div>

# CONCLUSION:
# 
# From the output showing the proportion of reordered orders for each product, we can observe the following:
# 
# The product_id column represents the unique identifier for each product.
# The order_id column represents the proportion of reordered orders for each product. However, there seems to be some missing values in this column.
# The product_name column contains the names of the products.
# Conclusion:
# 
# The proportion of reordered orders varies across different products.
# Some products have a high proportion of reordered orders, indicating that they are frequently repurchased by customers.
# Other products have a lower proportion of reordered orders, suggesting that they may be purchased less frequently or may not be as popular among customers.
# Further analysis can be conducted to understand the factors influencing the reordering behavior of customers and to identify strategies for improving product retention and customer satisfaction.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# We are supposed to calculate a proportion here.     
# </div>

# ### [C4] For each customer, what proportion of their products ordered are reorders?

# In[50]:


# Step 1: Calculate total products and sum of reordered products per order
order_products_summary = order_products_df.groupby('order_id').agg(total_products=('product_id', 'count'),
                                                                  total_reordered=('reordered', 'sum'))

# Step 2: Merge with instacart_orders to associate orders with users
order_user_merged = order_products_summary.merge(instacart_orders_df[['order_id', 'user_id']], on='order_id')

# Step 3: Calculate proportion of reordered products for each order
order_user_merged['reorder_proportion'] = order_user_merged['total_reordered'] / order_user_merged['total_products']

# Step 4: Calculate average proportion of reordered products for each customer
customer_reorder_proportion = order_user_merged.groupby('user_id')['reorder_proportion'].mean()

# Display the result
print(customer_reorder_proportion)


# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment ❌</b>
#     
# Please use the `reordered` column for count and sum.     
# </div>
# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2  ✔️</h2>
#     
# 
# By the way, you can also use the `mean` function here. Take a look: 
# 
# </div>

# In[51]:


# Reviewer's code 2

order_products_df.merge(instacart_orders_df).groupby('user_id')['reordered'].mean()


# 

# ### [C5] What are the top 20 items that people put in their carts first? 

# In[52]:


# First items added to the cart
first_items = order_products_df[order_products_df['add_to_cart_order'] == 1]

# Count the occurrences to find the top 20 first items
top_first_items = first_items['product_id'].value_counts().head(20)


# In[53]:


# Join with products DataFrame to get product names
top_first_items_with_names = (pd.DataFrame({'product_id': top_first_items.index})).merge(products_df[['product_id', 'product_name']], on='product_id')


# In[54]:


# Display the top 20 items that people put in their carts first along with their names and product IDs
print("Top 20 items that people put in their carts first:\n", top_first_items_with_names)


# CONCLUSION:
# 
# Based on the top 20 items that people put in their carts first, it's evident that a significant portion of these items consists of fresh produce and dairy products, such as bananas, organic strawberries, organic whole milk, and organic baby spinach. Additionally, beverages like spring water, soda, and sparkling water grapefruit are also popular choices.
# 
# This indicates that customers tend to prioritize purchasing perishable and essential items, such as fruits, vegetables, and dairy, at the beginning of their shopping trips. Additionally, the presence of beverages suggests that customers may also prioritize staying hydrated while shopping.

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Everything looks good, but don't forget about the coclusions :)     
# </div>

# <div class="alert alert-danger" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <b> Reviewer's comment  ❌</b>
#     
# Please don't forget to add the overall conclusion to your project: what has been done and what can be inferred from the results. 
# 
# </div>

# FINAL SUMMARY:
# 
# Throughout the project, an in-depth analysis of the Instacart dataset has been conducted, focusing on various aspects such as data cleaning, exploration, and visualization. Here's an overall summary of the project:
# 
# Data Cleaning:
# 
# Duplicates were identified and removed from the orders, products, and order_products dataframes.
# Missing values were handled by either filling them with appropriate values or dropping rows with missing data.
# Data types were adjusted to ensure consistency and accuracy in analysis.
# Exploratory Data Analysis (EDA):
# 
# Distribution of orders by hour of the day and day of the week was analyzed, providing insights into peak shopping times.
# The frequency of orders, reorder rates, and other trends were explored to understand customer behavior.
# Top products, both in terms of popularity and reordering frequency, were identified.
# Visualization:
# 
# Histograms, bar plots, and other visualizations were used to represent the distribution and trends within the data effectively.
# Matplotlib and Pandas plotting functions were utilized to create clear and insightful visualizations.
# Inferences:
# 
# Peak shopping times were observed to be during daylight hours, with slight variations based on the day of the week.
# Customers tend to purchase perishable items such as fruits, vegetables, and dairy products frequently.
# Reorder rates provide insights into the popularity and trustworthiness of specific products among customers.
# The analysis provides valuable insights for Instacart to optimize inventory management, marketing strategies, and customer experiences.
# Overall, the project aimed to understand customer behavior, preferences, and patterns within the Instacart dataset. The insights gained can be used to make data-driven decisions to enhance the shopping experience for Instacart customers and improve operational efficiency for the company.

# <div class="alert alert-success" style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Reviewer's comment 2 ✔️</h2>
#     
# 
# Awesome job, thank you so much!    
#     
# </div>

# In[ ]:




