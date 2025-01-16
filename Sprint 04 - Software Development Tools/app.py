import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('vehicles_us.csv')

# Header
st.header('GSK Garage')

# Sidebar for filtering options
st.sidebar.header('Filter Cars by Year, Fuel Type, and Price')

# Year selection with an 'All' option
year_options = df['model_year'].dropna().unique()
year_options_sorted = sorted(year_options)
year_options_sorted.insert(0, 'All')
selected_years = st.sidebar.multiselect('Select Year(s)', year_options_sorted, default='All')

# Fuel type selection with an 'All' option
fuel_options = df['fuel'].dropna().unique()
fuel_options_sorted = sorted(fuel_options)
fuel_options_sorted.insert(0, 'All')
selected_fuels = st.sidebar.multiselect('Select Fuel Type(s)', fuel_options_sorted, default='All')

# Update selection logic to handle 'All' option
if 'All' in selected_years:
    selected_years = year_options
if 'All' in selected_fuels:
    selected_fuels = fuel_options

# Price range selection
min_price = 0
max_price = int(df['price'].max())
selected_price_range = st.sidebar.slider('Select Price Range', min_price, max_price, (min_price, max_price))

# Filter the dataframe based on selections
filtered_df = df[(df['model_year'].isin(selected_years)) &
                 (df['fuel'].isin(selected_fuels)) &
                 (df['price'] >= selected_price_range[0]) &
                 (df['price'] <= selected_price_range[1])]

# Display the filtered dataframe
st.subheader('Filtered Cars')
if not filtered_df.empty:
    st.write(filtered_df)
    # Display some statistics about the filtered data
    st.write(f"Number of cars: {filtered_df.shape[0]}")
    st.write(f"Average price: ${filtered_df['price'].mean():,.2f}")
else:
    st.write('No cars match the selected criteria.')

# Histogram for 'price'
st.subheader('Price Distribution')
fig_histogram_price = px.histogram(filtered_df, x='price', title='Price Distribution', nbins=30)
fig_histogram_price.update_layout(xaxis_title='Price (USD)', yaxis_title='Count')
st.plotly_chart(fig_histogram_price)

# Histogram for 'odometer'
st.subheader('Odometer Distribution')
fig_histogram_odometer = px.histogram(filtered_df, x='odometer', title='Odometer Distribution', nbins=30)
fig_histogram_odometer.update_layout(xaxis_title='Odometer (miles)', yaxis_title='Count')
st.plotly_chart(fig_histogram_odometer)

# Histogram for 'model_year'
st.subheader('Model Year Distribution')
fig_histogram_model_year = px.histogram(filtered_df, x='model_year', title='Model Year Distribution', nbins=20)
fig_histogram_model_year.update_layout(xaxis_title='Model Year', yaxis_title='Count')
st.plotly_chart(fig_histogram_model_year)

# Scatter plot of 'price' vs 'odometer'
st.subheader('Price vs Odometer')
fig_scatter_price_odometer = px.scatter(filtered_df, x='odometer', y='price', title='Price vs Odometer')
fig_scatter_price_odometer.update_layout(xaxis_title='Odometer (miles)', yaxis_title='Price (USD)')
show_scatter_price_odometer = st.checkbox('Show Scatter Plot: Price vs Odometer')

if show_scatter_price_odometer:
    st.plotly_chart(fig_scatter_price_odometer)

# Scatter plot of 'price' vs 'model_year'
st.subheader('Price vs Model Year')
fig_scatter_price_model_year = px.scatter(filtered_df, x='model_year', y='price', title='Price vs Model Year')
fig_scatter_price_model_year.update_layout(xaxis_title='Model Year', yaxis_title='Price (USD)')
show_scatter_price_model_year = st.checkbox('Show Scatter Plot: Price vs Model Year')

if show_scatter_price_model_year:
    st.plotly_chart(fig_scatter_price_model_year)

# Scatter plot of 'odometer' vs 'model_year'
st.subheader('Odometer vs Model Year')
fig_scatter_odometer_model_year = px.scatter(filtered_df, x='model_year', y='odometer', title='Odometer vs Model Year')
fig_scatter_odometer_model_year.update_layout(xaxis_title='Model Year', yaxis_title='Odometer (miles)')
show_scatter_odometer_model_year = st.checkbox('Show Scatter Plot: Odometer vs Model Year')

if show_scatter_odometer_model_year:
    st.plotly_chart(fig_scatter_odometer_model_year)