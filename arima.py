import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.express as px
import joblib


# Load Data
df = pd.read_csv("data/Quantity_data.csv")
df_raw = df.copy()

# Data Cleaning
def clean_data(df):
    df["weekend_date"] = df["weekend_date"].astype(str).str[:10]  
    df["weekend_date"] = pd.to_datetime(df["weekend_date"])  
    df = df.sort_values(by="weekend_date")  
    df.set_index("weekend_date", inplace=True)  
    return df

df = clean_data(df)
df_graph = df.copy()

# Load Pre-trained ARIMA Model
arima_model_fit = joblib.load("ARIMA_MODELdel.pkl")

df = df[["quantity"]]
df = pd.DataFrame(df.groupby(df.index)["quantity"].sum())


# Feature Engineering
def feature_engineering(df):
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.astype("float") 

    df['lag_1'] = df['quantity'].shift(1)
    df['lag_2'] = df['quantity'].shift(2)
    df['lag_3'] = df['quantity'].shift(3)

    df["rolling_mean_4"] = df["quantity"].rolling(window=4).mean()
    df["rolling_std_4"] = df["quantity"].rolling(window=4).std()   

    return df

df = feature_engineering(df)
df.dropna(inplace=True)


def outliars_removal(df,col):
    Q1 = df["quantity"].quantile(.25)
    Q3 = df["quantity"].quantile(.75)
    IQR = Q3 - Q1

    lower_limit = Q1 + 1.5 * IQR
    upper_limit = Q3 - 1.5* IQR

    df[col] = df[col].clip(lower_limit,upper_limit)

    return df

df = outliars_removal(df,"quantity")

features = ['day', 'month', 'year', 'week', 'lag_1', 'lag_2', 'lag_3',
       'rolling_mean_4', 'rolling_std_4']
target = "quantity"

# Train-Test Split
split_date = pd.Timestamp("2024-06-01")
train = df[df.index < split_date]
test = df[df.index >= split_date]


# Validation
arima_Validation = arima_model_fit.forecast(steps=len(test), exog=test[features])

# Sidebar - Calendar for Future Predictions
st.sidebar.header("📅 Forecast Future Sales")
num_weeks = st.sidebar.slider("Select weeks to predict:", 1, 50, 10)

# Title
st.title("🔮 Time Series Forecasting Dashboard")

# Display DataFrame at Top
st.subheader("📊 Raw Data")
st.dataframe(df_raw)


# Load ARIMA Predictions 
test['Validation_data'] = np.array(arima_Validation)
future_dates = pd.date_range(start=df.index[-1], periods=num_weeks+1, freq='W')[1:]

# Forcasting data
future_exog = pd.DataFrame({
    "day": future_dates.day,
    "month": future_dates.month,
    "year": future_dates.year,
    "week": future_dates.isocalendar().week
})


future_exog["lag_1"] = df[target].iloc[-1]
future_exog["lag_2"] = df[target].iloc[-2]
future_exog["lag_3"] = df[target].iloc[-3]

future_exog["rolling_mean_4"] = df[target].iloc[-4:].mean()
future_exog["rolling_std_4"] = df[target].iloc[-4:].std()

# Ensure column order
future_exog = future_exog[features]

future_exog.index = pd.to_datetime(future_exog.index)
future_exog["week"] = future_exog["week"].astype("float")

arima_forecast_values = arima_model_fit.forecast(steps=len(future_exog), exog=future_exog)

# Create Plotly Graph
st.subheader("📈 ARIMA Forecasting Visualization")

fig = go.Figure()

fig.add_trace(go.Scatter(x=train.index, y=train['quantity'], 
                         mode='lines', name='Training Data',
                         line=dict(color='blue')))

fig.add_trace(go.Scatter(x=test.index, y=test['quantity'], 
                         mode='lines', name='Validation Data',
                         line=dict(color='green')))

fig.add_trace(go.Scatter(x=test.index, y=test['Validation_data'], 
                         mode='lines', name='Validation Forecast',
                         line=dict(color='orange', dash='dot')))

fig.add_trace(go.Scatter(x=future_dates, y=arima_forecast_values, 
                         mode='lines', name='Future Prediction',
                         line=dict(color='red', dash='dash')))

fig.update_layout(title="📊 ARIMA Forecasting",
                  xaxis_title="Date",
                  yaxis_title="Quantity",
                  template="plotly_dark",
                  legend=dict(x=0, y=1),
                  height=600,
                  width=1000)

st.plotly_chart(fig)

# Button to Download Forecast Data
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Quantity': arima_forecast_values})
csv = forecast_df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="📥 Download Forecast Data",
    data=csv,
    file_name="sales_forecast.csv",
    mime="text/csv"
)
st.subheader("📊 EDA - Qunatity Treds, Brand, Category & Sub-Category Trends")


# Create a 2-column layout
col1, col2 = st.columns(2)

# Histogram with KDE
with col1:
    # Create a histogram with KDE (Kernel Density Estimate) using Plotly Express
    fig1 = px.histogram(
        df_graph, 
        x="quantity", 
        nbins=50, 
        marginal="violin",  # Adds KDE-like visualization
        opacity=0.7, 
        color_discrete_sequence=["royalblue"]
    )

    # Update layout for better readability
    fig1.update_layout(
        title_text="Quantity Distribution with KDE",
        xaxis_title="Quantity",
        yaxis_title="Density",
        bargap=0.1
    )

    # Display plot in Streamlit
    st.plotly_chart(fig1, use_container_width=True)

# Boxplot
with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=df_graph["quantity"], marker_color="crimson", boxmean=True))
    fig2.update_layout(title_text="Quantity Boxplot", yaxis_title="Quantity")
    st.plotly_chart(fig2, use_container_width=True)

# Create two equal-sized columns with spacing
col1, col2 = st.columns(2, gap="medium")

# Brand Distribution Chart
with col1:
    st.markdown('<p style="font-size:16px; font-weight:bold;">🔍 Brand-wise Quantity</p>', unsafe_allow_html=True)

    # Get top 10 brands by quantity
    brand_counts = df_graph["brand"].value_counts().nlargest(10).reset_index()
    brand_counts.columns = ["Brand", "Quantity"]

    # Create an interactive pie chart with fixed size
    fig1 = px.pie(
        brand_counts,
        names="Brand",
        values="Quantity",
        color_discrete_sequence=px.colors.qualitative.Bold,  # Bright color scheme
        hole=0.4,  # Donut chart effect
    )

    fig1.update_layout(height=350, width=350)

    st.plotly_chart(fig1, use_container_width=True)

# Channel Distribution Chart
with col2:
    st.markdown('<p style="font-size:16px; font-weight:bold;">📡 Channel-wise Quantity</p>', unsafe_allow_html=True)

    # Aggregate sales by channel
    channel_counts = df_graph["channel"].value_counts().reset_index()
    channel_counts.columns = ["Channel", "Quantity"]

    # Create an interactive pie chart with fixed size
    fig2 = px.pie(
        channel_counts,
        names="Channel",
        values="Quantity",
        color_discrete_sequence=px.colors.qualitative.Vivid,  # Vibrant color scheme
        hole=0.4,
    )

    fig2.update_layout(height=350, width=350)

    st.plotly_chart(fig2, use_container_width=True)

# Create two columns for better layout
col1, col2 = st.columns(2, gap="medium")

# Category Sales Trends (Horizontal Bar Chart)
with col1:
    st.markdown('<p style="font-size:16px; font-weight:bold;">📌 Category Quantity Trends</p>', unsafe_allow_html=True)

    # Aggregate sales by category
    category_trend = df_graph.groupby("category")["quantity"].sum().reset_index()

    # Create an interactive horizontal bar chart
    fig3 = px.bar(
        category_trend,
        x="quantity",
        y="category",
        orientation="h",
        color="category",
        labels={"quantity": "Total Sales", "category": "Category"},
        color_discrete_sequence=px.colors.qualitative.Prism,  # Bright complementary colors
    )

    fig3.update_layout(height=400, width=500)

    st.plotly_chart(fig3, use_container_width=True)

# Sub-category Sales Trends (Horizontal Bar Chart)
with col2:
    st.markdown('<p style="font-size:16px; font-weight:bold;">📈 Subcategory Quantity Trends</p>', unsafe_allow_html=True)

    # Aggregate sales by subcategory and sort
    subcategory_trend = df_graph.groupby("sub_category")["quantity"].sum().reset_index()
    subcategory_trend = subcategory_trend.sort_values(by="quantity", ascending=True)

    # Create an interactive horizontal bar chart
    fig4 = px.bar(
        subcategory_trend,
        x="quantity",
        y="sub_category",
        orientation="h",
        color="sub_category",
        labels={"quantity": "Total Sales", "sub_category": "Subcategory"},
        color_discrete_sequence=px.colors.qualitative.Dark24,  # Bold dark tones for contrast
    )

    fig4.update_layout(height=400, width=500)

    st.plotly_chart(fig4, use_container_width=True)

# Yearly Quantity Trends
st.write("### 📊 Yearly Quantity Trends")

# Ensure 'date' column is in datetime format if not already
df["year"] = df.index.year
df["month"] = df.index.month

# Convert month numbers to names (e.g., 1 → Jan, 2 → Feb)
df["month_name"] = df.index.strftime("%b")

# Aggregate Quantity by year and month
monthly_Quantity = df.groupby(["year", "month_name"])["quantity"].sum().reset_index()

# Sort the months correctly (since strings are sorted alphabetically by default)
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_Quantity["month_name"] = pd.Categorical(monthly_Quantity["month_name"], categories=month_order, ordered=True)
monthly_Quantity = monthly_Quantity.sort_values("month_name")

# Create interactive multi-line chart
fig5 = px.line(
    monthly_Quantity,
    x="month_name",
    y="quantity",
    color="year",
    markers=True,
    title="Monthly Quantity Trends for Each Year",
    labels={"quantity": "Total Quantity", "month_name": "Month", "year": "Year"},
    color_discrete_sequence=px.colors.qualitative.Set1,  # Bright consistent colors
)

# Increase line thickness and marker size for better visibility
fig5.update_traces(line=dict(width=4), marker=dict(size=8))

# Show the interactive plot in Streamlit
st.plotly_chart(fig5, use_container_width=True)

# Insights Section
st.header("📈 Insights")
st.markdown("""
- The **distribution of quantity** is right-skewed, as observed in the **distplot**, while the **boxplot highlights the presence of significant outliers.**

- **Outlier Handling**: Given the presence of extreme values, further **investigation and possible treatment** of outliers may help improve predictive modeling accuracy.

- After **cleaning and aggregating** the data, we observed that the **minimum quantity sold is 203, the maximum is 11,816, and the standard deviation is 2,094**, indicating high variability in the dataset.

- The **trend analysis** of quantity sold shows a **gradual upward movement**, with **no clear seasonality** detected in the data.

- **Brand B1 dominates sales**, contributing to **70% of the total quantity sold**, while **Channel 2 emerges as the top-performing sales channel.**

- Within the **category segmentation**, face **products account for over 95% of total sales**, with **face cleansers and face serums being the most popular items.**
""")

st.markdown("                                                  BY **TANUJ CHAUHAN**")


st.sidebar.write("\n🔹Evaluation Scores🔹"
                    "\n\n **→  Weekly** RMSE: 714"
                    "\n\n **→  Monthly** RMSE: 851"
                    )
