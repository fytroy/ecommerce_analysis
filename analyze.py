import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# --- Configuration and Setup ---
# Set plot style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Default figure size
plt.rcParams['font.size'] = 12 # Default font size

print("Libraries imported successfully!")
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Location: Nairobi, Kenya")

# --- Step 1: Data Simulation ---
# This function will generate a synthetic E-commerce dataset.
# In a real project, you would replace this with loading your actual data.

def generate_ecommerce_data(num_records=250000, start_date=datetime(2023, 1, 1), end_date=datetime(2024, 12, 31)):
    """
    Generates a synthetic e-commerce transaction dataset.
    """
    product_categories = ['Electronics', 'Apparel', 'Home & Kitchen', 'Books', 'Beauty & Health', 'Sports & Outdoors']
    product_names = {
        'Electronics': ['Smartphone X', 'Laptop Pro', 'Smartwatch', 'Headphones', 'Tablet'],
        'Apparel': ['T-Shirt Basic', 'Jeans Slim Fit', 'Winter Jacket', 'Sneakers', 'Dress Casual'],
        'Home & Kitchen': ['Coffee Maker', 'Blender', 'Cookware Set', 'Vacuum Cleaner', 'Smart Speaker'],
        'Books': ['Fiction Bestseller', 'Self-Help Guide', 'Cooking Book', 'Science Fiction Novel'],
        'Beauty & Health': ['Face Cream', 'Shampoo', 'Protein Powder', 'Electric Toothbrush'],
        'Sports & Outdoors': ['Yoga Mat', 'Dumbbell Set', 'Camping Tent', 'Running Shoes']
    }
    shipping_cities = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 'Malindi', 'Karen']
    payment_methods = ['Credit Card', 'M-Pesa', 'PayPal', 'Bank Transfer', 'Cash on Delivery']

    data = []
    # Fewer unique customers than records to simulate repeat purchases
    customer_ids = [f'CUST_{i:05d}' for i in range(1, num_records // 10 + 1)]

    for i in range(num_records):
        order_id = f'ORD_{i:07d}'
        customer_id = random.choice(customer_ids)
        order_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

        # Simulate some seasonality and price variance
        if order_date.month in [11, 12]: # Higher sales in Nov/Dec
            quantity = random.randint(1, 5)
            base_price = random.uniform(50, 500)
        elif order_date.month in [1, 2]: # Lower sales in Jan/Feb
            quantity = random.randint(1, 3)
            base_price = random.uniform(30, 300)
        else:
            quantity = random.randint(1, 4)
            base_price = random.uniform(40, 400)

        category = random.choice(product_categories)
        product_name = random.choice(product_names[category])
        price = round(base_price * random.uniform(0.9, 1.1), 2) # Add some price variance

        shipping_city = random.choice(shipping_cities)
        payment_method = random.choice(payment_methods)

        data.append([order_id, customer_id, product_name, category, price, quantity, order_date, shipping_city, payment_method])

    df = pd.DataFrame(data, columns=[
        'OrderID', 'CustomerID', 'ProductName', 'Category', 'Price', 'Quantity',
        'OrderDate', 'ShippingCity', 'PaymentMethod'
    ])
    return df

print("\n--- Step 1: Data Simulation ---")
print("Generating synthetic E-commerce data (approx. 250,000 records)... This may take a moment.")
df = generate_ecommerce_data(num_records=250000)
print("Data generation complete!")
print(f"Initial Dataset shape: {df.shape}")
print("First 5 rows of the generated dataset:")
print(df.head())

# --- Step 2: Data Loading & Initial Inspection ---
# (Partially covered by simulation, but crucial for real datasets)

print("\n--- Step 2: Initial Data Inspection ---")
print("Dataset Information:")
df.info()

print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

print("\nNumber of unique values in key categorical columns:")
for col in ['Category', 'ShippingCity', 'PaymentMethod', 'CustomerID', 'OrderID']:
    print(f"- {col}: {df[col].nunique()} unique values")

print("\nChecking for missing values:")
print(df.isnull().sum())

# --- Step 3: Data Cleaning ---

print("\n--- Step 3: Data Cleaning ---")

# 1. Ensure 'OrderDate' is in datetime format
# This is usually handled by generate_ecommerce_data, but good practice for real data
df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
print(f"OrderDate Dtype after conversion: {df['OrderDate'].dtype}")
# Handle cases where conversion failed (e.g., 'coerce' turned invalid dates to NaT)
if df['OrderDate'].isnull().any():
    initial_null_dates = df['OrderDate'].isnull().sum()
    df.dropna(subset=['OrderDate'], inplace=True)
    print(f"Removed {initial_null_dates} rows due to invalid OrderDate after conversion.")

# 2. Handle missing CustomerID (if any).
# For synthetic data, we designed it not to have missing CustomerIDs.
# In a real scenario, if CustomerID is critical and missing:
if df['CustomerID'].isnull().any():
    print("Warning: Missing CustomerIDs found. Assigning unique anonymous IDs.")
    # Assign unique anonymous ID for missing CustomerIDs
    missing_cust_count = df['CustomerID'].isnull().sum()
    df.loc[df['CustomerID'].isnull(), 'CustomerID'] = [f'ANON_CUST_{i}' for i in range(missing_cust_count)]
else:
    print("No missing CustomerIDs found (as expected from simulation).")

# 3. Identify and correct inconsistencies in categorical spellings (e.g., leading/trailing spaces, inconsistent casing)
print("Standardizing categorical columns (removing extra spaces, consistent casing)...")
for col in ['Category', 'PaymentMethod', 'ShippingCity', 'ProductName']:
    if df[col].dtype == 'object': # Only apply to string columns
        df[col] = df[col].astype(str).str.strip().str.capitalize()
print("Categorical standardization complete.")


# 4. Remove duplicate transaction entries
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
duplicates_removed = initial_rows - df.shape[0]
print(f"Removed {duplicates_removed} duplicate rows.")

# 5. Ensure Price and Quantity are numeric and handle potential errors
# This is crucial for calculations and robust against real-world dirty data
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

# Drop rows where Price or Quantity became NaN after conversion (invalid numeric data)
initial_invalid_numeric_rows = df[(df['Price'].isnull()) | (df['Quantity'].isnull())].shape[0]
df.dropna(subset=['Price', 'Quantity'], inplace=True)
print(f"Removed {initial_invalid_numeric_rows} rows due to invalid Price or Quantity values.")

# 6. Remove transactions with zero or negative price/quantity (often data entry errors)
initial_zero_negative_rows = df[(df['Price'] <= 0) | (df['Quantity'] <= 0)].shape[0]
df = df[(df['Price'] > 0) & (df['Quantity'] > 0)]
print(f"Removed {initial_zero_negative_rows} rows with zero or negative Price/Quantity.")


# Final check for any remaining missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print(f"Final Dataset shape after cleaning: {df.shape}")
print("Data cleaning complete.")

# --- Step 4: Feature Engineering ---

print("\n--- Step 4: Feature Engineering ---")

# 1. Calculate TotalSales for each transaction
df['TotalSales'] = df['Price'] * df['Quantity']
print("Calculated 'TotalSales' column.")

# 2. Extract Year, Month, DayOfWeek, Hour from OrderDate
df['OrderYear'] = df['OrderDate'].dt.year
df['OrderMonth'] = df['OrderDate'].dt.month
df['OrderDayOfWeek'] = df['OrderDate'].dt.day_name() # Monday, Tuesday, etc.
df['OrderHour'] = df['OrderDate'].dt.hour
print("Extracted temporal features (Year, Month, DayOfWeek, Hour).")

# Create a customer summary for later RFM and general customer analysis
customer_summary = df.groupby('CustomerID').agg(
    first_purchase=('OrderDate', 'min'),
    last_purchase=('OrderDate', 'max'),
    total_orders=('OrderID', 'nunique'),
    total_spent=('TotalSales', 'sum')
).reset_index()

print("\nCustomer summary created for further segmentation.")
print(customer_summary.head())

print("\nFeature engineering complete.")

# --- Step 5: Exploratory Data Analysis (EDA) ---

print("\n--- Step 5: Exploratory Data Analysis (EDA) ---")
print("Generating visualizations... Please close each plot window to proceed.")

# 1. Sales Trends: Total sales over time (monthly)
print("\nAnalyzing Sales Trends (Monthly)...")
monthly_sales = df.set_index('OrderDate').resample('M')['TotalSales'].sum()

plt.figure(figsize=(14, 7))
monthly_sales.plot(kind='line', marker='o', linestyle='-')
plt.title('Monthly Total Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Sales (KES)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show() # Close this plot to continue

# Sales by Day of Week
print("Analyzing Sales Trends (Day of Week)...")
sales_by_day = df.groupby('OrderDayOfWeek')['TotalSales'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_day.index, y=sales_by_day.values, palette='viridis')
plt.title('Total Sales by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Total Sales (KES)', fontsize=14)
plt.tight_layout()
plt.show() # Close this plot to continue

# Sales by Hour of Day
print("Analyzing Sales Trends (Hour of Day)...")
sales_by_hour = df.groupby('OrderHour')['TotalSales'].sum()
plt.figure(figsize=(12, 6))
sns.barplot(x=sales_by_hour.index, y=sales_by_hour.values, palette='magma')
plt.title('Total Sales by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Total Sales (KES)', fontsize=14)
plt.xticks(range(0, 24)) # Ensure all hours are displayed
plt.tight_layout()
plt.show() # Close this plot to continue


# 2. Product Performance: Top N products by sales revenue and quantity
print("\nAnalyzing Product Performance (Top Products & Categories)...")
top_products_sales = df.groupby('ProductName')['TotalSales'].sum().nlargest(10)
top_products_quantity = df.groupby('ProductName')['Quantity'].sum().nlargest(10)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.barplot(x=top_products_sales.values, y=top_products_sales.index, palette='crest', ax=axes[0])
axes[0].set_title('Top 10 Products by Sales Revenue', fontsize=16)
axes[0].set_xlabel('Total Sales (KES)', fontsize=14)
axes[0].set_ylabel('Product Name', fontsize=14)

sns.barplot(x=top_products_quantity.values, y=top_products_quantity.index, palette='flare', ax=axes[1])
axes[1].set_title('Top 10 Products by Quantity Sold', fontsize=16)
axes[1].set_xlabel('Total Quantity Sold', fontsize=14)
axes[1].set_ylabel('Product Name', fontsize=14)

plt.tight_layout()
plt.show() # Close this plot to continue

# Sales by Category
sales_by_category = df.groupby('Category')['TotalSales'].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 7))
sns.barplot(x=sales_by_category.index, y=sales_by_category.values, palette='plasma')
plt.title('Total Sales by Product Category', fontsize=16)
plt.xlabel('Product Category', fontsize=14)
plt.ylabel('Total Sales (KES)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() # Close this plot to continue


# 3. Customer Analysis: Average Order Value & Spending Distribution
print("\nAnalyzing Customer Behavior (AOV & Spending Distribution)...")
# Average Order Value (AOV) per customer
overall_aov = df['TotalSales'].sum() / df['OrderID'].nunique()
print(f"Overall Average Order Value: KES {overall_aov:.2f}")

# Distribution of Total Spent per Customer
plt.figure(figsize=(10, 6))
sns.histplot(customer_summary['total_spent'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Total Spending per Customer', fontsize=16)
plt.xlabel('Total Spent (KES)', fontsize=14)
plt.ylabel('Number of Customers', fontsize=14)
# Limit x-axis to better visualize the main distribution, ignoring extreme outliers
plt.xlim(0, customer_summary['total_spent'].quantile(0.99))
plt.tight_layout()
plt.show() # Close this plot to continue


# 4. Geographic Analysis: Sales distribution by ShippingCity
print("\nAnalyzing Geographic Sales Distribution...")
sales_by_city = df.groupby('ShippingCity')['TotalSales'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_city.values, y=sales_by_city.index, palette='cividis')
plt.title('Total Sales by Shipping City', fontsize=16)
plt.xlabel('Total Sales (KES)', fontsize=14)
plt.ylabel('Shipping City', fontsize=14)
plt.tight_layout()
plt.show() # Close this plot to continue


# 5. Payment Method Analysis
print("\nAnalyzing Payment Method Popularity...")
sales_by_payment = df.groupby('PaymentMethod')['TotalSales'].sum().sort_values(ascending=False)
count_by_payment = df['PaymentMethod'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.barplot(x=sales_by_payment.index, y=sales_by_payment.values, palette='mako', ax=axes[0])
axes[0].set_title('Total Sales by Payment Method', fontsize=16)
axes[0].set_xlabel('Payment Method', fontsize=14)
axes[0].set_ylabel('Total Sales (KES)', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(x=count_by_payment.index, y=count_by_payment.values, palette='rocket', ax=axes[1])
axes[1].set_title('Number of Transactions by Payment Method', fontsize=16)
axes[1].set_xlabel('Payment Method', fontsize=14)
axes[1].set_ylabel('Number of Transactions', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show() # Close this plot to continue

print("\nEDA complete. Visualizations generated.")

# --- Step 6: RFM Analysis (Recency, Frequency, Monetary) ---

print("\n--- Step 6: Performing RFM Analysis ---")

# Define a snapshot date (the day after the last transaction in your dataset)
# Ensure the snapshot date is always after the latest order date
snapshot_date = df['OrderDate'].max() + timedelta(days=1)
print(f"Snapshot date for RFM calculation: {snapshot_date.strftime('%Y-%m-%d')}")

# Calculate Recency, Frequency, and Monetary for each customer
rfm = df.groupby('CustomerID').agg(
    Recency=('OrderDate', lambda date: (snapshot_date - date.max()).days), # Days since last purchase
    Frequency=('OrderID', 'nunique'),                                   # Number of unique orders
    Monetary=('TotalSales', 'sum')                                      # Total spent
).reset_index()

print("\nRFM scores calculated for each customer (first 5 rows):")
print(rfm.head())

# Segmenting RFM scores into quintiles (5 segments)
# R_Score: Lower recency (fewer days) is better, so rank descending
# F_Score, M_Score: Higher values are better, so rank ascending
print("\nSegmenting RFM scores into quintiles...")
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop') # Higher score for lower recency
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop') # Higher score for higher frequency
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop') # Higher score for higher monetary

# Convert scores to integer type for easier manipulation
rfm['R_Score'] = rfm['R_Score'].astype(int)
rfm['F_Score'] = rfm['F_Score'].astype(int)
rfm['M_Score'] = rfm['M_Score'].astype(int)


# Combine RFM scores to create an RFM Segment
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Define customer segments based on RFM scores
# This is a common heuristic; you might adjust based on business needs and insights
def rfm_segment(row):
    r = row['R_Score']
    f = row['F_Score']
    m = row['M_Score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions' # Bought recently, buy often, spend a lot
    elif r >= 4 and f >= 3:
        return 'Loyal Customers' # Buy recently, buy often
    elif r >= 3 and m >= 3:
        return 'Potential Loyalist' # Recent, good amount, but not frequent
    elif r >= 3 and f >= 2:
        return 'Promising' # Recent, fairly frequent
    elif r >= 2 and f >= 2:
        return 'Customers Needing Attention' # Below average recency, frequency
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Lost Customers' # Old, infrequent, low spending
    else:
        return 'Others' # Catch-all for less distinct segments

rfm['Customer_Segment'] = rfm.apply(rfm_segment, axis=1)

print("\nRFM scores segmented and customer segments assigned (first 5 rows):")
print(rfm.head())
print("\nDistribution of Customer Segments:")
print(rfm['Customer_Segment'].value_counts())

# Visualize Customer Segments
plt.figure(figsize=(12, 7))
sns.countplot(y='Customer_Segment', data=rfm, order=rfm['Customer_Segment'].value_counts().index, palette='viridis')
plt.title('Distribution of Customer Segments (RFM)', fontsize=16)
plt.xlabel('Number of Customers', fontsize=14)
plt.ylabel('Customer Segment', fontsize=14)
plt.tight_layout()
plt.show() # Close this plot to continue

# Analyze average R, F, M values per segment
segment_means = rfm.groupby('Customer_Segment')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(by='Monetary', ascending=False)
print("\nAverage RFM values per Customer Segment:")
print(segment_means.round(2)) # Round for cleaner output

print("\nRFM Analysis complete.")
print("\n--- Project Execution Finished ---")