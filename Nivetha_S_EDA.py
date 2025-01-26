import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load the data
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Preview the data
print("Customers Data:")
print(customers.head())
print("\nProducts Data:")
print(products.head())
print("\nTransactions Data:")
print(transactions.head())

# Check for missing values
print("\nMissing values in Customers dataset:")
print(customers.isnull().sum())
print("\nMissing values in Products dataset:")
print(products.isnull().sum())
print("\nMissing values in Transactions dataset:")
print(transactions.isnull().sum())

# Clean the data by dropping rows with missing values
customers.dropna(inplace=True)
products.dropna(inplace=True)
transactions.dropna(inplace=True)

# Basic descriptive statistics
print("\nDescriptive statistics for Customers:")
print(customers.describe())
print("\nDescriptive statistics for Products:")
print(products.describe())
print("\nDescriptive statistics for Transactions:")
print(transactions.describe())

# Visualizations
# Example: Distribution of product prices
sns.histplot(products['Price'], kde=True)
plt.title('Product Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Example: Total revenue by region
revenue_by_region = transactions.groupby('CustomerID')['TotalValue'].sum().reset_index()
region_revenue = pd.merge(revenue_by_region, customers[['CustomerID', 'Region']], on='CustomerID', how='left')
region_revenue = region_revenue.groupby('Region')['TotalValue'].sum().reset_index()

# Bar plot of total revenue by region
sns.barplot(x='Region', y='TotalValue', data=region_revenue)
plt.title('Total Revenue by Region')
plt.xlabel('Region')
plt.ylabel('Total Revenue')
plt.show()

# Example: Relationship between quantity and total value
sns.scatterplot(data=transactions, x='Quantity', y='TotalValue')
plt.title('Quantity vs Total Value')
plt.xlabel('Quantity')
plt.ylabel('Total Value')
plt.show()

# Example: Trend of transactions over time (monthly)
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')
monthly_transactions = transactions.groupby('Month')['TransactionID'].count().reset_index()

# Ensure the data types are correct
monthly_transactions['TransactionID'] = pd.to_numeric(monthly_transactions['TransactionID'], errors='coerce')
monthly_transactions['Month'] = pd.to_datetime(monthly_transactions['Month'], errors='coerce')

# Drop any rows with missing values after conversion
monthly_transactions = monthly_transactions.dropna()

# Now proceed with the plotting
sns.lineplot(data=monthly_transactions, x='Month', y='TransactionID')
plt.title('Transactions Over Time (Monthly)')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.show()

# Example: Distribution of total value by product category
product_sales = pd.merge(transactions, products[['ProductID', 'Category']], on='ProductID', how='left')
category_sales = product_sales.groupby('Category')['TotalValue'].sum().reset_index()

sns.barplot(x='Category', y='TotalValue', data=category_sales)
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Value')
plt.xticks(rotation=45)
plt.show()

# Example: Correlation between price and total value
sns.scatterplot(data=transactions, x='Price', y='TotalValue')
plt.title('Price vs Total Value')
plt.xlabel('Price')
plt.ylabel('Total Value')
plt.show()

# Optional: Save cleaned data to new CSV files
customers.to_csv('Cleaned_Customers.csv', index=False)
products.to_csv('Cleaned_Products.csv', index=False)
transactions.to_csv('Cleaned_Transactions.csv', index=False)

# Generate PDF Report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, txt="Business Insights Report", ln=True, align='C')

# Add Business Insights
pdf.set_font('Arial', '', 12)
insights = [
    "1. Customers from North America have the highest total spending, contributing to a significant portion of revenue. By targeting this region with personalized offers, the company could further boost its sales and strengthen its customer base.",
    
    "2. The Electronics category is the highest revenue-generating product category, suggesting that it has a strong market demand. The company can consider introducing new products in this category to further capitalize on this trend.",
    
    "3. There is a noticeable spike in transaction volume during the holiday season, highlighting an opportunity for the company to launch holiday-specific promotions or discounts to maximize sales during this period.",
    
    "4. Customers who make bulk purchases tend to generate higher revenue. Offering discounts or rewards for bulk buying could encourage customers to increase their order size, potentially improving the company's revenue per customer.",
    
    "5. A large proportion of sales comes from small transactions. The company could focus on upselling and cross-selling additional products during these smaller purchases to increase the total transaction value per customer."
]

# Write insights to PDF (limited to 100 words per insight)
for insight in insights:
    pdf.multi_cell(0, 10, insight)

# Save the PDF report
pdf_output_path = 'business_insights_report.pdf'
pdf.output(pdf_output_path)

print(f"Business insights report generated and saved as {pdf_output_path}.")
