import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Step 1: Load the datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Step 2: Preprocess the data

# Convert dates to datetime
transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])

# Aggregate transaction data for each customer
customer_transactions = transactions.groupby("CustomerID").agg(
    TotalSpending=('TotalValue', 'sum'),
    TotalTransactions=('TransactionID', 'count'),
    AvgSpending=('TotalValue', 'mean'),
).reset_index()

# Merge customer profile data with aggregated transaction data
customer_data = customers.merge(customer_transactions, on="CustomerID", how="left")

# Step 3: Feature Engineering (Normalization)
features = customer_data[['TotalSpending', 'TotalTransactions', 'AvgSpending']]

# Check for missing values
if features.isnull().any().any():
    print("There are missing values in the dataset.")
    # Handle missing values - Option 1: Remove rows with missing values from both features and customer_data
    customer_data = customer_data.dropna(subset=['TotalSpending', 'TotalTransactions', 'AvgSpending'])
    features = features.dropna()  # Remove missing values from the feature set

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Choose Clustering Algorithm (KMeans)
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust n_clusters based on your preference (between 2 and 10)
customer_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 5: Clustering Metrics (DB Index)
db_index = davies_bouldin_score(scaled_features, customer_data['Cluster'])

# Step 6: Visualize the clusters using PCA (2D plot)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Create a DataFrame with PCA components and cluster labels
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = customer_data['Cluster']

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set1', s=100, marker='o')
plt.title('Customer Segmentation Using KMeans (PCA Visualization)', fontsize=16)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot as a PNG file (so it can be embedded into the PDF)
plt.savefig('clustering_plot.png')
plt.close()

# Step 7: Create a PDF to store the results using fpdf
pdf_filename = "Customer_Segmentation_Report.pdf"
pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, "Customer Segmentation Report", ln=True, align='C')

# Add DB Index and number of clusters
pdf.set_font("Arial", '', 12)
pdf.ln(10)
pdf.cell(200, 10, f"DB Index: {db_index}", ln=True)
pdf.cell(200, 10, f"Number of Clusters: {len(customer_data['Cluster'].unique())}", ln=True)

# Add the image (clustering plot)
pdf.ln(10)
pdf.image('clustering_plot.png', x=30, y=pdf.get_y(), w=150)

# Save the PDF
pdf.output(pdf_filename)

# Step 8: Save results to CSV
customer_data.to_csv("Clustered_Customers.csv", index=False)

print(f"Clustering report saved to {pdf_filename}")
