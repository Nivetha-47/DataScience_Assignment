import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Step 1: Load the datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Step 2: Preprocess the data

# Convert dates to datetime format
customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])
transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])

# Step 3: Merge the dataframes
merged_data = transactions.merge(customers, on="CustomerID", how="left") \
    .merge(products, on="ProductID", how="left")

# Step 4: Create customer profiles

# Aggregate data to create customer profiles
customer_profiles = merged_data.groupby('CustomerID').agg(
    AvgSpending=('TotalValue', 'mean'),
    TotalSpending=('TotalValue', 'sum'),
    TotalTransactions=('TransactionID', 'count'),
    PreferredCategory=('Category', lambda x: x.mode()[0])
).reset_index()

# Normalize the numerical features (AvgSpending, TotalSpending, TotalTransactions)
customer_profiles["AvgSpending"] = customer_profiles["AvgSpending"] / customer_profiles["AvgSpending"].max()
customer_profiles["TotalSpending"] = customer_profiles["TotalSpending"] / customer_profiles["TotalSpending"].max()
customer_profiles["TotalTransactions"] = customer_profiles["TotalTransactions"] / customer_profiles["TotalTransactions"].max()

# Step 5: Calculate the similarity between customers

# Select features for similarity calculation
features = customer_profiles[["AvgSpending", "TotalSpending", "TotalTransactions"]]

# Calculate cosine similarity
similarity_matrix = cosine_similarity(features)

# Convert the similarity matrix to a DataFrame for easier handling
similarity_df = pd.DataFrame(
    similarity_matrix, 
    index=customer_profiles["CustomerID"], 
    columns=customer_profiles["CustomerID"]
)

# Step 6: Get the top 3 similar customers for each of the first 20 customers

def get_top_3_similar(customers_df, target_customer_id):
    similar_customers = customers_df.loc[target_customer_id].sort_values(ascending=False)
    return similar_customers.iloc[1:4].index.tolist(), similar_customers.iloc[1:4].values.tolist()

# Generate recommendations for customers C0001 to C0020
lookalike_results = []

for customer_id in customer_profiles["CustomerID"][:20]:
    similar_customers, scores = get_top_3_similar(similarity_df, customer_id)
    for similar_customer, score in zip(similar_customers, scores):
        lookalike_results.append([customer_id, similar_customer, score])

# Step 7: Save the results to Lookalike.csv

output_filename = "Lookalike.csv"

with open(output_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["CustomerID", "SimilarCustomerID", "SimilarityScore"])  # Column headers
    writer.writerows(lookalike_results)

print(f"Lookalike recommendations have been saved to {output_filename}.")
