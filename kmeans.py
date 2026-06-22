# STEP 1 - Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# STEP 2 - Load Dataset
df = pd.read_csv("Mall_Customers.csv")
print("Dataset loaded!")
print(df.head())

# STEP 3 - EDA
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nInfo:")
print(df.describe())

# STEP 4 - Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# STEP 5 - Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 6 - Find best K using Elbow Method
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(1,11), inertia, 'o-', color='red')
plt.title('Elbow Method - Find Best K')
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# STEP 7 - Train Final Model with K=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# STEP 8 - Plot the Clusters
plt.figure(figsize=(8,6))
colors = ['red','blue','green','orange','purple']
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'],
                cluster_data['Spending Score (1-100)'],
                c=colors[i], label=f'Cluster {i}', s=60)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# STEP 9 - Print Cluster Summary
print("\nCluster Summary:")
print(df.groupby('Cluster')[['Annual Income (k$)',
      'Spending Score (1-100)']].mean().round(2))

# STEP 10 - Save Results
df.to_csv('customer_segments.csv', index=False)
print("\nDone! Results saved to customer_segments.csv")