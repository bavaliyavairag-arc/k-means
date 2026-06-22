# Customer Segmentation using K-Means Clustering

## 📌 Task
Task-02 of Data Science Internship at Prodigy Infotech

## 📝 Description
This project uses K-Means Clustering algorithm to group
customers of a retail store based on their purchase history
like Annual Income and Spending Score.

## 📊 Dataset
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- File: Mall_Customers.csv

## 🛠️ Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📁 Files in this Repository
- kmeans.py → Main Python code
- Mall_Customers.csv → Dataset
- customer_segments.csv → Output with cluster labels

## 🔢 Steps Performed
1. Imported Libraries
2. Loaded Dataset
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Data Preprocessing (StandardScaler)
6. Finding Optimal K (Elbow Method)
7. Training K-Means Model
8. Visualizing Clusters
9. Analyzing Cluster Results
10. Saving Output

## 📈 Results
The customers were divided into 5 clusters:
- Cluster 0 → High Income, High Spending (Target Customers)
- Cluster 1 → Low Income, Low Spending (Budget Customers)
- Cluster 2 → High Income, Low Spending (Careful Customers)
- Cluster 3 → Low Income, High Spending (Impulsive Customers)
- Cluster 4 → Average Income, Average Spending (Normal Customers)

## 👨‍💻 Author
- Name: Vairag
- Internship: Prodigy Infotech
- Task: 02
