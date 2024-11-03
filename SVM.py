import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

# Generate synthetic data for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)  # Adjust parameters as needed
svm.fit(X)

# Predict outliers/anomalies
y_pred = svm.predict(X)

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, edgecolors='k')

plt.title('One-Class SVM for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/airaasyahira98/JIE43203./refs/heads/main/ifood_df.csv')

#check missing data
a = data.isnull().any()
b = data.isnull().sum()
print(a)
print()
print(b)

data.shape

data.duplicated().sum()  #count duplicated bcs no col that unique

data.drop_duplicates(inplace=True)  # Use drop_duplicates() to remove duplicates in-place

data.duplicated().sum()  #count duplicated bcs no col that unique

# handle missing data (delete column in df)
newdata = pd.DataFrame(data)
del newdata['AcceptedCmp1']
del newdata['AcceptedCmp2']
del newdata['AcceptedCmp3']
del newdata['AcceptedCmp4']
del newdata['AcceptedCmp5']
del newdata['marital_Divorced']
del newdata['marital_Married']
del newdata['marital_Single']
del newdata['marital_Together']
del newdata['marital_Widow']
del newdata['education_Basic']
del newdata['education_Graduation']
del newdata['education_Master']
del newdata['education_PhD']
del newdata['MntWines']
del newdata['MntFruits']
del newdata['MntMeatProducts']
del newdata['MntFishProducts']
del newdata['MntSweetProducts']
del newdata['MntGoldProds']
del newdata['Customer_Days']
del newdata['education_2n Cycle']
del newdata['Z_CostContact']
del newdata['Z_Revenue']

# check again the null value to ensure it already drop the column
newdata.isnull().any()

newdata.shape

#Creating subplot of each column with its own scale
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, axs = plt.subplots(1, len(newdata.columns), figsize=(20,10))

# detecting outliers in every column in dataset
for i, ax in enumerate(axs.flat):
    ax.boxplot(newdata.iloc[:,i], flierprops=red_circle)
    ax.set_title(newdata.columns[i], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
