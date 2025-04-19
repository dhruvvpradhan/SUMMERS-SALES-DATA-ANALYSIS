import pandas as pd

try:
    df = pd.read_csv('Computed insight - Success of active sellers.csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'Computed insight - Success of active sellers.csv' not found.")
except pd.errors.ParserError:
    print("Error: Unable to parse the CSV file. Please check the file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print(df.dtypes)

print(df.isnull().sum())


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(df['totalunitssold'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Total Units Sold')
plt.ylabel('Frequency')
plt.title('Distribution of Total Units Sold')


plt.subplot(2, 2, 2)
plt.hist(df['rating'], bins=20, color='lightcoral', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Rating')


plt.subplot(2, 2, 3)
plt.hist(df['averagediscount'], bins=20, color='lightgreen', edgecolor='black')
plt.xlabel('Average Discount')
plt.ylabel('Frequency')
plt.title('Distribution of Average Discount')

plt.tight_layout()
plt.show()

print("Shape of the DataFrame:", df.shape)

print(df['merchantid'].nunique())
print(df['merchantid'].value_counts().head(10))

for col in ['totalurgencycount', 'urgencytextrate']:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)


df = df.drop_duplicates(keep='first')


print(df.isnull().sum())
print(df.shape)
import matplotlib.pyplot as plt
import seaborn as sns


print(df.describe())

numeric_cols = df.select_dtypes(include=['number']).columns # Select only numeric columns
correlation_matrix = df[numeric_cols].corr() # Calculate correlation for numeric columns only
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['totalunitssold'], df['averagediscount'], color='blue', alpha=0.5)
plt.xlabel('Total Units Sold')
plt.ylabel('Average Discount')
plt.title('Total Units Sold vs. Average Discount')


plt.subplot(1, 2, 2)
plt.scatter(df['totalunitssold'], df['rating'], color='green', alpha=0.5)
plt.xlabel('Total Units Sold')
plt.ylabel('Rating')
plt.title('Total Units Sold vs. Rating')

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16, 12))
plt.subplot(2, 3, 1)
plt.hist(df['totalunitssold'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Total Units Sold')
plt.ylabel('Frequency')
plt.title('Distribution of Total Units Sold')

plt.subplot(2, 3, 2)
plt.hist(df['rating'], bins=20, color='lightcoral', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Rating')

plt.subplot(2, 3, 3)
plt.hist(df['averagediscount'], bins=20, color='lightgreen', edgecolor='black')
plt.xlabel('Average Discount')
plt.ylabel('Frequency')
plt.title('Distribution of Average Discount')

plt.subplot(2, 3, 4)
plt.hist(df['meanproductprices'], bins=20, color='orange', edgecolor='black')
plt.xlabel('Mean Product Prices')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Product Prices')

plt.subplot(2, 3, 5)
plt.hist(df['meanretailprices'], bins=20, color='purple', edgecolor='black')
plt.xlabel('Mean Retail Prices')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Retail Prices')

plt.tight_layout()
plt.show()

numeric_cols = ['totalunitssold', 'rating', 'averagediscount', 'meanproductprices', 'meanretailprices']
sns.pairplot(df[numeric_cols], diag_kind='kde')
plt.suptitle('Scatter Plot Matrix of Numerical Features', y=1.02)
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['totalunitssold'], color='skyblue')
plt.xlabel('Total Units Sold')
plt.title('Box Plot of Total Units Sold')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['rating'], color='lightcoral')
plt.xlabel('Rating')
plt.title('Box Plot of Rating')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.regplot(x='averagediscount', y='totalunitssold', data=df, scatter_kws={"color": "skyblue"}, line_kws={"color": "red"})
plt.xlabel('Average Discount')
plt.ylabel('Total Units Sold')
plt.title('Relationship between Average Discount and Total Units Sold')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
