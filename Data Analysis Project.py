import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from statsmodels.stats.weightstats import ztest # type: ignore

# Load CSV file
df = pd.read_csv(r"C:\Users\asus\OneDrive\Desktop\DataScienceToolbox\BankChurnDataset.csv", encoding='latin-1')

print(df.head()) # Display rows
print(df.info()) # Display coiumn names and data types
print(df.shape) # For finding shape of dataset
print(df.describe()) # Display statistics of dataset 
print(df.isnull().sum()) # Missing value 
print(df.groupby('EstimatedSalary')['Age'].mean()) # Average value of Estimated salary by age
print(df.sort_values('EstimatedSalary',ascending=True)) # Data sorting by electric range in ascending order
print(df['CreditScore'].unique()) # Unique value 
print(df['Exited'].value_counts()) # Churn count

# Visualization using pie chart
label = ['Stayed', 'Churned']
color = ['skyblue', 'lightgreen']
plt.figure(figsize=(8, 6))
plt.pie(df['Exited'].value_counts(), labels=label, autopct='%1.2f%%', colors=color, startangle=90)
plt.title("Customer Churn Distribution")
plt.show()

# Visualization using histogram
plt.figure(figsize=(8,6))
plt.hist(df['Age'], bins=20, color='yellow', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency') # To find number of count of that particular age
plt.title('Distribution of age')
plt.show()

# Compute correlation matrix using scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Age'], y=df['CreditScore'], color='red', alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Credit Score")
plt.title("Correlation Scatter Plot of Age vs Credit Score")
plt.show()

# Correlation using heatmap
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Boxplot for Age vs Churn
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Exited'], y=df['Age'])
plt.title("Age Distribution by Churn Status")
plt.show()

# Countplot for Geography vs Churn
plt.figure(figsize=(8, 6))
sns.countplot(x='Geography', hue='Exited', data=df)
plt.title("Churn Count by Geography")
plt.legend(title="Exited", labels=["No", "Yes"])
plt.show()

# Performing Z-test Hypothesis testing
salary = df['EstimatedSalary'].dropna()# Drop NaN values if any
balance = df['Balance'].dropna()# Drop NaN values if any
z_stat, p_value = ztest(salary, balance) # Z-test between EstimatedSalary and Balance
print(f"Z-statistic: {z_stat:.2f}")
print(f"P-value: {p_value:.2f}")

alpha = 0.05 # Hypothesis testing decision
if p_value < alpha:
    print("Reject the null hypothesis: Significant difference between Salary and Balance.")
else:
    print("Fail to reject the null hypothesis: No significant difference between Salary and Balance.")

# KDE plot visualization
plt.figure(figsize=(8, 6))
sns.kdeplot(df['EstimatedSalary'].dropna(), label='EstimatedSalary', fill=True)
sns.kdeplot(df['Balance'].dropna(), label='Balance', fill=True)
plt.title('Distribution of EstimatedSalary vs Balance')
plt.xlabel('Amount')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()