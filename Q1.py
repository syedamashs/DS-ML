import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

print(df['Age'].mean())

sns.countplot(x='Survived',data = df)
plt.title("Survival Count")
plt.show()

sns.histplot(df['Age'],kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(x='Sex', hue='Survived',data = df)
plt.title("Survival Count")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival based on Class")
plt.show()

sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title("Age vs Fare")
plt.show()
