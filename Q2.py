import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Titanic-Dataset.csv")

df = df.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


df = pd.get_dummies(df, drop_first = True)

X = df.drop('Survived',axis=1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,lr_pred))

cm = confusion_matrix(y_test,lr_pred)
sns.heatmap(cm , annot=True, cmap='Blues')
plt.show()




from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(y_pred[:10])

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R2 Score:", r2)



import numpy as np

print("\nEnter Passenger Details:")

pclass = int(input("Pclass (1/2/3): "))
age = float(input("Age: "))
sibsp = int(input("SibSp: "))
parch = int(input("Parch: "))
fare = float(input("Fare: "))
sex = input("Sex (male/female): ").lower()
embarked = input("Embarked (Q/S): ").upper()

sex_male = 1 if sex == 'male' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

sample = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])
sample_scaled = scaler.transform(sample)

pred_class = lr.predict(sample_scaled)
print("Classification Result:", "Survived" if pred_class[0]==1 else "Not Survived")

pred_reg = reg.predict(sample_scaled)
print("Survival Probability:", round(pred_reg[0], 3))




import numpy as np
from sklearn.cluster import KMeans

# -------------------------
# TRAIN KMEANS
# -------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# -------------------------
# USER INPUT
# -------------------------
print("\nEnter Passenger Details:")

pclass = int(input("Pclass (1/2/3): "))
age = float(input("Age: "))
sibsp = int(input("SibSp: "))
parch = int(input("Parch: "))
fare = float(input("Fare: "))
sex = input("Sex (male/female): ").lower()
embarked = input("Embarked (Q/S): ").upper()

# Encoding
sex_male = 1 if sex == 'male' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Create sample
sample = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S]])

# Scale input
sample_scaled = scaler.transform(sample)

# -------------------------
# PREDICT CLUSTER
# -------------------------
cluster = kmeans.predict(sample_scaled)

print("Cluster Assigned:", cluster[0])
