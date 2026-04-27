import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("spam.csv",encoding='latin-1')

df = df[['v1','v2']]

df.columns = ['label','text']

df['label'] = df['label'].map({'ham':0,'spam':1})

print(df.head())

tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train,y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

print(acc)

print(classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)
sns.heatmap(cm,annot=True, cmap='Blues')
plt.show()


msg = input("Enter the msage to check it is spam or ham: ")

msg_vec = tfidf.transform([msg])
ans = model.predict(msg_vec)

print("Prediction: ", "spam" if ans[0]==1 else "Ham")
