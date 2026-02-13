import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("train.csv")

# Data cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

# Features
X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

print("Accuracy:", model.score(X_test,y_test)) 
