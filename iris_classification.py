# ======================================
# Iris Flower Classification Project
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Iris Flower Classification Started")

# 1️⃣ Load Dataset
df = pd.read_csv("Iris.csv")

# 2️⃣ Remove Id column
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# 3️⃣ Dataset Overview
print(df.head())
print(df.info())

# =========================
# 📊 GRAPH 1: Species Count
# =========================
plt.figure(figsize=(5,4))
sns.countplot(x='Species', data=df)
plt.title("Count of Iris Species")
plt.show()

# =========================
# 📊 GRAPH 2: Sepal Length vs Sepal Width
# =========================
plt.figure(figsize=(6,4))
sns.scatterplot(
    x='SepalLengthCm',
    y='SepalWidthCm',
    hue='Species',
    data=df
)
plt.title("Sepal Length vs Sepal Width")
plt.show()

# =========================
# 📊 GRAPH 3: Petal Length vs Petal Width
# =========================
plt.figure(figsize=(6,4))
sns.scatterplot(
    x='PetalLengthCm',
    y='PetalWidthCm',
    hue='Species',
    data=df
)
plt.title("Petal Length vs Petal Width")
plt.show()

# =========================
# Feature & Target
# =========================
X = df.drop('Species', axis=1)
y = df['Species']

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Model
# =========================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# =========================
# Prediction
# =========================
y_pred = model.predict(X_test)

# =========================
# Model Evaluation
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 📊 GRAPH 4: Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 🌸 Predict New Flower
# =========================
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)

print("Predicted Iris Species:", prediction[0])

print("Project Completed Successfully")
