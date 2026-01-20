# ======================================
# Car Price Prediction with ML (Fixed)
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("Car Price Prediction Project Started")

# 1️⃣ Load Dataset
df = pd.read_csv("car data.csv")  
# ⚠️ जर तुझं file नाव वेगळं असेल तर इथे बदल

# 2️⃣ View data
print(df.head())
print(df.info())

# =========================
# 📊 GRAPH 1: Selling Price Distribution
# =========================
plt.figure(figsize=(6,4))
sns.histplot(df['selling_price'], kde=True)
plt.title("Distribution of Car Selling Price")
plt.xlabel("Selling Price")
plt.show()

# =========================
# Feature Engineering
# =========================

# Car Age
df['Car_Age'] = 2025 - df['year']

# Drop unnecessary column
df.drop(['name', 'year'], axis=1, inplace=True)

# =========================
# 📊 GRAPH 2: Car Age vs Price
# =========================
plt.figure(figsize=(6,4))
sns.scatterplot(x=df['Car_Age'], y=df['selling_price'])
plt.title("Car Age vs Selling Price")
plt.xlabel("Car Age (Years)")
plt.ylabel("Selling Price")
plt.show()

# =========================
# Convert categorical → numerical
# =========================
df = pd.get_dummies(df, drop_first=True)

# =========================
# Split Data
# =========================
X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# Prediction
# =========================
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

# =========================
# 📊 GRAPH 3: Actual vs Predicted
# =========================
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()

print("Project Finished")
