# Practical-1 (Uber)

Problem Statement: Predict the price of the Uber ride from a given pickup point to the agreed drop-off location.
Perform following tasks:
1. Pre-process the dataset.
2. Identify outliers.
3. Check the correlation.
4. Implement linear regression and random forest regression models.
5. Evaluate the models and compare their respective scores like R2, RMSE, etc.

> [!NOTE]
> Dataset available in [Datasets](../Datasets/uber.csv) directory.

---
 
## Steps

1. Importing Libraries
1. Data Loading and Pre-processing
2. Outlier Detection
3. Correlation Analysis
4. Model Implementation (Linear Regression & Random Forest)
5. Model Evaluation and Comparison

---

## Code

### 0. Importing Libraries:

```python3
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import radians, cos, sin, asin, sqrt
```

### 1. Data Loading & Preprocessing:

```python3
# Load the dataset
df = pd.read_csv("uber.csv")   # change to your local path if needed
print("Initial Data Shape:", df.shape)
print(df.head())

# Drop rows with missing values
df.dropna(inplace=True)
print("After dropping missing values:", df.shape)

# Rename columns for easier reference
df.rename(columns={'pickup_datetime': 'pickup_datetime'}, inplace=True)

# Convert pickup_datetime to datetime object
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Extract useful datetime features
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

# Drop datetime column (not needed as a direct feature)
df.drop(['pickup_datetime', 'key'], axis=1, inplace=True, errors='ignore')

print("\nColumns after feature extraction:\n", df.columns)
```

### 2. Outlier Detection & Removal:

```python3
# Remove entries with unrealistic fares
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]

# Remove unrealistic latitude and longitude values
df = df[(df['pickup_latitude'] <= 90) & (df['pickup_latitude'] >= -90)]
df = df[(df['dropoff_latitude'] <= 90) & (df['dropoff_latitude'] >= -90)]
df = df[(df['pickup_longitude'] <= 180) & (df['pickup_longitude'] >= -180)]
df = df[(df['dropoff_longitude'] <= 180) & (df['dropoff_longitude'] >= -180)]

print("Data shape after removing outliers:", df.shape)
```

### 3. Feature Engineering - Distance Calculation:

```python3
# Define Haversine function to calculate distance between pickup and drop-off
def haversine(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# Apply the Haversine formula
df['distance_km'] = df.apply(lambda x: haversine(x['pickup_latitude'], x['pickup_longitude'],
                                                 x['dropoff_latitude'], x['dropoff_longitude']), axis=1)

# Remove zero-distance trips
df = df[df['distance_km'] > 0]
```

### 4. Correlation Analysis:

```python3
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
```

### 5. Model Training:

```python3
# Define features and target
X = df[['distance_km', 'hour', 'day', 'month', 'year', 'day_of_week']]
y = df['fare_amount']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Linear Regression --------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# -------------------- Random Forest Regression --------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
```

### 6. Model Evaluation:

```python3
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nModel: {model_name}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    return r2, rmse, mae

# Evaluate both models
lr_scores = evaluate_model(y_test, y_pred_lr, "Linear Regression")
rf_scores = evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")
```

### 7. Comparison:

```python3
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest Regressor'],
    'R2': [lr_scores[0], rf_scores[0]],
    'RMSE': [lr_scores[1], rf_scores[1]],
    'MAE': [lr_scores[2], rf_scores[2]]
})

print("\nModel Comparison:")
print(results)
```

```python3
# Plot comparison
plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='R2', data=results)
plt.title("R² Score Comparison between Models")
plt.show()
```

---

## Miscellaneous

- [Dataset source](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)

---
