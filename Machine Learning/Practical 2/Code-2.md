# Practical-2 (Spam Email Detection)

Problem Statement: Classify the email using the binary classification method. Email Spam detection has two states: a) Normal State – Not Spam, b) Abnormal State – Spam. Use K-Nearest Neighbors and Support Vector Machine for classification. Analyze their performance. 

> [!NOTE]
> Dataset available in [Datasets](../Datasets/emails.csv) directory.

---
 
## Steps

1. Import libraries
2. Load dataset
3. Data splitting (training and testing)
4. KNN
5. SVM
6. Plotting

---

## Code

### 1. Import libraries:

```python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Load dataset:

```python3
df = pd.read_csv("emails.csv", encoding="ISO-8859-1")  # Adjust path if needed

# Drop unnecessary columns if present
if "Email No." in df.columns:
    df = df.drop(columns=["Email No."])

# Ensure label is integer
df["Prediction"] = df["Prediction"].astype(int)

# Features & target
X = df.drop(columns=["Prediction"])
y = df["Prediction"]

# Print basic info
print(df.columns)
print(df.head(5))
```

### 3. Data splitting (training and testing):

```python3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 4. KNN:

```python3
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n--- KNN Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
```

### 5. SVM:

```python3
svm = SVC(kernel='linear', random_state=42)  # Linear kernel for binary classification
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\n--- SVM Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
```

### 6. Plotting:

```python3
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("KNN Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("SVM Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.show()
```

---

## Miscellaneous

- [Dataset source](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

---
