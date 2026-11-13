
# Importing Libraries
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# Load dataset
data_path = Path("script/data/PS_20174392719_1491204439457_log.csv")

df = pd.read_csv(data_path)

print("Dataset loaded successfully")


# Select relevant columns
df_new = df[
    ['step', 'type', 'amount', 'isFraud']
].copy()

# Convert column names to lowercase
df_new.columns = df_new.columns.str.lower()

print("\nDataset shape:", df_new.shape)


# Separate features and target
X = df_new.drop(columns=['isfraud'])
y = df_new['isfraud']

print("\nTarget distribution:")
print(y.value_counts())


# Split into training and test sets
df_full_train, df_test = train_test_split(
    df_new,
    test_size=0.2,
    random_state=1,
    stratify=df_new.isfraud
)

# Split training set into training and validation sets
df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=1,
    stratify=df_full_train.isfraud
)


# Reset indexes
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# Separate features and target
y_train = df_train.isfraud.values
y_val = df_val.isfraud.values
y_test = df_test.isfraud.values

X_train = df_train.drop(columns=['isfraud'])
X_val = df_val.drop(columns=['isfraud'])
X_test = df_test.drop(columns=['isfraud'])


# Define categorical and numerical features
categorical = ['type']
numerical = ['step', 'amount']


# Initialize DictVectorizer
dv = DictVectorizer(sparse=False)


# Convert training data into dictionaries
train_dict = X_train[categorical + numerical].to_dict(orient='records')

# Fit and transform training data
X_train = dv.fit_transform(train_dict)


# Transform validation data
val_dict = X_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# Transform test data
test_dict = X_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dict)


# Check transformed data
print("\nTraining matrix shape:", X_train.shape)
print("Validation matrix shape:", X_val.shape)
print("Test matrix shape:", X_test.shape)


# Calculate class imbalance ratio
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("\nScale positive weight:", scale_pos_weight)


# Initialize XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)


# Train the model
model.fit(X_train, y_train)

print("\nXGBoost model trained successfully")


# Predict fraud probabilities on validation data
y_pred_val = model.predict_proba(X_val)[:, 1]

# Apply selected threshold
threshold = 0.8
isfraud_val = (y_pred_val >= threshold).astype(int)


# Predict fraud probabilities on test data
y_pred_test = model.predict_proba(X_test)[:, 1]

# Apply selected threshold
isfraud_test = (y_pred_test >= threshold).astype(int)


# Evaluate test performance
print("\nXGBoost Test Results")
print("Accuracy:", accuracy_score(y_test, isfraud_test))
print("Precision:", precision_score(y_test, isfraud_test))
print("Recall:", recall_score(y_test, isfraud_test))
print("F1 Score:", f1_score(y_test, isfraud_test))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, isfraud_test))

# Save trained model and preprocessing objects
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Save XGBoost model in native JSON format
model_path = models_dir / "xgboost_model.json"
model.save_model(model_path)

# Save preprocessing objects and threshold separately
artifact = {
    "dv": dv,
    "threshold": threshold
}

preprocessing_path = models_dir / "preprocessing.joblib"
joblib.dump(artifact, preprocessing_path)

print("\nXGBoost model saved successfully!")
print("Model location:", models_dir)