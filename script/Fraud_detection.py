#!/usr/bin/env python
# coding: utf-8

# # FRAUD DETECTION SYSTEM
# #### The goal is to build a machine learning-powered fraud detection system that evaluates financial transactions that identifies potentially fraudulent activity, and flags suspicious transactions before they are completed.

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import shutil
from pathlib import Path


# In[4]:


# Creating Data folder where we will store our csv file
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)


# In[5]:


# Coping the dataset
source = Path(r"C:\Users\USER\Downloads\PS_20174392719_1491204439457_log.csv")

destination = data_folder / source.name

shutil.copy2(source, destination)


# In[6]:


# Confirming if file is copied
destination.exists()


# In[7]:


# Loading Datatset
df = pd.read_csv(destination)

df.head()


# # EDA

# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


# Checking for any duplicates
df[df.duplicated()]


# #### What we want is to build a fraud detection that's able to detect a fraud before a tranasaction. So we would not use columns like 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest' & 'newbalanceDest'

# In[11]:


df.columns


# In[12]:


df_new = df[['step', 'type', 'amount', 'nameOrig', 
             'nameDest', 'isFraud', 'isFlaggedFraud']]


# In[13]:


df_new.head()


# In[14]:


# Renaming columns
df_new.columns = df_new.columns.str.lower()

df_new.columns


# In[15]:


# Renaming my str variables
strings = df_new.dtypes[df_new.dtypes == 'str'].index

for col in strings:
    df_new[col] = df_new[col].str.lower().str.replace(' ', '_')

df_new.head()


# In[16]:


# Checking unique values in each feature
for col in df_new.columns:
    print(col)
    print(df_new[col].unique())
    print("--------------------")


# In[17]:


df_new['isflaggedfraud'].nunique()


# #### isFlaggedFraud contains only zeros in our dataset, it has zero variance.
# 
# #### This means the column cannot help my model distinguish fraudulent transactions from legitimate ones.

# # FEATURE IMPORTANCE: MUTUAL INFORMATION

# In[18]:


categorical = ['type', 'nameorig', 'namedest']

from sklearn.metrics import mutual_info_score

def mutual_info_isfraud_score(series):
    return mutual_info_score(df_new.isfraud, series)

mi = df_new[categorical].apply(mutual_info_isfraud_score)
mi.sort_values(ascending=False)


# #### I would still exclude 'nameorig' & 'namedest' from our initial model. Why:
# #### - Unseen accounts: A model may encounter account IDs it has never seen during training.
# #### - Overfitting: The model may memorise account-specific patterns instead of learning general fraud behaviour.

# # FEATURE IMPORTANCE: CORRELATION

# In[19]:


numerical = ['step', 'amount']

df_new[numerical].corrwith(df_new.isfraud)


# ## Observations:
# 
# #### Step (0.045030)
# #### - There is a very weak positive linear relationship between transaction time and fraud.
# #### - This suggests that step alone does not explain much about fraudulent transactions.
# #### Amount (0.128862)
# #### - There is a weak positive linear relationship between transaction amount and fraud.
# #### - This suggests that larger transaction amounts tend to be associated with fraud, although the relationship is not strong.

# In[20]:


# Our new dataframe
df_new = df_new[['step', 'type', 'amount', 'isfraud']]


# In[21]:


df_new.head()


# In[22]:


# Lets check the distribution of isfraud feature
df_new['isfraud'].value_counts()


# In[23]:


df_new['isfraud'].value_counts(normalize=True) * 100


# In[24]:


# Plot our distribution
sns.countplot(data=df_new, x='isfraud')

plt.title('Distribution of Fraudulent and Legitimate Transactions')
plt.xlabel('Fraud (0 = No, 1 = Yes)')
plt.ylabel('Number of Transactions')
plt.show()


# ##### The dataset has many more legitimate transactions than fraudulent ones. About 99.89% of the transactions are legitimate, while only 0.11% are fraudulent. This means a model could achieve high accuracy by simply predicting all transactions as legitimate, even though it would fail to detect fraud. For this reason, we will not rely on accuracy alone. Instead, we will use precision, recall, F1-score and PR-AUC to better evaluate how well our models detect fraudulent transactions.

# # SPLITTING OUR DATA OR VALIDATION FRAMEWORK

# In[25]:


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df_new, test_size=0.2, random_state=1, stratify=df_new.isfraud)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1, stratify=df_full_train.isfraud)


# #### We use stratify to ensure that the training, validation and test sets maintain approximately the same proportion of fraudulent and legitimate transactions because its highly imbalanced.

# In[26]:


len(df_train), len(df_val), len(df_test)


# In[27]:


df_train = df_train.reset_index(drop='True')
df_val = df_val.reset_index(drop='True')
df_test = df_test.reset_index(drop='True')


# In[28]:


y_train = df_train.isfraud.values
y_test = df_test.isfraud.values
y_val = df_val.isfraud.values


# In[29]:


del df_train['isfraud']
del df_val['isfraud']
del df_test['isfraud']


# In[30]:


# Lets check if startification worked
for name, y in [
    ('Training', y_train),
    ('Validation', y_val),
    ('Testing', y_test)
]:
    print(name)
    print(pd.Series(y).value_counts(normalize=True) * 100)
    print('--------------------')


# ##### We used stratified sampling when splitting the dataset to ensure that the training, validation and testing sets maintain approximately the same proportion of legitimate and fraudulent transactions. The results show that all three datasets have almost identical distributions. This will help us train and evaluate our models fairly.

# # ONEHOTENCODING

# In[31]:


categorical = ['type']

numerical = ['step', 'amount']


# In[32]:


# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)


# In[33]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')

X_train = dv.fit_transform(train_dict)

# Same for validation data
val_dict = df_val[categorical + numerical].to_dict(orient='records')

X_val = dv.transform(val_dict)


# In[34]:


print(X_train.shape)
print(X_val.shape)
print(dv.get_feature_names_out())


# # LOGISTIC REGRESSION

# In[35]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# In[36]:


# Soft predictions. Comes into 2 columns left = 0 & Right = 1. We only need the right part
model.predict_proba(X_val)


# In[37]:


y_pred = model.predict_proba(X_val)[:,1]

y_pred


# In[38]:


# If the model thinks the probability of isfraud is 50% or more, we predict that the customer will fraud.
isfraud_decision = (y_pred >= 0.5)


# In[39]:


y_val[0:30]


# In[40]:


isfraud_decision.astype(int)[0:30]


# ##### We used predict_proba() to obtain the probability of each transaction being legitimate or fraudulent. We selected the fraud probabilities and applied a threshold of 0.5 to classify transactions. Any transaction with a fraud probability of 0.5 or higher is classified as fraudulent, while the rest are classified as legitimate.

# # MODEL EVALUATION

# In[41]:


from sklearn.metrics import accuracy_score

accuracy_score(y_val, isfraud_decision)


# # CONFUSSION MATRIX

# In[42]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, isfraud_decision)


# ### Confusion Matrix
# **Observations:**
# 
# - The model correctly identified most legitimate transactions.
# - Out of 229 actual fraudulent transactions, it detected only 16.
# - It failed to detect 213 fraudulent transactions, classifying them as legitimate.
# 
# #### Conclusion
# 
# The confusion matrix reveals that Logistic Regression performs well at identifying legitimate transactions but struggles to detect fraudulent activity. We need to explore other models that can improve fraud detection.
# 

# # PRECISION, RECALL & F1-SCORE

# In[43]:


from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision:', precision_score(y_val, isfraud_decision))
print('Recall:', recall_score(y_val, isfraud_decision))
print('F1-score:', f1_score(y_val, isfraud_decision))


# ### Evaluation Metrics – Logistic Regression
# 
# #### 1. Precision – 76.19%
# 
# **What does precision measure?**
# 
# Precision tells us how often the model is correct when it predicts that a transaction is fraudulent.
# 
# - The model flagged 21 transactions as fraudulent.
# - Out of these, 16 were actually fraudulent.
# 
# **Interpretation:** The model was correct approximately 76% of the time when it flagged a transaction as fraud.
# 
# #### 2. Recall – 6.98%
# 
# **What does recall measure?**
# 
# Recall tells us how many of the actual fraudulent transactions the model successfully detected.
# 
# - There were 229 actual fraudulent transactions.
# - The model detected only 16 of them.
# 
# **Interpretation:** The model detected only about 7 out of every 100 fraudulent transactions, meaning it missed most fraudulent activity.
# 
# #### 3. F1-Score – 12.80%
# 
# **What does F1-score measure?**
# 
# F1-score combines precision and recall into a single metric. It is the harmonic mean of the two:
# 
# **Interpretation:** The F1-score is low because the model has reasonably good precision but very poor recall. Although most transactions flagged as fraud are genuinely fraudulent, the model fails to detect the majority of actual fraud cases.
# 
# #### Conclusion
# 
# The Logistic Regression model achieves good precision but performs poorly in detecting fraudulent transactions. Its low recall means that many fraudulent transactions go undetected.
# 
# This suggests that we need to explore other machine learning models that can improve fraud detection, particularly recall, while maintaining reasonable precision.
# 

# # DECISSION TREE

# In[44]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

isfraud_decision = (y_pred >= 0.5)


# In[45]:


print('Accuracy:', accuracy_score(y_val, isfraud_decision))
print('Confusion Matrix:\n', confusion_matrix(y_val, isfraud_decision))
print('Precision:', precision_score(y_val, isfraud_decision))
print('Recall:', recall_score(y_val, isfraud_decision))
print('F1-score:', f1_score(y_val, isfraud_decision))


# ### Decision Tree Classifier
# 
# #### 1. Accuracy – 99.94%
# **Interpretation:** The model correctly classified approximately 99.94% of all transactions. However, because our dataset is highly imbalanced, accuracy alone is not sufficient to evaluate the model's performance.
# 
# #### 2. Confusion Matrix
# 
# The confusion matrix shows how the model classified legitimate and fraudulent transactions.
# **Observations:**
# 
# - **True Negatives (209,485):** Correctly identified legitimate transactions.
# - **False Positives (1):** One legitimate transaction was incorrectly classified as fraudulent.
# - **False Negatives (115):** Failed to detect 115 actual fraudulent transactions.
# - **True Positives (114):** Correctly identified 114 fraudulent transactions.
# 
# **Interpretation:** The model produces very few false fraud alerts, but it still misses approximately half of the actual fraudulent transactions.
# 
# #### 3. Precision – 99.13%
# **Interpretation:** Approximately 99% of transactions flagged as fraudulent were actually fraudulent. This means the model produces very few false fraud alerts.
# 
# #### 4. Recall – 49.78%
# **Interpretation:** The model detected approximately 50% of all fraudulent transactions. However, it still missed 115 out of 229 actual fraud cases.
# 
# #### 5. F1-Score – 66.28%
# **Interpretation:** The F1-score shows a considerable improvement compared with Logistic Regression. The Decision Tree achieves an excellent precision of 99.13% and a much better recall of 49.78%.
# 
# #### Conclusion
# 
# Compared with Logistic Regression, the Decision Tree significantly improves fraud detection.
# 
# - Precision increased from 76.19% to 99.13%.
# - Recall increased from 6.98% to 49.78%.
# - F1-score increased from 12.80% to 66.28%.
# 
# The Decision Tree is much better at identifying fraudulent transactions while producing very few false alerts. However, it still misses approximately half of all fraudulent transactions, so we should explore whether another model can improve recall without significantly reducing precision.
# 

# # RANDOM FOREST

# In[46]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# In[47]:


# Make predictions
y_pred = model.predict_proba(X_val)[:, 1]

isfraud_decision = (y_pred >= 0.5)


# In[48]:


# Model evaluation
print('Accuracy:', accuracy_score(y_val, isfraud_decision))
print('Confusion Matrix:\n', confusion_matrix(y_val, isfraud_decision))
print('Precision:', precision_score(y_val, isfraud_decision))
print('Recall:', recall_score(y_val, isfraud_decision))
print('F1-score:', f1_score(y_val, isfraud_decision))


# 
# ### Random Forest Observations
# 
# The Random Forest model achieved an accuracy of 99.93%. However, accuracy alone is not sufficient to evaluate the model because the dataset is highly imbalanced.
# 
# The confusion matrix shows that the model correctly identified 129 fraudulent transactions and 209,444 legitimate transactions. It incorrectly classified 42 legitimate transactions as fraudulent and failed to identify 100 fraudulent transactions.
# 
# The precision of 75.44% indicates that approximately 75% of the transactions predicted as fraudulent were actually fraudulent.
# 
# The recall of 56.33% indicates that the model successfully identified approximately 56% of all fraudulent transactions, missing 100 out of 229 fraudulent transactions.
# 
# The F1-score of 64.5% shows the balance between precision and recall.
# 
# ### Comparison of the Three Models
# 
# | Metric | Logistic Regression | Decision Tree | Random Forest |
# |---|---:|---:|---:|
# | Accuracy | 99.90% | 99.94% | 99.93% |
# | Precision | 76.19% | 99.13% | 75.44% |
# | Recall | 6.99% | 49.78% | 56.33% |
# | F1-score | 12.80% | 66.28% | 64.50% |
# | Fraud detected | 16 | 114 | 129 |
# | Fraud missed | 213 | 115 | 100 |
# 
# ### Conclusion
# 
# Random Forest achieved the highest recall, identifying 129 fraudulent transactions compared to 114 for Decision Tree and 16 for Logistic Regression.
# 
# Decision Tree achieved the highest precision and F1-score, indicating a better balance between precision and recall at the default classification threshold of 0.5.
# 
# Overall, Random Forest is better at identifying fraudulent transactions, while Decision Tree produces fewer false fraud alerts.
# 
# Further experiments with classification thresholds are necessary to determine whether we can improve recall while maintaining reasonable precision.
# 

# # THRESHHOLDS

# In[49]:


thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    isfraud_decision = (y_pred >= threshold)

    print(f'Threshold: {threshold}')
    print('Precision:', precision_score(y_val, isfraud_decision))
    print('Recall:', recall_score(y_val, isfraud_decision))
    print('F1-score:', f1_score(y_val, isfraud_decision))
    print()


# ### Classification Threshold Observations
# 
# We initially used a classification threshold of 0.5 to classify transactions as fraudulent or legitimate. However, we experimented with different thresholds to understand how they affect precision, recall and F1-score.
# 
# At a threshold of 0.1, the model achieved the highest recall of 60.70%, meaning it identified more fraudulent transactions. However, precision dropped to 35.46%, resulting in more false fraud alerts.
# 
# As the threshold increased, precision generally improved while recall decreased. This happens because the model becomes more selective when classifying transactions as fraudulent.
# 
# At a threshold of 0.7, the model achieved the highest F1-score of 69.30%, with precision of 97.62% and recall of 53.71%.
# 
# At a threshold of 0.9, precision reached 100%, but recall decreased to 50.66%, meaning the model missed more fraudulent transactions.
# 
# ### Conclusion
# 
# A threshold of 0.7 produced the highest F1-score among the thresholds tested. It provides a better balance between precision and recall compared to the default threshold of 0.5.
# 
# However, the choice of threshold depends on the business objective. In fraud detection, missing fraudulent transactions can be costly, so recall is also an important consideration.
# 
# Further evaluation is required before selecting the final threshold.
# 

# In[50]:


thresholds = np.arange(0.60, 0.81, 0.01)

results = []

for threshold in thresholds:
    isfraud_decision = (y_pred >= threshold)

    precision = precision_score(y_val, isfraud_decision)
    recall = recall_score(y_val, isfraud_decision)
    f1 = f1_score(y_val, isfraud_decision)

    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

df_thresholds = pd.DataFrame(results)

df_thresholds.sort_values(
    by='f1_score',
    ascending=False
)


# ### Observations
# 
# The threshold of 0.67 provides the best balance between precision and recall among the thresholds tested.
# 
# We will use 0.67 as our selected classification threshold for the Random Forest model, subject to final evaluation on the test dataset.
# 

# In[51]:


test_dict = df_test[categorical + numerical].to_dict(orient='records')

X_test = dv.transform(test_dict)

y_pred_test = model.predict_proba(X_test)[:, 1]

isfraud_test = (y_pred_test >= 0.67)


# In[52]:


# Final Metrics
print('Accuracy:', accuracy_score(y_test, isfraud_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, isfraud_test))
print('Precision:', precision_score(y_test, isfraud_test))
print('Recall:', recall_score(y_test, isfraud_test))
print('F1-score:', f1_score(y_test, isfraud_test))


# # XGBOOST

# In[53]:


from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

model_xgb.fit(X_train, y_train)

# Make predictions
y_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]

isfraud_decision_xgb = (y_pred_xgb >= 0.5).astype(int)


# In[54]:


print('Accuracy:', accuracy_score(y_val, isfraud_decision_xgb))
print('Confusion Matrix:\n', confusion_matrix(y_val, isfraud_decision_xgb))
print('Precision:', precision_score(y_val, isfraud_decision_xgb))
print('Recall:', recall_score(y_val, isfraud_decision_xgb))
print('F1-score:', f1_score(y_val, isfraud_decision_xgb))


# ### Observations
# 
# The XGBoost model achieved an accuracy of **96.39%**, precision of **2.36%**, recall of **79.48%**, and an F1-score of **4.59%**.
# 
# From the confusion matrix:
# 
# - **True Negatives (201,970):** The model correctly classified 201,970 legitimate transactions.
# - **False Positives (7,516):** The model incorrectly classified 7,516 legitimate transactions as fraudulent.
# - **False Negatives (47):** The model failed to identify 47 actual fraudulent transactions.
# - **True Positives (182):** The model correctly identified 182 fraudulent transactions.
# 
# **Key observations:**
# 
# 1. **High recall (79.48%):** The model successfully detected approximately 79 out of every 100 actual fraudulent transactions. This is an improvement over Random Forest, which achieved 54.59% recall.
# 
# 2. **Low precision (2.36%):** Only about 2 out of every 100 transactions flagged as fraudulent were actually fraudulent. This indicates a high number of false alarms.
# 
# 3. **False positives:** Out of 209,486 legitimate transactions, 7,516 (approximately 3.59%) were incorrectly flagged as fraud.
# 
# 4. **Accuracy limitation:** Although the model achieved 96.39% accuracy, this metric alone does not provide a reliable measure of performance because the dataset is highly imbalanced, with very few fraudulent transactions.
# 
# **Conclusion:**
# 
# XGBoost detected more fraudulent transactions than Random Forest, identifying 182 fraud cases compared to Random Forest's 125. It also reduced the number of missed fraud cases from 104 to 47. However, this improvement came at the cost of significantly more false positives.
# 
# The results demonstrate that XGBoost is better at capturing fraudulent transactions at the 0.5 threshold, but its precision needs improvement to reduce unnecessary fraud alerts.
# 

# ## Fine tuning

# In[55]:


thresholds = np.arange(0.60, 0.81, 0.01)

results = []

for threshold in thresholds:
    isfraud_decision = (y_pred_xgb >= threshold)

    precision = precision_score(y_val, isfraud_decision)
    recall = recall_score(y_val, isfraud_decision)
    f1 = f1_score(y_val, isfraud_decision)

    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

df_thresholds = pd.DataFrame(results)

df_thresholds.sort_values(
    by='f1_score',
    ascending=False
)


# ## Model Evaluation on test data

# In[58]:


test_dict = df_test[categorical + numerical].to_dict(orient='records')

X_test = dv.transform(test_dict)

y_pred_test_xgb = model_xgb.predict_proba(X_test)[:, 1]

isfraud_test_xgb = (y_pred_test_xgb >= 0.5).astype(int)


# In[60]:


# Final Metrics
print('Accuracy:', accuracy_score(y_test, isfraud_test_xgb))
print('Confusion Matrix:\n', confusion_matrix(y_test, isfraud_test_xgb))
print('Precision:', precision_score(y_test, isfraud_test_xgb))
print('Recall:', recall_score(y_test, isfraud_test_xgb))
print('F1-score:', f1_score(y_test, isfraud_test_xgb))


# In[61]:


test_dict = df_test[categorical + numerical].to_dict(orient='records')

X_test = dv.transform(test_dict)

y_pred_test_xgb = model_xgb.predict_proba(X_test)[:, 1]

isfraud_test_xgb = (y_pred_test_xgb >= 0.8).astype(int)


# In[62]:


# Final Metrics
print('Accuracy:', accuracy_score(y_test, isfraud_test_xgb))
print('Confusion Matrix:\n', confusion_matrix(y_test, isfraud_test_xgb))
print('Precision:', precision_score(y_test, isfraud_test_xgb))
print('Recall:', recall_score(y_test, isfraud_test_xgb))
print('F1-score:', f1_score(y_test, isfraud_test_xgb))


# ### XGBoost Test Evaluation – Threshold Comparison
# 
# I evaluated the XGBoost model using two classification thresholds: 0.5 (standard threshold) and 0.8 (selected through validation threshold tuning).
# 
# **Observations:**
# 
# 1. **Threshold 0.5:** The model achieved 84.21% recall, correctly identifying 192 out of 228 fraudulent transactions. However, it generated 7,538 false positives, resulting in a low precision of 2.48%.
# 
# 2. **Threshold 0.8:** Precision improved from 2.48% to 10.57%, while recall decreased from 84.21% to 72.37%. False positives also reduced significantly, from 7,538 to 1,396.
# 
# 3. **F1-score:** The F1-score increased from 4.83% to 18.45%, indicating a better balance between precision and recall at the 0.8 threshold.
# 
# **Conclusion:**
# 
# Increasing the threshold to 0.8 made XGBoost more selective when identifying fraudulent transactions. Although it detected 27 fewer fraud cases, it reduced false alarms by approximately 81.5%.
# 
# The 0.8 threshold provides a more balanced result than 0.5 for this experiment. However, the final threshold should depend on the financial cost of missed fraud versus the operational cost of investigating false alerts.
# 

# In[ ]:




