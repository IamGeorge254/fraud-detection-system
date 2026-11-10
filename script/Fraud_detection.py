#!/usr/bin/env python
# coding: utf-8

# # FRAUD DETECTION SYSTEM
# #### The goal is to build a machine learning-powered fraud detection system that evaluates financial transactions that identifies potentially fraudulent activity, and flags suspicious transactions before they are completed.

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import shutil
from pathlib import Path


# In[5]:


# Creating Data folder where we will store our csv file
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)


# In[6]:


# Coping the dataset
source = Path(r"C:\Users\USER\Downloads\PS_20174392719_1491204439457_log.csv")

destination = data_folder / source.name

shutil.copy2(source, destination)


# In[7]:


# Confirming if file is copied
destination.exists()


# In[8]:


# Loading Datatset
df = pd.read_csv(destination)

df.head()


# # EDA

# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


# Checking for any duplicates
df[df.duplicated()]


# #### What we want is to build a fraud detection that's able to detect a fraud before a tranasaction. So we would not use columns like 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest' & 'newbalanceDest'

# In[12]:


df.columns


# In[13]:


df_new = df[['step', 'type', 'amount', 'nameOrig', 
             'nameDest', 'isFraud', 'isFlaggedFraud']]


# In[14]:


df_new.head()


# In[15]:


# Renaming columns
df_new.columns = df_new.columns.str.lower()

df_new.columns


# In[16]:


# Renaming my str variables
strings = df_new.dtypes[df_new.dtypes == 'str'].index

for col in strings:
    df_new[col] = df_new[col].str.lower().str.replace(' ', '_')

df_new.head()


# In[17]:


# Checking unique values in each feature
for col in df_new.columns:
    print(col)
    print(df_new[col].unique())
    print("--------------------")


# In[18]:


df_new['isflaggedfraud'].nunique()


# #### isFlaggedFraud contains only zeros in our dataset, it has zero variance.
# 
# #### This means the column cannot help my model distinguish fraudulent transactions from legitimate ones.

# # FEATURE IMPORTANCE: MUTUAL INFORMATION

# In[19]:


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

# In[20]:


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

# In[21]:


# Our new dataframe
df_new = df_new[['step', 'type', 'amount', 'isfraud']]


# In[22]:


df_new.head()


# In[23]:


# Lets check the distribution of isfraud feature
df_new['isfraud'].value_counts()


# In[24]:


df_new['isfraud'].value_counts(normalize=True) * 100


# In[25]:


# Plot our distribution
sns.countplot(data=df_new, x='isfraud')

plt.title('Distribution of Fraudulent and Legitimate Transactions')
plt.xlabel('Fraud (0 = No, 1 = Yes)')
plt.ylabel('Number of Transactions')
plt.show()


# ##### The dataset has many more legitimate transactions than fraudulent ones. About 99.89% of the transactions are legitimate, while only 0.11% are fraudulent. This means a model could achieve high accuracy by simply predicting all transactions as legitimate, even though it would fail to detect fraud. For this reason, we will not rely on accuracy alone. Instead, we will use precision, recall, F1-score and PR-AUC to better evaluate how well our models detect fraudulent transactions.

# # SPLITTING OUR DATA OR VALIDATION FRAMEWORK

# In[26]:


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df_new, test_size=0.2, random_state=1, stratify=df_new.isfraud)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1, stratify=df_full_train.isfraud)


# #### We use stratify to ensure that the training, validation and test sets maintain approximately the same proportion of fraudulent and legitimate transactions because its highly imbalanced.

# In[27]:


len(df_train), len(df_val), len(df_test)


# In[28]:


df_train = df_train.reset_index(drop='True')
df_val = df_val.reset_index(drop='True')
df_test = df_test.reset_index(drop='True')


# In[29]:


y_train = df_train.isfraud.values
y_test = df_test.isfraud.values
y_val = df_val.isfraud.values


# In[30]:


del df_train['isfraud']
del df_val['isfraud']
del df_test['isfraud']


# In[31]:


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

# In[32]:


categorical = ['type']

numerical = ['step', 'amount']


# In[33]:


# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)


# In[34]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')

X_train = dv.fit_transform(train_dict)

# Same for validation data
val_dict = df_val[categorical + numerical].to_dict(orient='records')

X_val = dv.transform(val_dict)


# In[35]:


print(X_train.shape)
print(X_val.shape)
print(dv.get_feature_names_out())


# # LOGISTIC REGRESSION

# In[36]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# In[37]:


# Soft predictions. Comes into 2 columns left = 0 & Right = 1. We only need the right part
model.predict_proba(X_val)


# In[38]:


y_pred = model.predict_proba(X_val)[:,1]

y_pred


# In[39]:


# If the model thinks the probability of isfraud is 50% or more, we predict that the customer will fraud.
isfraud_decision = (y_pred >= 0.5)


# In[40]:


y_val[0:30]


# In[41]:


isfraud_decision.astype(int)[0:30]


# ##### We used predict_proba() to obtain the probability of each transaction being legitimate or fraudulent. We selected the fraud probabilities and applied a threshold of 0.5 to classify transactions. Any transaction with a fraud probability of 0.5 or higher is classified as fraudulent, while the rest are classified as legitimate.

# # MODEL EVALUATION

# In[42]:


from sklearn.metrics import accuracy_score

accuracy_score(y_val, isfraud_decision)


# # CONFUSSION MATRIX

# In[43]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_val, isfraud_decision)


# ### Confusion Matrix
# **Observations:**
# 
# - The model correctly identified most legitimate transactions.
# - Out of 229 actual fraudulent transactions, it detected only 16.
# - It failed to detect 213 fraudulent transactions, classifying them as legitimate.
# 
# #### Interpretation
# 
# Although the model achieved high accuracy, it failed to identify most fraudulent transactions.
# 
# This highlights a major challenge when working with an imbalanced dataset: accuracy alone can be misleading because the majority class (legitimate transactions) dominates the results. Therefore, we need to evaluate the model using additional metrics such as:
# 
# - **Precision:** How often fraud predictions are correct.
# - **Recall:** How many actual fraudulent transactions are detected.
# - **F1-score:** The balance between precision and recall.
# 
# #### Conclusion
# 
# The confusion matrix reveals that Logistic Regression performs well at identifying legitimate transactions but struggles to detect fraudulent activity. We need to explore other models that can improve fraud detection.
# 

# # PRECISION, RECALL & F1-SCORE

# In[44]:


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

# In[45]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

isfraud_decision = (y_pred >= 0.5)


# In[46]:


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

# In[47]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# In[48]:


# Make predictions
y_pred = model.predict_proba(X_val)[:, 1]

isfraud_decision = (y_pred >= 0.5)


# In[49]:


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

# In[50]:


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

# In[51]:


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


# ### Threshold Fine-Tuning Observations
# 
# After testing thresholds between 0.6 and 0.8 using increments of 0.01, the threshold of 0.67 achieved the highest F1-score of 69.44%.
# 
# At this threshold, the model achieved:
# - Precision: 95.42%
# - Recall: 54.59%
# - F1-score: 69.44%
# 
# This indicates that approximately 95% of transactions predicted as fraudulent were actually fraudulent, while the model successfully identified approximately 55% of all fraudulent transactions.
# 
# Compared to the threshold of 0.7, the threshold of 0.67 produced a slightly higher F1-score.
# 
# ### Conclusion
# 
# The threshold of 0.67 provides the best balance between precision and recall among the thresholds tested.
# 
# We will use 0.67 as our selected classification threshold for the Random Forest model, subject to final evaluation on the test dataset.
# 

# In[52]:


df_test


# In[53]:


test_dict = df_test[categorical + numerical].to_dict(orient='records')

X_test = dv.transform(test_dict)

y_pred_test = model.predict_proba(X_test)[:, 1]

isfraud_test = (y_pred_test >= 0.67)


# In[54]:


# Final Metrics
print('Accuracy:', accuracy_score(y_test, isfraud_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, isfraud_test))
print('Precision:', precision_score(y_test, isfraud_test))
print('Recall:', recall_score(y_test, isfraud_test))
print('F1-score:', f1_score(y_test, isfraud_test))


# ### Model Evaluation on Test Dataset
# 
# After selecting a classification threshold of 0.67 using the validation dataset, we evaluated the Random Forest model on the unseen test dataset.
# 
# The model achieved an accuracy of 99.95%.
# 
# The confusion matrix shows that the model correctly identified 128 fraudulent transactions and 209,479 legitimate transactions. It incorrectly classified 8 legitimate transactions as fraudulent and failed to identify 100 fraudulent transactions.
# 
# The precision of 94.12% indicates that approximately 94% of transactions predicted as fraudulent were actually fraudulent.
# 
# The recall of 56.14% indicates that the model successfully identified approximately 56% of all fraudulent transactions.
# 
# The F1-score was 70.33%, showing a reasonable balance between precision and recall.
# 
# ### Validation and Test Comparison
# 
# The validation F1-score was 69.44%, while the test F1-score was 70.33%.
# 
# The similarity between the validation and test results suggests that the model generalises reasonably well to unseen data.
# 
# ### Conclusion
# 
# The Random Forest model, using a classification threshold of 0.67, achieved an F1-score of 70.33% on the test dataset.
# 
# It demonstrated high precision, correctly identifying most transactions classified as fraudulent. However, its recall of 56.14% indicates that it still misses approximately 44% of fraudulent transactions.
# 
# Overall, the model performs well in minimising false fraud alerts.

# In[ ]:




