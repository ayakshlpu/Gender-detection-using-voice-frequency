#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#**iNTRODUCTION **

# WE LEARN PRACICE AND COMPARE A6 CLASSIFICATION MODELS IN THIS PROJECTS, SO YOU WILL IN THIS KERNEL:
# 1. Test- train dates split
# 2. Support Vector Machines (SVM) Classification
# 3. Naive Bayes classification
# 4. Decision Tree classification
# 5. Random Forest classification
# 6. Compare all of these classfication Models

# PROBLEM- STATEMENT: - GENDER RECOGNITION BY VOICE AND SPEECH ANALYSIS**


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


df=pd.read_csv("E:\youtube course\gender-recognition-using-voice.csv")


# In[14]:


df.head()


# In[16]:


df.shape


# In[17]:


print(df.columns)


# In[18]:


df.isnull().sum()


# In[29]:


df.info()


# In[19]:


# DATA INSIGHT AND ADDING LABEL TO MALE/FEMALE

a=df.label.value_counts
print(a)
print("TOTAL", df.label.count())


# In[20]:


# CORRELATION BETWEEN DIFFERENT INPUT VARIABLES
df.corr()


# In[21]:


df.label = [1 if each =="female" else 0 for each in df.label]
# this assign 1 to female , 0 to male


# In[34]:


df.label


# In[22]:


# SPLITING DATASET INTO TRAINING AND TESTING SET
y = df.label.values
x = df.drop(['label'],axis ='columns')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=50)
#test_size=0.2 means %20 test dates, %80 train dates

algo_names=[]  #Empty LIST
algo_scores=[] #Empty LIST


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


X_train.head()


# In[27]:


X_train.tail()


# In[53]:


# APPLYING DIFFERENT MODELS NOW!
# DECISION TREE

from sklearn.tree import DecisionTreeClassifier
#IMPORTING DECISION TREE CLASSIFIER FOR APPLYING MODEL
dec_tree = DecisionTreeClassifier(random_state = 50)
dec_tree.fit(X_train, y_train)
print("Decision Tree Classification score: ", dec_tree.score(X_test, y_test))  #ACCURACY
algo_names.append("Decision Tree")
algo_scores.append(dec_tree.score(X_test, y_test))
p1=dec_tree.predict(X_test)   #PREDICTED VALUES


# In[54]:


# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
#IMPORTING RANDOM FOREST CLASSIFIER FOR APPLYTING MODEL
rand_forest = RandomForestClassifier(random_state = 50)
rand_forest.fit(X_train, y_train)
print("Random Forest Classification Score: ", rand_forest.score(X_test, y_test))
algo_names.append("Random Forest")
algo_scores.append(rand_forest.score(X_test, y_test))
p2= rand_forest.predict(X_test)


# In[55]:


#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_test, y_test)
print("Naive Bayes Classification Score: ", format(naive_bayes.score(X_test, y_test)))
algo_names.append("Naive Bayes")
algo_scores.append(naive_bayes.score(X_test, y_test))
p4= naive_bayes.predict(X_test)


# In[70]:


#SVM

from sklearn.svm import SVC
#IMPORTING THE SVC CLASS FOR APPLYTING THE MODEL
svm= SVC(random_state = 50)
svm.fit(X_train, y_train)
print("SVM Classification score is: ",format(svm.score(X_test, y_test)))
algo_names.append("SVM")
algo_scores.append(svm.score(X_test, y_test))
p3=svm.predict(X_test)


# In[56]:


#COMPARING THE CLASSFICATION SCORE OF DIFFERENT ALGORITHM USING BAR GRAPH'

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xlabel([])
    plt.ylabel([])
    plt.grid(False)
plt.show()

#PRINTING THE CLASSIFICATION REPORT OF DIFFERENT MODELS USED USING SKLEARN.METRICS LIBRARY AND CLASSFICATION REPORT FUNCTION
# In[64]:


#IMPORTING THE APPROPRIATE FUNCTION

from sklearn.metrics import classification_report


# #DECISION TREE CLASSFICAITON REPORT

# In[66]:


report_dec_tree=classification_report(y_test,p1)
print(report_dec_tree)


# #RANDOM FOREST CLASSIFICATION REPORT
# 

# In[69]:


report_random_tree=classification_report(y_test,p2)
print(report_random_tree)


# # SVM CLASSIFICATION REPORT

# In[71]:


report_svm=classification_report(y_test,p3)
print(report_svm)


# NAIVE BAYES CLASSIFICATION REPORT

# In[73]:


report_naive_bayes=classification_report(y_test,p4)
print(report_naive_bayes)


# # PLOTTING PIE CHARTS FOR THE WRONG VS CORRECT PREDICTIONS OF EACH MODEL

# DECISION TREE PREDICTIONS PIE CHART

# In[77]:


comp1=[31.7,602.3]
l1=['predicted wrong','predicted correct']
plt.pie(comp1, labels= l1,autopct= '%0.1f%%',colors=['red','pink'])
plt.title("DECISION TREE PREDICTIONS")
plt.show()


# RANDOM FOREST PREDICTION PIE CHART

# In[78]:


comp1=[12.68,621.32]
l2=['predicted wrong','predicted correct']
plt.pie(comp1, labels= l2,autopct= '%0.1f%%',colors=['red','pink'])
plt.title("RANDOM FOREST PREDICTIONS")
plt.show()


# SVM PREDICTIONS PIE CHART
# 

# In[80]:


comp3=[171.18,462.82]
l3=['predicted wrong','predicted correct']
plt.pie(comp3, labels= l3,autopct= '%0.1f%%',colors=['red','pink'])
plt.title("SVM PREDICTIONS")
plt.show()


# NAIVE BAYES PREDICTION PIE CHART 

# In[82]:


comp4=[76.08,557.92]
l4=['predicted wrong','predicted correct']
plt.pie(comp4, labels= l4,autopct= '%0.1f%%',colors=['red','pink'])
plt.title("NAIVE BAYES PREDICTIONS")
plt.show()


# # PLOTTING CONFUSION MATRIX USING SKLEARN.METRICS.CONFUSION_MATRIX AND HEATMAP OF SEABORN LIBRARY

# In[83]:


from sklearn.metrics import confusion_matrix


# DECISION TREE CONFUSTION MATRIX

# In[84]:


# IMPORTING TEST VALUES AND PREDICTED VALUES AS PARAMETERS
conf_mat1 = confusion_matrix(y_test, p1)
plt.figure(figsize=(6,6))
# HEATMAP USED FOR PLOTTING
sns.heatmap(conf_mat1, annot=True, fmt = ".0f" )
plt.ylabel("ACTUAL")
plt.xlabel("PREDICTED")
plt.title("DECISION TREE CONFUSION MATRIX")
plt.show()


# RANDOM FOREST CONFUSION MATRIX

# In[89]:


# IMPORTING TEST VALUES AND PREDICTED VALUES AS PARAMETERS
conf_mat2 = confusion_matrix(y_test, p2)
plt.figure(figsize=(6,6))
# HEATMAP USED FOR PLOTTING
sns.heatmap(conf_mat2, annot=True, fmt = ".0f" )
plt.ylabel("ACTUAL")
plt.xlabel("PREDICTED")
plt.title("RANDOM FOREST TREE CONFUSION MATRIX")
plt.show()


# SVM CONFUSION MATRIX 

# In[90]:


# IMPORTING TEST VALUES AND PREDICTED VALUES AS PARAMETERS
conf_mat3 = confusion_matrix(y_test, p3)
plt.figure(figsize=(6,6))
# HEATMAP USED FOR PLOTTING
sns.heatmap(conf_mat3, annot=True, fmt = ".0f" )
plt.ylabel("ACTUAL")
plt.xlabel("PREDICTED")
plt.title("TREE CONFUSION MATRIX")
plt.show()


# NAIVE BAYES CONFUSION MATRIX

# In[91]:


# IMPORTING TEST VALUES AND PREDICTED VALUES AS PARAMETERS
conf_mat4 = confusion_matrix(y_test, p4)
plt.figure(figsize=(6,6))
# HEATMAP USED FOR PLOTTING
sns.heatmap(conf_mat1, annot=True, fmt = ".0f" )
plt.ylabel("ACTUAL")
plt.xlabel("PREDICTED")
plt.title("NAIVE BAYES CONFUSION MATRIX")
plt.show()


# # PLOTTING ROC CURVE FOR DIFFERENT MODELS USING ROC_CURVE OF SKLEARN.METRICS

# DECISION TREE ROC CURVE

# In[101]:


from sklearn.metrics import roc_curve, roc_auc_score, auc
b=[]
# CACULATING PREDICTION PROBABLITY
probs1=dec_tree.predict_proba(X_test)
preds1= probs1[:,1]
# CALCULATING TRUE POSITIVE RATE AND FALSE POSITIVE RATE
fpr1, tpr1, threeshold1 = roc_curve(y_test, preds1)
roc_auc1 = auc(fpr1, tpr1)
plt.title("Receive Operating Characteristics")
plt.plot(fpr1, tpr1, 'b', label='AUC= %0.2f' %roc_auc1)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')

plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("DECISION TREE ROC CURVE")
plt.show()
b.append(roc_auc1)
print("DECISION TREE AUC1: ", roc_auc_score(y_test, preds1))


# RANDOM FOREST ROC CURVE

# In[105]:


probs2=rand_forest.predict_proba(X_test)
preds2= probs2[:,1]
# CALCULATING TRUE POSITIVE RATE AND FALSE POSITIVE RATE
fpr2, tpr2, threeshold2 = roc_curve(y_test, preds2)
roc_auc2 = auc(fpr2, tpr2)
plt.title("Receive Operating Characteristics")
plt.plot(fpr2, tpr2, 'b', label='AUC= %0.2f' %roc_auc2)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')

plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("RANDOM FOREST ROC CURVE")
plt.show()
b.append(roc_auc2)
print("RANDOM FOREST AUC1: ", roc_auc_score(y_test, preds2))


#  SVM ROC CURVE

# In[106]:


# CALCULATING TRUE POSITIVE RATE AND FALSE POSITIVE RATE
fpr3, tpr3, threeshold3 = roc_curve(y_test, p3)
roc_auc3 = auc(fpr3, tpr3)
plt.title("Receive Operating Characteristics")
plt.plot(fpr3, tpr3, 'b', label='AUC= %0.2f' %roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')

plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("SVM ROC CURVE")
plt.show()
b.append(roc_auc3)
print("SVM AUC1: ", roc_auc_score(y_test, p3))


# NAIVE BAYES ROC CURVE

# In[140]:


# CALCULATING TRUE POSITIVE RATE AND FALSE POSITIVE RATE
probs3=rand_forest.predict_proba(X_test)
preds3= probs3[:,1]
fpr4, tpr4, threeshold4 = roc_curve(y_test, preds3)
roc_auc4 = auc(fpr4, tpr4)
plt.title("Receive Operating Characteristics")
plt.plot(fpr4, tpr4, 'b', label='AUC= %0.2f' %roc_auc4)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1], 'r--')

plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("NAIVE BAYES ROC CURVE")
plt.show()
b.append(roc_auc4)
print("NAIVE BAYES AUC1: ", roc_auc_score(y_test, preds3))


# # COMPARING AREA UNDER CURVE FOR ALL MODEL USING LINE GRAPH

# In[148]:


a=["DECISION TREE", "RANDOM FOREST", "SVM", "NAIVE BAYES"]
plt.plot(a,color="red", )
plt.title('AUC OF ALL MODELS')
plt.xlabel('MODEL')
plt.ylabel('AUC SCORE')
plt.show()


# # CONCLUSION

# BASED ON PRECISION, F1-SCORE, RECALL, VALUES, PREDICITING PIE CHART, ACCURACY SCORE, CONFUSION MATRIX, ROC CURVE AND AUC LINE GRAPH, WE CAN SAY THAT RANDOM FOREST CLASSFIFICATION IS BEST SUITED FOR THIS PROBLEM WHERE WE ARE PREDICTING THE PERSON AS MALE/FEMALE BASED UPON THEIR VOICE SIGNALS, DECISION TREE IS LEAST SUITABLE BASED ON SAME ANALYSIS.

# In[ ]:




