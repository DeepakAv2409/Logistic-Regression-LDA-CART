#!/usr/bin/env python
# coding: utf-8

# #### Name : Deepak AV 
# ##### Logistic regression , Linear Discriminant Analysis, CART - Decision Tree
# ###### Problem 2 - Contraceptive Method Dataset

# ###### 2.1 Data Ingestion: Read the dataset. Do the descriptive statistics and do null value condition check, check for duplicates and outliers and write an inference on it. Perform Univariate and Bivariate Analysis and Multivariate Analysis.

# In[1]:


#importing all basic necessary packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 


# In[2]:


os.getcwd()


# In[3]:


#Reading the Dataset
df = pd.read_excel("Contraceptive_method_dataset.xlsx")
df.head()


# In[4]:


#getting the information of the dataset
df.info()


# In[5]:


#calculating all the statistical values like mean, median etc.
df.describe()


# In[6]:


#Shows the Shapeof the datasaet
df.shape


# In[7]:


#checking the null values whether there are not
df.isnull().sum()


# In[8]:


#calculating the mean for the numerical variables present in the dataset
df.mean(axis=0)


# In[9]:


#Replacing the the null values with the calculated mean
df["Wife_age"].fillna(value = 32, inplace = True)


# In[10]:


#Replacing the the null values with the calculated mean
df["No_of_children_born"].fillna(value = 3, inplace = True)


# In[11]:


#Again checking with the null values
df.isnull().sum()


# In[12]:


#Checking for the duplicates present in the dataset
df.duplicated().sum()


# In[13]:


#Displaying the total number of values present in the each category under each variable
for column in df.columns:
    if df[column].dtype == 'object':
        print(column.upper(),': ',df[column].nunique())
        print(df[column].value_counts().sort_values())
        print('\n')


# In[14]:


#Replacing the categorical variable into the numerical variable according the description given in the question
df['Wife_ education'].replace(['Uneducated','Primary','Secondary','Tertiary'],[1,2,3,4], inplace=True)
df['Husband_education'].replace(['Uneducated','Primary','Secondary','Tertiary'],[1,2,3,4], inplace=True)
df['Wife_religion'].replace(['Scientology','Non-Scientology'],[1,0], inplace=True)
df['Husband_education'].replace(['Uneducated','Primary','Secondary','Tertiary'],[1,2,3,4], inplace=True)
df['Wife_Working'].replace(['No', 'Yes'],[0, 1], inplace=True)
df['Standard_of_living_index'].replace(['Very Low','Low','Very High','High'],[1,2,3,4], inplace=True)
df['Media_exposure '].replace(['Not-Exposed', 'Exposed'],[0, 1], inplace=True)
df['Contraceptive_method_used'].replace(['No', 'Yes'],[0, 1], inplace=True)


# In[15]:


#Checking whether the dataset as been changed in to the numerical dataset 
df.head()


# In[16]:


#checking for the information after replacing with numerical variable in the dataset
df.info()


# In[17]:


df.isnull().sum()


# In[18]:


df.shape


# In[19]:


#checks for the duplicates in the dataset after replacing into the numerical variable
df.duplicated().sum()


# In[20]:


#Deleting the duplicates from the dataset 
df = df.drop_duplicates()


# In[21]:


#checks whether duplicates are there or not
df.duplicated().sum()


# In[22]:


#checking for the shape of dataset after done with the data cleaning
df.shape


# In[23]:


#Calculating the Statistical Analysis after done with the Data Cleaning
df.describe()


# #### Univariate Analysis

# In[24]:


#Showing the Univariate Data Visulization
plt.figure(figsize=(20,10))
sns.boxplot(data=df)


# #### Bivariate Analysis

# In[25]:


#Showing the Bivariate Data Visulization
plt.figure(figsize=(20,10))
sns.scatterplot(data=df)


# #### Multivariate Analysis

# In[26]:


#Showing the Multivariate Data Visulization
plt.figure(figsize=(10,5))
sns.pairplot(data=df,size=2)


# In[27]:


#Checking whether the outliers are present are not using the box plot
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.boxplot(data=df)


# In[28]:


#Removing the outliers present in the data
def remove_outlier(col):
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[29]:


for i in df.columns:
    LL, UL = remove_outlier(df[i])
    df[i] = np.where(df[i] > UL, UL, df[i])
    df[i] = np.where(df[i] < LL, LL, df[i])


# In[30]:


#Checking for outliers after removing it 
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.boxplot(data=df)


# In[31]:


#Correlating the dataset in the pearson method
df_corr = df.corr(method='pearson')
df_corr


# In[32]:


#Dropping the null values columns present afte correlation
df = df.drop(df.columns[[4,8]], axis=1)


# In[33]:


df


# In[34]:


#Correlatio is done after dropping the columns present null values
df_corr = df.corr(method='pearson')
df_corr


# In[35]:


#Showing the Heatmap of the dataset
plt.figure(figsize=(20,20))
sns.heatmap(df_corr, annot=True)


# ###### 2.2 Do not scale the data. Encode the data (having string values) for Modelling. Data Split: Split the data into train and test (70:30). Apply Logistic Regression and LDA (linear discriminant analysis) and CART.

# #### Split the data into train and test

# In[36]:


#Libraries for Splitting the dataset
from sklearn.model_selection import train_test_split


# In[37]:


df.columns


# In[38]:


#Assigning the variables to X and Y 
X=df[['Wife_age', 'Wife_ education', 'Husband_education',
       'No_of_children_born', 'Wife_Working', 'Husband_Occupation',
       'Standard_of_living_index', 'Contraceptive_method_used']]
y=df['Husband_Occupation']


# In[39]:


#Splitting the data to train and test labels
array = df.values
X = array[:,0:7] # select all rows and first 8 columns which are the attributes
Y = array[:,7]   # select all rows and the 8th column which is the classification "Yes", "No" for Contraceptive_method
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
type(X_train)


# #### LOGISTIC REGRESSION

# In[40]:


#Libraries for the Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[41]:


# Fit the model on original data
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))
print(metrics.classification_report(y_test, y_predict))


# In[42]:


#Calculating the Actual and Predicted value for the Test data
cm = metrics.confusion_matrix(y_test, y_predict)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['No', 'Yes']
plt.title('Confusion Matrix - Test Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['G1', 'G2'], ['G1','G2']]
 
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


# #### Linear Discriminent Analysis

# In[43]:


#Importing the libraries necessary for the LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler


# In[44]:


df.head()


# In[45]:


#Scalling the Data for the variable Contraceptive_method
scaler=StandardScaler()
X = scaler.fit_transform(df.drop(['Contraceptive_method_used'],axis=1))
Y = df['Contraceptive_method_used']


# In[46]:


#Checkng the total values present in the binary operation
Y.value_counts()


# In[47]:


#Calling the Linear discriinant analysis
clf = LinearDiscriminantAnalysis()
model=clf.fit(X,Y)
model


# In[48]:


pred_class = model.predict(X)
df['Prediction'] = pred_class 


# In[49]:


#Confusion matrix is for the prediction class
confusion_matrix(Y, pred_class)


# In[50]:


#Ploting the Actual and Predicted values of the LDA 
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(Y, pred_class),annot=True,fmt='.4g'),'\n\n'
plt.ylabel('Actual Value')
plt.xlabel('Predicted Value')
plt.show();


# In[51]:


#Generating the Classification report for the Linear discriminent analysis
from sklearn.metrics import classification_report
print(classification_report(Y, pred_class))


# In[52]:


X.shape


# In[53]:


#Getting the Coefficients values of the Model
model.coef_


# In[54]:


#Getting the Intercept value of the model
model.intercept_


# In[93]:


''' LDF=0.2656+ X1*(-0.5507) + X2*(0.5949) + X3*(0.0463) + X4*(0.6890) + X5*(-.04953) + X6*(0.08202) + X7*(0.16487) 
'''


# In[56]:


DS=[]
coef=[-0.5507807 ,  0.59496081,  0.04638562,  0.6890541 , -0.04953592,
         0.08202671,  0.16487058] # Coefficients 
for p in range(len(X)):
    s3=0
    for q in range(X.shape[1]):
        s3=s3+(X[p,q]*coef[q]) # Building the LDF equation 
    s3=s3+0.2656
    DS.append(s3)
    


# In[57]:


#Calculating the Probability values
s1=0
s2=0
for i in range(len(X)):
    if DS[i]>=0:
        print("FOR Row:",i," ",X[i,:])
        print()
        #print("-->","{ prob(Y=1|X) =",pred_prob[:,1][i],">0.5 is True}")
        print("-->","{ DS: ",DS[i],">=0 , Classify as 1}")
        print("------------------------------------------------------------------------------------------")
        s1+=1
    elif DS[i]<0:
        print("FOR Row:",i," ",X[i,:])
        print()
        #print("-->","{ prob(Y=1|X) =",pred_prob[:,1][i],">0.5 is True}")
        print("-->","{ DS: ",DS[i],"<0 , Classify as 0}")
        print("------------------------------------------------------------------------------------------")
        s2+=1


# In[58]:


#Printing the total values classified in the variable
print(s1," rows classified as 1 (Yes) ")
print(s2," rows classified as 0 (NO) ")


# In[59]:


pred_prob=model.predict_proba(X)#Posterior Probability for each row


# In[60]:


pred_prob[:,1]


# In[61]:


'''
Classification Rule :

if prob(Y=1|X) >=0 then Classify as 1 
else ifprob(Y=1|X) <0 then Classify as 0 
'''
s3,s4=0,0
for i in range(len(pred_prob[:,1])):
    if pred_prob[:,1][i]>=0.5:
        print("FOR Row:",i," ",X[i,:])
        print()
        print("-->","{ prob(Y=1|X) =",pred_prob[:,1][i],">=0.5 , Classify as 1 }")
        print("------------------------------------------------------------------------------------------")
        s3+=1
    elif pred_prob[:,1][i]<0.5:
        print("FOR Row:",i," ",X[i,:])
        print()
        print("-->","{ prob(Y=1|X) =",pred_prob[:,1][i],"< 0.5 , Classify as 0 }")
        print("------------------------------------------------------------------------------------------")
        s4+=1


# In[62]:


print(s3," rows classified as 1 (Yes) ")
print(s4," rows classified as 0 (No) ")


# #### CART - Decision Tree

# In[63]:


#importing the necessary library for the Decision tree classifier
from sklearn.tree import DecisionTreeClassifier


# In[64]:


af = pd.read_excel("Contraceptive_method_dataset.xlsx")
af.head()


# In[65]:


af.info()


# In[66]:


af.isnull().sum()


# In[67]:


af.mean(axis=0)


# In[68]:


af["Wife_age"].fillna(value = 32, inplace = True)


# In[69]:


af["No_of_children_born"].fillna(value = 3, inplace = True)


# In[70]:


af.isnull().sum()


# In[71]:


af.info()


# In[72]:


#Assigning all the Categorical variable to the feature 
for feature in af.columns: 
    if af[feature].dtype == 'object': 
        af[feature] = pd.Categorical(af[feature]).codes


# In[73]:


af.info()


# In[74]:


X = af.drop("Contraceptive_method_used" , axis=1)
y = af.pop("Contraceptive_method_used")


# In[75]:


#Splitting the Dataset
X_train, X_test, train_labels, test_labels = train_test_split(X, y, test_size=.30, random_state=1)


# In[76]:


#Decision tree classifier using the gini index
dt_model = DecisionTreeClassifier(criterion = 'gini' )


# In[77]:


dt_model.fit(X_train, train_labels)


# In[78]:


from sklearn import tree

train_char_label = ['No', 'Yes']
Credit_Tree_File = open('d:\credit_tree.dot','w')
dot_data = tree.export_graphviz(dt_model, out_file=Credit_Tree_File, feature_names = list(X_train), class_names = list(train_char_label))

Credit_Tree_File.close()


# In[79]:


# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(dt_model.feature_importances_, columns = ["Importnce"], index = X_train.columns))


# In[80]:


#claculating the prediction value for the test data
y_predict = dt_model.predict(X_test)


# In[81]:


reg_dt_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 7,min_samples_leaf=10,min_samples_split=30)
reg_dt_model.fit(X_train, train_labels)


# In[82]:


credit_tree_regularized = open('d:\credit_tree_regularized.dot','w')
dot_data = tree.export_graphviz(reg_dt_model, out_file= credit_tree_regularized , feature_names = list(X_train), class_names = list(train_char_label))

credit_tree_regularized.close()

print (pd.DataFrame(dt_model.feature_importances_, columns = ["Importance"], index = X_train.columns))


# In[83]:


ytrain_predict = reg_dt_model.predict(X_train)
ytest_predict = reg_dt_model.predict(X_test)


# ###### 2.3 Performance Metrics: Check the performance of Predictions on Train and Test sets using Accuracy, Confusion Matrix, Plot ROC curve and get ROC_AUC score for each model Final Model: Compare Both the models and write inference which model is best/optimized.

# In[84]:


# predict probabilities
probs = reg_dt_model.predict_proba(X_train)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(train_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# In[85]:


# AUC and ROC for the test data


# predict probabilities
probs = reg_dt_model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# In[86]:


#classification report for the Train labels
print(classification_report(train_labels, ytrain_predict))


# In[87]:


#classification report for the Test labels
print(classification_report(test_labels, ytest_predict))


# In[88]:


#Confusion matric for the Train labels
confusion_matrix(train_labels, ytrain_predict)


# In[89]:


#Confusion matric for the Test labels
confusion_matrix(test_labels, ytest_predict)


# In[90]:


#calculating the model score for the train labels
reg_dt_model.score(X_train,train_labels)


# In[91]:


#calculating the model score for the test labels
reg_dt_model.score(X_test,test_labels)


# In[ ]:





# In[ ]:





# In[ ]:




