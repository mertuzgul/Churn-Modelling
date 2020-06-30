#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # IMPORT DATA

# In[2]:


data = pd.read_csv("Churn_Modelling.csv")
data.head()


# In[3]:


data.index = data['RowNumber']
data.drop(['RowNumber'],axis=1,inplace=True)


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.head()


# # DATA PREPARATION AND CLEANING

# In[7]:


data.dropna()
X = data.iloc[:,2:12]
y = data.iloc[:,12]


# In[8]:


X.head()


# In[9]:


y.head()


# In[10]:


Gender = pd.get_dummies(X['Gender'])
#Gender.head()
X.drop(['Gender'],axis=1,inplace=True)
X = pd.concat([X,Gender],axis=1)
X.head()


# In[11]:


Geography = pd.get_dummies(X['Geography'])
#Geography.head()
X.drop(['Geography'],axis=1,inplace=True)
X = pd.concat([X,Geography],axis=1)
X.head()


# # Exploratory Data Analysis

# In[12]:


X.head()


# In[13]:


X.describe(include='all')


# In[14]:


X.info()


# In[15]:


CreditScore = pd.DataFrame(data.CreditScore)
Age = pd.DataFrame(data.Age)
Balance = pd.DataFrame(data.Balance)
NumOfProducts = pd.DataFrame(data.NumOfProducts)
EstimatedSalary = pd.DataFrame(data.EstimatedSalary)
Gender = pd.DataFrame([1 if i=="Male" else 0 for i in data.Gender])



CreditScore.plot.hist()
Age.plot.hist()
Balance.plot.hist()
NumOfProducts.plot.hist()
EstimatedSalary.plot.hist()
Gender.plot.hist(title="1:Male\n0:Female",bins=3)


# In[16]:


NumOfProducts.plot.box()


# In[17]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[18]:


CountryGroup = data.groupby('Geography')
CountryGroup.head(1)


# In[19]:


Country_data_avg = CountryGroup.mean()
Country_data_count = CountryGroup.count()


# In[20]:


Country_data_avg


# In[21]:


Country_data_count


# In[22]:


from sklearn.preprocessing import LabelEncoder
dataframe_y = pd.DataFrame(y)
le = LabelEncoder()
encoder = dataframe_y.apply(le.fit_transform)
color = []
for l in encoder.values:
    if l == 0:
        color.append("red")
    else:
        color.append("blue")            

plt.figure(figsize=(10,10))
plt.scatter(data['CreditScore'],data['EstimatedSalary'], color=color, marker='o', alpha=0.2,s=20)
axes = plt.gca()
axes.set_xlim([0,2000])
axes.set_ylim([0,3000])
plt.xlabel('CreditScore')
plt.ylabel('EstimatedSalary')

plt.show()


# # MODEL BUILDING

# Logistic Regression,Naive Bayes, SVM,KNN,RandomForest

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)

sc = StandardScaler()#if i dont use scandart Scaler my accuracy willl drop
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

LR = LogisticRegression()
LR.fit(x_train,y_train)

LR_pred = LR.predict(x_test)
score_LR = accuracy_score(y_test,LR_pred)*100
print("accuracy score of Logistic regression is :{}".format(score_LR))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, LR_pred)
print("Logistic Regression")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(y_test, LR_pred).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# In[25]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

SVM = svm.SVC()
SVM.fit(x_train,y_train)

SVM_pred = SVM.predict(x_test)
score_SVM = accuracy_score(y_test,SVM_pred)*100
print("accuracy score of SVM is :{}".format(score_SVM))


cm = confusion_matrix(y_test, LR_pred)
print("Support Vector Machine")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(y_test, SVM_pred).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# I also Use PARAMETER TUNING in this Model
# (After I represent KNN Model)

# In[26]:


from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

scores = []

#I used parameter tuning EXTRA in KNN
for neighbor in range(10):
    KNN = KNeighborsClassifier(n_neighbors = neighbor+1, metric = 'minkowski')
    KNN.fit(x_train, y_train)

    KNN_pred = KNN.predict(x_test)
    score_KNN = accuracy_score(y_test,KNN_pred)*100
    print("accuracy score of KNN is :{} with neighbor {}".format(score_KNN,neighbor+1))
    scores.append([score_KNN,neighbor+1])


# In[27]:


scores = sorted(scores,key=lambda l:l[0], reverse=True)
BestNeighbors = scores[0][1]

KNN = KNeighborsClassifier(n_neighbors = BestNeighbors, metric = 'minkowski')
KNN.fit(x_train, y_train)

KNN_pred = KNN.predict(x_test)
score_KNN = accuracy_score(y_test,KNN_pred)*100
print("accuracy score of Best KNN is :{} with neighbor {}".format(score_KNN,BestNeighbors))


# In[28]:


from sklearn.naive_bayes  import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

NB = GaussianNB()
NB.fit(x_train,y_train)

NB_pred = NB.predict(x_test)
score_NB = accuracy_score(y_test,NB_pred)*100
print("accuracy score of Naive Bayes is :{}".format(score_NB))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, NB_pred)
print("Naive Bayes")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(y_test, NB_pred).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# In[29]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=50) 
RFC.fit(x_train, y_train)
RFC_pred = RFC.predict(x_test)
score_RFC = accuracy_score(y_test,RFC_pred)*100
print("accuracy score of Random Forest is :{}".format(score_RFC))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, RFC_pred)
print("Random Forest")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(y_test, RFC_pred).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# <h2>Comparative Performance Analysis</h2>
# 
# Accuracy, Recall, Precision, F1 Score, AUC Score

# In[30]:


from sklearn import metrics

Model_Pred = [LR_pred,SVM_pred,NB_pred,RFC_pred,KNN_pred]
Model_Name = ["Logistic_Regression","SupportVectorMachines","NaiveBayes","RandomForest","KNN"]

for Pred,Name in zip(Model_Pred,Model_Name):
    print("\t\t{} Metrics".format(Name))
    print("---------------------------------------------")
    print("\n")
    print("Accuracy: " + str(metrics.accuracy_score(y_test,Pred)))
    print("Precision late: " + str(metrics.precision_score(y_test,Pred)))
    print("Precision early: " + str(metrics.precision_score(y_test,Pred,pos_label=0)))
    print("Recall: " + str(metrics.recall_score(y_test,Pred)))
    print("F1 score: " + str(metrics.f1_score(y_test,Pred)))
    print("AUC score: " + str(metrics.roc_auc_score(y_test,Pred)))
    print("\n")
    print(metrics.classification_report(y_test,Pred))
    print("\n\n")
    


# <h2>Ensemble Learning</h2>
# 
# 
# Our method, Hard voting with VotingClassifier, Soft voting with VotingClassifier

# In[31]:


Ensamble_pred = LR_pred+SVM_pred+NB_pred+RFC_pred+KNN_pred
Ensamble_pred = [1 if (i >= 3) else 0 for i in Ensamble_pred]

cm = confusion_matrix(y_test, Ensamble_pred)
print("Ensemble")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))


# In[32]:


from sklearn.ensemble import VotingClassifier

estimators=[('knn', KNN), ('rf', RFC), ('log_reg',LR),('gnb', NB),('svm',SVM)]

EC_hard = VotingClassifier(estimators=estimators,voting='hard')
EC_hard.fit(x_train, y_train)

EC_hard_pred = EC_hard.predict(x_test)
cm = confusion_matrix(y_test, EC_hard_pred)
print("Ensemble Hard Vote")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))


# In[33]:


from sklearn.ensemble import VotingClassifier

estimators=[('knn', KNN), ('rf', RFC), ('log_reg',LR),('gnb', NB)]

EC_soft = VotingClassifier(estimators=estimators,voting='soft')
EC_soft.fit(x_train, y_train)

EC_soft_pred = EC_soft.predict(x_test)
cm = confusion_matrix(y_test, EC_soft_pred)
print("Ensemble Soft Vote")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))


# In[34]:


from sklearn import metrics

Ensamble_Model_Pred = [Ensamble_pred,EC_hard_pred,EC_soft_pred]
Ensamble_Model_Name = ["Our Method","Hard Vote","Soft Vote"]

for Pred,Name in zip(Ensamble_Model_Pred,Ensamble_Model_Name):
    print("\t\t{} Metrics".format(Name))
    print("---------------------------------------------")
    print("\n")
    print("Accuracy: " + str(metrics.accuracy_score(y_test,Pred)))
    print("Precision late: " + str(metrics.precision_score(y_test,Pred)))
    print("Precision early: " + str(metrics.precision_score(y_test,Pred,pos_label=0)))
    print("Recall: " + str(metrics.recall_score(y_test,Pred)))
    print("F1 score: " + str(metrics.f1_score(y_test,Pred)))
    print("AUC score: " + str(metrics.roc_auc_score(y_test,Pred)))
    print("\n")
    print(metrics.classification_report(y_test,Pred))
    print("\n\n")
    


# <h2>Parameter Tuning</h2>
# 
# 
# Optimizing parameters using GridSearchCV for KNN and Random Forest

# In[35]:


from sklearn.model_selection import GridSearchCV


# In[36]:


p = [{'n_neighbors':[i for i in range(3,30,2)], 'metric':['minkowski','manhattan','euclidean']}]


gs = GridSearchCV(estimator=KNN,
                  param_grid=p,
                  scoring='accuracy',
                  cv=10)
grid_search = gs.fit(x_train, y_train)
print("Best Estimator:\n")
print(grid_search.best_estimator_)
print("\nBest Parameters:\n")
print(grid_search.best_params_)
print("\nBest Score: " + str(grid_search.best_score_))


# In[37]:


p_rf = [{'n_estimators':[i for i in range(10,100,10)], 'criterion':['gini','entropy'] , 
                         'max_depth':[None,10,50,100,200]}]

gs = GridSearchCV(estimator=RFC,
                  param_grid=p_rf,
                  scoring='accuracy',
                  cv=10)
grid_search = gs.fit(x_train, y_train)
print("Best Estimator:\n")
print(grid_search.best_estimator_)
print("\nBest Parameters:\n")
print(grid_search.best_params_)
print("\nBest Score: " + str(grid_search.best_score_))


# # CREATING DEEP LEARNING MODEL

# In[38]:


data.head()


# In[39]:


X.head()


# In[40]:


InputDeep = np.asarray(X)
InputDeep


# In[41]:


y.head()

Y = pd.get_dummies(y)
Y.columns = ["Not_Exit","Exit"]

#Y = y
Y.head()


# In[42]:


label = np.asarray(Y)
label.shape


# In[43]:


label


# In[44]:


print(InputDeep.shape)
print(InputDeep.shape[1])

Inputshape = InputDeep.shape[1]

#InputDeep = InputDeep.astype(int)


# In[45]:


from keras.layers import Input, Dense, Dropout,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LeakyReLU

input_layer = Input(shape=(Inputshape,))
x = Dense(units=128)(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Dense(units=64)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Dropout(rate=0.25)(x)

x = Dense(units=32)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

output_layer = Dense(units=label.shape[1], activation = 'softmax')(x)

DeepModel = Model(input_layer, output_layer)
DeepModel.summary()

opt = Adam(lr=0.0005)
DeepModel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[46]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

xtrain, xtest,ytrain,ytest = train_test_split(InputDeep,label,test_size=0.20, random_state=0)

sc = StandardScaler()#if i dont use scandart Scaler my accuracy willl drop
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

history = DeepModel.fit(xtrain,ytrain,validation_split=0.10,batch_size=16,epochs=100,shuffle=True)


# In[47]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[48]:


evaluating = DeepModel.evaluate(xtest, ytest)
Deep_pred = DeepModel.predict(xtest)

#Deep_pred = np.asarray([0 if i < 0.5 else 1 for i in Deep_pred])

CLASSES = np.asarray([0,1])
preds_single = CLASSES[np.argmax(Deep_pred, axis = 1)]
actual_single = CLASSES[np.argmax(ytest, axis = 1)]


# In[49]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

score_Deep = accuracy_score(actual_single,preds_single)*100
print("accuracy score of Deep Learning is :{}".format(score_Deep))

cm = confusion_matrix(actual_single, preds_single)
print("Deep Learning Model")
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
print("Accuracy : " + str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])))

tn, fp, fn, tp = confusion_matrix(actual_single, preds_single).ravel()
print("True positive: " + str(tp))
print("True negative: " + str(tn))
print("False positive: " + str(fp))
print("False negative: " + str(fn))


# <h2> Conclusion </h2>
# 
# 
# I have tried to Create a model to predict if this person will exit the bank or not. It vas a categorization problem and i have binary label.Firs I did to extract data and create train and test label. I have a datasets with 10000 varibales and use 80 oercent of them to train model. I use pandas.getDummies for country and Gender to make our model more accuracy. After I split the data i use standart scaler for both train and test data.
# After I prepare data I used 5 model:Logistic Regression,Naive Bayes, SVM,KNN,RandomForest. in Knn model İ also use parameter tuning and test model with 10 different neighbors. according to result best neighbor for KNN is 8.
# If we compare all Models performance wee see
# <br>NaiveBayes Metrics has accuracy :81.2%</br>
# <br>RandomForest Metrics has accuracy :86.6%</br>
# <br>KNN Metrics has accuracy :83.9%</br>
# <br>SupportVectorMachines Metrics has accuracy: 86.4%</br>
# <br>Logistic_Regression Metrics has accuracy: 81.3%</br>
# <br>Even all of them are better for categorization we can see</br> RandomForest and SVM are ver close each other and have best perform for categorization this problem.Unfortunately NaiveBayes and Logistic_Regression has good result compare the other result(of course their accuracies are not bad). 
# As we can see soft or hard classifiers, hard classifiers have slightly better performance than soft classifiers<br></br>
# Finaly I generate Deep Leraning model and in order to increase his accuracy i used BatchNormalization and Dropout layer. As a result , this model has a great performance compare to other models, almost 87%
