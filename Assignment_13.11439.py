
# coding: utf-8

# ## Predicting Survival in the Titanic Data Set

# <p>I will be using a decision tree to make predictions about the Titanic data set from
# Kaggle. This data set provides information on the Titanic passengers and can be used to
# predict whether a passenger survived or not.
# 
# <p>I will use only Pclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard),
# and Fare to predict whether a passenger survived.

# ## Importing Modules

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sb

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pylab import rcParams

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

import pydotplus
from IPython.display import Image, display
from matplotlib.colors import ListedColormap


# ## Data Reading from URL

# In[2]:


url= "https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(url) #Read CSV file into DataFrame


# ## Data Exploration/Analysis

# In[3]:


titanic.head() #Returns the first 5 rows of titanic dataframe 


# In[4]:


titanic.columns #Columns of titanic dataframe


# In[5]:


titanic.info() #Prints information about titanic DataFrame.


# In[6]:


titanic.describe() #The summary statistics of the titanic dataframe


# In[7]:


titanic.shape #Return a tuple representing the dimensionality of titanic DataFrame.


# In[8]:


titanic.isnull().values.any() #Check for any NA’s in the dataframe.


# In[9]:


total = titanic.isnull().sum().sort_values(ascending=False)
per_1 = titanic.isnull().sum()/titanic.isnull().count()*100
per_2 = (round(per_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, per_2], axis=1, keys=['Total', '%'])
missing_data


# <p>There are missing data in cabin, Age and Embarked coulmns.</p>

# ## Data Visualization

# ### Age and Sex:

# In[10]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 9))
women = titanic[titanic['Sex']=='female']
men = titanic[titanic['Sex']=='male']
ax = sb.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0], kde =False)
ax = sb.distplot(women[women['Survived']==0].Age.dropna(), bins=30, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sb.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1], kde = False)
ax = sb.distplot(men[men['Survived']==0].Age.dropna(), bins=30, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# <p>We can see that men have a high probability of survival when they are between 18 and 30 years old, which is also a little bit true for women but not fully. For women the survival chances are higher between 14 and 40.</p>

# ### Embarked, Pclass and Sex:

# In[11]:


#FacetGrid class helps in visualizing distribution of one variable as well as the relationship 
#between multiple variables separately within subsets of your dataset using multiple panels
FacetGrid = sb.FacetGrid(titanic, row='Embarked', size=5, aspect=2)
FacetGrid.map(sb.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# <p>Embarked seems to be correlated with survival, depending on the gender. </p>

# <p>Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port C. Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S.</p>

# ### Pclass:

# In[12]:


sb.barplot(x='Pclass', y='Survived', data=titanic)


# <p>Pclass is contributing to a persons chance of survival, especially if this person is in class 1. </p>

# In[13]:


grid = sb.FacetGrid(titanic, col='Survived', row='Pclass', size=3, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=30)
grid.add_legend()


# <p>There is a high probability that a person in pclass 3 will not survive. </p>

# ### SibSp and Parch

# In[14]:


titanic_temp = titanic
titanic_temp['relatives'] = titanic_temp['SibSp'] + titanic_temp['Parch']
titanic_temp.loc[titanic_temp['relatives'] > 0, 'not_alone'] = 0
titanic_temp.loc[titanic_temp['relatives'] == 0, 'not_alone'] = 1
titanic_temp['not_alone'] = titanic_temp['not_alone'].astype(int)
titanic_temp['not_alone'].value_counts()


# In[15]:


axes = sb.factorplot('relatives','Survived', data=titanic_temp, aspect = 3, )


# ## Data Preprocessing

# ### Missing Data: Cabin
# <p>Observing the dataframe: A cabin number looks like ‘C123’ and the letter refers to the deck.</p>
# <p>I am going to extract observed cabin feature and create a new feature, that contains a persons deck. Afterwords I will convert the feature into a numeric variable. The missing values will be converted to zero. </p>

# In[16]:


titanic_new = titanic

import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

titanic_new['Cabin'] = titanic_new['Cabin'].fillna("U0")
titanic_new['Deck'] = titanic_new['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
titanic_new['Deck'] = titanic_new['Deck'].map(deck)
titanic_new['Deck'] = titanic_new['Deck'].fillna(0)
titanic_new['Deck'] = titanic_new['Deck'].astype(int)

#I will now drop the cabin feature
titanic_new = titanic_new.drop(['Cabin'], axis=1)


# ### Missing Data: Age

# <p> I will create an array that contains random numbers, which are computed based on the mean age value in regards to the standard deviation and is_null. <p>

# In[17]:


mean = titanic_new["Age"].mean()
std = titanic_new["Age"].std()
is_null = titanic_new["Age"].isnull().sum()

#Computing random numbers between the mean, std and is_null
rand_age = np.random.randint(mean - std, mean + std, size = is_null)

#Filling NaN values in Age column with random values generated
age_slice = titanic_new["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
titanic_new["Age"] = age_slice
titanic_new["Age"] = titanic_new["Age"].astype(int)


# In[18]:


titanic_new["Age"].isnull().sum() #Check for any null in the Age column.


# ### Missing Data: Embarked

# <p>I will fill Embarked misssing values with the most common one. </p>

# In[19]:


titanic_new['Embarked'].describe()  #The summary statistics of the Embarked column


# In[20]:


common_value = 'S' #By Observing the Embarked column
titanic_new['Embarked'] = titanic_new['Embarked'].fillna(common_value)


# In[21]:


titanic_new.info() #Prints information about titanic_new DataFrame.


# In[22]:


titanic_new.isnull().values.any() #Check for any NA’s in the dataframe.


# ### Converting Features: Sex

# <p> Converting ‘Sex’ feature into numeric. </p>

# In[23]:


genders = {"male": 1, "female": 0}
titanic_new['Sex'] = titanic_new['Sex'].map(genders)


# ### Converting Features: Fare

# <p>Converting “Fare” from float to int64.</p>

# In[24]:


titanic_new['Fare'] = titanic_new['Fare'].fillna(0)
titanic_new['Fare'] = titanic_new['Fare'].astype(int)


# In[25]:


titanic_new.info() #Prints information about titanic_new DataFrame.


# ## titanic and titanic_new dataframe

# In[26]:


titanic.head() #Returns the first 5 rows of the titanic dataframe 


# In[27]:


titanic_new.head() #Returns the first 5 rows of the titanic_new dataframe 


# In[28]:


titanic.describe() #The summary statistics of the titanic dataframe


# In[29]:


titanic_new.describe() #The summary statistics of the titanic_new dataframe


# In[30]:


titanic.info() #Prints information about titanic DataFrame


# In[31]:


titanic_new.info() #Prints information about titanic_new DataFrame


# ## Train, Test & Split

# In[32]:


#Selecting features
X = titanic_new[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']] 
Y = titanic_new['Survived']


# In[33]:


#Spliting data randomly into 70% training and 30% test
from sklearn import tree, metrics, model_selection, preprocessing
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)
print(X_train.shape) #Training data shape (predictor values) : 70%
print(X_test.shape) #Test data shape (predictor values) : 30%
print(Y_train.shape) #Training data shape (target values) : 70%
print(Y_test.shape) #Test data shape (target values) : 30%


# ## Creating and Training the Model

# In[34]:


#Training the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, Y_train)


# ## Predicting Survived using Test Data

# In[35]:


#Using the model to make predictions with the test data
Y_pred = dtree.predict(X_test)


# In[36]:


Y_pred.shape #Return a tuple representing the dimensionality of Y_pred.


# In[37]:


Y_pred 


# ## Visualization

# ### Visualize how the tree splits using GraphViz

# In[38]:


import os     
os.environ["PATH"] += os.pathsep + 'C:\\Anaconda3\\Library\\bin\\graphviz'
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'],  
                                class_names=['Survived','Died'])
graph = pydotplus.graph_from_dot_data(dot_data)  
display(Image(graph.create_png()))


# ## Evaluate the model's performance

# ### How did created model perform?

# In[39]:


#How did created model perform?
count_misclassified = (Y_test != Y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(Y_test, Y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# ### Holdout Validation
# 

# In[40]:


from sklearn.cross_validation import KFold

cv = KFold(n=len(X), #Number of elements
           n_folds=10, #Desired number of cv folds
           random_state=12) 

fold_accuracy = []

for train_fold, valid_fold in cv:
    train = X.loc[train_fold] #Extract train data with cv indices
    valid = X.loc[valid_fold] #Extract valid data with cv indices
    
    train_y = Y.loc[train_fold]
    valid_y = Y.loc[valid_fold]
    
    model = dtree.fit(X = train, y = train_y)
    valid_acc = model.score(X = valid, y = valid_y)
    fold_accuracy.append(valid_acc)    

print("Accuracy per fold: ", fold_accuracy, "\n")
print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))


# ### Cross Validation

# In[41]:


from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator= dtree, #Model to test
                X = X,  
                y = Y,                #Target variable
                scoring = "accuracy", #Scoring metric    
                cv=10)                #Cross validation folds

print("Accuracy per fold: ")
print(scores)
print("Average accuracy: ", scores.mean())

