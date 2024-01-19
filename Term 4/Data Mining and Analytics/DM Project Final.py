#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore")
from sklearn.impute import SimpleImputer


# # Data Exploration and Data Cleaning

# ### Loading Dataset

# In[2]:


data = pd.read_csv(r'C:\Users\Verina\Documents\data mining project\adult.csv',na_values='?')
data
imported_data = data.copy()
data


# ### Exploring the Data

# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


numeric_data = data.select_dtypes(include=['number']).columns
data[numeric_data].isnull().sum()


# In[6]:


categorical_data = data.select_dtypes(include=['object']).columns
data[categorical_data].isnull().sum()


# #### Conclusion:
# Only categorical data have missing values

# In[7]:


# Statistics about data
data[numeric_data].describe().transpose()


# In[8]:


# Unique values
data[categorical_data].describe().transpose()


# ### Data Handling and Cleaning

# #### Use SimpleImputer to handle missing data:
# This method is used to handle missing data by replacing all Null values with the top frequent value in its feature.

# In[9]:


# Using most frequent
# Missing data is categorical
# Initialize the imputer model
imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values using the most frequent strategy
imputer.fit(data[categorical_data])
imputed_data = imputer.transform(data[categorical_data])

# Convert the imputed data back to a DataFrame
imputed_data = pd.DataFrame(imputed_data, columns=data[categorical_data].columns)
data = pd.concat([imputed_data,data[numeric_data]],axis=1) 


# In[10]:


data.head()


# In[11]:


data.isnull().sum()
# Now our Data doesn't have missing values


# #### Handling Duplications 

# In[12]:


# Removing duplicates(repetition)
# Check if there are any duplicates using duplicated() method
data.duplicated().sum()


# In[13]:


data = data.drop_duplicates()


# In[14]:


data.duplicated().sum()


# ##### Visualizing Data Distributions After Cleaning

# In[15]:


data[numeric_data].hist(figsize=(12,9),layout=(3,3),color='mediumpurple',edgecolor='indigo',grid=False)
plt.suptitle('Numeric Data Distributions')


# In[16]:


fig = plt.figure(figsize=(10,12),layout='constrained')
fig.suptitle('Categorical Data Distributions')

gs = fig.add_gridspec(3,3)


ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(list(data[categorical_data[0]].value_counts(sort=False).keys()),
             list(data[categorical_data[0]].value_counts(sort=False).values),color='slateblue')
ax1.set_title(categorical_data[0])
ax1.tick_params(axis='x', rotation=90)

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(list(data[categorical_data[1]].value_counts(sort=False).keys()),
             list(data[categorical_data[1]].value_counts(sort=False).values),color='slateblue')
ax2.set_title(categorical_data[1])
ax2.tick_params(axis='x', rotation=90)

ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(list(data[categorical_data[2]].value_counts(sort=False).keys()),
             list(data[categorical_data[2]].value_counts(sort=False).values),color='slateblue')
ax3.set_title(categorical_data[2])
ax3.tick_params(axis='x', rotation=90)

ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(list(data[categorical_data[3]].value_counts(sort=False).keys()),
             list(data[categorical_data[3]].value_counts(sort=False).values),color='slateblue')
ax4.set_title(categorical_data[3])
ax4.tick_params(axis='x', rotation=90)

ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(list(data[categorical_data[4]].value_counts(sort=False).keys()),
             list(data[categorical_data[4]].value_counts(sort=False).values),color='slateblue')
ax5.set_title(categorical_data[4])
ax5.tick_params(axis='x', rotation=90)

ax6 = fig.add_subplot(gs[1, 2])
ax6.bar(list(data[categorical_data[5]].value_counts(sort=False).keys()),
             list(data[categorical_data[5]].value_counts(sort=False).values),color='slateblue')
ax6.set_title(categorical_data[5])
ax6.tick_params(axis='x', rotation=90)

ax7 = fig.add_subplot(gs[2, 0:2])
ax7.bar(list(data[categorical_data[7]].value_counts(sort=False).keys()),
             list(data[categorical_data[7]].value_counts(sort=False).values),color='slateblue')
ax7.set_title(categorical_data[7])
ax7.tick_params(axis='x', rotation=90)

ax8 = fig.add_subplot(gs[2, 2])
ax8.bar(list(data[categorical_data[6]].value_counts(sort=False).keys()),
             list(data[categorical_data[6]].value_counts(sort=False).values),color='slateblue')
ax8.set_title(categorical_data[6])
ax8.tick_params(axis='x', rotation=90)



plt.show()


# In[17]:


# correlation between numeric attributes
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(),annot=True,cmap = 'Oranges',vmin=-1, vmax=1)


# ##### Visualizing Relationship between income and other attributes

# In[18]:


sns.set_palette("Spectral")
data.plot.box(subplots=True,layout=(2,3),by='income',figsize=(15,10),patch_artist=True,title='Relationship Between Income and Numeric Data')


# In[19]:


fig = plt.figure(figsize=(15,15),layout='tight')
fig.suptitle('Relationship Between Income and Categorical Data')

gs = fig.add_gridspec(3,3)

sns.set_palette("Spectral")

ax1 = fig.add_subplot(gs[0, 0])
sns.countplot(ax=ax1, data=data, x=categorical_data[0],hue='income')
ax1.tick_params(axis='x', rotation=90)

ax2 = fig.add_subplot(gs[0, 1])
sns.countplot(ax=ax2, data=data, x=categorical_data[1],hue='income')
ax2.tick_params(axis='x', rotation=90)

ax3 = fig.add_subplot(gs[0, 2])
sns.countplot(ax=ax3, data=data, x=categorical_data[2],hue='income')
ax3.tick_params(axis='x', rotation=90)

ax4 = fig.add_subplot(gs[1, 0])
sns.countplot(ax=ax4, data=data, x=categorical_data[3],hue='income')
ax4.tick_params(axis='x', rotation=90)

ax5 = fig.add_subplot(gs[1, 1])
sns.countplot(ax=ax5, data=data, x=categorical_data[4],hue='income')
ax5.tick_params(axis='x', rotation=90)

ax6 = fig.add_subplot(gs[1, 2])
sns.countplot(ax=ax6, data=data, x=categorical_data[5],hue='income')
ax6.tick_params(axis='x', rotation=90)

ax7 = fig.add_subplot(gs[2, 0:2])
sns.countplot(ax=ax7, data=data, x=categorical_data[7],hue='income')
ax7.tick_params(axis='x', rotation=90)

ax8 = fig.add_subplot(gs[2, 2])
sns.countplot(ax=ax8, data=data, x=categorical_data[6],hue='income')
ax8.tick_params(axis='x', rotation=90)

plt.show()


# In[20]:


plt.figure(figsize=(8,6),layout='tight')
plt.subplot(2,1,1)
sns.countplot(x=data['education.num'],palette='plasma',
             order=data['education.num'].value_counts().index)
plt.subplot(2,1,2)
sns.countplot(x=data['education'],palette='plasma',
             order=data['education'].value_counts().index)
plt.xticks(rotation=45)


# ###### by plotting the counts of education and education.num columns, we can see they represent the same thing

# In[21]:


data.drop(columns=['education.num'],inplace=True)
data


# In[22]:


# Note that age is skewed and have outliers 
sns.set(rc={'figure.figsize':(11.7,8.27),'figure.dpi':90})
sns.distplot(x=data["age"],kde=True,bins=90,color="#0072B2")
plt.xticks(np.arange(15,95,5))
plt.xlabel("age")
plt.show()


# ### Binning
#  Binning can be a useful technique for reducing the impact of noise in data
#  age need to be binned.

# In[23]:


data['Age Groups'] = pd.cut(data['age'],7)


# In[24]:


data['Age Groups'].unique()


# In[25]:


data = data.drop('age',axis=1)


# In[26]:


# Visualization for grouped data
sns.countplot(data['Age Groups'], hue = data['income'], palette='flare')
plt.show()


# # Data Transformation

# ### Feature Engineering

# In[27]:


# Education feature
data.education= data.education.replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th'], 'School')
data.education = data.education.replace('HS-grad', 'High School')
data.education = data.education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'Higher')
data.education = data.education.replace('Bachelors', 'Undergrad')
data.education = data.education.replace('Masters', 'Graduated')
data.education = data.education.replace('Doctorate', 'Doc')
# We replace the similar labels in Education feature with the same, decreasing the number that will not affect the accuracy.


# In[28]:


# Martial status
data['marital.status']= data['marital.status'].replace(['Married-civ-spouse', 'Married-AF-spouse'], 'Married')
data['marital.status']= data['marital.status'].replace(['Never-married'], 'Not-Married')
data['marital.status']= data['marital.status'].replace(['Divorced', 'Separated','Widowed',
                                                   'Married-spouse-absent'], 'Other')
# We do as the previous line


# ##### Visualize the effect of data transformation

# In[29]:


# group education by value counts and reorder rows
edu1 = imported_data[['education']].value_counts().to_frame()
edu1=edu1.reindex([['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th','HS-grad',
              'Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college','Bachelors','Masters','Doctorate']]).reset_index()
edu1.columns = ['education','count']
edu2 = data[['education']].value_counts().to_frame()
edu2=edu2.reindex([['School','High School','Higher','Undergrad','Graduated','Doc']]).reset_index()
edu2.columns = ['education','count']

#define colors
colours = {'School':'deepskyblue',
           'Preschool':'powderblue',
           '1st-4th':'lightblue',
           '5th-6th':'skyblue', 
           '7th-8th':'lightsteelblue', 
           '9th':'lightcyan',
           '10th':'lightskyblue',
           '11th':'aliceblue',
           '12th':'paleturquoise',
           'High School':'mediumpurple',
           'HS-grad':'mediumpurple',
           'Higher':'sandybrown',
           'Assoc-voc':'peachpuff',
           'Assoc-acdm':'linen',
           'Prof-school':'bisque',
           'Some-college':'wheat',
           'Undergrad':'hotpink',
           'Bachelors':'hotpink',
           'Graduated':'silver',
           'Masters':'silver',
           'Doc':'palegreen',
           'Doctorate':'palegreen'}

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.pie(edu1['count'],labels=edu1['education'],colors=[colours[key] for key in list(edu1['education'])])
plt.title('education (before transformation)')
plt.subplot(1,2,2)
plt.pie(edu2['count'],labels=edu2['education'],colors=[colours[key] for key in list(edu2['education'])])
plt.title('education (after transformation)')


# In[30]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xticks(rotation=20)
plt.title('marital.status (before transformation)')
sns.countplot(x=imported_data['marital.status'],hue=imported_data['income'],palette="viridis")
plt.subplot(1,2,2)
plt.xticks(rotation=20)
sns.countplot(x=data['marital.status'],hue=data['income'],palette="viridis")
plt.title('marital.status (after transformation)')


# ### Encoding
# Encoding is a method from the sklearn library, It is used to transform categorical or optimal data to numerical data based on which method will be chosen.
# LabelEncoder( ) is better with optimal data or categorical with less number of labels in one feature.
# 
# In the One-Hot encoder, each category is represented by a binary vector with a length equal to the number of categories present in the data. We replace the similar labels in Education feature with the same, decreasing the number that will not affect the accuracy.

# In[31]:


from sklearn import preprocessing

tr = preprocessing.LabelEncoder()
data['sex'] = tr.fit_transform(data['sex'])
data['income'] = tr.fit_transform(data['income'])


# In[32]:


# Select the categorical columns
cat_cols = data.select_dtypes(include=['object', 'category']).columns


# In[33]:


# Perform one-hot encoding on the categorical columns
data = pd.get_dummies(data, columns=cat_cols)


# In[34]:


data
# Now, features are numeric data that is better for the model and its accuracy.


# In[35]:


# one benefit of separating dummy features into columns is to decrease correlation between features
plt.figure(figsize=(12,8))
filter_col = [col for col in data if col.startswith('occupation')]
sns.heatmap(data[filter_col].corr(), annot=True, cmap='vlag',vmin=-1, vmax=1)


# # Data Normalization

# In[36]:


numeric_data = data.select_dtypes(include=['number']).columns
numeric_data
for col in numeric_data:
    max_col = data[col].max()
    min_col = data[col].min()
    new_max = 1
    new_min = 0
    data[col]=((data[col] - min_col)/(max_col - min_col))*(new_max-new_min) + new_min
data[numeric_data].head()


# In[37]:


data.head()


# ##### Visualize the effect of normalization

# In[38]:


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.boxplot(data=imported_data[['fnlwgt','capital.gain','capital.loss','hours.per.week']])
plt.title('numeric data before normalization')
plt.subplot(1,2,2)
sns.boxplot(data=data[['fnlwgt','capital.gain','capital.loss','hours.per.week']])
plt.title('numeric data after normalization')


# # Feature Selection and Data Reduction

# ### Select KBest Method

# In[39]:


from sklearn.feature_selection import SelectKBest, f_regression
# Split the data into features and target
X = data.drop('income', axis=1)
y = data['income']

# Use SelectKBest to select the top 65 features based on f_regression scores
select = SelectKBest(score_func=f_regression, k=65)
select.fit(X, y)

# Get the selected features
selected_feature = X.columns[select.get_support()]

# Print the selected features
print(selected_feature)


# In[40]:


correl = data[selected_feature].corrwith(data["income"]).to_frame()
fig=plt.figure(figsize=(5,15))
fig.suptitle('Correlation between Selected Features and Target Variable (Income)')
sns.heatmap(correl, annot=True, cmap='RdYlGn',vmin=-1, vmax=1)


# # Logistic regression

# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

lr = LogisticRegression()


# ## training the model 
# 

# In[43]:


model = lr.fit(X_train, y_train)
print("Acc on training data: {:,.3f}".format(lr.score(X_train, y_train)))


# ## testing the model 

# In[44]:


prediction = model.predict(X_test)
print("Acc on test data: {:,.3f}".format(lr.score(X_test, y_test)))


# ## Applying k-Fold Cross-Validation to Logistic Regression

# In[45]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold

cv =RepeatedKFold(n_splits=10, n_repeats=2, random_state=0)


# In[46]:


scores = cross_val_score(lr, X, 
         y, cv=cv)
print(np.mean(scores))


# ## Model evaluation

# In[47]:


accuracy_score(prediction, y_test.values)


# In[48]:


sns.set(rc={'figure.figsize':(9,4)})
cfm = confusion_matrix(prediction, y_test.values)
sns.heatmap(cfm, annot=True,cmap="Blues")
print(cfm)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')


# In[49]:


print(classification_report(y_test, prediction))


# ## ROC curve

# In[50]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, prediction)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, prediction)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# # Logistic Regression From Scratch

# In[51]:


import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression23():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred


# In[52]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = LogisticRegression23(lr=0.09)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)


# # Random Forest

# In[53]:


from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[54]:


x = data.drop('income', axis=1)
y = data['income']


# In[55]:


x_train, x_test, y_train, y_test = train_test_split(x[selected_feature], y, test_size=0.3, random_state=1) # 70% training and 30% test


# In[56]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy' ,random_state = 51)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)
accuracy_score(y_test, y_pred)


# In[57]:


cm=confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(6, 4))
plt.title('Heatmap of Confusion Matrix', fontsize = 12)
sns.heatmap(cm, annot=True, fmt="d", cmap="pink_r")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[58]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[59]:


print(classification_report(y_test, y_pred))


# In[60]:


param_grid = {
    'n_estimators': [10, 20, 60],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3]
}


# Perform grid search
grid_search = GridSearchCV(rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Print best hyperparameters and corresponding accuracy score
print("Best hyperparameters:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)


# # Neural Network

# In[61]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf


# In[62]:


tf.random.set_seed(20) 
np.random.seed(20)
inputs = tf.keras.Input(shape=(65,))
x = tf.keras.layers.Dense(30, activation='relu')(inputs)
x = tf.keras.layers.Dense(10, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fit model
history = model.fit(x_train, y_train, epochs=30)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[63]:


y_predN = np.round(model.predict(x_test))
cm_N=confusion_matrix(y_test, y_predN )
fig = plt.figure(figsize=(6, 4))
plt.title('Heatmap of Confusion Matrix', fontsize = 12)
sns.heatmap(cm_N, annot=True, fmt="d", cmap="Pastel1_r")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[64]:


fpr, tpr, thresholds = roc_curve(y_test, y_predN)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_predN)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[65]:


print(classification_report(y_test, y_predN))

