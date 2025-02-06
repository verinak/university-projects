#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ## Exploratory data analysis

# In[2]:


data = pd.read_csv(r'C:\Users\olama\Downloads\data01.csv')
data.head()


# In[3]:


print(data.dtypes)


# ### Display basic statistics for each numerical column
# 

# In[4]:


data.describe()


# In[5]:


data.info()


# ## Data Cleaning

# In[6]:


data_missing= (data== '?').sum()
# data_missing


# In[7]:


data_missing2 = data.isna().sum()
# print(data_missing2)


# In[8]:


# Replace the values with NaN
data[data == '?'] = np.nan

# Iterate over each column in data_missing2
for col in data_missing2.index:
    data[col].fillna(data[col].mode()[0], inplace=True)


# In[9]:


data.duplicated().sum()


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your dataset is stored in a DataFrame called 'df' with a column named 'outcome'
outcome_counts = data['outcome'].value_counts()
total_samples = outcome_counts.sum()

outcome_0 = outcome_counts[0]
outcome_1 = outcome_counts[1]

proportion_0 = outcome_0 / total_samples
proportion_1 = outcome_1 / total_samples

# Plotting the bar plot
labels = ['0', '1']
values = [outcome_0, outcome_1]

plt.bar(labels, values)
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Distribution of Outcome')

plt.show()


# In[11]:


# Calculate quartiles and IQR for each column
outliers = pd.DataFrame()

for column in data.columns:
    Q1 = data[column].describe()['25%']
    Q3 = data[column].describe()['75%']
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Check for outliers in the current column
    column_outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    if not column_outliers.empty:
        # Print the column name and outlier values
        print(f"Column: {column}")
        print(column_outliers[column])
        print("\n")


# ### Data Normalization

# In[12]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Normalize the columns of the DataFrame
normalized_data = scaler.fit_transform(data)

# Create a new DataFrame with normalized values
normalized_data = pd.DataFrame(normalized_data, columns=data.columns)
normalized_data


# ## Feature Selection
# 
# Filter model: Univariate Feature Selection, specifically the SelectKBest method. This method is a type of filter-based feature selection technique.
# 
# Filter-based feature selection methods evaluate each feature independently of the others. They assign a score or rank to each feature based on their relationship with the target variable. Features with higher scores are considered more important or relevant to the target variable.

# In[13]:


X = normalized_data.drop("outcome", axis=1)  # Features
y = normalized_data["outcome"]  # Target variable

# Apply SelectKBest class to extract feature scores
best_features = SelectKBest(score_func=f_classif, k='all')
fit = best_features.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concatenate the data frames for better visualization
feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
feature_scores.columns = ['Feature', 'Score']

# Print feature scores
print("Feature Scores:")
print(feature_scores.to_string(index=False))

# Select top 10 features based on scores
selected_features = feature_scores.nlargest(40, 'Score')
print("\nTop 40 Features:")
print(selected_features.to_string(index=False))


# In[14]:


selected_feature_names = selected_features['Feature'].tolist()

# Filter the data dataframe to only include the selected features
filtered_data = normalized_data[selected_feature_names]


# In[15]:


filtered_data


# ###### Wrapper model: 
# Recursive Feature Elimination (RFE) with logistic regression. RFE is a wrapper-based feature selection technique that uses a machine learning model to rank the features based on their importance.

# In[16]:


X = normalized_data.drop("outcome", axis=1)  # Features
y = normalized_data["outcome"]  # Target variable

# Feature extraction using RFE with logistic regression
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, n_features_to_select=40)
fit = rfe.fit(X, y)

# Get rankings of all features
feature_rankings = pd.DataFrame({'Feature': X.columns, 'Ranking': fit.ranking_})

# Print rankings of all features
print("Rankings of All Features:")
print(feature_rankings.to_string(index=False))

# Get selected features
selected_features2 = feature_rankings[feature_rankings['Ranking'] == 1]

# Print selected features
print("\nSelected Features:")
print(selected_features2.to_string(index=False))


# In[17]:


selected_feature_names2 = selected_features2['Feature'].tolist()

# Filter the data dataframe to only include the selected features
filtered_data2 = normalized_data[selected_feature_names2]


# In[18]:


filtered_data2


# # PCA

# In[19]:


data_without_outcome= data.drop("outcome", axis=1)


# In[20]:


data_without_outcome


# In[21]:


scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(data_without_outcome), columns=data_without_outcome.columns)


# In[22]:


features_scaled


# In[23]:


# Apply PCA
pca = PCA(n_components=2)
pca_features = pd.DataFrame(pca.fit_transform(features_scaled))


# In[24]:


pca_features


# In[25]:


# Access PCA components (feature importance)
component_names = ['PC1', 'PC2']
explained_variance_ratio = pca.explained_variance_ratio_
component_df = pd.DataFrame(pca.components_, columns=features_scaled.columns.tolist())
component_df.index = component_names

print("Feature Importance:")
component_df


# In[26]:


correlation_matrix = np.corrcoef(features_scaled.T)
correlation_matrix


# In[27]:


import seaborn as sns
sns.heatmap(features_scaled.corr())


# In[28]:


sns.heatmap(pca_features.corr())


# In[29]:


# Calculate the absolute sum of PCA component weights for each original feature
feature_weights = np.abs(component_df).sum(axis=0)

# Sort the features based on their weights
sorted_features = feature_weights.sort_values(ascending=False)

# Determine the least important features to drop
num_features_to_drop = 10  # Specify the number of least important features to drop
features_to_drop = sorted_features.tail(num_features_to_drop).index.tolist()

print("Columns to drop for dimensionality reduction:")
print(features_to_drop)


# In[30]:


import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create the feature matrix X by excluding the 'ID' and 'outcome' columns
X = data.drop(['ID', 'outcome'], axis=1)

# Calculating VIF for each variable
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Print the VIF values
print("VIF values:")
print(vif_data)

# Create a bar plot for VIF values
plt.figure(figsize=(10, 6))
plt.bar(vif_data["Variable"], vif_data["VIF"])
plt.title("VIF Values for Each Variable")
plt.xlabel("Variable")
plt.ylabel("VIF Value")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[31]:


columns_to_drop_vif = vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
df = data.drop(columns=columns_to_drop_vif)
df


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load a sample dataset (you can replace this with your own dataset)
X = data.drop(['ID', 'outcome'], axis=1)
y = data["outcome"]

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i in range(len(np.unique(y))):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], c=colors[i], label=str(i))

plt.title('t-SNE Dimensionality Reduction')
plt.legend()
plt.show()


# # Modeling
# 
# 

# In[33]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from scipy.stats import randint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.metrics import Precision, Recall, AUC
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score


# In[34]:


X = filtered_data

Y=data['outcome']


# ### standardization

# In[35]:


scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
X=standardized_data


# In[36]:


print(X.shape)


# In[37]:


y.value_counts()


# In[38]:



# Assuming you have the test set labels
# ...

# Count the occurrences of each class
class_counts = y.value_counts()

# Set colors for each class using a rose color palette
class_colors = sns.color_palette("pastel", n_colors=len(class_counts))

# Bar chart for class distribution
plt.bar(class_counts.index, class_counts, color=class_colors)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution in Test Set')

# Set x-ticks to only 0 and 1
plt.xticks([0, 1])

plt.show()


# ## Handling Imabalanced Data using SMOTE

# In[39]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='minority')


# In[40]:


X_sm,y_sm=smote.fit_resample(X,y)


# In[41]:


X_sm.shape


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, train_size = 0.70, test_size = 0.30, random_state = 109)


# In[43]:



print("X_Train Shape:", X_train.shape)
print("X_Test Shape:", X_test.shape)


# In[44]:


y_train.value_counts()


# In[45]:



# Count the occurrences of each class
class_counts = y_train.value_counts()

# Set colors for each class using a rose color palette
class_colors = sns.color_palette("pastel", n_colors=len(class_counts))

# Bar chart for class distribution
plt.bar(class_counts.index, class_counts, color=class_colors)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution in Test Set')

# Set x-ticks to only 0 and 1
plt.xticks([0, 1])

plt.show()


# ### ClassifierMetricsVisualizer

# In[46]:



from scipy.stats import randint 
from collections import Counter
from statsmodels.formula.api import ols
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from IPython.display import display_html 
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve, classification_report
from scipy.stats import randint 
from collections import Counter

class ReusableUtils():
    
    def Generate_Model_Test_Classification_Report(self, model, X_test, y_test, model_name=""):

        y = 1.05
        # Report Title & Classification Mterics Abbreviations...
        fig, axes = plt.subplots(3, 1, figsize = (8, 3))
        axes[0].text(9, 1.8, " MODEL TEST REPORT", fontsize=30, horizontalalignment='center', 
                     color='DarkBlue', weight = 'bold')

        axes[0].axis([0, 10, 0, 10])
        axes[0].axis('off')

        axes[1].text(9, 4, "Model Name: " + model_name, style='italic', 
                             fontsize=18, horizontalalignment='center', color='DarkOrange', weight = 'bold')

        axes[1].axis([0, 10, 0, 10])
        axes[1].axis('off')

        axes[2].text(0, 4, "* 1 - Not Survived\t\t\t\t\t\t\t * 0 - Survived\n".expandtabs() +
                     "* MCC - Matthews Correlation Coefficient\t\t* AUC - Area Under The Curve\n".expandtabs() +
                     "* ROC - Receiver Operating Characteristics     " + 
                     "\t* AUROC - Area Under the Receiver Operating    Characteristics".expandtabs(), 
                     style='italic', fontsize=10, horizontalalignment='left', color='orangered')

        axes[2].axis([0, 10, 0, 10])
        axes[2].axis('off')

        scores = []
        metrics = ['F1       ', 'MCC      ', 'Precision', 'Recall   ', 'Accuracy ',
                   'AUC_ROC  ', 'AUC_PR   ']

        # Plot ROC and PR curves using all models and test data...
        y_pred = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)[:, 1:]

        fpr, tpr, thresholds = roc_curve(y_test.values.ravel(), y_pred)
        precision, recall, th = precision_recall_curve(y_test.values.ravel(), y_pred_probs)

        # Calculate the individual classification metic scores...
        model_f1_score = f1_score(y_test, y_pred)
        model_matthews_corrcoef_score = matthews_corrcoef(y_test, y_pred)
        model_precision_score = precision_score(y_test, y_pred)
        model_recall_score = recall_score(y_test, y_pred)
        model_accuracy_score = accuracy_score(y_test, y_pred)
        model_auc_roc = auc(fpr, tpr)
        model_auc_pr = auc(recall, precision)

        scores.append([model_f1_score,
                       model_matthews_corrcoef_score,
                       model_precision_score,
                       model_recall_score,
                       model_accuracy_score,
                       model_auc_roc,
                       model_auc_pr])

        sampling_results = pd.DataFrame(columns = ['Classification Metric', 'Score Value'])
        for i in range(len(scores[0])):
            sampling_results.loc[i] = [metrics[i], scores[0][i]]

        sampling_results.index = np.arange(1, len(sampling_results) + 1)

        class_report = classification_report(y_test, y_pred)
        conf_matx = confusion_matrix(y_test, y_pred)

        # Display the Confusion Matrix...
        fig, axes = plt.subplots(1, 3, figsize = (20, 4))
        sns.heatmap(conf_matx, annot=True, annot_kws={"size": 16},fmt='g', cbar=False, cmap="GnBu", ax=axes[0])
        axes[0].set_title("1. Confusion Matrix", fontsize=21, color='darkgreen', weight = 'bold', 
                          style='italic', loc='left', y=y)

        # Classification Metrics
        axes[1].text(5, 1.8, sampling_results.to_string(float_format='{:,.4f}'.format, index=False), style='italic', 
                     fontsize=20, horizontalalignment='center')
        axes[1].axis([0, 10, 0, 10])
        axes[1].axis('off')
        axes[1].set_title("2. Classification Metrics", fontsize=20, color='darkgreen', weight = 'bold', 
                          style='italic', loc='center', y=y)

        # Classification Report
        axes[2].text(0, 1, class_report, style='italic', fontsize=20)
        axes[2].axis([0, 10, 0, 10])
        axes[2].axis('off')
        axes[2].set_title("3. Classification Report", fontsize=20, color='darkgreen', weight = 'bold', 
                          style='italic', loc='center', y=y)

        plt.tight_layout()
        plt.show()

        # AUC-ROC & Precision-Recall Curve
        fig, axes = plt.subplots(1, 2, figsize = (14, 4))

        axes[0].plot(fpr, tpr, label = f"auc_roc = {model_auc_roc:.3f}")
        axes[1].plot(recall, precision, label = f"auc_pr = {model_auc_pr:.3f}")

        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].legend(loc = "lower right")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("4. AUC - ROC Curve", fontsize=15, color='darkgreen', ha='right', weight = 'bold', 
                          style='italic', loc='center', pad=1, y=y)

        axes[1].legend(loc = "lower left")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("5. Precision - Recall Curve", fontsize=15, color='darkgreen', ha='right', weight = 'bold', 
                          style='italic', loc='center', pad=3, y=y)

        plt.subplots_adjust(top=0.95) 
        plt.tight_layout()
        plt.show()
        
        return None


# In[47]:


utils = ReusableUtils()


# ## First Model: Decision Tree Model

# In[48]:


# Decision Tree with Default Parameters

tree = DecisionTreeClassifier(random_state = 100)

tree = tree.fit(X_train, y_train)

#tree_pred = tree.predict(X_test)

# Generate the model test classification report
utils.Generate_Model_Test_Classification_Report(tree, X_test, y_test, model_name="Default Decision Tree")


# ### Second Model : XGBoost

# In[49]:


xgb_best = XGBClassifier(random_state=123, seed=100)
xgb_best.fit(X_train, y_train)
utils.Generate_Model_Test_Classification_Report(xgb_best, X_test, y_test, model_name="XGBoost with balanced data")


# ## Combining the Models

# ### Stacking Method

# In[50]:


Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[51]:


xgb_classifier = XGBClassifier()
dt_classifier = DecisionTreeClassifier()


# In[52]:


bal_stacking_classifier = StackingClassifier(
    estimators=[ ('xgb', xgb_classifier), ('dt', dt_classifier)],
    final_estimator=RandomForestClassifier(n_estimators=20),
    stack_method='auto'
)


# In[53]:


# Train the stacking model
bal_stacking_classifier.fit(X_train, y_train)


# In[54]:


# Make predictions
y_pred_bal_stacking = bal_stacking_classifier.predict(X_test)


# In[55]:


utils.Generate_Model_Test_Classification_Report(bal_stacking_classifier, X_test, y_test, model_name="Balanced Stacking Model")


# ## Third Model : SVM

# In[56]:


# SVM with Default Parameters 

svm = SVC(C=1, gamma=0.01,kernel = 'rbf', probability = True, random_state = 100)

svm = svm.fit(X_train, y_train)

# Generate the model test classification report
utils.Generate_Model_Test_Classification_Report(svm, X_test, y_test, model_name="Default Support Vector Machine")


# ### Fine-Tuning the SVM Model using GridSearchCV

# In[57]:


# GridSearchCV to find best parameters for svm
svm = SVC(kernel = 'rbf', probability = True, random_state = 100)

# parameters to build the model on
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}  

grid_search = GridSearchCV(estimator = svm, param_grid = param_grid, 
                  scoring = 'accuracy', n_jobs = -1, cv = 5, verbose = 2)

# fitting the model for grid search 
grid_search.fit(X_train, y_train)
    
# get the best parameter from gird search 
print("Best Parameters:", grid_search.best_params_) 
  
# get the best classifier model after hyper-parameter tuning 
print("\nBest Classifier:", grid_search.best_estimator_) 


# In[58]:


best_svc = grid_search.best_estimator_

best_svc.fit(X_train, y_train)


# Generate the model test classification report
utils.Generate_Model_Test_Classification_Report(best_svc, X_test, y_test, model_name="Tuned Support Vector Machine")


# In[59]:


# Create models
best_tree = tree.fit(X_train, y_train)
best_svc =best_svc.fit(X_train, y_train)
best_xgb =xgb_best.fit(X_train, y_train)


y_pred_tree = best_tree.predict(X_test)
y_pred_svc = best_svc.predict(X_test)
y_pred_xgb=best_xgb.predict(X_test)


# In[60]:



# Create a DataFrame to store prediction results
results_df = pd.DataFrame({'True Label': y_test, 'SVM Prediction': y_pred_svc, 'Tree Prediction': y_pred_tree, 'XGBoost Prediction': y_pred_xgb})

# Count the occurrences of correct and incorrect predictions
correct_predictions = results_df[results_df['True Label'] == results_df['SVM Prediction']]
incorrect_predictions = results_df[results_df['True Label'] != results_df['SVM Prediction']]

# Set colors for correct and incorrect predictions
correct_color = '#B19CD9'
incorrect_color = 'salmon'

# Bar chart for prediction comparison - SVM
plt.bar(['Correct Predictions', 'Incorrect Predictions'], [len(correct_predictions), len(incorrect_predictions)], color=[correct_color, incorrect_color])
plt.ylabel('Count')
plt.title('Prediction Comparison - SVM')
plt.show()


# ### comparing models by visualization

# In[61]:


svm_precision = precision_score(y_test, y_pred_svc)
svm_recall = recall_score(y_test, y_pred_svc)
svm_f1 = f1_score(y_test, y_pred_svc)
svm_accuracy = accuracy_score(y_test, y_pred_svc)

tree_precision = precision_score(y_test, y_pred_tree)
tree_recall = recall_score(y_test, y_pred_tree)
tree_f1 = f1_score(y_test, y_pred_tree)
tree_accuracy = accuracy_score(y_test, y_pred_tree)

xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)


# Set pastel colors
mauve_color = '#B19CD9'
rose_color = '#E6B0AA'

# Visualize the results using a grouped bar chart with pastel colors
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

bar_width = 0.25
index = np.arange(4)

# Bar chart for SVM
plt.bar(index, [svm_precision, svm_recall, svm_f1,svm_accuracy], bar_width, label='SVM', color=mauve_color)
# Bar chart for Decision Tree
plt.bar(index + bar_width, [tree_precision, tree_recall, tree_f1,tree_accuracy], bar_width, label='Decision Tree', color=rose_color)

xgb_color = '#AED6F1'  # Choose a color for XGBoost bars
plt.bar(index + 2 * bar_width, [xgb_precision, xgb_recall, xgb_f1, xgb_accuracy], bar_width, label='XGBoost', color=xgb_color)



plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Evaluation Metrics Comparison: SVM vs. Decision Tree vs. XGBOOST')
plt.xticks(index  + bar_width, ['Precision', 'Recall', 'F1-Score',"Accuracy"])
plt.legend()
plt.show()


# In[62]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate confusion matrices for each classifier
cm_svm = confusion_matrix(y_test, y_pred_svc)
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Plot confusion matrix heatmaps
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 3, 2)
sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Oranges", cbar=False, annot_kws={"size": 14})
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 3, 3)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", cbar=False, annot_kws={"size": 14})
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()


# In[63]:



# Set colors
colors = ['purple', 'orange', 'green']

# Set metrics and classifiers
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
classifiers = ['SVM', 'Decision Tree', 'XGBoost']

# Create data for radar chart
svm_data = [svm_precision, svm_recall, svm_f1, svm_accuracy]
tree_data = [tree_precision, tree_recall, tree_f1, tree_accuracy]
xgb_data = [xgb_precision, xgb_recall, xgb_f1, xgb_accuracy]

# Set up angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

# Plot radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, svm_data, color=colors[0], alpha=0.25, label='SVM')
ax.fill(angles, tree_data, color=colors[1], alpha=0.25, label='Decision Tree')
ax.fill(angles, xgb_data, color=colors[2], alpha=0.25, label='XGBoost')

# Add labels and legend
ax.set_thetagrids(angles * 180/np.pi, metrics)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title('Radar Chart: Model Comparison')
plt.show()


# In[64]:


# Apply PCA
pca = PCA(n_components=2)
pca_features=pca.fit_transform(features_scaled)


# # Clustering

# In[65]:


# Create dict to store results of clustering algorithms for comparision
best_model = pd.DataFrame({'Model':[], 'Sil_score':[], 'CH_score':[], 'setting':[]})


# In[66]:


import seaborn as sns

def custom_palette(custom_colors):
    """Show color palette that use in this notebook"""
    customPalette = sns.set_palette(sns.color_palette(custom_colors))
    sns.palplot(sns.color_palette(custom_colors),size=0.8)
    plt.tick_params(axis='both', labelsize=0, length = 0)

 # for plot title
FONT = {'fontsize':30, 'fontstyle':'normal', 'fontfamily':'Georgia', 'backgroundcolor':'#B41B10', 'color':'#E4C09E'}

# Create List of Color Palletes
colors1 = ["#D1382F", "#392E37", "#F1F0F0", "#34C5B3", "#D79C4C", "#999EA2", "#D59A99", "#E4C09E"]
colors2 = ["#7D1F3F","#303336","#ECE3E2","#D24E49","#B41B10"]
dark_colors = ["#B41B10", "#D79C4C", "#8e18be", "#34C5B3", "#7D1F3F", "#a2c723", "#42575B"]
# Plot Color Palletes
for color in [colors1, colors2, dark_colors]:
    custom_palette(color)


# In[67]:


def plot_evaluation(sh_score, ch_score, name, x=range(2,11)):
    """
    for draw evaluation plot include silhouette_score and calinski_harabasz_score.
        sh_score(list): include silhouette_score of models
        ch_score(list): include calinski_harabasz_score of models
        name(string): name of clustering algorithm
        x(list): has range of number for x axis
    """
    
    fig, ax = plt.subplots(1,2,figsize=(15,7), dpi=100)
    ax[0].plot(x, sh_score, color=dark_colors[0], marker='o', ms=9, mfc=dark_colors[-1])
    ax[1].plot(x, ch_score, color=dark_colors[0], marker='o', ms=9, mfc=dark_colors[-1])
    ax[0].set_xlabel("Number of Clusters", labelpad=20)
    ax[0].set_ylabel("Silhouette Coefficient", labelpad=20)
    ax[1].set_xlabel("Number of Clusters", labelpad=20)
    ax[1].set_ylabel("calinski Harabasz Coefficient", labelpad=20)
    plt.suptitle(f'Evaluate {name} Clustering',y=0.9, **FONT)
    plt.tight_layout(pad=3)
    plt.show()


# ## Birch clustering

# In[68]:


# Check optimom n_clusters for implement birch by using silhouette and calinski coefficient
from sklearn.cluster import Birch
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score 
from kneed import KneeLocator
from sklearn import metrics

silhouette_coef = []
for k in range(2,11):
    birch = Birch(n_clusters=k, threshold=0.0001)
    birch.fit(pca_features)
    score = silhouette_score(pca_features, birch.labels_)
    silhouette_coef.append(score)

calinski_harabasz_coef = []
for k in range(2,11):
    birch = Birch(n_clusters=k, threshold=0.0001)
    birch.fit(pca_features)
    score = calinski_harabasz_score(pca_features, birch.labels_)
    calinski_harabasz_coef.append(score)


# In[69]:


plot_evaluation(silhouette_coef, calinski_harabasz_coef, 'Birch')


# In[70]:


# Implement Birch algorithm
birch = Birch(n_clusters=3, threshold=0.0001)
birch.fit(pca_features)
# Store result of Birch
pred = birch.labels_


# In[71]:


# Creating a scatter plot
plt.figure(figsize=(6, 4))  
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=pred, cmap='YlOrBr_r', alpha=0.7, edgecolors='b')
plt.title('Birch Clustering Result ')
plt.show()


# In[72]:


score_br = metrics.silhouette_score(pca_features,pred)
print("Score of Birch = ", score_br)


# In[73]:


# Store results obtained from Birch
best_model.loc[len(best_model.index)] = [
    f"BIRCH",
    silhouette_score(pca_features, pred),
    calinski_harabasz_score(pca_features, pred),
    {"n_clusters":3, "threshold":0.0001}]


# ## k-means clustering

# In[74]:


from sklearn.cluster import KMeans
# Kmeans algorithm settings
kmeans_set = {"init":"random", "n_init":10, "max_iter":300, "random_state":42}
# Find inertia for k cluster
inertias = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, **kmeans_set)
    kmeans.fit(pca_features)
    inertias.append(kmeans.inertia_)


# In[75]:


from kneed import KneeLocator
def elbow_optimizer(inertias,name):
    """ Find optimom k for clustering algorithm
        inertias (list): list that has inertia for each selected k
        name (string): name of clustering algorithm
    """

    kl = KneeLocator(range(1,11), inertias, curve='convex', direction="decreasing")
    plt.style.use("fivethirtyeight")
    sns.lineplot(x=range(1,11), y=inertias, color=dark_colors[0])
    plt.xticks(range(1,11))
    plt.xlabel("Number of Clusters", labelpad=20)
    plt.ylabel("Inertia", labelpad=20)
    plt.title(f"Elbow Method for {name}", y=1.09, fontdict=FONT)
    plt.axvline(x=kl.elbow, color=dark_colors[-1], label='axvline-fullheight', ls='--')
    plt.show()


# In[76]:


elbow_optimizer(inertias, 'Kmeans')


# In[77]:


plot_evaluation(silhouette_coef, calinski_harabasz_coef, 'Kmeans')


# In[78]:


# Implement kmeans for n_clusters=3
kmeans = KMeans(n_clusters=3, **kmeans_set).fit(pca_features)
# Store result of kmeans
pred = kmeans.labels_
plt.figure(figsize=(6, 4))  
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=pred,cmap="YlOrBr_r",alpha=.7,edgecolors="b") 
plt.title('K-means Clustering Result ')
plt.show


# In[79]:


# Store resluts of kmeans
best_model.loc[len(best_model.index)] = [
    f"Kmeans",
    silhouette_score(pca_features, pred),
    calinski_harabasz_score(pca_features, pred),
    {"n_clusters":3, **kmeans_set}]


# ## Hierarchical Clustering 
# ### The dendrogram is a tree-like diagram that illustrates the arrangement of clusters produced by the hierarchical clustering algorithm. It shows how data points are merged at each step and can help you decide on the number of clusters by looking at the vertical lines where they cross. More the distance of the vertical lines in the dendrogram, more the distance between the clusters.

# In[80]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create a dendrogram
linkage_matrix = linkage(pca_features, method='ward')

# Plot the dendrogram
plt.figure(figsize=(11, 5))
dendrogram(linkage_matrix, orientation="top",  show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distance')
plt.show()


# In[81]:


plt.figure(figsize=(11, 5))  
plt.title("Dendrograms")  
dend = dendrogram(linkage(pca_features, method='ward'))
plt.axhline(y=45, color='r', linestyle='--')


# ### To determine the number of clusters 
# #### We can set a threshold distance and draw a horizontal line (Generally, we try to set the threshold so that it cuts the tallest vertical line). Let’s set this threshold as 45 and draw a horizontal line. 

# In[82]:



# Find optimom bandwidth for implement agglomerative clustering by using silhouette and calinski coefficient
silhouette_coef = []
for k in range(2,15):
    agg = AgglomerativeClustering(n_clusters=k)
    agg.fit(pca_features)
    score = silhouette_score(pca_features, agg.fit_predict(pca_features))
    silhouette_coef.append(score)

calinski_harabasz_coef = []
for k in range(2,15):
    agg = AgglomerativeClustering(n_clusters=k)
    agg.fit(pca_features)
    score = calinski_harabasz_score(pca_features, agg.fit_predict(pca_features))
    calinski_harabasz_coef.append(score)


# In[83]:


# Draw plots of silhouette_score and calinski_harabasz_score for AgglomerativeClustering models
plot_evaluation(silhouette_coef, calinski_harabasz_coef, 'Agglomerative', x=range(2,15))


# In[84]:


# Perform hierarchical clustering 
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
clusters = model.fit_predict(pca_features)


# In[85]:


plt.figure(figsize=(6, 4))  
plt.scatter(x=pca_features[:, 0],  y=pca_features[:, 1])
plt.title('Original Data Before HC Clustering ')
plt.show()


# In[86]:


plt.figure(figsize=(6, 4))  
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters,alpha=.7,cmap="YlOrBr_r") 
plt.title('Hierarchical Clustering Result ')
plt.show


# In[87]:


# Store results obtained from AgglomerativeClustering
best_model.loc[len(best_model.index)] = [
    f"Agglomerative",
    silhouette_score(pca_features, clusters),
    calinski_harabasz_score(pca_features, clusters),
    {"n_clusters":3}]


# In[88]:


best_model


# In[89]:


# Show results obtained from different models in plot
fig, ax = plt.subplots(1,2,figsize=(15,7), dpi=100)
ax[0].plot(best_model.Model, best_model.Sil_score, marker='o', c=dark_colors[0], ms=9, mfc=dark_colors[-1])
ax[1].plot(best_model.Model, best_model.CH_score, marker='o', c=dark_colors[0], ms=9, mfc=dark_colors[-1])
ax[0].set_xlabel("Models", labelpad=20)
ax[0].set_ylabel("Silhouette Score", labelpad=20)
ax[1].set_xlabel("Models", labelpad=20)
ax[1].set_ylabel("Calinski-Harabasz Score", labelpad=20)
ax[0].tick_params(labelrotation=90) 
ax[1].tick_params(labelrotation=90) 
plt.suptitle(f'Comparision Models', **FONT)
plt.show()


# ### we conclude the the K-mean clustering in the best according to silhouette_score and calinski_harabasz_score

# ## With our dataset labeled as 1s and 0s, we're diving into clustering to group similar data points. The plan is to see how well these clusters align with the actual data labels, 

# In[90]:


kmeans = KMeans(n_clusters=2, **kmeans_set).fit(pca_features)
# Store result of kmeans
pred = kmeans.labels_


# In[91]:


# cross tabulation table
df = pd.DataFrame({'labels':pred,"class":normalized_data['outcome']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# In[92]:


birch = Birch(n_clusters=2, threshold=0.0001)
birch.fit(pca_features)
# Store result of Birch
pred = birch.labels_


# In[93]:


# cross tabulation table
df = pd.DataFrame({'labels':pred,"class":normalized_data['outcome']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# In[ ]:





# In[ ]:




