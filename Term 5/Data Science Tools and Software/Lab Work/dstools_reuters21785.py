# -*- coding: utf-8 -*-
"""DSTools_Reuters21785.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_Th0K8PUWRxiQlKKke8zXw6COE98qpJt

# Data Science Tools and Software - Lab Assignment 1

Part II - Reuters21785 Dataset
---

## **Reading Data**

### Load Data Into Colab Environment
"""

!wget https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz
!mkdir /content/reuters21578
!tar -xzf reuters21578.tar.gz -C /content/reuters21578

"""### Read Data From Files Into Dataframe"""

import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics



# Directory containing all the SGML files
sgml_directory = "/content/reuters21578"

file_list = os.listdir(sgml_directory)
file_list.sort()  # Sort the files inside the directory

# Initialize an empty list to store data
data = []

# Iterate over all SGML files in the directory
for filename in file_list:
    if filename.endswith(".sgm"):
        # Construct the full path to the file
        full_path = os.path.join(sgml_directory, filename)

        # Read the SGML file
        with open(full_path, "r", encoding="latin-1") as file:
            content = file.read()
        # soup contain all the content of the file
        soup = BeautifulSoup(content, "html.parser")
        # let s get only all the tags named Reuters using find_all() function
        reuters_tags = soup.find_all("reuters")

        # we want to find each tag named Reuters and save each tag in a dictionnary ,
        #then finally we will have a list of dictionnaries , each one represent a specific tag

        for reuters_tag in reuters_tags:
            topic = reuters_tag['topics']
            lewissplit = reuters_tag['lewissplit']
            newId = reuters_tag['newid']
            oldId = reuters_tag['oldid']

            date = reuters_tag.find("date").text.strip()
            topics = reuters_tag.find("topics").text.strip()
            places = reuters_tag.find("places").text.strip()
            body = reuters_tag.find("body")
            title = reuters_tag.find("title")
            if topic == "YES"  or topics != "" :
                data.append({
                "Topic": topic,
                "lewissplit": lewissplit,
          #      "oldid": oldId,
          #      "newid": newId,
                "Date": date,
                "Topics": topics,
                "Places": places,
                "Body": body,
                "Title": title
            })




# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)

df

"""## **Data Cleaning**

### Handling Missing Values
"""

df.info()

df.shape

df.isna().sum()

# Drop rows with null values in a "Body"
df = df.dropna(subset=["Body"])

df.shape

# converting articles and topic columns to their text format
df["Body"]=df["Body"].apply(lambda x: x.text)
df["Title"]=df["Title"].apply(lambda x: x.text)

df["Topics"].value_counts()

# Replace missing Topic values with 'earn'
df['Topics'] = [value if value != '' else 'earn' for value in df['Topics']]

df["Topics"].iloc[20]

df["Topics"].value_counts()

df["Body"].iloc[0]

df.shape

"""### Datetime Conversion"""

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y %H:%M:%S.%f', errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df

"""### Cleaning Text Data"""

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
nltk.download('stopwords')

# Get the stop words
stop_words = stopwords.words('english')

def clean_Body(row):
    # remove url links
    row = re.sub("http\s+", '', row)
    # remove html tags
    row = re.sub("<[^<]+?>", '', row)
    # remove special characters, numbers and punctuations
    row = re.sub("[^A-za-z+ ]", ' ', row)
    # lower the text
    row = row.lower()
    # remove stopwords and words that have length of 1
    row = " ".join(word for word in nltk.wordpunct_tokenize(row) if len(word)!=2 and word not in stop_words)

    return row

df["Body"]=df["Body"].apply(lambda x:clean_Body(x))
df["Title"]=df["Title"].apply(lambda x:clean_Body(x))

"""## **Exploratory Data Analysis**

### Data Visualization
"""

df['Year'].unique()

"""All dates have the year 1987, as the Reuters21785 dataset was collected in 1987."""

monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Count')

# Set 'Year' and 'Month' as datetime for better plotting
monthly_counts['Date'] = pd.to_datetime(monthly_counts[['Year', 'Month']].assign(day=1))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(monthly_counts['Date'], monthly_counts['Count'], marker='o')
plt.title('Monthly Collected Articles Count Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.grid(True)
plt.show()

df["Topics"].value_counts(normalize=True)

"""We can see that 'earn' is the most common topic in the documents with 46%"""

from wordcloud import WordCloud

# Filter data for 'earn' and 'acq' topics
earn_data = df[df['Topics'] == 'earn']['Body'].str.cat(sep=' ')
acq_data = df[df['Topics'] == 'acq']['Body'].str.cat(sep=' ')
trade_data=df[df['Topics']=="trade"]['Body'].str.cat(sep=' ')
# Generate word clouds
earn_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(earn_data)
acq_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(acq_data)
trade_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(trade_data)


# Plot the word clouds
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(earn_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for "earn" Topic')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(acq_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for "acq" Topic')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(trade_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for "trade" Topic')
plt.axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Filter DataFrame to only include rows with "earn" topics
df_earn = df[df["Topics"] == "earn"]

# Get the counts of places posting about "earn" articles
place_counts = df_earn["Places"].value_counts()

# Select the top 10 countries
top_10_places = place_counts.head(10)
colors = ['teal', 'pink', 'green', 'purple', 'orange', 'red' , 'cyan', 'brown', 'gray', 'blue']

# Plotting the histogram for top 10 places
plt.figure(figsize=(10, 6))
plt.bar(top_10_places.index, top_10_places.values, color=colors)
plt.xlabel('Places')
plt.ylabel('Count')
plt.title('Top 10 Places Posting about "Earn" Articles')
plt.xticks(rotation=90)
plt.show()

# Count the number of articles published in each place
places_counts = df['Places'].value_counts()

# Select the top 5 places with the highest publication frequency
top_5_places = places_counts.head(5)

top_5_places

custom_colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#FFE6E6']

# Create a pie chart of the top 5 places with the highest publication frequency
plt.figure(figsize=(6, 6))
plt.pie(top_5_places, labels= top_5_places.index, autopct='%1.1f%%', startangle=140, colors= custom_colors)
plt.title('Top 5 Places Publishing Articles')
plt.show()

# Select the top 5 places with the highest publication frequency
top_5_places = places_counts.head(5).index

custom_colors = ['#66b3ff','#ffcc99', '#FFE6E6']
# Create subplots for each place
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Top 3 Topics in the Top 5 Places for Article Publication')

for i, place in enumerate(top_5_places):
    # Filter the data for the current place
    filtered_data = df[df['Places'] == place]

    # Count the number of articles in each topic
    topic_counts = filtered_data['Topics'].value_counts()

    # Select the top 3 topics for the current place
    top_3_topics = topic_counts.head(3)

    # Create a pie chart for the current place
    row, col = divmod(i, 3)
    ax = axes[row, col]
    ax.pie(top_3_topics, labels=top_3_topics.index, autopct='%1.1f%%', startangle=90 ,  colors=custom_colors)
    ax.set_title(place)

axes[1, 2].axis('off')  # hide empty plot grid at the end

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

import seaborn as sns
# create word count column
df['WordCount'] = [len(s.split()) for s in df['Body']]

# plot word count histogram
g = sns.displot(df, x="WordCount", binwidth=9, stat="density",kde=True)
new_ticks = [i for i in range(0, df['WordCount'].max() + 1, 50)] # add ticks every 50 words
g.set(xticks=new_ticks)
g.fig.set_figwidth(14)
g.fig.set_figheight(6)
plt.title("Distribution of Articles' Word Count")

"""We can see that the majority of articles are between 10-70 words."""

# get top 10 topics
topics_count = df['Topics'].value_counts()
highest_topics = topics_count[0:10].index

# plot boxplots
plt.figure(figsize=(14,8))
sns.boxplot(
    data=df[df['Topics'].isin(highest_topics)],
    x="Topics",
    y="WordCount",
    palette="Set3"
)
plt.title("Boxplot of Word Count Distribution for the Top 10 Topics")

!pip install scattertext

df_earn=df[df["Topics"]=="earn"]    # all the documents that its topic is earn
df_others=df[df["Topics"]!="earn"]  # the rest of the documents (not earn)

import scattertext as st
from IPython.display import HTML

scatter_data = pd.concat([df_earn, df_others.assign(Topics='other')])[['Topics', 'Title']]

# Create a Scattertext Corpus
corpus = st.CorpusFromPandas(scatter_data,
                              category_col='Topics',
                              text_col='Title',
                              nlp=st.whitespace_nlp_with_sentences
                             ).build()

# Create a Scattertext plot
html = st.produce_scattertext_explorer(corpus,
                                       category='earn',
                                       category_name='earn',
                                       not_category_name='other',
                                       width_in_pixels=1000)

HTML(html)

"""### Dissimilarity Matrix"""

unique_places = df["Places"].unique()
print(unique_places)

df.head(30)

from sklearn.feature_extraction.text import CountVectorizer

# create a document-term matrix for the first 10 documents
vect = CountVectorizer(max_features=20)
transform_vect = vect.fit_transform(df["Places"][21:30])

# rename the indices as doc_#
index = ['doc_{}'.format(i) for i in range(9)]

# create a dataframe
dist_df = pd.DataFrame(
      transform_vect.todense()
    , columns=vect.get_feature_names_out()
    , index=index
)

# view two docs
dist_df.loc[['doc_0', 'doc_2', 'doc_3', 'doc_4', 'doc_8' ], :]

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix in a specific column
similarity_matrix = cosine_similarity(transform_vect)

# Convert the similarity matrix to a dataframe
similarity_df = pd.DataFrame(similarity_matrix, index=index, columns=index)

# Print the similarity matrix
print(similarity_df)

import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances

# calculate jaccard distance on all documents
jaccard = pairwise_distances(dist_df.values, metric='jaccard')

sns.heatmap(pairwise_distances(dist_df.values, Y=None, metric='jaccard'));

# save dissimilarity matrix into csv file
pd.DataFrame(jaccard).to_csv('dissimilarity_8.csv', index=False)

for doc_ind in range(0,5):
    print('jaccard distance {}'.format(jaccard[0][doc_ind])) # focus on the first doc (index 0)
    print(dist_df.iloc[[0, doc_ind], :], '\n\n')  # print the first doc and the comparison document

from sklearn.feature_extraction.text import CountVectorizer

# create a document-term matrix for the first 10 documents
vect = CountVectorizer(max_features=20)
transform_vect = vect.fit_transform(df["Topics"][3:30])

# rename the indices as doc_#
index = ['doc_{}'.format(i) for i in range(27)]

# create a dataframe
dist_df = pd.DataFrame(
      transform_vect.todense()
    , columns=vect.get_feature_names_out()
    , index=index
)

# view two docs
dist_df.loc[['doc_1', 'doc_15',  'doc_25','doc_26'], :]

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(transform_vect)

# Convert the similarity matrix to a dataframe
similarity_df = pd.DataFrame(similarity_matrix, index=index, columns=index)

# Print the similarity matrix
print(similarity_df)

import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances

# calculate jaccard distance on all documents
jaccard = pairwise_distances(dist_df.values, metric='jaccard')

sns.heatmap(pairwise_distances(dist_df.values, Y=None, metric='jaccard'));

# save dissimilarity matrix into csv file
pd.DataFrame(jaccard).to_csv('dissimilarity_26.csv', index=False)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
nltk.download('punkt')

#using cosine similarity in all the document not just a specific column

# Assign the DataFrame to df_doc
df_doc = df

# Preprocess the text data
doc_corp = [' '.join(word_tokenize(str(row).lower())) for row in df_doc.values]

# Compute the TF-IDF matrix
tfidf_df = TfidfVectorizer(stop_words='english')
tf_idf_matrix = tfidf_df.fit_transform(doc_corp)

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

# Convert the similarity matrix to a DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=df_doc.index, columns=df_doc.index)

# Print the similarity matrix
print(similarity_df)

"""## **Data Transformation**"""

# segregating the data into train and test dataframes
df_train = df[df["lewissplit"]=="TRAIN"]
df_test  = df[df["lewissplit"]=="TEST"]

#initializing the vectorizer.
tfidf = TfidfVectorizer(min_df=5)

# vectorizing train and test datasets
X_train = tfidf.fit_transform(df_train["Body"].tolist())
X_test  = tfidf.transform(df_test["Body"].tolist())

# coverting the labels to binary- 1 for earn and 0 for others
y_train = df_train["Topics"].apply(lambda x: 1 if x=="earn" else 0).tolist()
y_test  = df_test["Topics"].apply(lambda x: 1 if x=="earn" else 0).tolist()

print(f"The vectorizer has {X_train.shape[0]} rows and {X_train.shape[1]} features")

from collections import Counter

counter=Counter(y_train)
counter     # Imbalanced data

from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train,y_train)

counter=Counter(y_train)
counter

"""## **Modelling**"""

def make_confusion_matrix(y_actual,y_predict,type_data):
    cm = metrics.confusion_matrix(y_actual, y_predict)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['others', 'earn']
    plt.title(f'Confusion Matrix - {type_data} Data')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(cm[i][j]))
    plt.show()
    print(f"\n Classification Matrix for {type_data} data:\n",metrics.classification_report(y_actual, y_predict))

##  Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model,flag=True):
    '''
    model : classifier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list=[]

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)

    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)

    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)

    train_f1 = metrics.f1_score(y_train,pred_train)
    test_f1 = metrics.f1_score(y_test,pred_test)

    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision,train_f1,test_f1))

    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True:
        print("Accuracy on training set : ", model.score(X_train,y_train))
        print("Accuracy on test set : ",     model.score(X_test,y_test))
        print("\nRecall on training set : ", metrics.recall_score(y_train,pred_train))
        print("Recall on test set : ",       metrics.recall_score(y_test,pred_test))
        print("\nPrecision on training set : ", metrics.precision_score(y_train,pred_train))
        print("Precision on test set : ",     metrics.precision_score(y_test,pred_test))
        print("\nF1 score on training set : ", metrics.f1_score(y_train,pred_train))
        print("F1 score on test set : ",     metrics.f1_score(y_test,pred_test))


    return score_list # returning the list with train and test scores

"""### Linear SVC Model"""

svc_model = LinearSVC(random_state=42)
svc_model.fit(X_train, y_train)

svc_score=get_metrics_score(svc_model,flag=True)

y_pred_train=svc_model.predict(X_train)
y_pred_test=svc_model.predict(X_test)
## training data
make_confusion_matrix(y_train,y_pred_train,"train")
## training data
make_confusion_matrix(y_test,y_pred_test,"test")