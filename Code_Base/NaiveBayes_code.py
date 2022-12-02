# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

# %%
train_data1 = pd.read_csv('train.csv')
test_data1 =  pd.read_csv('test.csv')

train_data = train_data1[:2000]
test_data = test_data1[:2000]
train_data.shape

# %%
test_data.info()

# %%
train_data.info()

# %%
train_data.head()

# %%
from wordcloud import WordCloud

# %%
# check the dataframe for missing values
train_data.isnull().sum()


# %%

train_data_source = 'train.csv'
test_data_source = 'test.csv'

train_df = pd.read_csv(train_data_source, header=None)
test_df = pd.read_csv(test_data_source, header=None)

train_df1 = train_df[:4000]


for df in [train_df1]:
    df[1] = df[1] + df[2]
    df = df.drop([2], axis=1)

# convert text_string to lower case 
train_texts = train_df1[1].values[:2000]
train_texts = [s.lower() for s in train_texts] 


df.head()
df.shape

# %%

train_data_source = 'train.csv'
test_data_source = 'test.csv'

train_df = pd.read_csv(train_data_source, header=None)
test_df = pd.read_csv(test_data_source, header=None)

test_df1 = test_df[:4000]
for tf in [test_df1]:
    tf[1] = tf[1] + tf[2]
    tf = tf.drop([2], axis=1)

# convert text_string to lower case 
test_texts = test_df1[1].values
test_texts = [s.lower() for s in test_texts] 

tf.head()
tf.shape



# %%
df.shape

# %%
import nltk
nltk.download('omw-1.4')
# Removing all empty spaces
df[1].dropna(inplace=True) 
df[2] = df[1]
# Converting the given text into lower letter
df[1] = [entry.lower() for entry in df[1]]
# Removing all given stop words, special characters and numbers
# xy is considered to be one tag map
xy = defaultdict(lambda : wn.NOUN)
xy['J'] = wn.ADJ
xy['V'] = wn.VERB
xy['R'] = wn.ADV

df.head()

# %%
from nltk import pos_tag
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

# %%
train_data_X, test_data_X, train_data_Y, test_data_Y = model_selection.train_test_split(df[1],df[1],test_size=0.3)


# %%
Labelenc = LabelEncoder()
train_data_Y = Labelenc.fit_transform(train_data_Y)
test_data_Y = Labelenc.fit_transform(test_data_Y)

# %%
termfrequencyvector = TfidfVectorizer(max_features=50)
termfrequencyvector.fit(df[1])
train_data_X_tfv = termfrequencyvector.transform(train_data_X)
test_data_Y_tfv = termfrequencyvector.transform(test_data_X)

print(termfrequencyvector.vocabulary_)

# %%
print(train_data_X_tfv)

# %%
train_data_Y.shape

# %%
nb = naive_bayes.MultinomialNB()
nb.fit(train_data_X_tfv, train_data_Y)
pred_nb = nb.predict(test_data_Y_tfv)
print("accuracy score using SVM-Naive Baye's",accuracy_score(pred_nb, test_data_Y))


