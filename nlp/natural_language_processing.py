# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords')

# stopwords contains list of irrelevant words
from nltk.corpus import stopwords

#for stemming(taking root of the words)
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    #sub returns only letters from the text
    review = re.sub('[^a-zA-z]' ," ", dataset["Review"][i])
    review = review.lower()
    # removing irrelevant words
    review = review.split()
    
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    
    # changing list back to string
    review = " ".join(review)
    
    corpus.append(review)
    
# creating bag of words model
#tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

#splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20, random_state = 0)

#fitting classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#predicting the test set results
Y_pred = classifier.predict(X_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


    
    
    
    
    
