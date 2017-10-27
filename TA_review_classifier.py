#---------------------------------------------------------------------------------------------
# Name:          TA_review_classifier.py
# Purpose:       This .py parses hotel reviews and then uses a Naive Bayes classifier to
#                classify those reviews as negative or positive and extract the most infor-
#                mative words in each review that determine the result.
#                It outputs the Classifier's accuracy as a percentage and the 10 most infor-
#                mative features.
#
# Required libs: nltk, requests, random, re, bs4
# Author:        Konstantinos Oikonomou
#
# Created:       25/02/2017
#
#---------------------------------------------------------------------------------------------

#-----------IMPORTS-----------
import nltk
import requests
import random
import re

from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#-----------------------------

#----------- Parsing and Getting the Data -----------

#the hotel's review urls (each page)
urls = ['https://www.tripadvisor.com/Hotel_Review-g187147-d586832-Reviews-Sibour_Hotel-Paris_Ile_de_France.html#REVIEWS',
        'https://www.tripadvisor.com/Hotel_Review-g187147-d586832-Reviews-or10-Sibour_Hotel-Paris_Ile_de_France.html#REVIEWS',
        'https://www.tripadvisor.com/Hotel_Review-g187147-d586832-Reviews-or20-Sibour_Hotel-Paris_Ile_de_France.html#REVIEWS']

#documents is a list of tuples containing the review and the classification
documents = []

#iterate through each page
for url in urls:
    #get the source code
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'html.parser')

    #find the reviews
    revs = soup.findAll("p", { "class" : "partial_entry" })

    #normalise a bit
    reviews = [str(review).strip('<p class="partial_entry">').strip('</p>') for review in revs]

    #find the ratings
    rat = soup.findAll("div", {'class' : 'rating reviewItemInline'})

    temp = [re.search(r'ui_bubble_rating bubble_..', str(i)).group(0) for i in rat]

    #convert the bubbles into negative and positive
    # 1-3 Negative , 4-5 Positive
    ratings = []
    for item in temp:
        if int(item[-2]) <= 3:
            ratings.append('neg')
        else:
            ratings.append('pos')

    for x, y in zip(reviews, ratings):
        documents.append((x, y))


#-------------------------------------------------------

#--------------- Further Normalisation -----------------

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

#Tokenize
for index, item in enumerate(documents):
    documents[index] = (tokenizer.tokenize(item[0]), item[1])

#Lemmatize, Lowercase and remove StopWords
for index, item in enumerate(documents):
    documents[index] = ([lemmatizer.lemmatize(word).lower() for word in item[0] if word not in stopwords.words('english')], item[1])

all_words = []
for item in documents:
    all_words.extend(item[0])

all_words = nltk.FreqDist(all_words)

#-------------------------------------------------------

#---------- Classification with NB Classifier ----------

#shuffle for better distribution of categories
random.shuffle(documents)

word_features = list(all_words.keys())
def find_features(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

#since all of the documents are 25, we use the first 19 for training and the rest for testing
# 75/25
training_set = featuresets[:19]
testing_set = featuresets[19:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('The classifier is {} percent accurate.'.format(nltk.classify.accuracy(classifier, testing_set) * 100))
classifier.show_most_informative_features(10)

#---------------------------------------------------------