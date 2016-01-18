'''
The submission to hackerrank. This is my first submission that passed,
with tuned parameter found in modelTuning.py.

Created on 18 Jan 2016

@author: chris
'''

# Enter your code here. Read input from STDIN. Print output to STDOUT
# from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np


# Load in training data

fname = 'trainingdata.txt'

targets = []
data    = [] 

with open(fname) as f:
    for line in f:
        targets.append(line[0])
        data.append(line[2:])

# train the classifier

textClf = Pipeline([('vect', CountVectorizer(max_df = 8.5)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier()),
])
_ = textClf.fit(data,targets)


# load in test data

docs = []
for _ in range(input()):
    docs.append(raw_input().strip())

# predict

predicted = textClf.predict(docs)

# output in hackerrank format

for pred in predicted:
    print pred
docs = []
for _ in range(input()):
    docs.append(raw_input().strip())