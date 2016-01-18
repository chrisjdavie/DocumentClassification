'''
Developing and tuning the model using the hackerrank training data. 

Created on 17 Jan 2016

@author: chris
'''
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

# load in data

fname = 'trainingdata.txt'

targets = []
data    = [] 

with open(fname) as f:
    for line in f:
        targets.append(line[0])
        data.append(line[2:])


# test-train split

docsTrain, docsTest, yTrain, yTest = train_test_split(data, targets, test_size = 0.5)

# build a pipeline

textClf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier()),
])
_ = textClf.fit(docsTrain,yTrain)
predicted = textClf.predict(docsTest)


# tune the parameters

parameters = { 'vect__max_df': [ 1.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5 ]
              
              #'vect__ngram_range': [(1, 1), (1, 2), (2, 2)]
}

gsClf = GridSearchCV(textClf, parameters)
gsClf = gsClf.fit(data,targets)


# show the best parameters

best_parameters, score, _ = max(gsClf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))


