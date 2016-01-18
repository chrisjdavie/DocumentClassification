# Report on the hackerrank.com Document Classification Puzzle
Chris Davie, 17/1/2016
## Summary
The hackerrank.com document classification puzzle requires that, using Machine Learning techniques, 94.7% of a set of hidden test documents be accurately catagorised. A set of data is provided to train the machine learning algorithm.

A standard Machine Learning approach, bag-of-words with the SGD classifier, passed this threshold after some tuning of the model. Further improvements were made using the Passive Aggressive classifier.
## Report
### The document classification puzzle
Hackerrank puzzle link: https://www.hackerrank.com/challenges/document-classification

The hackerrank document classification puzzle is straight forwards. Hackerrank provides a corpus of training documents, each classified into one of eight categories. Using this training data, the puzzle is to correctly categorise 94.7% of the hidden test documents. 

I chose this puzzle as I have solved a similar problem before, and saw that the problem was likely tractable within the time frame. I was also curious to see how other people solved the same problem; once the problem is solved hackerrank allows you to see other solutions.
### Initial solution
Code for the solution: https://github.com/chrisjdavie/DocumentClassification/tree/master

This type of problem can be tackled using a machine learning approach, using pre-built algorithms that can learn from and make predictions on data. I used both Python and the scikit-learn machine learning library; I am familiar with these and they are provided in the hackerrank environment. 

I used the ‘bag of words’ approach, where the individual words in each document are counted, and a statistical weight assigned to each word, indicating the category that word most likely appears in. 

I first found the occurrences of the words in each document using the scikit-learn vectorisor ‘CountVectorisor’. I then found the tf-idf, the ‘term frequency inverse document frequency’; the term frequency takes into account that longer documents have more words in them, making it more likely for any given word to appear. Term frequency then divides the counts of individual words by the total number of words in a document. Inverse document frequency decreases the weight of words that appear often across the corpus; these are less informative than words that appear rarely.

For the initial classifier, the algorithm that predicts the category of a document, I used the SGD classifier, a quite common and effective classifier for this type of problem.

This simple pipeline using tf-idf and an SGD classifier wasn’t quite good enough to pass the hackerrank threshold, catagorising 92.5% of the documents correctly, below the 94.7% required.
### Model tuning
The vectorisor, tf-idf transformer and classifier each have a number of parameters that can be varied, with the aim of producing better classifications. Sckit-learn has a tool for doing this, GridSearchCV; once the parameters are specified, it compares each combination of the parameters, finding how successfully each improve the predictions. In this case, it was found that setting the tdf_max = 0.85 in the tf-idf transformer improved the prediction sufficiently to pass the hackerrank threshold.

df_max refers to a maximum proportion of the documents a word can appear in if it is to be used. The logic for this limit is that very common words are less helpful in classifying documents.

With this change, the 95.2% of the test documents were correctly catagorised, passing the threshold required for hackerrank. This was not reliable - rerunning with this would sometimes fail to pass the threshold.
### Improvements from other examples
A feature of hackerrank is that once a puzzle has been solved, the users can look at other solutions, allowing improvements in the code and models used. In this case, the best solutions used the ‘PassiveAggressive’ classifier and a number of tuned parameters. Replicating this in my solution improved the best result to 95.5% of the documents correctly classified, and it consistently exceeded the hackerrank accuracy threshold.

This is consistent with other sources that shows the Passive-Agressive classifier predictions can exceed those of an SGD classifier, http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html



# Document Classification

Solving the hackerrank DocumentClassification puzzle

https://www.hackerrank.com/challenges/document-classification

## Running

$ cd solution/

$ python modelTuning.py < Input00.txt 

## Notes

There is a random aspect to the hackerrank assessment, so this version doesn't always pass
