# ME-Tagger

It is a POS tagger trained on the treebank corpus and uses Penn Treebank notations in tagging. 
The programming is divided into 2 parts, training and tagging. It is a flexible tagger which has the capability to train and save the trained model.
While tagging the program takes in the saved model and the untagged text file as command line arguments and writes the tagged result into a .txt file.

ME.py and Classifier.py are used to train the model, Tagger.py is used in tagging.
Classifiers used while training and their associated accuracies are given as follows:
Using Decision Tree Classifier, the accuracy observed is 89%
Using Stochastic Gradient Descent, the accuracy observed is 91%

Libraries Used are:
1. NumPy
2. sklearn
3. scikit
4. NLTK
