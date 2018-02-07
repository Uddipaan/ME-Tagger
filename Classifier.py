__author__ = 'uddipaan'

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from ME import *
from sklearn.externals import joblib

#Pipeline class is a useful toll for encapsulating multiple different transforms alongside an estimator into one obj.
clsf = Pipeline([
	('vectorizer', DictVectorizer(sparse=False)),
	('classifier', DecisionTreeClassifier(criterion='entropy'))
])
#We call our estimator instance clf, as it is a classifier. It now must be fitted to the model, i.e., it must learn from the model which is done by passing the training set to the fit method.
clsf.fit(x[:1000], y[:1000])

print "Training complete!"

x_test, y_test = transform_to_dataset(test)

print "Accuracy: ", clsf.score(x_test, y_test)

#Saving the model so that i dont have to train it every time i run the program.
filename = 'ME_Model.sav'
joblib.dump(clsf,filename)

##The class DictVectorizer can be used to convert feature arrays represented as list of stndrd Python "dict" objects to NumPy/SciPy representations used by scikit-learn estimators.
'''vec = DictVectorizer()
t = vec.fit_transform(x[:5]).toarray()
print t
r = vec.get_feature_names()
print r'''
