__author__ = 'uddipaan'

import nltk
import pprint

#The corpora is assigned to "tagged_sentence".
tagged_sentences = nltk.corpus.treebank.tagged_sents()

#Displays the sentences under consideration.
#This will be used to train the ME tagger.
##print tagged_sentences[0]


#this removes the POS tags in the training corpora
def untag(tagged_sentences):
	return [w for w, t in tagged_sentences]

#Extracting the features(hassle incoming -_- )
def features(sentence, index):
	##print('\n')
	return {
		'word': sentence[index],
		'first_word': index == 0,
		'last_word': index == len(sentence) - 1,
		'prev-word': '' if index == 0 else sentence[index-1],
		'next-word': '' if index == len(sentence) - 1 else sentence[index + 1],
		'has_hyphen': '-' in sentence[index],
		'is_numeric': sentence[index].isdigit(),
		'capital_inside': sentence[index][1:].lower() != sentence[index][1:],
		'is_capitalized': sentence[index][0].upper() == sentence[index][0],
		'is_all_caps': sentence[index].upper() == sentence[index],
		'is_all_lower': sentence[index].lower() == sentence[index],
		'prefix-1': sentence[index][0],
		'prefix-2': sentence[index][:2],
		'prefix-3': sentence[index][:3],
		'suffix-1': sentence[index][-1],
		'suffix-2': sentence[index][-2:],
		'suffix-3': sentence[index][-3:]
		}
##pprint.pprint(features(untag(tagged_sentences[0]),0))

#split the training corpus into training sentences and testing sentences
def split(sentences):
	cutoff = int(0.75 * len(sentences))
	training = sentences[:cutoff]
	test = sentences[cutoff:]
	
	
	return training, test

training, test = split(tagged_sentences)

#transform the training sentences to a dataset
#in the dataset list x defines the word in terms of the feature and list y defines the POS tag of the feature
#this transformation enables us to train the classifier
def transform_to_dataset(sentences):
	x,y = [],[]
	
	for sentc in sentences:
		for index in range(len(sentc)):
			x.append(features(untag(sentc), index))
			y.append(sentc[index][1])
	return x, y

x,y = transform_to_dataset(training)


