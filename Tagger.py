__author__ = 'uddipaan'

import nltk
import argparse
from sklearn.externals import joblib
from ME import features

if __name__ == '__Tagger__':
    if len(sys.argv)<1:
        print("Error. Quiting!!");
        exit(1);

parser = argparse.ArgumentParser() #This is a constructor where ArgumentParser is the class name
parser.add_argument(dest='mod_name',help="Specify the trained model file name.")
parser.add_argument(dest='txt_file',help="Name of the text file where tagging must be done.")
args, unknown = parser.parse_known_args() #using parse_known_args rather than parse_args enables the use of ArgumentParser in code within the scope of if __name__ == 'main':


mod_name = args.mod_name
txt_file = args.txt_file


clsf = joblib.load(mod_name)

file_strm = open(txt_file, 'r') # filestream to read the textdocument provided

sentence = ''

while True:
	curline = file_strm.readline()
	if not curline:
		break
	print curline
	sentence = curline.strip()	

sents = str(sentence)
#print sents #(this is just to check the sentence.) 

def tag(sentence):
	tagged_sentence = []
	tags = clsf.predict([features(sentence,index) for index in range(len(sentence))])
	return zip(sentence, tags)

tags = ""
tags = tag(nltk.word_tokenize(sents)) #to make 'word tokenize' work, please get punkt library.

file_strm.close()

#save the tagged document in a text file
tagged_file = "Tagged.txt"
file_strm = open(tagged_file, 'w')
file_strm.write(str(tags))

file_strm.close()
