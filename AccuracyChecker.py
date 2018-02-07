from sklearn.externals import joblib
from ME import test, transform_to_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest='mod_name',help="Specify the trained model file name.")
args, unknown = parser.parse_known_args() 

mod_name = args.mod_name

clsf = joblib.load(mod_name)

x_test, y_test = transform_to_dataset(test)         
print "Accuracy: ", clsf.score(x_test, y_test)
