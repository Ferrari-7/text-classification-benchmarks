# Import packages
# system tools
import os
import sys
sys.path.append("..")
# data munging tools
import pandas as pd
import utils.classifier_utils as clf
# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
# for saving models
from joblib import dump, load

# load data
def load_data():
    filename = os.path.join("..", "in", "fake_or_real_news.csv")
    data = pd.read_csv(filename)
    # get text and labels
    X = data["text"]
    y = data["label"]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def doc_vectorizer(X_train, X_test):
    # vectorizer
    vectorizer = TfidfVectorizer(ngram_range = (1,2),    # unigrams and bigrams (1 word and 2 word units)
                                lowercase =  True,       # lowercase
                                max_df = 0.95,           # remove very common words
                                min_df = 0.05,           # remove very rare words
                                max_features = 100)      # keep only top 100 features
    # fit and transform data to training data
    X_train_feats = vectorizer.fit_transform(X_train)
    # fit and transform data to test data
    X_test_feats = vectorizer.fit_transform(X_test)
    return  X_train_feats, X_test_feats

def mlp_clf(X_train_feats, X_test_feats, y_train):
    classifier = MLPClassifier(activation = "logistic",
                                hidden_layer_sizes = (20,),
                                max_iter=1000,
                                random_state = 42)
    # fit classifier to data
    classifier.fit(X_train_feats, y_train)
    # get predictions
    y_pred = classifier.predict(X_test_feats)
    # save model in folder called "models"
    clf_name = os.path.join("..", "models", "mlp_classifier.joblib")
    dump(classifier, clf_name)
    return y_pred

def clf_report(y_test, y_pred):
    # making a classification report
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    
    # saving the report as a txt file
    report_path = os.path.join("..", "out", "mlp_report.txt")
    text_file = open(report_path, "w")
    text_file.write(classifier_metrics)
    text_file.close()


def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train_feats, X_test_feats = doc_vectorizer(X_train, X_test)
    y_pred = mlp_clf(X_train_feats, X_test_feats, y_train)
    clf_report(y_test, y_pred)

if __name__=="__main__":
    main()