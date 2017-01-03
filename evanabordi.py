import csv
import pdb
import nltk
from sklearn.feature_extraction.text import CountVectorizer


with open('input/biology.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    bodies, tags = zip(*((x[2], x[3]) for x in csv_reader))
    cv_w = CountVectorizer(tokenizer=nltk.word_tokenize)
    w = cv_w.fit_transform(bodies)

    cv_t = CountVectorizer(tokenizer=nltk.word_tokenize)
    t = cv_t.fit_transform(tags).T

    tw = t.dot(w)
    pdb.set_trace()
