import csv
import pdb
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


with open('input/biology.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
    bodies, tags = zip(*((x[2], x[3]) for x in csv_reader))

    bodies_training, bodies_validating = bodies[:-10], bodies[-10:]
    tags_training, tags_validating = tags[:-10], tags[-10:]

    cv_w = CountVectorizer(tokenizer=nltk.word_tokenize)
    w = cv_w.fit_transform(bodies_training)
    cv_t = CountVectorizer(tokenizer=nltk.word_tokenize)
    t = cv_t.fit_transform(tags_training).T

    uki1 = t.dot(w)
    uki3 = uki1.sum(axis = 1)
    epsilon = 1
    n = uki1.shape[1]
    gnb = np.log((uki1.toarray() + epsilon) / (uki3 + n * epsilon))

    num_docs = len(bodies_training)
    prior = np.log(t.sum(axis=1)/num_docs)

    x = cv_w.transform([bodies_validating[0]])
    gnb.dot(x)

    pdb.set_trace()
