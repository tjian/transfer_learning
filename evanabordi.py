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

    epsilon = 1

    absolute_word_prob = w.sum(axis = 0).A[0].astype(np.float) + epsilon
    absolute_word_prob /= sum(absolute_word_prob)
    np.log(absolute_word_prob, out = absolute_word_prob)

    uki1 = t.dot(w)
    uki3 = uki1.sum(axis = 1).A[:, 0].astype(np.float)
    n = uki1.shape[0]
    gnb = np.log((uki1.A.astype(np.float) + epsilon) / (uki3[:, None] + n * epsilon))

    num_docs = len(bodies_training)
    prior = np.log(t.sum(axis=1).A[:, 0].astype(np.float)/num_docs)

    x = cv_w.transform([bodies_validating[0]]).astype(np.float).toarray()[0]
    likelihood = gnb.dot(x)
    absolute_prob = absolute_word_prob.dot(x)

    posterior = prior + likelihood - absolute_prob
    print([k for k, v in cv_t.vocabulary_.items() if posterior[v] > np.log(0.5)])
    #print(cv_t.inverse_transform(posterior > np.log(0.5)))
    print(tags_validating[0])


    pdb.set_trace()
