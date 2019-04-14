#from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
import os
import re

#load imdb.vocab into a list
def read_vocab():
    with open(os.path.join('aclImdb', 'imdb.vocab'), 'r', encoding='utf8') as vocab:
        return vocab.read().split()
def tokenized(reviews):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    NO_SPACE = ""
    SPACE = " "
    reviews = reviews.lower()
    reviews = REPLACE_NO_SPACE.sub(NO_SPACE, reviews)
    reviews = REPLACE_WITH_SPACE.sub(SPACE, reviews)
    corpus = reviews.split()
    #bag of words
    #corpus = set(corpus)
    return corpus

def countfreq(vocb, filenames):
    total_file_tokens = []
    for filename in filenames:#filenames
        f = open(filename)
        s = f.read()
        data_set = tokenized(s)
        total_file_tokens.append(data_set)
        f.close()

    vector = dict.fromkeys(vocb, 0)
    i = 0
    for file_tokens in total_file_tokens:
        i += 1
        if i % 1000 == 0: print(f"\tprocessed files: #{i}")
        for word in file_tokens:
            if word not in vector: continue
            else: vector[word] += 1
    return vector, i


