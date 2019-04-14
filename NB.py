from collections import Counter, defaultdict
from glob import glob
import os
import re
import pre_process as pre2
import math
class NBClassifier:
    def __init__(self):
        #self.python2 = sys.version_info < (3,)
        self.data = []
        self.label = []
        self.collections_words = []
        self.word_vector = []

    def train_data_real_task(self, vocab, neg_vector, num_neg_labels, pos_vector,num_pos_labels):
        total_num_labels = num_neg_labels + num_pos_labels
        
        #calulate prior parameters
        labels = {"neg" : num_neg_labels, 'pos': num_pos_labels}
        prior_prob = dict.fromkeys(labels, 0)
        for label in labels:
            prior_prob[label] = math.log2(labels[label] / total_num_labels)
        print(prior_prob)

        #calculate word prob for neg
        total_neg = 0
        for word in neg_vector:
            total_neg += neg_vector[word]
        print(f"total tokens in neg files: {total_neg}")
        #calculate word prob for neg
        neg_word_prob = dict.fromkeys(neg_vector, 0)
        for word in neg_vector:
            #neg_word_prob[word] = (neg_vector[word] + 1)/(labels['neg'] + 1)
            neg_word_prob[word] = math.log2((neg_vector[word] + 1)/(len(neg_vector)+ total_neg))
        #print(neg_word_prob)

        total_pos = 0
        for word in neg_vector:
            total_pos += pos_vector[word]
        print(f"total tokens in pos files: {total_pos}")
        #calculate word prob for pos
        pos_word_prob = dict.fromkeys(pos_vector, 0)
        for word in pos_vector:
            pos_word_prob[word] = math.log2((pos_vector[word] + 1)/(len(pos_vector)+ total_pos))

        f3 = open('movie-review-BOW.NB','w')
        title = "NEG\t\t\t\t\tPOS\n"
        f3.write(title)
        for word in vocab:
            s = word + ": " + str(neg_word_prob[word]) + "\t\t" + word + ": "+ str(pos_word_prob[word])+"\n"
            f3.write(s)
        f3.close()

        return prior_prob, neg_word_prob, pos_word_prob

    def predict_real_task(self,filestr, prior_prob, neg_word_prob, pos_word_prob):
        corpus = pre2.tokenized(filestr)
        labels = ["neg", "pos"]

        pred_labels = dict.fromkeys(labels, 0)
        #calculate neg porb
        pred_labels['neg'] = prior_prob['neg']
        for word in corpus:
            if word not in neg_word_prob:
                neg_word_prob[word] = 1
            pred_labels['neg'] = pred_labels['neg'] + neg_word_prob[word]
        
        pred_labels['pos'] = prior_prob['pos']
        for word in corpus:
            if word not in pos_word_prob:
                pos_word_prob[word] = 1
            pred_labels['pos'] = pred_labels['pos'] + pos_word_prob[word]

        values = pred_labels.values()
        max_value = max(values)
        max_value_label = [l for l, v in pred_labels.items() if v == max_value]
        return max_value_label

def main():
    #########real task############
    classifier = NBClassifier()
    filenames_neg = glob("aclImdb/train/neg/*.txt")
    filenames_pos = glob("aclImdb/train/pos/*.txt")
    vocab = pre2.read_vocab()
    print("\n####### REAL TASK #######")
    print("Pre processing TRAIN folder NEG data...(wait couple seconds)")
    neg_vector, num_neg_labels = pre2.countfreq(vocab, filenames_neg)

    print("\nPre processing TRAIN folder POS data...(wait couple seconds)")
    pos_vector, num_pos_labels = pre2.countfreq(vocab, filenames_pos)
    prior_prob, neg_word_prob, pos_word_prob = classifier.train_data_real_task(vocab, neg_vector, num_neg_labels, pos_vector,num_pos_labels)

    ########predict########

    f1 = open('neg_predition.txt','w') 
    filenames_neg = glob("aclImdb/test/neg/*.txt")
    print("\nPredict Test folder NEG data: ")
    i = 0
    total_neg_pred = 0
    for filename in filenames_neg:#filenames
        i += 1
        if i % 1000 == 0: print(f"\tpredicted files #: {i}")
        f = open(filename)
        s = f.read()
        label = classifier.predict_real_task(s, prior_prob, neg_word_prob, pos_word_prob)
        s = filename +"\tneg"+ "\t" + str(label[0]) + "\n"
        f1.write(s)
        if label[0] == "neg":
            total_neg_pred += 1
    f1.close()

    print(f"For {i} files in NEG folder, {total_neg_pred} of files are predicted as NEG")
    print(f"Accuracy is {total_neg_pred/i}")

    filenames_pos = glob("aclImdb/test/pos/*.txt")
    print("\nPredict Test folder POS data: ")
    j = 0
    total_pos_pred = 0
    for filename in filenames_pos:#filenames
        j += 1
        if j % 1000 == 0: print(f"\tpredicted files #: {j}")
        f = open(filename)
        s = f.read()
        label = classifier.predict_real_task(s, prior_prob, neg_word_prob, pos_word_prob)
        if label[0] == "pos":
            total_pos_pred += 1
    print(f"For {j} files in POS folder, {total_pos_pred} of files are predicted as POS")
    print(f"Accuracy is {total_pos_pred/j}")
    print(f"Overall prediction: {(total_neg_pred+total_pos_pred)/(i + j)}")
    ##########predict###########

    return
if __name__ == '__main__':
    main()