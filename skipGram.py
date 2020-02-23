from __future__ import division
import argparse
import pandas as pd
import re  # regular expression
# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

__authors__ = ['Antoine Guiot', 'Arthur Claude', 'Armand Margerin']
__emails__ = ['antoine.guiot@supelec.fr', 'arthur.claude@supelec.fr', 'armand.margerin@gmail.com']

import pickle
from spacy.lang.en import English

nlp = English()


# pour ajouter un mot en stop word nlp.vocab[word].is_stop = True

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            # l = re.sub('[^a-zA-Z]', '', l)
            # regex = re.compile('[^a-zA-Z]')
            # First parameter is the replacement, second parameter is your input string
            sentences.append(re.sub('[^a-zA-Z]', ' ', l).lower().split())
    # removing stopwords and punctuation
    for sentence in sentences:

        for word in sentence:
            lexeme = nlp.vocab[word]

            if lexeme.is_stop == True:
                sentence.remove(word)
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=10, git=5, winSize=5, minCount=5):
        self.minCount = minCount
        self.winSize = winSize

        # dictionnary containing the nb of occurrence of each word

        sentences_concat = np.concatenate(sentences)
        unique, frequency = np.unique(sentences_concat, return_counts=True)
        self.occ = dict(zip(unique, frequency))
        self.vocab = {k: v for k, v in self.occ.items() if v > self.minCount}
        self.w2id = dict(zip(self.vocab.keys(), np.arange(0, len(self.vocab))))

        self.trainset = sentences  # set of sentences
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed

        id = self.w2id.values()
        vect = np.random.random((self.nEmbed, len(self.w2id)))

        self.U = dict(zip(id, vect.T))
        self.V = dict(zip(id, vect.T))
        # self.U = np.random.random((self.nEmbed, len(self.w2id)))
        # self.V = np.random.random((self.nEmbed, len(self.w2id)))
        self.loss = []
        self.trainWords = 0
        self.accLoss = 0.
        self.q = {}
        s = 0
        for w in self.w2id.keys():
            f = self.occ[w] ** (3 / 4)
            s += f
            self.q[self.w2id[w]] = f
        self.q = {k: v / s for k, v in self.q.items()}  # dictionary with keys = ids and values = prob q

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        w2id_list = list(self.w2id.values())
        # [w2id_list.remove(omit_word_id) for omit_word_id in omit]
        # q_list = []
        # for i in w2id_list:
        #	q_list.append(self.q[i])
        q_list = list(self.q.values())
        negativeIds = np.random.choice(w2id_list, size=self.negativeRate, p=q_list)
        for i in range(len(negativeIds)):
            if negativeIds[i] in omit:
                while negativeIds[i] in omit:
                    negativeIds[i] = np.random.choice(w2id_list, p=q_list)
        return negativeIds

    def train(self, nb_epochs):
        eta = 0.25
        for epoch in range(nb_epochs):
            eta = 0.9 * eta
            for counter, sentence in enumerate(self.trainset):
                sentence = list(filter(lambda word: word in self.vocab, sentence))

                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))
                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word]
                        if ctxtId == wIdx: continue
                        negativeIds = self.sample({wIdx, ctxtId})
                        self.trainWord(wIdx, ctxtId, negativeIds, eta)
                        self.trainWords += 1
                        self.accLoss += self.compute_loss(wIdx, ctxtId)
                if counter % 100 == 0:
                    #print(' > training %d of %d' % (counter, len(self.trainset)))
                    self.loss.append(self.accLoss / self.trainWords)
                    self.trainWords = 0
                    self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds, eta):
        # we want to maximize the log likelihood l = sum[sigma(gamma(i,j)*u_i*v_j)]
        # U = self.U
        # V = self.V

        # compute gradients of l
        U1 = self.U[wordId]
        V2 = self.V[contextId]
        scalar = U1.dot(V2)
        gradl_word = 1 / (1 + np.exp(scalar)) * V2
        gradl_context = 1 / (1 + np.exp(scalar)) * U1  # modifié le signe

        # update representations
        U1 += eta * gradl_word
        V2 += eta * gradl_context

        # update U and V
        self.U[wordId] = U1
        self.V[contextId] = V2

        for negativeId in negativeIds:
            # compute gradients of l
            U1 = self.U[wordId]
            V2 = self.V[negativeId]
            scalar = U1.dot(V2)
            gradl_word = -1 / (1 + np.exp(-scalar)) * V2
            gradl_context = -1 / (1 + np.exp(-scalar)) * U1

            # update representations
            U1 += eta * gradl_word
            V2 += eta * gradl_context

            # update U and V
            self.U[wordId] = U1
            self.V[negativeId] = V2

        # update self.U and self.V

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.U, self.w2id, self.vocab], f)

    def compute_loss(self, id_word_1, id_word_2):
        w1 = self.U[id_word_1]
        w2 = self.U[id_word_2]
        scalair = w1.dot(w2)
        similarity = 1 / (1 + np.exp(-scalair))
        return similarity

    def compute_score(self):

    def similarity(self, word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        common_vect = +np.ones(self.nEmbed) * 10000
        if word1 not in self.vocab and word2 in self.vocab:
            id_word_2 = self.w2id[word2]
            w1 = common_vect
            w2 = self.U[id_word_2]
        elif word1 in self.vocab and word2 not in self.vocab:
            id_word_1 = self.w2id[word1]
            w1 = self.U[id_word_1]
            w2 = common_vect
        elif word1 not in self.vocab and word2 not in self.vocab:
            w1 = common_vect
            w2 = common_vect
        else:
            id_word_1 = self.w2id[word1]
            id_word_2 = self.w2id[word2]
            w1 = self.U[id_word_1]
            w2 = self.U[id_word_2]

        # scalair = w1.dot(w2)/np.linalg.norm(w1,w2)
        similarity = w1.dot(w2) / (np.linalg.norm(w1)*np.linalg.norm(w2))

        # similarity = 1 / (1 + np.exp(-scalair))
        # similarity = scalair / (np.linalg.norm(w1) * np.linalg.norm(w2))
        return similarity

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            U_l, w2id_l, vocab_l = pickle.load(f)
            sg = SkipGram([])
            sg.U = U_l
            sg.w2id = w2id_l
            sg.vocab = vocab_l
        return sg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(1)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
