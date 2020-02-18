from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

__authors__ = ['Anoine Guiot', 'Arthur Claude', 'Armand Margerin']
__emails__ = ['armand.margerin@gmail.com', 'antoine.guiot@supelec.fr', 'athur.claude@supelec.fr']

import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import random

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

nlp = English()


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(l.lower().split())
    # removing stopwords and punctuation
    sentences_pre = []
    for sentence in sentences:
        for word in sentence:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                sentences_pre.append(word)
    return sentences_pre


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        nbsentences = len(sentences)
        self.w2id = {}
        idx = 1
        for sentence in sentences:
            for word in sentence:
                if word not in self.w2id.keys():
                    self.w2id[word] = idx
                    idx += 1

        self.trainset = sentences  # set of sentences
        self.winSize = winSize
        self.vocab = 0  # list of valid words
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed
        self.U = np.random.rand((len(self.w2id), self.nEmbed))
        self.V = np.random.rand((len(self.w2id), self.nEmbed))

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        omit_id = []
        for word in omit:
            omit_id.append(self.w2id[word])
        w2id_list = list(self.w2id.values())
        [w2id_list.remove(omit_word_id) for omit_word_id in omit_id]
        negativeIds = random.choices(w2id_list, len(w2id_list) / self.negativeRate)
        return negativeIds

    def train(self):
        self.U = np.random((len(self.w2id), self.nEmbed))
        self.V = np.random((len(self.w2id), self.nEmbed))

        for counter, sentence in enumerate(self.trainset):

        sentence = filter(lambda word: word in self.vocab, sentence)

        for wpos, word in enumerate(sentence):
            wIdx = self.w2id[word]
            winsize = np.random.randint(self.winSize) + 1
            start = max(0, wpos - winsize)
            end = min(wpos + winsize + 1, len(sentence))

            for context_word in sentence[start:end]:
                ctxtId = self.w2id[context_word]
                if ctxtId == wIdx: continue
                negativeIds = self.sample({wIdx, ctxtId})
                self.trainWord(wIdx, ctxtId, negativeIds)
                self.trainWords += 1

        if counter % 1000 == 0:
            print
            ' > training %d of %d' % (counter, len(self.trainset))
            self.loss.append(self.accLoss / self.trainWords)
            self.trainWords = 0
            self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds):
        # we want to maximize the log likelihood l = sum[sigma(gamma(i,j)*u_i*v_j)]
        eta = 0.025  # learning rate
        U = self.U
        V = self.V

        # compute gradients of l
        U1 = U[:, wordId]
        V2 = V[:, contextId]
        scalar = U1.dot(V2)
        gradl_word = 1 / (1 + np.exp(-scalar)) * V2
        gradl_context = 1 / (1 + np.exp(-scalar)) * U1

        # update representations
        U1 += eta * gradl_word
        V2 += eta * gradl_context

        for negativeId in negativeIds:
            # compute gradients of l
            U1 = U[:, wordId]
            V2 = V[:, negativeId]
            scalar = U1.dot(V2)
            gradl_word = -1 / (1 + np.exp(scalar)) * V2
            gradl_context = -1 / (1 + np.exp(scalar)) * U1

            # update representations
            U1 += eta * gradl_word
            V2 += eta * gradl_context

        # update self.U and self.V
        self.U = U
        self.V = V

    def save(self, path):
        raise NotImplementedError('implement it!')

    def similarity(self, word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        id_word_1 = self.w2id[word1]
        id_word_2 = self.w2id[word2]
        U1 = U[:, id_word_1]
        V2 = V[:, id_word_2]
        scalair = U1.dot(V2)
        similarity = 1 / (1 + np.exp(-scalair))
        return similarity

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(...)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
