import numpy as np
import jieba

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# load wiki.zh 300d word2vec embedding provided by Facebook
with open('../wiki.zh/wiki.zh.vec', "r", encoding='utf-8') as lines:
    w2v = {line.split()[0]: np.asarray(line.split()[1:], dtype='float32') for line in lines}


sent2vec = MeanEmbeddingVectorizer(w2v)

# test sentences
sents_str = [u'我来到北京清华大学',
         u'小明硕士毕业于中国科学院',
         u'小明后在日本京都大学深造']

# calling jieba word segmenter to prepare sent2vec input
sents_tokenized = [list(jieba.cut(sent, cut_all=False)) for sent in sents_str]
vecs = sent2vec.transform(sents_tokenized)


for jieba_out, vec in zip(sents_tokenized, vecs):
    print(jieba_out)
    print(vec[0:5])
