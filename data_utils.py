import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def max_doc_len(doc):
    return max([len(line.split()) for line in doc])


def conv(fld):
    return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)


class DataHandler():
    def __init__(self, idxs=None):
        self.corpus = []
        self.n_classes = None
        if idxs is not None:
            self.idxs = idxs
        else:
            self.idxs = None
        self.path = "./DL-interpretability/"
        self.word2vec = np.load("./data/word_embeds.npy")
        self.embedding_dim = self.word2vec.shape[1]

        with open("./models/idx2word.pkl", "rb") as f:
            self.idx2word = pickle.load(f)
        self.word2idx = {v: k for k, v in self.idx2word.items()}
        self.vocab = self.word2idx

    def get_data(self, get_test=False):
        # get the data, shuffled and split between train and test sets
        print("\n Loading newsgroup data")

        categories = ['alt.atheism', 'comp.graphics',
                      'sci.space', 'talk.religion.misc']
        self.idx2label = {i: c for i, c in enumerate(categories)}
        self.label2idx = {c: i for i, c in enumerate(categories)}

        if "<pad>" not in self.vocab.keys():
            self.vocab.update({"<unk>":  len(self.vocab)})
            self.vocab.update({"<pad>": len(self.vocab)})
        self.vocab_len = len(self.vocab.keys())

        # with open(self.path + "/models/count_vec.pkl", "rb") as f:
        #     self.vec = pickle.load(f)
        #     self.vocab_len = len(self.vec.vocabulary_)

        # if "<pad>" not in self.vec.vocabulary_.keys():
        #     self.vec.vocabulary_.update({"<unk>":  len(self.vec.vocabulary_)})
        #     self.vec.vocabulary_.update({"<pad>": len(self.vec.vocabulary_)})
        #     self.vocab_len = len(self.vec.vocabulary_)

        if get_test:
            self.X_tok = np.load("./data/X_test.npy")
            self.y = np.load("./data/y_test.npy")
        else:
            self.X_tok = np.load("./data/X_train.npy")
            self.y = np.load("./data/y_train.npy")

        self.n_classes = np.unique(self.y).shape[0]
        print(" X ", self.X_tok.shape)
        print(" y {}, num classes {}".format(self.y.shape, self.n_classes))
        self.n_samples = self.y.shape[0]

        return categories

    def tokens2doc(self, doc):

        words = [self.idx2word[word] for word in doc]
        return words

    def make_corpus(self, docs=None):
        if docs is None:
            docs = self.X_tok

        for doc in docs:
            doc_words = [self.idx2word[word] for word in doc if word != self.word2idx["<pad>"]]
            tmp_doc = " ".join(doc_words)
            self.corpus.append(tmp_doc)
        return None

    def get_embeddings(self, numpy=True):
        if numpy:
            self.doc_w2v = np.empty((len(self.X_tok), self.X_tok[0].shape[0], 128))
            for i, x in enumerate(self.X_tok):
                for j, x1 in enumerate(x):
                    self.doc_w2v[i, j] = self.tok2wordvec(x1)
        else:
            self.doc_w2v = []
            for x in self.X_tok:
                self.doc_w2v.append(np.asarray([self.tok2wordvec(tok) for tok in x]).flatten())
        return None

    def tok2wordvec(self, tok):
        if tok in self.idx2word.keys():
            word = self.idx2word[tok]
            embedding_dim = self.word2vec.shape[1]

            if word in ["<unk>", "<pad>"]:
                return np.zeros((embedding_dim))
            else:
                return self.word2vec[tok]
        else:
            return np.zeros((self.embedding_dim))

    def make_dtm(self, ngrams, vec=None):
        if len(self.corpus) == 0:
            self.make_corpus()
        if self.word2idx is None:
            self.get_embeddings()
        if vec is None:
            vec = CountVectorizer(ngram_range=ngrams,
                                  vocabulary=self.word2idx.keys())
        X = vec.fit_transform(self.corpus)
        return X, vec
