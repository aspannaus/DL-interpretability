"""

Module for preprocessing raw text data.

"""

import os
import pickle
import re

import gensim
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


def clean_text(data):
    """Cleans raw text data
    Args:
        data: list of text strings
    Returns:
        List of cleaned text
    """
    cleaned_text = []
    for text in data.split():
        x = re.sub('[^\w]|_', ' ', text)  # only keep numbers and letters and spaces
        x = x.lower()
        x = re.sub(r"[^\x00-\x7f]", r'', x)  # remove non ascii texts
        x = [y for y in x.split(" ") if y]  # remove empty words
        cleaned_text.append(''.join(x))
    return ' '.join(cleaned_text)


def max_doc_len(docs):
    """Gets max len from list of lists."""
    return max(len(doc) for doc in docs)


def pad_docs(data, vocab, max_len):
    """Pads docs to consistent length.

        Args:
        data: list of raw text docs as strings
        vocab: dict of (word, int) as (key, value) pairs
        max_len: int, max doc length

        Returns:
        np.ndarray of int tokens padded to max len 
    """
    X = np.ones((len(data), max_len), dtype=np.int_) * vocab["<pad>"]

    for i, x in enumerate(data):
        n = len(x)
        X[i, :n] = x
    return X


def word2int(wds, model):
    """Get list of int tokens.

        Args:
        wds: list of raw text strings
        model: gensim word2vec model

        Returns:
        list of tokenized data

    """
    ints = [model.wv.key_to_index.get(wd.lower()) for wd in wds]
    X = [x for x in ints if x is not None]
    return X


def tokenize_numeric(data):
    """Convert numbers to tokens.

        Args:
        data: list of raw text

        Returns:
        list with numeric values replaced with int or float tokens.

    """
    out_wds = []
    for w in data.split():
        if w.isdigit():
            out_wds.append('int_token')
        elif w.replace('.','',1).isdigit():  # float
            out_wds.append('float_token')
        else:
            out_wds.append(w)
    return out_wds


def main():
    categories = ['alt.atheism', 'comp.graphics',
                  'sci.space', 'talk.religion.misc']
    remove = ('headers', 'footers')

    seed = 42
    test_split = 0.10
    embed_dim = 300

    data = fetch_20newsgroups(subset='all', remove=remove,
                              categories=categories, random_state=42)
    ys = data['target']
    
    dat = [clean_text(d) for d in data['data']]
    dat = [gensim.parsing.preprocessing.strip_tags(d) for d in dat]
    dat = [gensim.parsing.preprocessing.strip_non_alphanum(d) for d in dat]
    dat = [gensim.parsing.preprocessing.remove_stopwords(d) for d in dat]

    dat = [gensim.parsing.preprocessing.split_alphanum(d) for d in dat]
    dat = [gensim.parsing.preprocessing.strip_short(d, minsize=3) for d in dat]
    dat = [tokenize_numeric(d) for d in dat]

    print("Creating vocab and word embeddings")
    model = gensim.models.word2vec.Word2Vec(vector_size=embed_dim,
                                            min_count=5, epochs=25, workers=8)
    model.build_vocab(dat)
    model.train(dat, total_examples=model.corpus_count, epochs=model.epochs)

    # create train and test splits, val is split from test in model training loop
    x_train, x_test, y_train, y_test = train_test_split(dat, ys, test_size=test_split,
                                                        random_state=seed)

    # add <unk> token
    word_vecs = [model.wv.vectors[index] for index in model.wv.key_to_index.values()]
    rng = np.random.default_rng(seed)
    w2v = np.append(word_vecs, rng.normal(size=(2, embed_dim), scale=0.1), axis=0)
    id2word = {v: k for k, v in model.wv.key_to_index.items()}
    id2word[len(model.wv.key_to_index)] = "<unk>"
    id2word[len(id2word.keys())] = "<pad>"
    word2id = {str(v): k for k, v in id2word.items()}

    X_train = [word2int(x, model) for x in x_train]
    X_test = [word2int(x, model) for x in x_test]

    max_len = max(max_doc_len(X_train), max_doc_len(X_test))

    X_train = pad_docs(X_train, word2id, max_len)
    X_test = pad_docs(X_test, word2id, max_len)

    if not os.path.exists('data/'):
        os.mkdir('data')

    np.save("./data/X_train.npy", X_train)
    np.save("./data/y_train.npy", y_train)
    np.save("./data/X_test.npy", X_test)
    np.save("./data/y_test.npy", y_test)

    if not os.path.exists('models/'):
        os.mkdir('models')

    with open("models/word2idx.pkl", "wb") as f:
        pickle.dump(word2id, f)

    with open("models/idx2word.pkl", "wb") as f:
        pickle.dump(id2word, f)

    np.save('./data/word_embeds.npy', w2v)
    print("Train shape ", X_train.shape)
    print("Test shape ", X_test.shape)


if __name__ == "__main__":
    main()
