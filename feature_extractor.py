import math
import os
import pickle

from collections import Counter

import kmapper as km
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import pynndescent

import cnn_preds
import data_utils
import make_mapper_graph


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.disable_eager_execution()


class MapperData():
    def __init__(self, graph=None, data=None):
        self.path = "./models/"

        if graph is None:
            print(f"\n Loading Mapper data from {self.path}mapper_data.pkl\n")
            with open(self.path + "mapper_data.pkl", "rb") as f_in:
                self.graph = pickle.load(f_in)
        else:
            self.graph = graph
        if data is None:
            self.X_proj = np.load(self.path + "X_proj_iso.npy")
        else:
            self.X_proj = data
        self.index = pynndescent.NNDescent(self.X_proj, metric='cosine')
        self.index.prepare()
        # self.feat = np.zeros((n_feat, 100))
        self.k_hat = None
        self.dists = None
        self.embedding_dim = None
        self.f = None

        self.nerve = km.GraphNerve()
        self.edges, self.simplices = self.nerve.compute(self.graph["nodes"])
        with open(self.path + "cover.pkl", "rb") as f:
            self.cover = pickle.load(f)

    def data_from_cluster_id(self, cluster_id):
        if cluster_id in self.graph["nodes"]:
            cluster_members = self.graph["nodes"][cluster_id]
            members_data = self.X_proj[cluster_members]
            return cluster_members, members_data
        return []

    def filter_graph(self, filter_perc, label):
        n_simpl = sum(len(v) for _, v in self.edges.items())
        print(f"Graph has {n_simpl} edges and {len(self.graph['nodes'])} nodes")

        subgraph = []
        ctr = 0
        proj_dim0 = 0
        ids = []
        one_dim = [s for s in self.simplices if len(s) > 1]
        for n in one_dim:
            for s in n:
                tmp_docids, tmp_data = self.data_from_cluster_id(s)
                # filter here on avg softmax value, class, etc
                # indices are: [embedding 0, embedding 1,
                # Predicted class, Ground Truth class, Prob of GT]
                idxs = np.nonzero(tmp_data[:, 3] == label)[0]
                if np.any(tmp_data[idxs, 4] > filter_perc) and len(idxs) > 0:
                    ctr += 1
                    ids.append(np.asarray(tmp_docids)[idxs])
                    subgraph.append(tmp_data)
                    proj_dim0 += tmp_data.shape[0]
        doc_ids = np.asarray([i for lst in ids for i in lst])
        tmp_data = np.asarray(ids, dtype=object).flatten()
        print(f"Number of filtered nodes {ctr}")
        print(f"Num of docs = {np.unique(doc_ids).shape}")

        if np.unique(doc_ids).shape[0] > 0:
            proj_X = subgraph[0]
            for i in range(1, len(subgraph)):
                proj_X = np.concatenate((proj_X, subgraph[i]), axis=0)
            return (np.unique(doc_ids), np.unique(proj_X, axis=0))
        if np.unique(doc_ids).shape[0] == 0:
            return (None, None)

    def assign_nodes(self, data_values, nodes):
        label = data_values[2]
        label_idxs = np.where(self.X_proj[:, 2] == label)[0]
        index = pynndescent.NNDescent(self.X_proj[label_idxs])
        index.prepare()
        tree_data = []
        tree_node_ids = []

        for node, data_ids in nodes.items():
            ids = np.intersect1d(data_ids, label_idxs)
            node_data = self.X_proj[ids]
            tree_data.append(node_data)
            tree_node_ids.append([node]*len(node_data))

        tree_data = np.vstack(tree_data)
        tree_node_ids = np.concatenate(tree_node_ids)

        nn_ids, _ = index.query(np.array([data_values]), k=5)

        return tree_node_ids[nn_ids]

    def find(self, data_point, eps=0.2):
        """Finds the hypercubes that contain the given data point.
        Parameters
        ===========
        data_point: array-like
            The data point to locate.
        Returns
        =========
        cube_ids: list of int
            list of hypercube indices, empty if the data point is outside the cover.
        """
        cube_ids = []
        for i, center in enumerate(self.cover.centers_):
            lb = center[:2] - self.cover.radius_[:2] - eps
            ub = center[:2] + self.cover.radius_[:2] + eps
            if np.all(data_point[:2] >= lb) and np.all(data_point[:2] <= ub):
                cube_ids.append(i)
        return cube_ids

    def find_nodes(self, cube_ids):
        nodes = {}
        for node, data_ids in self.graph['nodes'].items():
            if node.startswith(tuple(['cube'+str(i) for i in cube_ids])):
                nodes[node] = data_ids
        return nodes

    def data_from_clusterid(self, cluster_id, data):
        if cluster_id in self.graph["nodes"]:
            cluster_members = self.graph["nodes"][cluster_id]
            cluster_members_data = data[cluster_members]
            print(cluster_members_data)

    def compute_dtm(self, index, X_label, m=0.05):
        self.k_hat = max(math.floor(m * X_label.shape[0]), 250)
        print(f"m = {m:6.4f}, k = {self.k_hat}")
        _, self.dists = index.query(X_label, k=self.k_hat)
        return np.mean(self.dists[:, 1:]**2, axis=1)

    def predict(self, X, model, lens):
        probs, doc = model.cnn_output(X)
        pred = np.argmax(probs)
        x = lens.predict(doc)
        return [x[0], x[1], pred, pred, probs[0, pred]]

    def get_nodes_in_graph(self, x):
        nnbrs, _ = self.index.query(np.array([x[:2]]), k=20)
        scaler = 1.0
        for nbr in nnbrs.ravel():
            cube_ids = self.find(self.X_proj[nbr], eps=scaler*self.cover.radius_[:2])
            tmp_nodes = self.find_nodes(cube_ids)
            if len(tmp_nodes) > 0:
                nodes = self.assign_nodes(self.X_proj[nbr], tmp_nodes)
                tmp_ids = []
                if len(nodes) > 0:
                    for node in nodes.ravel():
                        tmp_ids.extend(self.graph["nodes"][node])
                    return np.unique(tmp_ids)
                continue
            scaler = 1.5 * scaler  # increase search radius
            continue

    def get_wordvecs(self, docs, dw, unique=True, ret_wds=True):
        self.embedding_dim = dw.embedding_dim

        if unique:
            tok = np.unique(dw.X_tok[docs])
            idxs = np.where(tok != dw.word2idx["<pad>"])[0]
            label_wds = tok[idxs]
            N = label_wds.shape[0]
            X = np.empty((N, self.embedding_dim))
            for i, ttok in enumerate(label_wds):
                tmp = dw.tok2wordvec(ttok)
                if np.all(tmp != 0):
                    X[i] = tmp
        else:
            tok = dw.X_tok[docs].flatten()
            idxs = np.where(tok != dw.word2idx["<pad>"])[0]
            test_wds = tok[idxs]
            M = test_wds.shape[0]
            X = np.empty((M, self.embedding_dim))
            for i, ttok in enumerate(test_wds):
                tmp = dw.tok2wordvec(ttok)
                if np.all(tmp != 0):
                    X[i] = tmp
        if ret_wds:
            return X, tok[idxs]
        return X

    def compute_density(self, log=True):
        d = self.embedding_dim
        p = 2
        num = np.sum(np.power(np.arange(self.k_hat), p / d))
        den = np.sum(np.power(self.dists, p), axis=1)
        self.f = np.zeros(den.shape)
        if log:
            self.f[den.nonzero()] = (p / d) * (np.log(num) - np.log(den[den.nonzero()]))
        else:
            tmp = num / den[den.nonzero()]
            self.f[den.nonzero()] = np.power(tmp, d/p)


class PredictFun():
    def __init__(self, training=False, verbose=1):
        self.path = "./DL-interpretability/"

        model_file = "./models/newsgroups_cnn.h5"
        print("\n Loading model from " + model_file)
        self.model = load_model(model_file, compile=True)
        if verbose:
            self.model.summary()
        self.predict = K.function(
            [self.model.layers[0].input, K.learning_phase()],
            self.model.layers[-1].output)
        self.fun = K.function([[self.model.layers[0].input],
                               K.learning_phase()],
                              [self.model.layers[-1].output,
                              self.model.layers[-2].output])

        # self.out is the number of predictions
        vocab_len = len(self.model.get_layer('embedding').get_weights()[0])
        n_classes = self.model.layers[-1].output.shape[1]
        self.out = np.empty((500, n_classes))
        self.X1 = np.zeros((1, vocab_len))  # 1 x vocab size
        self.training_phase = training
        self.idx2word = None
        self.word2idx = None
        self.word2vec = None
        self.remove = ['floattoken', 'largeinttoken', 'breakdoc',
                       'breaktoken', '<pad>', 'pad', '<>']

    def get_embeddings(self):

        print(" Loading dictionary mappings from {self.path}")
        self.word2vec = np.load('./models/vectors.npy', allow_pickle=True)

        with open('./models/word2idx.pkl', "rb") as f:
            self.word2idx = pickle.load(f)
        with open('./models/idx2word.pkl', "rb") as f:
            self.idx2word = pickle.load(f)

    def doc2tok(self, X):

        X = X.split()
        tmp = np.array([self.word2idx[x] if x not in self.remove else
                        len(self.word2idx) for x in X])
        N = tmp.shape[0]
        self.X1[0, :N] = tmp.copy()
        return self.X1

    def predict_prob(self, X):
        """Make prediction from document, as string."""
        if isinstance(X) is list:
            # this is for lime, the else is for our method
            for i, x in enumerate(X):
                X1 = self.doc2tok(x)
                self.out[i] = self.predict([X1, self.training_phase])
            return self.out
        N = X.shape[0]
        return self.predict([X.reshape((1, N)), self.training_phase])

    def cnn_output(self, X, training_phase=None):
        if training_phase is None:
            training_phase = self.training_phase
        N = X.shape[0]
        return self.fun([X.reshape((1, N)), training_phase])


def indices_of_k_smallest(arr, k):
    arr[arr < 1e-12] = np.inf
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
    # if you want it in a list of indices . . .
    # return np.array(np.unravel_index(idx, arr.shape))[:, range(k)].transpose().tolist()


def indices_of_k_greatest(arr, k):
    arr[arr < 1e-12] = np.inf
    idx = np.argpartition(arr.ravel(), -k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])


def feat_per_doc():
    """Compute local features per document."""

    categories = ['alt.atheism', 'comp.graphics',
                  'sci.space', 'talk.religion.misc']

    dw = data_utils.DataHandler()
    _ = dw.get_data(get_test=True)

    # get training data for interpretability
    dw_tr = data_utils.DataHandler()
    _ = dw_tr.get_data(get_test=False)

    lens = make_mapper_graph.Filter()

    X_proj = np.load(lens.model_path + "X_proj_iso.npy")
    with open(lens.model_path + "mapper_data.pkl", "rb") as f:
        graph = pickle.load(f)

    n_tests = 1
    n_samples = 1
    n_classes = len(categories)
    cnn = cnn_preds.DLPreds(n_tests, n_samples, n_classes, training=False)
    cnn.get_pred_funs()

    mapper = MapperData(data=X_proj, graph=graph)
    max_wds = 50

    for k, x in enumerate(dw.X_tok[:10]):
        doc = [dw.idx2word[w] for w in x if dw.idx2word[w] != "<pad>"]
        print(k, (" ").join(doc))
        mapper_pred = mapper.predict(x, cnn, lens)
        doc_ids = mapper.get_nodes_in_graph(mapper_pred)
        X_all = mapper.get_wordvecs(doc_ids, dw_tr, unique=False, ret_wds=False)
        X_new, wds = mapper.get_wordvecs(k, dw, unique=True, ret_wds=True)

        index = pynndescent.NNDescent(X_all,  metric='cosine')
        dtm = mapper.compute_dtm(index, X_new)
        ctr = Counter(dw_tr.X_tok[doc_ids].flatten())

        sorted_idxs = np.argsort(dtm, axis=None)
        zeros = np.where(dtm[sorted_idxs] < 0.5)[0]
        print(f"\n Pred: {categories[mapper_pred[2]]} " +\
              f"Label: {categories[dw.y[k]]} conf: {mapper_pred[-1]:4.4f}")

        for ictr, i in enumerate(zeros):
            if ictr > max_wds:
                break
            ii = sorted_idxs[i]
            if wds[ii] in dw_tr.idx2word.keys():
                print(dtm[ii], dw_tr.idx2word[wds[ii]], ctr[wds[ii]])
        print()


def global_features(bucket=0):
    """Compute global features per class."""

    # get training data for interpretability
    dw_tr = data_utils.DataHandler()
    _ = dw_tr.get_data(get_test=False)

    categories = ['alt.atheism', 'comp.graphics',
                  'sci.space', 'talk.religion.misc']

    dtm_params = {'alt.atheism': 0.045, 'comp.graphics': 0.05,
                  'sci.space': 0.05, 'talk.religion.misc': 0.05}

    category = categories[bucket]
    label = dw_tr.label2idx[category]
    print(f"\n Label: {category}")

    lens = make_mapper_graph.Filter()

    data = np.load(lens.model_path + "X_proj_iso.npy")
    with open(lens.model_path + "mapper_data.pkl", "rb") as f:
        graph = pickle.load(f)

    mapper = MapperData(data=data, graph=graph)

    filter_perc = 0.95
    docs, _ = mapper.filter_graph(filter_perc, label)

    # high-accuracy label specific words
    X_label, wds = mapper.get_wordvecs(docs, dw_tr, unique=True, ret_wds=True)

    # all words, given a label
    label_idxs = np.where(dw_tr.y == label)[0]
    X_all, all_wds = mapper.get_wordvecs(label_idxs, dw_tr, unique=False, ret_wds=True)

    w1 = []
    for w in all_wds:
        if w in dw_tr.idx2word.keys():
            word = dw_tr.idx2word[w]
        else:
            continue
        if word.isalnum() and len(word) > 1:
            if any(i.isdigit() for i in word):
                continue
            w1.append(word.lower())

    # count words
    wd_ctr = Counter(w1)
    total = np.sum(list(wd_ctr.values()))
    freq = np.asarray(list(wd_ctr.values())) / total
    freqs = {k: v / total for k, v in wd_ctr.items()}
    std = np.std(freq, ddof=1)
    zp = np.mean(freq) + 3 * std
    zm = np.mean(freq) - 3 * std

    filt_tok = [dw_tr.word2idx[w] for w in wd_ctr.keys() if freqs[w] < zp and freqs[w] > zm]
    word2tok = {dw_tr.idx2word[w]: w for w in filt_tok}
    ctr = Counter(dw_tr.X_tok[docs].flatten())

    print(f"\n Total words: {X_all.shape[0]}\n " +\
          f"Filerted words: {len(filt_tok)}\n Subset words: { X_label.shape[0]}")

    index = pynndescent.NNDescent(X_all[filt_tok], metric='cosine')

    index.prepare()
    dtm = mapper.compute_dtm(index, X_label, m=dtm_params[category])
    mapper.compute_density()
    sorted_idxs = np.argsort(dtm, axis=None)
    zeros = np.where(dtm[sorted_idxs] < 0.25)[0]
    wd_vec = []
    save_wds = []

    max_wds = 50
    ictr = 0

    for i in zeros:
        if ictr == max_wds:
            break
        ii = sorted_idxs[i]
        if wds[ii] in word2tok.values():
            ictr += 1
            print(f"{dtm[ii]:8.6f} {dw_tr.idx2word[wds[ii]]:12s} " +\
                  f"{ctr[wds[ii]]:4d} {mapper.f[ii]:7.6f}")
            wd_vec.append(X_label[ii])
            save_wds.append(dw_tr.idx2word[wds[ii]])

    out_f = "models/" + category + "_word2vec.npy"
    np.save(out_f, np.asarray(wd_vec))

    with open("models/" + category + "_wds.pkl", "wb") as f:
        pickle.dump(save_wds, f)

    print(" Density Stats")
    print(" Min\tMax\tMean\tStd")
    print(f"{np.min(mapper.f[mapper.f>0]):6.4f}  {np.max(mapper.f):6.4f}  " +\
         f"{np.mean(mapper.f[mapper.f>0]):6.4f}  {np.std(mapper.f[mapper.f>0], ddof=1):6.4f}")

    f_max = np.argmax(mapper.f)
    print(f"Max density word: {dw_tr.idx2word[wds[f_max]]}, dist {dtm[f_max]:6.4f}")


def main():
    # feat_per_doc()
    for i in range(4):
        global_features(bucket=i)


if __name__ == "__main__":
    main()
