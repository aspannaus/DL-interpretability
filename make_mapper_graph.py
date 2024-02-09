#!/usr/bin/env python3

"""
    File    : make_mapper_graph.py
    Author  : Adam Spannaus
    Created : 07/12/2020
    Modifid : Today

    Purpose: Create mapper graph of MTCNN prediction layer

"""
import math
import os
import pickle
import time

import numpy as np
import kmapper as km

from scipy.spatial.distance import pdist, squareform
from sklearn import cluster
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap

import hausdorff_dist
import data_utils


np.random.seed(42)


def find(cover, data_point):
    """Finds the hypercubes that contain the given data point.
        
        Args: 
        data_point: array-like
            The data point to locate.
        
        Returns:
            cube_ids: list of int
            list of hypercube indices, empty if the data point is outside the cover.
    """
    cube_ids = []
    for i, center in enumerate(cover.centers_):
        lb = center[:2] - cover.radius_[:2]
        ub = center[:2] + cover.radius_[:2]
        if np.all(data_point >= lb) and np.all(
            data_point[:2] <= ub
        ):
            cube_ids.append(i)
    return cube_ids


class Filter():
    """Class to create filter for Mapper graph construction.
        
        Attributes:
        -------------------------
        data_size = number of samples in dataset, as int
        pred_probs = array of prediction scores, as np.ndarray
        pred_class = array of prediction classes, as np.ndarray
        gt_probs = array of ground truth classes, as np.ndarray
        de = array of doc embeddings, as np.ndarray
        save_model = to save the Mapper model, as bool 

        model_path = path to saved models, as str
        filter_path = path to saved predictions, as str
        proj_X = projection of data into low-dimensional space, as np.ndarray
        preimage = data preimage, as np.ndarray

        svd = projection method for lens, step 1, as sklearn svd object
        emb = projection method for lens, step 2, as sklearn isomap object

        The projection step maps the 900 dim doc embedding first to 300 dims,
        then takes the first 100 dims of the svd projection, and maps to 2D via 
        isomap.

    """
    def __init__(self, data_size=0, save_model=True):
        self.data_size = data_size
        self.pred_probs = np.zeros((self.data_size))
        self.pred_class = np.zeros((self.data_size), dtype=np.int32)
        self.gt_probs = np.zeros((self.data_size))
        self.de = None  # doc embeddings
        self.save_model = save_model
        self.path = "./DL-interpretability/"

        if self.save_model:
            self.model_path = "./models/"

        self.filter_path = "./preds/"
        self.proj_X = None
        self.preimage = None

        if os.path.isfile(self.model_path + "svd.pkl"):
            with open(self.model_path + "svd.pkl", "rb") as f:
                self.svd = pickle.load(f)
        else:
            self.svd = None

        if os.path.isfile(self.model_path + "iso.pkl"):
            with open(self.model_path + "iso.pkl", "rb") as f:
                self.emb = pickle.load(f)
        else:
            self.emb = None

    def get_filter_vals(self):
        """Load saved prediction arrays from disk.

        Post-condition:
        -----------------------------
        Class attributes
            self.pred_probs
            self.pred_class
            self.gt_probs
            self.de
        are all populated with values.

        """
        loaded = np.load(self.filter_path + "cnn_preds.npz")
        self.pred_probs = loaded["pred_probs"]
        self.pred_class = loaded["pred_class"]
        self.gt_probs = loaded["gt_probs"]
        self.de = loaded["doc_embedding"]

    def set_filter(self, dw):
        """Computes the filter for Mapper graph.


        Params:
        -----------------
            dw: DataHandler class object

        Returns: None

        Post-condition:
            projection step objects saved, and preimage and projection class
            attributes populated.

        """

        if self.svd is not None:
            with open(self.model_path + "svd.pkl", "rb") as f:
                self.svd = pickle.load(f)
                X1 = self.svd.fit_transform(self.de)
        else:
            self.svd = TruncatedSVD(n_components=300, n_iter=7, random_state=42)
            X1 = self.svd.fit_transform(self.de)
            with open(self.model_path + "svd.pkl", "wb") as f:
                pickle.dump(self.svd, f)

        N = 100

        if self.emb is not None:
            with open(self.model_path + "iso.pkl", "rb") as f:
                self.emb = pickle.load(f)
                X_u = self.emb.fit_transform(X1[:, :N])
        else:
            self.emb = Isomap(n_components=2, n_jobs=16, n_neighbors=20)
            X_u = self.emb.fit_transform(X1[:, :N])
            with open(self.model_path + "iso.pkl", "wb") as f:
                pickle.dump(self.emb, f)

        # clustering in the doc embedding space is not computationlly tractable
        # for memory reasons, so we cluster in the reduced space X_u
        self.preimage = X_u
        self.proj_X = np.c_[X_u[:, 0], X_u[:, 1], self.pred_class, dw.y, self.gt_probs]

    def save_filter(self):
        """Save projected data points to disk."""
        np.save(self.model_path + "X_proj_iso.npy", self.proj_X)

    def predict(self, X):
        """Apply saved transformations to new data point X."""
        X_lsi = self.svd.transform(X)
        N = 100
        X = self.emb.transform(X_lsi[:, :N])

        return X.flatten()


class MakeMapper():
    """Class to create the Mapper graph.

        Attributes:
        ------------------
            proj_X: projection of data into low-dim space, as np.ndarray
            preimage: preimage of the dataset, as np.ndarray
            n_bootstrap: num bootstrap subsamples in Hausdorff dist, as int
            save_model: are we saving the model, as bool
            r: mapper resolution parameter, as int
            gain: mapper gain, as int
            cover: KeplerMapper cover class
            G: KeplerMapper mapper graph class

    """
    def __init__(self, proj_X=None, preimage=None, n_subsamples=250, save_model=True):
        self.proj_X = proj_X
        self.preimage = preimage
        self.n_bootstrap = n_subsamples
        self.save_model = save_model
        # self.path = "./DL-interpretability/"
        self.r = None
        self.G = None
        self.cover = None

        self.gain = 0.5 * np.ones(5)
        self.gain[3] = 0.0
        if not os.path.exists('./results'):
            os.mkdir("./results")

    def set_params(self):
        """Compute Mapper gain and resolution based on the dataset.
        
        
            Post-condition:
            --------------------
            self.r and self.G populated
        
        """
        print(" Estimating Mapper parameters...")
        dm = squareform(pdist(self.proj_X[:, :2]))
        params = hausdorff_dist.hausdorff_dist(self.proj_X[:, :2], dm,
                                               self.n_bootstrap, self.gain[0],
                                               self.proj_X[:, :2].shape[1])
        self.r = np.power(params[3], 1.0 / self.proj_X.shape[1])

        if self.r > 45:
            self.r = 45
        elif self.r < 2:
            self.r = 2
        else:
            self.r = math.ceil(self.r)

        print(f" Mapper params: gain = {self.gain}\tresolution = {self.r}")

    def compute_dm(self, dw):
        """Compute distance matrix for mapper construction.

            Args:
            --------------
            dw: DataHandler class object

            Converts categorical data to one-hot and uses Jaccard distance between entries.

            Returns:
            distance matrix between all inputs as np.ndarray

        """
        lb = preprocessing.LabelBinarizer()
        Y = lb.fit_transform(dw.y)
        N = self.proj_X.shape[0]
        euclidean_dm = np.zeros((N, N))
        prob_dm = np.zeros((N, N))
        class_dm = np.zeros((N, N))

        euclidean_dm = squareform(pdist(self.proj_X[:, :2], metric="euclidean"))
        prob_dm = squareform(pdist(self.proj_X[:, -1].reshape(-1, 1), metric="euclidean"))
        class_dm = squareform(pdist(Y, metric="jaccard"))
        return euclidean_dm + class_dm + prob_dm

    def make_graph(self, output_color_fn, dw):
        """Create mapper graph.


            Args:
            -------------
            output_color_fn: array of values to color Mapper nodes, as np.ndarray
            dw: DataHandler class object


            Returns: None

            Mapper object saved to disk and visualization saved as html file.

        """
        print("Creating mapper object")
        mapper = km.KeplerMapper(verbose=1)
        print("Creating graph")

        self.cover = km.Cover(perc_overlap=[0.5, 0.5, 0, 0, 0.5], n_cubes=[3, 3, 4, 4, 9])
        # cover class requires an index column in the first column
        # data_idx = np.arange(1, self.proj_X.shape[1] + 1)
        # idx = np.arange(self.proj_X.shape[0])
        # X_idx = np.c_[idx, self.proj_X]
        # # _ = self.cover.fit_transform(X_idx)
        # _ = self.cover.fit(self.proj_X)

        dm = self.compute_dm(dw)

        self.G = mapper.map(self.proj_X, X=dm,
                            cover=self.cover,
                            remove_duplicate_nodes=True,
                            precomputed=True,
                            clusterer=cluster.AgglomerativeClustering(metric="precomputed",
                                                                      n_clusters=2,
                                                                      linkage="single"))

        if self.save_model:
            with open("./models/cover.pkl", "wb") as f:
                pickle.dump(mapper.cover, f)

        if not os.path.exists("./results/"):
            os.mkdir("./results")

        today = time.strftime("%m%d")
        path = "./results/mapper_" + today + ".html"
        print(" Writing to disk\n")

        _ = mapper.visualize(self.G,
                             lens=self.proj_X,
                             lens_names=["Iso 0", "Iso 1", "Predicted", "Ground Truth", "Prob of GT"],
                             color_values=output_color_fn,
                             color_function_name=['True Label'],
                             title="Confidence Graph for MT CNN",
                             path_html=path, nbins=10)

    def save_graph(self):
        """Saves compted Mapper graph to disk."""
        out_path = "./models/mapper_data.pkl"
        print(f"\n Saving graph to {out_path}")
        with open(out_path, "wb") as f_out:
            pickle.dump(self.G, f_out)


def main():

    dw = data_utils.DataHandler()
    _ = dw.get_data(get_test=False)

    N = dw.X_tok.shape[0]

    lens = Filter(N)
    lens.get_filter_vals()
    lens.set_filter(dw)

    n_subsamples = 250
    mapper = MakeMapper(lens.proj_X, lens.preimage, n_subsamples)
    mapper.set_params()
    mapper.make_graph(dw.y, dw)

    mapper.save_graph()
    lens.save_filter()


if __name__ == "__main__":
    main()
