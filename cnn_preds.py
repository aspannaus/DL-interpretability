import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import data_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.keras.utils.disable_interactive_logging()
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()
# np.random.seed(42)


class DLPreds():
    """Class to make multiple predictions from trained model.


        Args:
            n_tests: number of predictions for each sample, int
            n_samples: number of data points, as int
            n_classes: number of possible classes, as int
            training: stochastic or deterministic inference, as bool


        Attributes:
            self.n_tests = number of predictions for each sample
            self.model_file = path to h5 file with trained model
            self.class_cts = number of possible classes
            self.n_samples = number of data points
            self.training_phase = stochastic or deterministic inference
            self.preds = array of predictions
            self.pred_class = array of predicted classes
            self.gt_prob = array of ground-truth scores
            self.pred_prob = array of predicted class scores
            self.pred_means = average of predicted class scores
            self.y = ground truth array

            self.model = trained model

            self.docs = array of document embeddings
            self.mean_doc = average document embedding
            self.fun = function handle returning doc embedding and softmax score per sample
            self.pred = function handle returning softmax score per sample
            self.doc_vec = function handle returning doc embedding score per sample

    """
    def __init__(self, n_tests, n_samples, n_classes, training=False):
        self.n_tests = n_tests
        self.path = "./DL-interpretability/"
        self.model_file = "./models/newsgroups_cnn.h5"
        self.class_cts = n_classes
        self.n_samples = n_samples
        self.training_phase = training
        self.preds = np.zeros((self.n_tests, self.class_cts))
        self.pred_class = np.zeros((self.n_samples), dtype=np.int_)
        self.gt_prob = np.zeros((self.n_samples))
        self.pred_prob = np.zeros((self.n_samples))
        self.pred_means = np.zeros((self.n_samples))
        self.y = np.zeros((self.n_samples), dtype=np.int_)

        print("\n Loading model from " + self.model_file)
        self.model = load_model(self.model_file, compile=True)

        N = self.model.layers[-2].output.shape[1]
        self.docs = np.zeros((self.n_tests, N))
        self.mean_doc = np.zeros((self.n_samples, N))
        self.fun = None
        self.pred = None
        self.doc_vec = None

    def get_pred_funs(self, verbose=True):
        """Defines function handles for prediction step."""
        if verbose:
            self.model.summary()
        self.fun = K.function([[self.model.layers[0].input],
                               K.learning_phase()],
                              [self.model.layers[-1].output,
                              self.model.layers[-2].output])

        self.pred = K.function([[self.model.layers[0].input],
                                K.learning_phase()],
                               self.model.layers[-1].output)

        self.doc_vec = K.function([[self.model.layers[0].input],
                                  K.learning_phase()],
                                  self.model.layers[-2].output)

    def predict(self, X, training_phase=None):
        """Define predict function.

            Args:
                X: np.ndarray of features
                training_phase: stochastic or deterministic inference

            Returns:
                prediction from saved model
        """
        if training_phase is None:
            training_phase = self.training_phase
        N = X.shape[0]
        return self.pred([X.reshape((1, N)), training_phase])

    def doc_embedding(self, X, training_phase=None):
        """Predict function returning doc embedding.

            Args:
                X: np.ndarray of features
                training_phase: stochastic or deterministic inference

            Returns:
                doc embedding from saved model
        """
        if training_phase is None:
            training_phase = self.training_phase
        N = X.shape[0]
        return self.doc_vec([X.reshape((1, N)), training_phase])

    def cnn_output(self, X, training_phase=None):
        """Function returning doc embedding and softmax scores.

            Args:
                X: np.ndarray of features
                training_phase: stochastic or deterministic inference

            Returns:
                doc embedding and softmax scores from saved model
        """
        if training_phase is None:
            training_phase = self.training_phase
        N = X.shape[0]
        return self.fun([X.reshape((1, N)), training_phase])

    def get_pred_stats(self, test_x, test_y):
        """Get summary statistics about the predictions."""

        # make predictions
        for i in range(self.n_tests):
            probs, doc = self.fun([test_x, self.training_phase])
            self.preds[i] = probs.flatten()
            self.docs[i] = doc.flatten()

        # get summary statistics
        self.pred_means = np.mean(self.preds, axis=0)
        self.gt_prob = self.pred_means[test_y]
        self.pred_class = np.argmax(self.pred_means, axis=0)
        self.pred_prob = self.pred_means[self.pred_class]
        self.mean_doc = self.docs.mean(axis=0)
        return (self.pred_class, self.pred_prob, self.gt_prob, self.mean_doc)


def progress_bar(value, end_value, bar_len=25):
    """Display progress bar during inference."""
    percent = float(value) / end_value
    arrow = '-' * int(round(percent * bar_len) - 1) + '>'
    spaces = ' ' * (bar_len - len(arrow))
    sys.stdout.write(f'\r Percent Complete : [{arrow+spaces}] {int(round(percent * 100))}%')
    sys.stdout.flush()


def main():
    n_predictions = 500

    dw = data_utils.DataHandler()
    _ = dw.get_data()

    cnn_preds = DLPreds(n_predictions, dw.n_samples, dw.n_classes, training=True)
    cnn_preds.get_pred_funs(verbose=True)

    pred_probs = []
    pred_class = []
    gt_probs = []
    doc_embedding = []

    print("\n Making predictions from trained model")
    for n in range(dw.n_samples):
        progress_bar(n, dw.n_samples)
        N = dw.X_tok[n].shape[0]
        X = dw.X_tok[n, :].reshape((1, N))
        cnn_out = cnn_preds.get_pred_stats(X, dw.y[n])
        pred_class.append(cnn_out[0])
        pred_probs.append(cnn_out[1])
        gt_probs.append(cnn_out[2])
        doc_embedding.append(cnn_out[3])

    print(f"\n Saving prediction arrays to: {cnn_preds.path}preds")
    if not os.path.exists("./preds/"):
        os.mkdir("./preds")

    np.savez_compressed('./preds/cnn_preds.npz',
                        pred_probs=pred_probs, pred_class=pred_class,
                        gt_probs=gt_probs, doc_embedding=doc_embedding)


if __name__ == "__main__":
    main()
