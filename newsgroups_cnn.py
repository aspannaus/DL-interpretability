"""
    Module for defining and training a deep learning model
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import concatenate

import data_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.keras.utils.disable_interactive_logging()
np.random.seed(42)


def plot_graphs(history, metric):
    """Plot training history."""
    plt.plot(history.history[metric])
    plt.plot(history.history["val_"+metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_"+metric])


def define_model(max_doc_len, embedding_matrix, n_classes, dropout=0.5):
    """Define a CNN model for raw text.

        Args:
            max_doc_len: longest doc, as int
            embedding_matrix: word embedding matric, as float32
            n_classes: number of classes, as int
            dropout: dropout rate, as float

        Returns:
            tensorflow model

    """
    # initializer = tf.keras.initializers.Constant(embedding_matrix)
    initializer = tf.keras.initializers.GlorotUniform()
    vocab_len = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]

    data_in = Input(shape=(max_doc_len,))
    embedding = Embedding(vocab_len, embedding_dim, trainable=True,
                          embeddings_initializer=initializer)(data_in)
    conv1 = Conv1D(filters=embedding_dim, kernel_size=3,
                   activation="relu",
                   padding="same")(embedding)
    conv2 = Conv1D(filters=embedding_dim,
                   kernel_size=4,
                   activation="relu",
                   padding="same")(embedding)
    conv3 = Conv1D(filters=embedding_dim,
                   kernel_size=5,
                   activation="relu",
                   padding="same")(embedding)

    concat = concatenate([conv1, conv2, conv3], axis=2)
    # doc_embeds = tf.math.reduce_max(concat, 1)
    doc_embeds = Dropout(dropout, name='concat_dropout')(tf.math.reduce_max(concat, 1, name='max_pool'))
    outputs = Dense(n_classes, activation="softmax")(doc_embeds)

    model = Model(inputs=data_in, outputs=outputs)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2.0e-4, amsgrad=True),
                  metrics=["accuracy"])
    print(model.summary())
    return model


def main():
    tf.config.threading.set_inter_op_parallelism_threads(0)
    dw = data_utils.DataHandler()
    _ = dw.get_data()

    model = define_model(dw.X_tok.shape[1], dw.word2vec, dw.n_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-2,
            patience=4,
            verbose=1,
        )
    ]

    history = model.fit(dw.X_tok,
                        dw.y,
                        epochs=35, batch_size=128,
                        callbacks=callbacks,
                        validation_split=0.2)

    dw.get_data(get_test=True)

    test_loss, test_acc = model.evaluate(dw.X_tok, dw.y, verbose=1)

    model.save("./models/newsgroups_cnn.h5")

    print(" Saving Model, Vocab, and word2vec")
    # save updated word embeddings
    np.save("./models/vectors.npy",
            model.get_layer("embedding").get_weights()[0])

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, "accuracy")
    plt.subplot(1, 2, 2)
    plot_graphs(history, "loss")
    if not os.path.exists("./results/"):
        os.mkdir("./results")
    plt.savefig("./results/newsgroups_cnn.png")


if __name__ == "__main__":
    main()
