"""

    plots.py: script to create visuals from output from feature_extrator.py

"""
import pickle

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.manifold import TSNE
from matplotlib.colors import to_hex


categories = ['alt.atheism', 'comp.graphics',
              'sci.space', 'talk.religion.misc']
d = []
t = []
lens = []
wds = []

for i, c in enumerate(categories):
    in_f = "./models/" + c + "_word2vec.npy"
    tmp = np.load(in_f)
    d.append(tmp)
    t.append(i * np.ones(tmp.shape[0], dtype=np.int_))
    lens.append(tmp.shape[0])
    with open("./models/" + c + "_wds.pkl", "rb") as f:
        wds.append(pickle.load(f))

data = np.concatenate((d[0], d[1], d[2], d[3]))
target = np.asarray(np.concatenate((t[0], t[1], t[2], t[3]))).flatten()
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = [categories[t1] for t1 in target]
colors = [color[t1] for t1 in target]
color_dict = {"alt.atheism": to_hex('tab:blue'),
              "comp.graphics": to_hex('tab:orange'),
              'sci.space': to_hex('tab:green'),
              'talk.religion.misc': to_hex('tab:red')}

words = [w for sublist in wds for w in sublist]

tsne = TSNE(metric="cosine", random_state=42, n_jobs=-1,
            method="exact").fit_transform(data)

df = pd.DataFrame({"tsne1": tsne[:, 0], "tsne2": tsne[:, 1], "label": labels,
                   "size": np.ones(tsne.shape[0]),
                   "words": words})

fig = px.scatter(df, x='tsne1', y='tsne2', color='label',
                 size='size', color_discrete_map=color_dict,
                 hover_data={'tsne1': False, 'tsne2': False, 'label': True,
                             'size': False, 'words': True})
fig.update_layout(plot_bgcolor='rgb(255,255,255)')
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(
    title="Newsgroups Word Embedding",
    xaxis_title="",
    yaxis_title="",
    legend_title="Newsgroup Classes",
    font={"family": "Times New Roman", "size": 18}
)

# fig.show()
fig.write_html("./results/newsgroup_words.html")
fig.write_image("./results/newsgroup_words.png", scale=1, width=2400, height=2400)
