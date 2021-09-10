# Title: GAT
# Author: Junghwan Kim
# Date: April 21, 2021

import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

########## Real data: Homogeneous graph from CSV files ##########
# the base_directory property tells us where it was downloaded to:
cora_cites_file = os.path.join("GAT/data/cora.cites")
cora_content_file = os.path.join("GAT/data/cora.content")

cora_cites = pd.read_csv(
    cora_cites_file,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["target", "source"],  # set our own names for the columns
)
print(cora_cites)

cora_feature_names = [f"w{i}" for i in range(1433)]
cora_raw_content = pd.read_csv(
    cora_content_file,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["id", *cora_feature_names, "subject"],  # set our own names for the columns
)
print(cora_raw_content)

cora_content_str_subject = cora_raw_content.set_index("id")
print(cora_content_str_subject)

cora_content_one_hot_subject = pd.get_dummies(
    cora_content_str_subject, columns=["subject"]
)
print(cora_content_one_hot_subject)

# One-hot encoding
cora_one_hot_subject = sg.StellarGraph(
    {"paper": cora_content_one_hot_subject}, {"cites": cora_cites}
)
print(cora_one_hot_subject.info())

########## Node classification with Graph ATtention Network (GAT) ##########

dataset = datasets.Cora()
GX, node_subjects = dataset.load()
set(node_subjects)

# Splitting the data
train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=200, test_size=None, stratify=node_subjects
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=700, test_size=None, stratify=test_subjects
)

from collections import Counter
Counter(train_subjects)

target_encoding = preprocessing.LabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)

generator = FullBatchNodeGenerator(cora_one_hot_subject, method="gat")

train_gen = generator.flow(train_subjects.index, train_targets)

gat = GAT(
    layer_sizes=[8, train_targets.shape[1]],
    activations=["elu", "softmax"],
    attn_heads=8,
    generator=generator,
    in_dropout=0.5,
    attn_dropout=0.5,
    normalize=None,
)

x_inp, predictions = gat.in_out_tensors()

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.005),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)

val_gen = generator.flow(val_subjects.index, val_targets)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

if not os.path.isdir("logs"):
    os.makedirs("logs")
es_callback = EarlyStopping(
    monitor="val_acc", patience=20
)  # patience is the number of epochs to wait before early stopping in case of no further improvement
mc_callback = ModelCheckpoint(
    "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
)

history = model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    callbacks=[es_callback, mc_callback],
)

sg.utils.plot_history(history)
model.load_weights("logs/best_model.h5")

test_gen = generator.flow(test_subjects.index, test_targets)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

model.save("my_model")



########## Making predictions with the model ##########

all_nodes = node_subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)

node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
print(df.head(20))

########## Node embeddings ##########

emb_layer = next(l for l in model.layers if l.name.startswith("graph_attention"))
print(
    "Embedding layer: {}, output shape {}".format(emb_layer.name, emb_layer.output_shape)
)

embedding_model = Model(inputs=x_inp, outputs=emb_layer.output)
emb = embedding_model.predict(all_gen)
print(emb.shape)

X = emb.squeeze()
y = np.argmax(target_encoding.transform(node_subjects), axis=1)

###############################################################################################
jX, labels_true = emb, y
n_cluster = len(set(labels_true))
print(n_cluster)

kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
labels = kmeans.labels_

from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

print("Mutual Information: %0.3f"
      % mutual_info_score(labels_true, labels))
print("Normalized Mutual Information: %0.3f"
      % normalized_mutual_info_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % adjusted_mutual_info_score(labels_true, labels))

import seaborn as sns
palette = np.array(sns.color_palette('muted', n_colors=len(np.unique(y))))
tsne = TSNE()
a, b, c = emb.shape
print(emb.shape)
emb2 = emb.reshape((b, c))
vis = tsne.fit_transform(emb2)

plt.figure(figsize=[10, 8])
plt.scatter(vis[:, 0], vis[:, 1], c=palette[labels], s=20, alpha=0.8)

plt.show()










# kmeans = KMeans(n_clusters=7, random_state=0)
# y_pred = kmeans.fit_predict(X)
#
# print(y_pred.shape)
#
# df = pd.DataFrame(dict(name=features.keys(), songs=features.values()))
# df['predict'] = y_pred



exit()




if X.shape[1] > 2:
    transform = TSNE  # PCA

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=list(GX.nodes()))
    emb_transformed["label"] = y
else:
    emb_transformed = pd.DataFrame(X, index=list(G.nodes()))
    emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
    emb_transformed["label"] = y

alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="jet",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of GAT embeddings for cora dataset".format(transform.__name__)
)
plt.show()