## DL Interpretability

## Description

This repository is the Tensorflow implementation of the paper [Topological Interpretability for Deep-Learning](https://arxiv.org/abs/2305.08642)
and reproduces the figures within. The code is model agnostic, and may be used with your own 
custom Tensroflow deep learning models; a pytorch version is in development.

## Quickstart Guide


### Install from source
Clone the repo
```shell
git clone https://code.ornl.gov/3t6/DL-interpretability.git
```
Load the working branch and pull in the subrepos
```shell
cd DL-interpretability
git checkout main
```

We've tested the code with python 3.8.15 and associated libraries. You can install the dependencies via
```shell
pip install -r requirements.txt
```

For further Tensorflow instructions or more details, head over to the [Tensorflow docs](https://www.tensorflow.org/api_docs).


## Workflow

The worflow for the code is best demonstated with an example on a subset 
of the 20newsgroups data.

First we need to download and pre-process the data
```
python data_setup.py

```
and create our CNN
```
python newsgroups_cnn.py
```

From our trained model, we need to make multiple predictions on each input to 
characterize the distribution of outcomes. This is done by running
```
python cnn_preds.py
```

Before creating our Mapper graph, we need to compile the distance function required to find good choices 
for the gain and resolution of the Mapper graph. The command is
```
python setup.py build_ext --inplace
```

The next step is to create the Mapper graph by
```
python make_mapper_graph.py
```
An inteactive version of the Mapper output is saved as an `html` file for model inspection. 

The reamaining scripts extract the relevant keywords and cluster them in 2D for easy visualization.
First, identify the relevant words for each class
```
python feature_extractor.py
```
and visualize the word clusters of relevant words
```
python plots.py
```

The resulting plots are in saved in the `results/` directory.

## Contributing
Get in touch if you would like to help in writing code, example notebooks, and documentation are essential aspects of the project. To contribute please fork the project, make your proposed changes and submit a pull request. We will do our best to sort out any issues and get your contributions merged into the main branch.

If you found a bug, have questions, or are just having trouble with the library, please open an issue in our issue tracker and we'll try to help resolve it.

