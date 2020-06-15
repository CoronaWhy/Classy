# Classy
Supervised classification of Study designs

## Our purpose:
https://docs.google.com/document/d/1OVDxoY1M1XJUMI6YG1g5dubGx3aB1ObfgE2gucnNEEg/edit

### Some useful links:

from the organiser/curators of the CORD-19 competition/dataset
- https://docs.google.com/spreadsheets/d/1t2e3CHGxHJBiFgHeW0dfwtvCG4x0CDCzcTFX7yz9Z2E/edit#gid=1217643351
- https://curation-taskfroce.slack.com/archives/C011DFP73AA/p1587604383302700

### Classifiers:
#### Catboost with text features
[Catboost](https://catboost.ai/) is a machine learning library that runs gradient boosting algorithm on decision trees. It supports numerical, categorical, and most interestingly, text features. Details of it's text processing algorithms can be found [here](https://catboost.ai/docs/concepts/algorithm-main-stages_text-to-numeric.html) and in [this demo notebook](https://github.com/catboost/tutorials/blob/2c16945f850503bfaa631176e87588bc5ce0ca1c/text_features/text_features_in_catboost.ipynb) and extra details around particular flavour of gradient boosting catboost uses can also be found in their docs.

**Input data:**
I used a dataset of approximately 1500 papers annotated by CoronaWhy volunteers, consisting of these classes: `Computational`,`Experimental - in vitro`, `Experimental - in vivo`, `Clinical-interventional`, `Clinical-observational`,`Systematic review and/or meta-analysis`, `Review`.

**Features**
For each of these papers, I used the 'title' and 'abstract' text from the metadata file for training. I also tried using specter embeddings (on their own and also with the title and abstract), and also article UMLs as features. However, the performance of the classifier was similar to using title and abstract on their own so in the end I went with the simpler features of title and abstract.

**Training**
I tuned the hyperparameters of the model using gridsearch on the base gradient boosting parameters (e.g. `learning_rate`, `depth`) with 5 fold cross validation. The text features hyperparameters (e.g. tokenizers, text-to-numeric transformers,), I mostly tested a few combinations manually. The metrics monitored are f1_macro, per class F1 and accuracy.The best params found in the tuning process was then used to train the final classifier over the whole dataset.

**performance**
The cross validation f1 scores for the different classes vary widely:
|class label| f1|
|:--|:--|
|Clinical-interventional| 0.74|
| Clinical-observational| 0.86 |
| Computational |0.45|
| Experimental - in vitro | 0.39|
| Experimental - in vivo'| 0.1|
| Review| 0.46|  
| Systematic review and/or meta-analysis| 0.89|

This is not altogether surprising since we have _very_ limited labelled data with high class imbalance. Getting more training data, and/or exploring semi supervised techniques should really help in improving classifer performance.
