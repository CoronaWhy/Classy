import sys
import os
sys.path.insert(0, '..')

from common.logging import create_logger
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from datasets import study_type_annotations_v2
from classifier import train_validate_catboost_model

from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm


if __name__=="__main__":
    logger = create_logger()
    df = study_type_annotations_v2()
    df["num_classes"] = df[["Computational",
                            "Experimental - in vitro", "Experimental - in vivo",
                            "Clinical-interventional", "Clinical-observational",
                            "Systematic review and/or meta-analysis", "Review"]].sum(axis=1)
    # currently we have:
    # num_classes value
    # 0.0         4864
    # 1.0         1541
    # 2.0         18
    # 3.0         3
    # so might as well only train on the ones that are single class, and ignore those that
    # are as yet unlabelled
    df = df[df["num_classes"]==1].copy()

    for category in ["Computational",
                            "Experimental - in vitro", "Experimental - in vivo",
                            "Clinical-interventional", "Clinical-observational",
                            "Systematic review and/or meta-analysis", "Review"]:
        df.loc[df[category] ==1, "label"] = category

    model_df = df[['abstract', 'title', 'label']].sample(frac=1.0)
    test_set_len = len(model_df) //5


    extra_text_params = {
        'dictionaries': [
            'Word:min_token_occurrence=2',
            'BiGram:gram_order=2',
            'UniGram:gram_order=1'
        ],
        'text_processing': [
            'BM25+Word|NaiveBayes+Word,BiGram|BoW:top_tokens_count=1000+Word,BiGram'
        ]
    }
    boosting_params = {
        #     'iterations': 2000,
        #         'learning_rate': 0.005,
    }

    cv_results = []

    grid = {
        'learning_rate': [0.1, 0.2, 0.5],
        'depth': [2, 4, 6],
        # 'l2_leaf_reg': [1, 4],
        "iterations": [1000, 2000, 3000]
            }

    for param_set in tqdm(list(product(grid["learning_rate"], grid["depth"],
                                       # grid["l2_leaf_reg"],
                                       grid["iterations"]))):
        boosting_params = {
            "learning_rate": param_set[0],
            "depth": param_set[1],
            # "l2_leaf_reg": param_set[2],
            "iterations": param_set[2]
        }

        params = {**extra_text_params, **boosting_params}
        results = []
        for i in tqdm(range(5)):

            test = model_df.iloc[i * test_set_len: (i + 1) * test_set_len]
            test_classes = np.sort(test.label.unique())
            train = model_df[~model_df.index.isin(test.index)]
            model, acc, f1, f1_macro, confusion_matrix, params = train_validate_catboost_model(
                train, test,
                ['abstract', 'title'],
                'label',
                text_features=['abstract', 'title'], params=params, verbose=False, logging=False)
            results.append({
                "i": i,
                "f1": {k: v for k, v in zip(test_classes, f1)},
                "acc": acc,
                'f1_macro': f1_macro,
                'confusion_matrix': confusion_matrix
            })


        results = pd.DataFrame(results)
        params_str = ",".join([f"{k}={v}" for k, v in params.items()])
        # logger.info(pd.DataFrame(results).to_string())

        mean_acc = results.acc.mean()
        mean_f1_macro = results.f1_macro.mean()
        mean_f1 = pd.DataFrame(results.f1.to_list()).mean().to_dict()
        logger.info(f"mean_f1_macro: {mean_f1_macro}, mean_acc: {mean_acc}")
        cv_results.append({**boosting_params, **{'mean_f1': mean_f1, 'mean_f1_macro': mean_f1_macro, 'mean_acc': mean_acc}})
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv("cv_results_catboost_design_classification_v2_trial2.csv")
    logger.info(cv_results.to_string())


