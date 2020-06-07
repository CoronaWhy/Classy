import sys
import os
sys.path.insert(0, '..')

from common.logging import create_logger
from pathlib import Path
import pandas as pd
import catboost
import numpy as np
from itertools import product
from datasets import study_type_annotations_v2
from classifier import train_validate_catboost_model
from catboost import Pool, CatBoostClassifier

from datetime import datetime
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# set this to True if you want to retrain the model e.g. with new data
RETRAIN_MODEL = False
CATBOOST_MODEL_NAME="study_design_catboost_classifier_7_June_2020.cbm"

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

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

    model_df = df[['abstract', 'title', 'label']]
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
        'learning_rate': 0.5,
        'depth': 4,
        "iterations": 2000
    }

    if RETRAIN_MODEL:
        params = {**extra_text_params, **boosting_params}
        model = train_validate_catboost_model(
            train_data=model_df, test_data=None,
            train_features=['abstract', 'title'],
            target_feature='label',
            text_features=['abstract', 'title'], params=params, verbose=False, logging=False)
        model.save_model(str(Path(__location__)/CATBOOST_MODEL_NAME))

    else:
        model =  CatBoostClassifier()
        model.load_model(str(Path(__location__)/CATBOOST_MODEL_NAME))

    test_data = pd.read_csv("/home/wwymak/coronawhy/pubmed_mining/data/metadata_v22.csv")

    test_pool = Pool(
        test_data[['abstract', 'title']].fillna(""),
        feature_names=['abstract', 'title'],
        text_features=['abstract', 'title'])
    predictions = model.predict(test_pool)
    test_data["label_catboost_classifier"] = predictions
    test_data.to_csv("/home/wwymak/coronawhy/pubmed_mining/data/metadata_v22.csv", index=False)