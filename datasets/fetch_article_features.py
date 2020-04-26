import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from fastcore.utils import *

def _enrich_annotations_one_subset(annotations_dataset, processed_article_data, extra_features, sections):
    """
    enrich annotations data with fields from the v8 processed data (e.g. UMLs etc)
    :param annotations_dataset: pd.DataFrame
    :param processed_article_data: pd.DataFrame from v8 processed data, the whole cord19 is divided into 20 files
    :param extra_features: List of fields from processed_article_data to append to annotations_dataset
    :return: pd.DataFrame annotations_dataset with added columns extra_features
    """
    enriched_dataset = annotations_dataset.copy()
    original_columns = enriched_dataset.columns
    for section in sections:
        new_feature_names = [f"{section}_{x}" for x in extra_features]
        enriched_dataset = enriched_dataset.merge(
            processed_article_data[processed_article_data.section==section],
            left_on="cord_uid", right_on="cord_uid",how="left")[original_columns + extra_features]
        enriched_dataset = enriched_dataset.rename(columns={
            k: v for k, v in zip(extra_features, new_feature_names)
        })
    return enriched_dataset


def enrich_annotations(annotations_dataset, processed_article_folder, extra_features, sections):
    enriched_dataset = annotations_dataset.copy()
    processed_text_filepaths = processed_article_folder.ls(file_type='.pkl')
    for filepath in tqdm(processed_text_filepaths):
        processed_article_data = pd.read_pickle(filepath)
        enriched_dataset = _enrich_annotations_one_subset(
            annotations_dataset, processed_article_data, extra_features, sections)

    return enriched_dataset
