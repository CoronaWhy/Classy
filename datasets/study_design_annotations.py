import pandas as pd
import numpy as np

def single_label_multiclass_annotated_study_design(annotations_filepath, metadata_filepath):
    """
    match david's annotated dataset (colums id, label) with the article metadata
    file from kaggle cord-19 https://www.kaggle.com/davidmezzetti/cord19-study-design
    :param annotations_filepath:
    :param metadata_filepath:
    :return: pd.DataFrame of annotations, with columns ["sha", "cord_uid",  "title", "abstract", "label", "label_str"]
    """

    label_number_to_study_name_mapping = {
        1: "Meta analysis", #systematic review and meta-analysis
        2: "Randomized control trial",
        3: "Non-randomized trial",
        4: "Prospective cohort", #prospective observational study
        5: "Time-series analysis",
        6: "Retrospective cohort", #retrospective observational study
        7: "Cross-sectional", # cross sectional study
        8: "Case control",
        9: "Case study", #case series
        10: "Simulation", #simulation
        0: "Other"
    }
    annotations = pd.read_csv(annotations_filepath)
    cord19_metadata = pd.read_csv(metadata_filepath)

    annotations = annotations.rename(columns={"id": "sha"}).dropna()
    annotations = annotations.merge(cord19_metadata, left_on="sha", right_on="sha")[
        ["sha", "cord_uid", "label", "title", "abstract"]]
    annotations = annotations.dropna(subset=['abstract', 'title']).drop_duplicates(
        subset=['abstract', 'title'])
    annotations['label_string'] = annotations["label"].apply(label_number_to_study_name_mapping.get)

    return annotations


def coronawhy_annotated_study_design(annotations_filepath, metadata_filepath):
    """
    match coronawhy's annotated dataset (colums id, label) with the article metadata
    file from https://docs.google.com/spreadsheets/d/1COg9MpzM4oQJ7EbbbeOIm12CewQ9TRHZihGV9cf4Teg/edit?usp=sharing
    :param annotations_filepath:
    :param metadata_filepath:
    :return: pd.DataFrame of annotations, with columns ["sha", "cord_uid",  "title", "abstract", "label", "label_str"]
    """

    # this is to handle the annoying 2 layer column headers
    annotations_col_name_mapping = {'Computational ': 'in_silico',
                                    'Non-human studies': 'in_vitro', 'Unnamed: 7': 'in_vivo',
                                    'Clinical studies': 'RCT_review', 'Unnamed: 9': 'RCT',
                                    'Unnamed: 10': 'controlled_trial_non_randomised',
                                    'Unnamed: 11': 'comparative_study',
                                    'Unnamed: 12': 'descriptive_study', 'Unnamed: 13': 'meta_study',
                                    'Unnamed: 14': 'others'}
    colname_to_number_mapping = {
        k: idx for idx, k in enumerate(['in_silico', 'in_vitro',
                           'in_vivo', 'RCT_review', 'RCT', 'controlled_trial_non_randomised',
                           'comparative_study', 'descriptive_study', 'meta_study'])
    }
    annotations = pd.read_csv(annotations_filepath)
    cord19_metadata = pd.read_csv(metadata_filepath)

    annotations = annotations.rename(columns=annotations_col_name_mapping)
    annotations = annotations.drop(0)

    annotations = annotations.merge(cord19_metadata, left_on="cord_uid",
                                    right_on="cord_uid", suffixes=("", "_duplicated"))[
        ["cord_uid",  "title", "abstract", 'in_silico', 'in_vitro',
       'in_vivo', 'RCT_review', 'RCT', 'controlled_trial_non_randomised',
       'comparative_study', 'descriptive_study', 'meta_study']]
    annotations = annotations.dropna(subset=['abstract', 'title']).drop_duplicates(
        subset=['abstract', 'title']).dropna(subset=['in_silico', 'in_vitro',
       'in_vivo', 'RCT_review', 'RCT', 'controlled_trial_non_randomised',
       'comparative_study', 'descriptive_study', 'meta_study'],  how='all')

    annotations[['in_silico', 'in_vitro', 'in_vivo', 'RCT_review', 'RCT',
                 'controlled_trial_non_randomised',
                 'comparative_study', 'descriptive_study', 'meta_study']] = \
        annotations[['in_silico', 'in_vitro','in_vivo', 'RCT_review', 'RCT',
                     'controlled_trial_non_randomised', 'comparative_study', 'descriptive_study',
                     'meta_study']].applymap(
        lambda x: 0 if type(x) == float and np.isnan(x) else 1)
    # select only those with unique lables
    class_instances = annotations[['in_silico', 'in_vitro',
                           'in_vivo', 'RCT_review', 'RCT', 'controlled_trial_non_randomised',
                           'comparative_study', 'descriptive_study', 'meta_study']].sum(axis=1)
    annotations = annotations.loc[class_instances[class_instances==1].index]
    #reverse one hot
    annotations['label_string'] = annotations[['in_silico', 'in_vitro',
                           'in_vivo', 'RCT_review', 'RCT', 'controlled_trial_non_randomised',
                           'comparative_study', 'descriptive_study', 'meta_study']].idxmax(1)

    annotations['label'] = annotations['label_string'].apply(colname_to_number_mapping.get)

    return annotations.reset_index(drop=True)[["cord_uid",  "title", "abstract", "label", "label_string"]]