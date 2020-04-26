import pandas as pd


def single_label_multiclass_annotated_study_design(annotations_filepath, metadata_filepath):
    """
    match david's annotated dataset (colums id, label) with the article metadata
    file from kaggle cord-19 https://www.kaggle.com/davidmezzetti/cord19-study-design
    :param annotations_filepath:
    :param metadata_filepath:
    :return: pd.DataFrame of annotations, with columns ["sha", "cord_uid",  "title", "abstract", "label", "label_str"]
    """

    label_number_to_study_name_mapping = {
        1: "Meta analysis",
        2: "Randomized control trial",
        3: "Non-randomized trial",
        4: "Prospective cohort",
        5: "Time-series analysis",
        6: "Retrospective cohort",
        7: "Cross-sectional",
        8: "Case control",
        9: "Case study",
        10: "Simulation",
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