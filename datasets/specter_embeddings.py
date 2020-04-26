import pandas as pd


def annotations_with_specter_embeddings(annotations_dataset, specter_embeddings_filepath):
    enriched_dataset = annotations_dataset.copy()
    specter_embeddings = pd.read_csv(
        specter_embeddings_filepath, header=None,
        names=['cord_uid'] + [f"{i}" for i in range(768)])
    enriched_dataset = enriched_dataset.merge(specter_embeddings, left_on="cord_uid", right_on="cord_uid")

    return enriched_dataset