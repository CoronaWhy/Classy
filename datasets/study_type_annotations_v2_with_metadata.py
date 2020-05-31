from pymongo import MongoClient
import pandas as pd
import os

from pathlib import Path

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Read-only credentials to CoronaWhy MongoDB service
mongouser = 'coronawhyguest'
mongopass = 'coro901na'
cordversion = 'v22'

def study_type_annotations_v2():
    df = pd.read_csv(Path(__location__) /"StudyTypeAnnotations2.csv")
    cord_uids = df.paper_id.to_list()
    client = MongoClient("mongodb://%s:%s@mongodb.coronawhy.org" % (mongouser, mongopass))
    db = client.get_database('cord19')
    collection = db[cordversion]
    metadata_df = pd.DataFrame(collection.find({"cord_uid": {"$in":cord_uids}}, {"cord_uid": 1, "title": 1, "abstract": 1}))
    metadata_df.abstract = metadata_df.abstract.apply(lambda x: " ".join(obj['text'] for obj in x ))
    df = df.merge(metadata_df, left_on="paper_id", right_on="cord_uid").drop(columns=["paper_id"]).set_index("cord_uid")
    return df