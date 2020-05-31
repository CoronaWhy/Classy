"""
PLEASE IGNORE (WIP using snorkel)
"""
import pandas as pd
from snorkel.labeling import labeling_function

#keywords as per the kaggle cord19 dict
#https://docs.google.com/spreadsheets/d/1t2e3CHGxHJBiFgHeW0dfwtvCG4x0CDCzcTFX7yz9Z2E/edit#gid=1217643351

RETROSPECTIVE_OBSERVATIONAL_STUDIES = [
    "retrospective",
    "medical records review",
    "chart review",
    "case control"
]

PROSPECTIVE_OBSERVATIONAL_STUDIES = [
    "prospective",
    "followed up",
    "baseline characteristics"
]

SIMULATION = [
    "computer model",
    "forecast",
    "mathematical model",
    "statistical model",
    "stochastic model",
    "simulation",
    "synthetic data",
    "monte carlo",
    "bootstrap",
    "machine learning",
    "deep learning",
    "in silico"
]

CASE_SERIES = [
    "case study",
    "case series",
    "case report",
]

PSEUDO_RCT = [
    "quasi-randomized",
    "pseudo-randomized",
    "non-randomized",
    "quasi-randomised",
    "pseudo-randomised",
    "non-randomised",
    "allocation method"
]

TIME_SERIES = [
    "survival analysis",
    "time-to-event analysis",
    "Weibull",
    "gamma",
    "lognormal",
    "Kaplan-Meier",
    "hazard ratio",
    "Cox proportional hazards",
    "median time to event",
    "longitudinal"
]

label_number_to_study_name_mapping = {
    1: "Systematic review",
    2: "Randomized control trial",
    3: "Non-randomized trial",
    4: "Prospective observational",
    5: "Time-to-event analysis",
    6: "Retrospective observational",
    7: "Cross-sectional",
    8: "Case series",
    9: "Modeling",
    0: "Other"
}
study_name_to_label_mapping = {v:k for k, v in label_number_to_study_name_mapping.items()}
ABSTAIN = -1


@labeling_function()
def simulation_keyword(x):
    return study_name_to_label_mapping["Modelling"] if x.text.lower() in SIMULATION else ABSTAIN