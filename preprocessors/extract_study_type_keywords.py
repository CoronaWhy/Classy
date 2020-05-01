import pandas as pd

#keywords as per the kaggle cord19 dict
#https://docs.google.com/spreadsheets/d/1t2e3CHGxHJBiFgHeW0dfwtvCG4x0CDCzcTFX7yz9Z2E/edit#gid=1217643351

RETROSPECTIVE_OBSERVATIONAL_STUDIES = [
    "retrospective",
    "medical records review",
    "chart review",
    "case control"
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
    "deep learning"
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