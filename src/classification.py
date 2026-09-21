#training the labels mutually decided by k_means and dbscan here
#using random_forest and xgboost here 
#since clustering doesnt scale well for real time detection so 
# we use classification to train a model that can label new traffic instantly

import os 
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from xgboost import XGBClassifer

PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"
RANDOM_STATE = 42