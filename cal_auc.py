import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT_DIR = "./LG_Aimers_6th/"
Data_DIR = "./data/"

def calculate_auc(y_pred, seed=1):
    train_data = pd.read_csv(os.path.join(ROOT_DIR, Data_DIR, "train.csv"))

    train, test = train_test_split(train_data, test_size=0.2, random_state=seed, stratify=train_data['임신 성공 여부'])
    y_test = test['임신 성공 여부'].copy()
    auc = roc_auc_score(y_test, y_pred)
    return auc



