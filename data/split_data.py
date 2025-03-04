import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_path = './LG_Aimers_6th/data/train.csv'
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

data = pd.read_csv(train_path)
for seed in seed_list:
    train, test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['임신 성공 여부'])
    train.to_csv(f'./LG_Aimers_6th/data/custom_train_{seed}.csv', index=False)
    test.drop(columns=['임신 성공 여부']).to_csv(f'./LG_Aimers_6th/data/custom_test_{seed}.csv', index=False)