import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def drop_columns(df):
    cols = [
        '불임 원인 - 여성 요인', # 고유값 1
        '불임 원인 - 정자 면역학적 요인' # '1'인 데이터가 train, test에서 모두 1개 >> 신뢰할 수 없음
    ]
    df = df.drop(cols, axis=1)
    return df

def 특정시술유형(train, test):
    # 범주화 함수 정의
    def categorize_procedure(proc):
        tokens = [token.strip() for token in proc.split(",") if token.strip() and not token.strip().isdigit()]

        # 우선순위에 따른 범주화
        if tokens.count("Unknown") >= 1:
            return "Unknown"
        if tokens.count("AH") >= 1:
            return "AH"
        if tokens.count("BLASTOCYST") >= 1:
            return "BLASTOCYST"
        if tokens.count("ICSI") >= 2 or tokens.count("IVF") >= 2:
            return "2ICSI_2IVF"
        if tokens.count("IVF") >= 1 and tokens.count("ICSI") >= 1:
            return "IVF_ICSI"
        if tokens == "ICSI":
            return "ICSI"
        if tokens == "IVF":
            return "IVF"
        return ",".join(tokens) if tokens else None

    for df in [train, test]:
        df['특정 시술 유형'] = df['특정 시술 유형'].str.replace(" / ", ",")
        df['특정 시술 유형'] = df['특정 시술 유형'].str.replace(":", ",")
        df['특정 시술 유형'] = df['특정 시술 유형'].str.replace(" ", "")

    counts = train['특정 시술 유형'].value_counts()
    allowed_categories = counts[counts >= 100].index.tolist()

    # allowed_categories에 속하지 않는 값은 "Unknown"으로 대체
    train.loc[~train['특정 시술 유형'].isin(allowed_categories), '특정 시술 유형'] = "Unknown"
    test.loc[~test['특정 시술 유형'].isin(allowed_categories), '특정 시술 유형'] = "Unknown"

    train['특정 시술 유형'] = train['특정 시술 유형'].apply(categorize_procedure)
    test['특정 시술 유형'] = test['특정 시술 유형'].apply(categorize_procedure)

    train['시술유형_통합'] = train['시술 유형'].astype(str) + '_' + train['특정 시술 유형'].astype(str)
    test['시술유형_통합'] = test['시술 유형'].astype(str) + '_' + test['특정 시술 유형'].astype(str)

    drop_cols = ['시술 유형', '특정 시술 유형']
    train = train.drop(drop_cols, axis=1)
    test = test.drop(drop_cols, axis=1)

    return train, test

def 해동된배아수(train, test, cap_value=7):
    train['해동된 배아 수'] = train['해동된 배아 수'].clip(upper=cap_value)
    test['해동된 배아 수'] = test['해동된 배아 수'].clip(upper=cap_value)
    return train, test

def 시술횟수(df_train):
    for col in [col for col in df_train.columns if '횟수' in col]:
        df_train[col] = df_train[col].replace({'6회 이상':'6회'})
        df_train[col] = df_train[col].str[0].astype(int)

    df_train['시술_임신'] = df_train['총 임신 횟수'] - df_train['총 시술 횟수']
    df_train = df_train.drop('총 시술 횟수', axis=1)
    return df_train

def encoding(train, test, seed=42):
    categorical_columns = [
        "시술 시기 코드",
        "시술 당시 나이",
        "배란 유도 유형",
        "배아 생성 주요 이유",

        ## 시술 횟수
        "클리닉 내 총 시술 횟수",
        "IVF 시술 횟수",
        "DI 시술 횟수",

        ## 임신 횟수
        "총 임신 횟수",
        "IVF 임신 횟수",
        "DI 임신 횟수",

        ## 출산 횟수
        "총 출산 횟수",
        "IVF 출산 횟수",
        "DI 출산 횟수",

        "난자 출처",
        "정자 출처",
        "난자 기증자 나이",
        "정자 기증자 나이",

        '시술유형_통합', # 특정시술유형()
    ]
    train[categorical_columns] = train[categorical_columns].astype(str)
    test[categorical_columns] = test[categorical_columns].astype(str)

    train[categorical_columns] = train[categorical_columns].astype('category')
    test[categorical_columns] = test[categorical_columns].astype('category')

    return train, test

def cb_all_process(train, val):
    train, val = drop_columns(train), drop_columns(val)

    train, val = 특정시술유형(train, val)

    train, val = 시술횟수(train), 시술횟수(val)

    train, val = encoding(train, val)

    return train, val
