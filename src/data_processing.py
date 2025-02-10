import os
import pandas as pd

def load_raw_data(file_path):
    """ 원본 데이터를 불러옵니다. """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """ 데이터 정제 및 필요한 전처리 작업을 수행합니다. """
    df.dropna(inplace=True)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def save_processed_data(df, output_path):
    """ 전처리된 데이터를 저장합니다. """
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw_data_path = "data/raw/financial_data.csv"
    processed_data_path = "data/processed/financial_data_cleaned.csv"
    
    df = load_raw_data(raw_data_path)
    df = preprocess_data(df)
    save_processed_data(df, processed_data_path)
