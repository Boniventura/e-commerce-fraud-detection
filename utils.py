"""
|--------------------------------------------------------------|
|                          Requirements                        |
|--------------------------------------------------------------|
"""
import pandas as pd
from pprint import pprint
import kagglehub
import yaml
import os
from ydata_profiling import ProfileReport

"""
|--------------------------------------------------------------|
|                          Maros                               |
|--------------------------------------------------------------|
"""

CONFIG_FILE = "config.yaml"
REPORT_PATH = "report.html"

"""
|--------------------------------------------------------------|
|                      Utils Functions                         |
|--------------------------------------------------------------|
"""

def _load_config_file():
  with open(CONFIG_FILE, 'r') as file:
    config = yaml.safe_load(file)
    dataset_kaggle_id = config['kaggle_data']['dataset_id']
    dataset_csv_name = config['kaggle_data']['csv_filename']

  return dataset_kaggle_id, dataset_csv_name

def load_dataset():
  dataset_kaggle_id, dataset_csv_name = _load_config_file()
  try:  
    path = kagglehub.dataset_download(dataset_kaggle_id)
    df = pd.read_csv(f"{path}/{dataset_csv_name}", encoding='cp1250', sep=",", low_memory=False)
    
  except Exception as e:
    print("Load data was not possible due to error.\n", e)

  return df, path

def _get_null_columns(df):
  return df.isnull().sum()

def _get_columns_with_nan(df):
  missing_data = df.isna().sum()
  return missing_data[missing_data > 0]

def _create_html_raport(df):
  profile = ProfileReport(df, title="Raport EDA", explorative=True)
  profile.to_file(REPORT_PATH)
  print(f"HTML raport saved to {REPORT_PATH}")

def first_view_into_data(df):
  
  print("\nFirst 5 rows of dataset:")
  pprint(df.head())
  
  print("\nDataset info:")
  pprint(df.info())

  print("\nDataset description for all columns:")
  pprint(df.describe(include='all'))

  print("\nDataset description for all numeric columns:")
  pprint(df.describe())

  print("\nNull values in each column:")
  pprint(_get_null_columns(df))

  print("\nColumns with NaN values:")
  pprint(_get_columns_with_nan(df))

  print("\n Generate HTML raport...")
  _create_html_raport(df)

