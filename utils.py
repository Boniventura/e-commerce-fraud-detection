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
from pathlib import Path

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
	profile = ProfileReport(df, title="Raport EDA", 
					explorative=True, correlations = {
					"pearson": {"calculate": True},
					"spearman": {"calculate": True},
					"kendall": {"calculate": True}})

	profile.to_file(REPORT_PATH)
	print(f"HTML raport saved to {REPORT_PATH}")


def _show_unique_values(df):
	for column in df.columns:
		unique_values = df[column].unique()
		print(f"Column '{column}' has {len(unique_values)} unique values: {unique_values}")

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

	print("\nUnique values in each column:")
	_show_unique_values(df)

	print("\n Generate HTML raport...")
	# _create_html_raport(df)

def clean_dataset_path(path):
	pass
	#tbd
	# path_to_delete = Path(path)
	# try:
	# 	path_to_delete.unlink()
	# 	return True, f"File {path} has been deleted."
	# except Exception as e:  
	# 	return False, str(e)

def _set_to_datetime(df):
	df['transaction_datetime'] = pd.to_datetime(df['transaction_time'])
	return df

def _check_if_night_transaction(df, column_name='transaction_datetime'):
	night_hours = (22, 23, 0, 1, 2, 3, 4, 5)
	df['is_night_transaction'] = df[column_name].dt.hour.isin(night_hours).astype(int)
	return df

def _check_if_weekend_transaction(df, column_name='transaction_datetime'):
	weekend_days = (5, 6)
	df['is_weekend_transaction'] = df[column_name].dt.dayofweek.isin(weekend_days).astype(int)
	return df

def _check_if_christmast_transaction(df, column_name='transaction_datetime'):
	christmas_start = pd.Timestamp(year=df[column_name].dt.year.min(), month=12, day=1)
	christmas_end = pd.Timestamp(year=df[column_name].dt.year.max(), month=12, day=26)
	df['is_christmas_transaction'] = df[column_name].apply(lambda x: christmas_start <= x <= christmas_end)
	return df

def _one_hot_encode_column(df, column_name):
	dummies = pd.get_dummies(df[column_name], prefix=column_name)

	pprint(df.head())
	df = pd.concat([df, dummies], axis=1)
	pprint(df.head())
	df.drop(column_name, axis=1, inplace=True)
	# print("1-hot encoding applied to column:")
	return df

def _datetime_featuring(df):
	df = _set_to_datetime(df)
	df = _check_if_night_transaction(df)
	df = _check_if_weekend_transaction(df)
	# df = _check_if_christmast_transaction(df)

	#delete original datetime column becase it is no longer used
	df.drop('transaction_time', axis=1, inplace=True)

	return df 


def _check_if_country_mismatch(df):
	df['is_country_mismatch'] = (df['country'] == df['bin_country']).astype(int)
	return df

def feature_engineering(df):
	df = _datetime_featuring(df)
	df = _check_if_country_mismatch(df)
	df = _one_hot_encode_column(df, 'merchant_category')
	df = _one_hot_encode_column(df, 'channel')
	return df

#Trzeba zrobic optiune i shap

