import pandas as pd
from pprint import pprint
import kagglehub
import yaml
import os
from ydata_profiling import ProfileReport

import utils 

CONFIG_FILE = "config.yaml"

def main():
	if __name__ == "__main__":

		df, path = utils.load_dataset()
		# utils.first_view_into_data(df)


		utils.feature_engineering(df)

		utils.first_view_into_data(df)
		utils.clean_dataset_path(path) 
main()
