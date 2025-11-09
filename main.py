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

    df, _ = utils.load_dataset()
    utils.first_view_into_data(df)

main()
