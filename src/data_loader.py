"""
|--------------------------------------------------------------|
|                          Requirements                        |
|--------------------------------------------------------------|
"""

import pandas as pd
import kagglehub
import yaml
import logging 

"""
|--------------------------------------------------------------|
|                          Maros                               |
|--------------------------------------------------------------|
"""

CONFIG_FILE = "config.yaml"
REPORT_PATH = "report.html"

logger = logging.getLogger(__name__)

class DataLoader():
    def __init__(self):
        self.config = CONFIG_FILE
        self.dataset_kaggle_id = None
        self.dataset_csv_name = None
        self.dataset = None

    def _load_config_file(self):
        try:
            with open(self.config, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(("Wczytano dane konfiguracyjne"))
                self.dataset_kaggle_id = config['kaggle_data']['dataset_id']
                self.dataset_csv_name = config['kaggle_data']['csv_filename']
        except FileNotFoundError:
            logger.info(f"Błąd: Plik nie został znaleziony pod ścieżką\n{self.path}")
            raise
        except Exception as e:
            logger.info(f"Wystąpił nieoczekiwany błąd podczas ładowania danych\n{e}")
            raise

    def _load_data(self) -> pd.DataFrame:
        try:
            path = kagglehub.dataset_download(self.dataset_kaggle_id)
            self.dataset = pd.read_csv(f"{path}/{self.dataset_csv_name}", encoding='cp1250', sep=",", low_memory=False)
            logger.info(f"Pomyślnie wczytano dataset")

        except FileNotFoundError:
            logger.error(f"Błąd: Plik nie został znaleziony")
            raise
        except Exception as e:
            logger.exception(f"Wystąpił błąd podczas ładowania datasetu:\n {e}")
            raise

    def _get_dataset_info(self) -> dict:
            if self.dataset is None:
                return {"status": "Dataset nie wczytany"}
                
            info = {
                "Rows": len(self.dataset),
                "Columns": len(self.dataset.columns),
                "Dtypes": self.dataset.dtypes.value_counts().to_dict()
            }
            return info

    def run_data_loader(self):
        self._load_config_file()
        self._load_data()

        return self.dataset
