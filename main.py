import utils
import logging

from src.data_loader import DataLoader
from src.exploratory_data_analysis import ExplaratoryDataAnalysis
from src.feature_engineering import FeatureEngineering

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    dataset = DataLoader().run_data_loader()
    ExplaratoryDataAnalysis(dataset).view_into_data(generate_html_report=False)
    dataset = FeatureEngineering(dataset).run_feature_engineering()
    ExplaratoryDataAnalysis(dataset).view_into_data(generate_html_report=False)
    #ModelTrainig.train()
    #ModelTrainig.evaluate()
    #ModelTrainig.optimize_hyperparameters()
    #Inferences.show_inferences()
    
    #PoC -Structure of the project 
    #Data Loader 
    # |
    # Exploratory Data Analysis before feature engineering
    # |
    # Feature Engineering
    # |
    # Exploratory Data Analysis after feature engineering (dont know how to create API - need consideration)
    # |
    # Model Training, Evaluation and Optimization
    # |
    # Inferences
 