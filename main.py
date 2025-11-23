import logging

from src.data_loader import DataLoader
from src.exploratory_data_analysis import ExplaratoryDataAnalysis
from src.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format    ='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
        logging.FileHandler('logs/pipeline.log') 
    ]
)

class Pipeline():
    def __init__(self):
        pass

    def run(self):
        logger.info("Pipeline Initalized")
        logger.info("Started data loading process...")
        dataset = DataLoader().run_data_loader()
        
        logger.info("Started exploratory data analysis...")
        eda = ExplaratoryDataAnalysis(dataset)
        eda.view_into_data()
        eda.create_report(generate_html_report=True, title ="Dataset report before Feature Engineering")
        logger.info("Started feature engineering process...")
        
        fe = FeatureEngineering(dataset)
        fe.run_feature_engineering()

        logger.info("Started exploratory data analysis after feature enginerring...")
        eda = ExplaratoryDataAnalysis(dataset)
        eda.get_plots(folder_name="AfterFeatureEngineering")
        eda.create_report(generate_html_report=True, title ="Dataset report after Feature Engineering")

        fe = FeatureEngineering(dataset)
        

        #ModelTrainig.train()
        #ModelTrainig.evaluate()
        #ModelTrainig.optimize_hyperparameters()
        #Inferences.show_inferences()
        

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()

    
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
 