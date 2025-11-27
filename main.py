#|--------------------------------------------------------------|
#|                          Requirements                        |
#|--------------------------------------------------------------|

import logging
from src.common import check_if_path_exists
from src.data_loader import DataLoader
from src.exploratory_data_analysis import ExplaratoryDataAnalysis
from src.feature_engineering import FeatureEngineering
from src.model_training import DataPreparation, XGBoostModelTraining
from src.model_training import XGBoostModelEvaluation, XGBoostModelOptimization, LogisticRegressionBenchmark

#|--------------------------------------------------------------|
#|                          Macors                              |
#|--------------------------------------------------------------|

LOG_DIR = "logs/"

#|--------------------------------------------------------------|
#|                          Logger                              |
#|--------------------------------------------------------------|

logger = logging.getLogger(__name__)
check_if_path_exists(folder_path=LOG_DIR)
file_handler = logging.FileHandler(f'{LOG_DIR}/pipeline.log', mode = 'w')
file_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(),
                file_handler])


#|--------------------------------------------------------------|
#|                          Main part                           |
#|--------------------------------------------------------------|

class Pipeline:
    def __init__(self):
        pass

    def run(self):
        logger.info("Pipeline Initialized")
        
        ## STEP 1: DATA LOADING
        logger.info("STEP 1: DATA LOADING")
        dataset = DataLoader().run_data_loader()
        
        ## STEP 2: EXPLORATORY DATA ANALYSIS (BEFORE FEATURE ENGINEERING)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS (Before Feature Engineering)")
        eda = ExplaratoryDataAnalysis(dataset)
        eda.view_into_data()  ## Podstawowe statystyki w konsoli
        eda.get_plots(folder_name="BeforeFeatureEngineering")  ## Wykresy do folderu
        eda.create_report(generate_html_report=False, title="Dataset report before Feature Engineering")  ## Pełny raport HTML
        
        ## STEP 3: FEATURE ENGINEERING
        logger.info("STEP 3: FEATURE ENGINEERING")
        fe = FeatureEngineering(dataset)
        dataset = fe.run_feature_engineering()

        ## STEP 4: EXPLORATORY DATA ANALYSIS (AFTER FEATURE ENGINEERING)
        logger.info("STEP 4: EXPLORATORY DATA ANALYSIS (After Feature Engineering)")
        eda = ExplaratoryDataAnalysis(dataset)
        eda.create_report(generate_html_report=False, title="Dataset report after Feature Engineering")
        
        ## STEP 5: DATA PREPARATION
        logger.info("STEP 5: DATA PREPARATION")
        dp = DataPreparation(dataset)
        prepared_data = dp.run_data_preparing()
        
        ## STEP 6: BENCHMARK MODEL - LOGISTIC REGRESSION
        logger.info("STEP 6: BENCHMARK MODEL (Logistic Regression)")
        benchmark_model = LogisticRegressionBenchmark(prepared_data)
        benchmark_model.train()
        benchmark_model_metrics = benchmark_model.evaluate()
        benchmark_model.save_model()
        
        ## STEP 7: MODEL TRAINING (XGBoost)
        logger.info("STEP 7: MODEL TRAINING (XGBoost)")
        trainer = XGBoostModelTraining(prepared_data)
        model = trainer.train_xgboost()
        trainer.save_model("xgboost_baseline.pkl")
        
        ## STEP 8: MODE EVALUATION (XGBoost)
        logger.info("STEP 8: MODEL EVALUATION (XGBoost)")
        evaluator = XGBoostModelEvaluation(model, prepared_data)
        metrics = evaluator.run_full_evaluation()
        
        ## STEP 9: CROSS-VALIDATION EVALUATION
        logger.info("STEP 9: CROSS-VALIDATION EVALUATION")
        cv_results = evaluator.evaluate_with_cross_validation(cv=5)
        
        ## STEP 10: MODEL INTERPRETATION (SHAP)
        logger.info("STEP 10: MODEL INTERPRETATION (C)")
        evaluator.interpret_model_shap()
        
        ## STEP 11: HYPERPARAMETER OPTIMIZATION
        logger.info("STEP 11: HYPERPARAMETER OPTIMIZATION")
        optimizer = XGBoostModelOptimization(prepared_data)
        best_params = optimizer.optimize_hyperparameters()
        optimizer.save_optimization_results()
        
        ## STEP 12: TRAINING OPTIMIZED MODEL
        logger.info("STEP 12: TRAINING OPTIMIZED MODEL")
        optimized_trainer = optimizer.train_optimized_model(prepared_data)
        optimized_model = optimized_trainer.get_model()
        optimized_trainer.save_model("xgboost_optimized.pkl")
        
        ## STEP 13: EVALUATING OPTIMIZED MODEL
        logger.info("STEP 13: EVALUATING OPTIMIZED MODEL")
        optimized_evaluator = XGBoostModelEvaluation(optimized_model, prepared_data)
        optimized_metrics = optimized_evaluator._evaluate_model()
        optimized_evaluator.save_metrics("xgboost_optimized_metrics.json")
        
        ## STEP 14: SUMMARY
        logger.info("WHOLE PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("Benchmark Model (Logistic Regression):")
        logger.info(f"  - ROC-AUC:  {benchmark_model_metrics['roc_auc']:.4f}")
        logger.info("XGBoost Baseline Model:")
        logger.info(f"  - ROC-AUC:  {metrics['roc_auc']:.4f}")
        logger.info(f"  - CV Mean ROC-AUC: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        logger.info("XGBoost Optimized Model:")
        logger.info(f"  - ROC-AUC:  {optimized_metrics['roc_auc']:.4f}")
        logger.info(f"Best Hyperparameters: {best_params}")

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
