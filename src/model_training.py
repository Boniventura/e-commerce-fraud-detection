from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class DataPreparation():
    def __init__(self, dataset):
        self.dataset = dataset 
        self.X = self.dataset.drop(columns=['is_fraud'])
        self.y = self.dataset['is_fraud']


    def _train_test_split(self, test_size=0.2, random_state=42):
    
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=y)
        
        train_test_dict = {
            "X_train" :X_train,
            "X_test" : X_test,
            "y_train" : y_train,
            "y_test" : y_test
        }

        return train_test_dict
    
    def _normalization(self, train_test_dict):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_test_dict["X_train"])
        X_test_scaled = scaler.transform(train_test_dict["X_test"])
        
        train_test_scaled_dict = {
            "X_train_scaled" : X_train_scaled,
            "X_test_scaled"  : X_test_scaled}

        return train_test_scaled_dict

    def _perform_pca(sel, train_test_scaled_dict, n_components = 2 ):
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(train_test_scaled_dict["X_train_scaled"])

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y, cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Visualization of Iris Dataset')
        plt.show()
        return X_pca
    
    def _normalize_dataset(self):
        pass

    def _train_test_split(self):
        pass

    def run_data_preparing(self):
        train_test_dict = self._train_test_split()
        train_test_scaled_dict = self._normalization(train_test_dict)
        self._perform_pca(train_test_scaled_dict)

class ModelTraining():
    def __init__(self):
        pass

class ModelEvaluation():
    def __init__(self):
        pass

class ModelOptimization():
    def __init__(self):
        pass
