from ydata_profiling import ProfileReport
from pathlib import Path
from pprint import pprint

REPORT_PATH = "report.html"

class ExplaratoryDataAnalysis():
    def __init__(self, dataset):
    
        self.dataset = dataset

    def _get_null_columns(self):
        return self.dataset.isnull().sum()

    def _get_columns_with_nan(self):
        missing_data = self.dataset.isna().sum()
        return missing_data[missing_data > 0]

    def _create_html_raport(self, title="Raport EDA"):
        profile = ProfileReport(self.dataset, title=title,
                        explorative=True, correlations = {
                        "pearson": {"calculate": True},
                        "spearman": {"calculate": True},
                        "kendall": {"calculate": True}})

        profile.to_file(REPORT_PATH)
        print(f"HTML raport saved to {REPORT_PATH}")

    def _show_unique_values(self):
        for column in self.dataset.columns:
            unique_values = self.dataset[column].unique()
            print(f"Column '{column}' has {len(unique_values)} unique values: {unique_values}")

    def view_into_data(self, generate_html_report=False):
        print("\nFirst 5 rows of dataset:")
        pprint(self.dataset.head())
        
        print("\nDataset info:")
        pprint(self.dataset.info())

        print("\nDataset description for all columns:")
        pprint(self.dataset.describe(include='all'))

        print("\nDataset description for all numeric columns:")
        pprint(self.dataset.describe())

        print("\nNull values in each column:")
        pprint(self._get_null_columns())

        print("\nColumns with NaN values:")
        pprint(self._get_columns_with_nan())

        print("\nUnique values in each column:")
        self._show_unique_values()
        
        if generate_html_report:
            print("\n Generate HTML raport...")
            print("temporary disabled")
            # self._create_html_raport(title="Dataset before Feature Engineering" )
