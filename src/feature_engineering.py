import pandas as pd
from pprint import pprint

class FeatureEngineering():
    def __init__(self, dataset):
        self.dataset = dataset

    def _set_to_datetime(self):
        self.dataset['transaction_datetime'] = pd.to_datetime(self.dataset['transaction_time'])

    def _check_if_night_transaction(self, column_name='transaction_datetime'):
        nightHours = (22, 23, 0, 1, 2, 3, 4, 5)
        self.dataset['is_night_transaction'] = self.dataset[column_name].dt.hour.isin(nightHours).astype(int)

    def _check_if_weekend_transaction(self, column_name='transaction_datetime'):
        weekendDays = (5, 6)
        self.dataset['is_weekend_transaction'] = self.dataset[column_name].dt.dayofweek.isin(weekendDays).astype(int)

    def _check_if_christmast_transaction(self, column_name='transaction_datetime'):
        isDecember = self.dataset[column_name].dt.month.isin([12]).astype(int)
        isNearChristmas = self.dataset[column_name].dt.day.isin(range(15,27)).astype(int)

        self.dataset['is_christmas_transaction'] = (isDecember & isNearChristmas).astype(int)

    def _one_hot_encode_column(self, column_name):
        dummies = pd.get_dummies(self.dataset[column_name], prefix=column_name)

        pprint(self.dataset.head())
        self.dataset = pd.concat([self.dataset, dummies], axis=1)
        pprint(self.dataset.head())
        self.dataset.drop(column_name, axis=1, inplace=True)

    def _datetime_featuring(self):
        self._set_to_datetime()
        self._check_if_night_transaction()
        self._check_if_weekend_transaction()
        self._check_if_christmast_transaction()

        self.dataset.drop('transaction_time', axis=1, inplace=True)

    def _check_if_country_mismatch(self):
        self.dataset['is_country_mismatch'] = (self.dataset['country'] != self.dataset['bin_country']).astype(int)

    def _drop_unused_columns(self, list_of_columns):
        self.dataset.drop(list_of_columns, axis=1, inplace=True, errors='ignore')

    def run_feature_engineering(self):
        # Main function to run all feature engineering steps and return modified dataset
        self._datetime_featuring()
        self._check_if_country_mismatch()
        self._one_hot_encode_column('merchant_category')
        self._one_hot_encode_column('channel')
        #add columns to drop if needed
        self._drop_unused_columns([])

        return self.dataset