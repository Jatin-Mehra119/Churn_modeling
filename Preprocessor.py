import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator

def safe_log_transform(x):
    return np.log(x + 1e-9)  # Adding a small constant to avoid log(0)

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        # lists of features for each transformation type
        self.num_list = ['Age', 'Tenure', 'NumOfProducts']
        self.log_list = ['CreditScore', 'Balance', 'EstimatedSalary']
        self.cat_list = ['Geography', 'Gender']

        # pipeline for numerical features
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler())
        ])

        # pipeline for logarithmic transformation features
        self.log_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('log_transformer', FunctionTransformer(safe_log_transform, validate=True, feature_names_out="one-to-one")),
            ('scaler', StandardScaler())
        ])

        # pipeline for categorical features
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder())
        ])

        # Combine all the pipelines into a ColumnTransformer
        self.preprocessor = ColumnTransformer([
            ('num', self.num_pipeline, self.num_list),
            ('log', self.log_pipeline, self.log_list),
            ('cat', self.cat_pipeline, self.cat_list)
        ], remainder='drop')

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.preprocessor.get_feature_names_out(input_features)
