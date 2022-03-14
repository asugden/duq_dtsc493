import catboost
import numpy as np
import pandas as pd
from typing import Union
import xgboost as xgb


class ParameterizedXGBoost():
    def __init__(self):
        self.continuous = False
        self.categorical = False
        self.model = None

    def train(self,
              features: Union[np.ndarray, pd.DataFrame],
              labels: Union[np.ndarray, pd.Series],
              categorical: bool = False):
        """Train an XGBRegressor or XGBClassifier

        Args:
            features (Union[np.ndarray, pd.DataFrame]): a feature set
            labels (Union[np.ndarray, pd.Series]): a list of labels of those featuers
            categorical (bool): whether the labels are categorical rather than continuous

        """
        if len(labels.unique()) > 2:
            if categorical:
                self.categorical = True
            else:
                self.continuous = True

        if self.continuous:
            self.model = xgb.XGBRegressor()
        elif self.categorical:
            self.model = catboost.CatBoostRegressor()
        else:
            self.model = xgb.XGBRFClassifier()

        self.model.fit(features, labels)

    def apply(self,
              features: Union[np.ndarray, pd.DataFrame],
              binarize: bool = False) -> np.ndarray:
        """Given a set of features, compute the best fit labels

        Args:
            features (Union[np.ndarray, pd.DataFrame]): features to be computed
            binarize (bool): binarize results if not continuous or categorical

        Returns:
            np.ndarray: the labels associated with each row

        """
        out = self.model.predict(features)
        if binarize and not self.continuous and not self.categorical:
            return out > 0.5
        else:
            return out
