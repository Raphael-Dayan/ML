import pandas as pd
import numpy as np


class CorrelatedFeatures:
    """Removes correlated features from a dataset using Pearson correlation"""

    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.correlations = None
        self.highly_correlated_features = None

    def fit(self, X):
        """calculates correlated features"""
        pd_corr = df.corr()
        pd_corr = pd_corr.where(np.triu(np.ones(pd_corr.shape), k=1).astype(np.bool))
        # passing to list and sorting values
        correlation_list = pd_corr.unstack().sort_values(ascending=False)
        correlation_list = (
            correlation_list.reset_index()
            .rename(
                columns={
                    "level_0": "feature_1",
                    "level_1": "feature_2",
                    0: "correlation",
                }
            )
            .dropna()
        )

        self.correlations = correlation_list

    def transform(self, X):
        """returns a dataframe without correlated features"""

        self.highly_correlated_features = self.correlations[
            (self.correlations["correlation"] > self.threshold)
            | (self.correlations["correlation"] < -self.threshold)
        ]
        return X.drop(columns=self.highly_correlated_features["feature_1"].unique())


class OutlierReplacerIQR:
    """Replaces outliers by their upper and lower bound as calculated using the interquartile range rule."""

    def __init__(
        self,
        lower_quantile=0.25,
        upper_quantile=0.75,
        multiplier=1.5,
        ignore_zero=False,
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.multiplier = multiplier
        self.ignore_zero = ignore_zero
        self.upper_limit = None
        self.lower_limit = None

    def fit(self, X):
        # calculates upper and lower boundary

        if not self.ignore_zero:
            q1 = X.quantile(self.lower_quantile)
            q3 = X.quantile(self.upper_quantile)

        elif self.ignore_zero:
            q1 = X[X != 0].quantile(self.lower_quantile)
            q3 = X[X != 0].quantile(self.upper_quantile)

        IQR = q3 - q1
        upper_limit = q3 + self.multiplier * IQR
        lower_limit = q1 - self.multiplier * IQR

        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def transform(self, X):
        for column in self.upper_limit.index:
            X[column] = np.where(
                X[column] > self.upper_limit[column],
                self.upper_limit[column],
                X[column],
            )
        for column in self.lower_limit.index:
            X[column] = np.where(
                X[column] < self.lower_limit[column],
                self.lower_limit[column],
                X[column],
            )

        return X
