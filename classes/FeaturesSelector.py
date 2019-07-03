import logging
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2

from classes.Dataset import Dataset


class FeaturesSelector:

    def __init__(self, dataset: Dataset):
        self.__features = dataset.get_features()
        self.__labels = dataset.get_labels()
        self.__best_features_ids = []

    def get_features(self) -> pd.DataFrame:
        return self.__features

    def get_best_features_ids(self) -> List[str]:
        return self.__best_features_ids

    def univariate_selection(self, num_features: int):
        """
        Apply sklearn.SelectKBest class to extract the top num_features best features

        :param num_features: the number of top features to be extracted
        """

        logging.info("Performing univariate selection...")

        # Perform univariate selection
        best_features = SelectKBest(score_func=chi2, k=num_features)
        fit = best_features.fit(self.__features, self.__labels)

        scores = pd.DataFrame(fit.scores_)
        columns = pd.DataFrame(self.__features.columns)

        # Concat two dataframes for better visualization
        feature_scores = pd.concat([columns, scores], axis=1)

        # Name the dataframe columns
        feature_scores.columns = ['Specs', 'Score']

        # Log the 10 best features
        logging.info("The {} best features are:".format(num_features))
        logging.info(feature_scores.nlargest(10, 'Score'))

        for feature in self.__features:
            if not feature_scores.nlargest(num_features, 'Score')['Specs'].str.contains(feature).any():
                self.__features.drop(feature, axis=1, inplace=True)
            else:
                logging.warning('Added FEATURE {}'.format(feature))
                self.__best_features_ids.append(feature)

    def features_importance(self, num_features: int, show: bool = False):
        logging.info("Calculating features importances...")

        model = ExtraTreesClassifier()
        model.fit(self.__features, self.__labels)

        # Use inbuilt class feature_importances of tree based classifiers
        feature_importances = model.feature_importances_

        logging.info("Features importances:")
        logging.info(feature_importances)

        # Plot graph of feature importances for better visualization
        importances = pd.Series(feature_importances, index=self.__features.columns)
        importances.nlargest(num_features).plot(kind='barh')

        if show:
            plt.show()
