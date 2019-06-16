from typing import List

import pandas as pd


class Preprocessor:

    def __init__(self, features: pd.DataFrame):
        """
        Build a preprocessing utility and performs the filling of the missing values.

        :param features: the whole set of features of the dataset
        """

        self.__features = features
        self.__fill_missing_values()

    def get_features(self) -> pd.DataFrame:
        return self.__features

    def __fill_missing_values(self):
        """ Fill missing values with the median value of the feature. """

        for feature in self.__features:
            median = self.__features[feature].median()
            self.__features[feature].fillna(median, inplace=True)

    def apply_features_scaling(self, to_be_preprocessed: List[str]):
        """
        Scale each feature within a [0, 1] range.

        :param to_be_preprocessed: the list of features (names) to be scaled
        """

        # For each axis take the maximum/minimum value and scale each feature value in [0, 1]
        for feature in self.__features:
            if feature in to_be_preprocessed:
                new_entries = []

                min_val = min(self.__features[feature])
                max_val = max(self.__features[feature])

                assert min_val != max_val, "The feature has the same value for all the rows of dataset!"

                for feature_val in self.__features[feature]:
                    new_entries.append(self.__map_value(feature_val, min_val, max_val, 0., 1.))

                self.__features[feature] = new_entries

    def apply_encoder(self, to_be_processed: List[str], thresholds: List[float]):
        for feature in self.__features:
            if feature in to_be_processed:
                pass

    @staticmethod
    def __map_value(value, low1, high1, low2, high2) -> float:
        """
        Scale the value in the range low1, high1 to the range low2, high2.

        :param value: value to map
        :param low1: min value of the mapping
        :param high1: max value of the mapping
        :param low2: min value to map to
        :param high2: max value to map to

        :return: a scale value in the given range
        """

        return low2 + (high2 - low2) * (value - low1) / (high1 - low1)

    def apply_sliding_window(self, to_be_preprocessed: List[str], width: int = 3):
        """
        Apply an averaging sliding window of given width to each feature.

        :param to_be_preprocessed:
        :param width: the left/right width of the sliding window,
        note that the total width of the sliding window is 2*width+1
        """

        # Iterate over features names
        for feature in self.__features:
            if feature in to_be_preprocessed:
                # Iterate over features values for a given feature
                for i in range(len(self.__features[feature])):

                    # Lower bound of the window
                    if i < width:
                        r = range(i, i + width + 1)
                    # Upper bound of the window
                    elif len(self.__features[feature]) - 1 - i < width:
                        r = range(i - width, i)
                    # Default case
                    else:
                        r = range(i - width, i + width + 1)

                    # Calculate the sum of all values in the window
                    values_sum = 0
                    for j in r:
                        values_sum += self.__features[feature][j]

                    # Set the current value to the average of the window
                    self.__features.at[i, feature] = values_sum / (1 + 2 * width)
