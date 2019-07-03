from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal.windows import gaussian
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Preprocessor:

    def __init__(self, features: pd.DataFrame, res_info: pd.DataFrame):
        """
        Build a preprocessing utility and performs the filling of the missing values.

        :param features: the whole set of features of the dataset
        """
        self.__features = features
        self.__info = res_info

    def get_features(self) -> pd.DataFrame:
        return self.__features

    def fill_missing_values(self, exclude: List[str] = []):
        """
        Fill missing values with the median value of the feature.
        :param exclude: list of columns to exclude
        """

        for feature in self.__features.keys():
            if feature not in exclude:
                self.__features[feature].fillna(0, inplace=True)

    def apply_one_hot_encoding(self, to_be_encoded: List[str],
                               encoders: Dict[str, OneHotEncoder] = dict()) -> Dict[str, OneHotEncoder]:
        """

        :param to_be_encoded: list of columns to be encoded
        :param encoders: dictionary containing for each column the encoder to use
        :return: the dictionary of encoders fitted for each column
        """
        for header in to_be_encoded:
            df = self.__features[header]
            df = df.fillna('-')
            X = np.asarray(df).reshape(-1, 1)
            if header not in encoders.keys():
                enc = OneHotEncoder(handle_unknown='ignore')
                enc.fit(X)
                encoders[header] = enc
            else:
                enc = encoders[header]
            Y = enc.transform(X).toarray()
            df = pd.DataFrame(Y, columns=enc.categories_[0].tolist())
            self.__features = self.__features.drop(columns=[header])
            self.__features = pd.concat([self.__features, df], axis=1)
        return encoders

    def apply_features_scaling(self, to_be_encoded: List[str], scaler=None) -> MinMaxScaler:
        """
        Scale each feature within a [0, 1] range.

        :param to_be_encoded: list of header string to be processed
        :param scaler: the previous scaler used for scaling the features of dataset
        """
        features_to_scale = self.__features[to_be_encoded]
        if scaler is None:
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features_to_scale)
        else:
            scaled_features = scaler.transform(features_to_scale)
        self.__features.loc[:, to_be_encoded] = scaled_features
        return scaler

    def apply_sliding_window(self, to_be_excluded: List[str], k: int = 3):
        """
        Apply an averaging sliding window of given width to each feature.

        :param to_be_excluded: list of header to exclude from sliding window
        :param k: the left/right width of the sliding window,
        note that the total width of the sliding window is 2*width+1
        """

        originals = self.__features[to_be_excluded]

        df_merged: DataFrame = pd.concat([self.__info, self.get_features()], axis=1)
        df_windows = df_merged.copy(deep=True)
        buff = pd.DataFrame(columns=self.__features.columns)

        for pdb_id in df_merged.pdb_id.unique():
            for chain in df_merged.chain[df_merged.pdb_id == pdb_id].unique():
                df_sliced = df_windows[(df_merged.pdb_id == pdb_id)
                                       & (df_merged.chain == chain)]
                info_sliced = df_sliced.iloc[:, 0:3]
                chain_len = len(df_merged.chain[(df_merged.pdb_id == pdb_id)
                                                & (df_merged.chain == chain)])
                df_windows_start = pd.DataFrame(np.array(df_sliced.iloc[1:(k // 2 + 1), ]),
                                                index=np.arange(-k // 2 + 1, 0, step=1),
                                                columns=list(df_merged.columns)).sort_index()
                df_windows_end = pd.DataFrame(
                    np.array(df_sliced.iloc[chain_len - (k // 2 + 1):chain_len - 1, ]),
                    index=np.arange(chain_len - 1 + k // 2, chain_len - 1, step=-1),
                    columns=list(df_merged.columns)).sort_index()

                df_with_start_sym = df_windows_start.append(df_sliced)
                df_win_k = df_with_start_sym.append(df_windows_end)

                sliced = df_win_k.iloc[:, 3:]
                window = gaussian(k, std=1)
                sliced = sliced.rolling(window=k, center=True).apply(lambda x: np.dot(x, window) / k,
                                                                     raw=True)
                sliced = sliced.iloc[k // 2:chain_len + k // 2, :]
                buff = buff.append(sliced, ignore_index=True)
        self.__features = buff
        self.__features[to_be_excluded] = originals
