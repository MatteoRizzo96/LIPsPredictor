import logging
import pprint
from statistics import mean
from typing import List, Dict, Union, Callable

from pandas import DataFrame
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate

from classes.ModelTrainer import ModelTrainer
from functions.model_persistence import dump_clf, load_clf


class EnsembleVotingPredictor:
    """
    This class can be used as a builder. It allows to add desired predictor to the ensemble model.
    Each predictor will be trained, evaluated (with a grid search and CV) and accuracy results and
    best parameters will be printed in the log file.
    When done, the predictor with optimized parameters are re-trained inside a VotingClassifier (
    method fit()).
    With predict_proba() it is possible to use the trained model to make predictions.
    """

    def __init__(self, x: Union[List[List[float]], DataFrame], y: Union[List[float], DataFrame],
                 scoring_f: Union[str, Callable]):
        """
        Initialize predictor

        :param x: the list of training samples and relative features
        :param y: list of labels
        """
        self.__predictor: VotingClassifier
        self.__builder = ModelTrainer(scoring_f)
        self.__trained = False
        self.__scoring_f = scoring_f
        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y,
        # test_size = 0.25,
        # random_state = 44)
        self.__x_train, self.__y_train = x, y

    def dump(self) -> None:
        """
        Write the classifier to file, inside configuration folder
        """
        dump_clf(self.__predictor, 'configuration/', 'voting_clf.joblib')
        logging.info('Saved predictor to file')

    def load(self) -> None:
        """
        Load the classifier from joblib file in configuration folder
        """
        self.__predictor = load_clf('configuration/voting_clf.joblib')
        self.__trained = True
        logging.info('Loaded predictor from file')

    def add_svm(self, param: Dict) -> None:
        """
        Train a svm classifier and adds the best model to the ensemble predictor

        :param x: training examples
        :param y: training labels
        :param param: grid search parameters to select best model
        """
        self.__builder.svm_clf(self.__x_train, self.__y_train, param)

    def add_random_forest(self, param: Dict) -> None:
        """
        Train a random forest classifier and adds the best model to the ensemble predictor

        :param x: training examples
        :param y: training labels
        :param param: grid search parameters to select best model
        """
        self.__builder.random_forest_clf(self.__x_train, self.__y_train, param)

    def add_logistic_classifier(self, param: Dict) -> None:
        """
        Train a logistic classifier and adds the best model to the ensemble predictor

        :param x: training examples
        :param y: training labels
        :param param: grid search parameters to select best model
        """
        self.__builder.logistic_clf(self.__x_train, self.__y_train, param)

    def fit(self) -> None:
        """
        Train the classifier.

        :param x: training examples
        :param y: training labels
        """
        logging.info('Training ensemble classifier')
        # Get best performing classifier to use in ensemble voting
        estimators, scores = self.__builder.get_classifiers()

        # Order estimators by score
        # est_weighted = [(e, s) for e, s in zip(estimators, scores)]
        # sorted_est = sorted(est_weighted, key=lambda el: el[1], reverse=True)
        # estimators = [el[0] for el in sorted_est]
        # Weights to assign to each classifier
        # weights = [0.7, 0.2, 0.1]

        assert len(estimators) != 0, 'Cannot train with no estimators.'

        # logging.info(
        #    'Assigning weights to each classifier.\nClassifier are: {}\n Weights are: {}'.format(
        #        [est[0] for est in estimators], weights))

        self.__predictor = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

        self.__predictor.fit(self.__x_train, self.__y_train)
        self.__trained = True

        # self.__predictor.score(self.__x_test, self.__y_test)

    def evaluate(self, x_test=None, y_test=None, k_fold: int = 10) -> Dict:
        assert self.__trained is True, 'Cannot predict with untrained model.'

        # Estimate of model accuracy
        metrics = cross_validate(self.__predictor, self.__x_train, self.__y_train, cv=k_fold,
                                 scoring={'accuracy': 'accuracy',
                                          'precision': 'precision',
                                          'recall': 'recall',
                                          'f1': 'f1',
                                          'f_beta': self.__scoring_f}, n_jobs=-1)

        avg = {
            'avg_accuracy': mean(metrics['test_accuracy']),
            'avg_precision': mean(metrics['test_precision']),
            'avg_recall': mean(metrics['test_recall']),
            'avg_f1': mean(metrics['test_f1']),
            'avg_f_beta': mean(metrics['test_f_beta'])
        }

        st = pprint.pformat(avg)

        logging.info('PRINTING METRICS FOR ENSEMBLE CLASSIFIER (AVERAGE):\n{}'.format(st))

        if x_test is not None:
            aa = self.__predictor.score(x_test, y_test)
            logging.info('Accuracy on test set: {}'.format(aa))

        return avg

    def predict_proba(self, x: List[List[float]]) -> List[List[float]]:
        """
        Return predicted probabilities for provided samples

        :param x: examples
        :return: weighted average probability for each class per sample
        """

        logging.info('Predicting...')

        assert self.__trained is True, 'Cannot predict with untrained model.'

        return self.__predictor.predict_proba(x)
