import logging
import pprint
from statistics import mean
from typing import List, Dict, Tuple, Callable, Union

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


class ModelTrainer:
    def __init__(self, scoring_f: Union[str, Callable]) -> None:
        self.__best_models: List[Tuple[str, BaseEstimator]] = []
        self.__scoring_f = scoring_f
        self.__weigths: List[float] = []

    def get_classifiers(self) -> Tuple[List[Tuple[str, BaseEstimator]], List[float]]:
        """
        Provides the list of classifiers as tuples (name, classifier) and relative list of score over
        scoring function

        :return: the list of classifier
        """
        return self.__best_models, self.__weigths

    def svm_clf(self, x: List[List[float]], y: List[float], params: Dict) -> None:
        """
        Train a svm classifier on provided samples performing a grid search with specified params

        :param x: training samples
        :param y: labels
        :param params: params to use in classifier (k_fold, grid, ...)
        """

        # Create pipeline to apply kernel and scale features and feed to an svm
        pipeline = Pipeline([
            ('ker', PolynomialFeatures()),  # apply kernel
            ('scaler', StandardScaler()),  # scale features
            ('clf', SVC(probability=True))
        ])
        k_fold = int(params['k_folds'])

        if params['grid_train'] is True:
            logging.info('Starting grid search with svm...')

            # Perform grid search over svm estimator
            clf = RandomizedSearchCV(estimator=pipeline,
                                     param_distributions=params['grid'],
                                     scoring=self.__scoring_f,
                                     cv=k_fold,
                                     n_jobs=-1)  # n of cores for parallelism (-1 is unbounded)

            # Start grid train
            clf.fit(x, y)

            logging.info('Best score for svm is: {}'.format(clf.best_score_))
            logging.info('Best parameters are: {}'.format(clf.best_params_))
            score = clf.best_score_
            clf = clf.best_estimator_

        else:
            logging.info('Loading params for svm...')
            clf = pipeline.set_params(**params['params'])

            logging.info('Evaluating svm with cross validation...')
            metrics = cross_validate(clf, x, y, cv=k_fold, scoring={'accuracy': 'accuracy',
                                                                    'precision': 'precision',
                                                                    'recall': 'recall',
                                                                    'f1': 'f1',
                                                                    'f_beta': self.__scoring_f},
                                     n_jobs=-1)

            avg = {
                'avg_accuracy': mean(metrics['test_accuracy']),
                'avg_precision': mean(metrics['test_precision']),
                'avg_recall': mean(metrics['test_recall']),
                'avg_f1': mean(metrics['test_f1']),
                'avg_f_beta': mean(metrics['test_f_beta'])
            }

            score = avg['avg_f_beta']

            st = pprint.pformat(avg)

            logging.info('PRINTING METRICS FOR SVM CLASSIFIER (AVERAGE):\n {}'.format(st))

        # Save best classifier
        self.__best_models.append(('svm', clf))
        self.__weigths.append(score)

    def random_forest_clf(self, x: List[List[float]], y: List[float], params: Dict) -> None:
        """
        Train a random forest classifier on provided samples performing a grid search with specified
        params

        :param x: training samples
        :param y: labels
        :param params: params to use in classifier (k_fold, grid, ...)
        """

        rf = RandomForestClassifier(n_jobs=-1)
        k_fold = int(params['k_folds'])

        if params['grid_train'] is True:
            logging.info('Starting grid search with random forest...')

            # Deals with None value as string that are not supported in json
            grid_params = params['grid']
            grid_params['max_depth'] = parse_None(grid_params['max_depth'])

            # Perform grid search
            clf = RandomizedSearchCV(estimator=rf,
                                     param_distributions=grid_params,
                                     cv=k_fold,
                                     scoring=self.__scoring_f,
                                     n_jobs=-1)

            # Start grid search
            clf.fit(x, y)

            logging.info('Best score (f-beta score) for random forest is: {}'.format(clf.best_score_))
            logging.info('Best parameters are: {}'.format(clf.best_params_))
            score = clf.best_score_
            clf = clf.best_estimator_

        else:
            # Prepare params dealing with 'None' values
            train_params = params['params']
            train_params['max_depth'] = parse_None(train_params['max_depth'])

            logging.info('Loading params for random forest...')
            clf = rf.set_params(**train_params)

            logging.info('Cross validating random forest...')
            # Cross validate to estimate accuracy
            metrics = cross_validate(clf, x, y, cv=k_fold,
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

            score = avg['avg_f_beta']

            st = pprint.pformat(avg)

            logging.info('PRINTING METRICS FOR RANDOM FOREST CLASSIFIER (AVERAGE):\n {}'.format(st))

        # Save best classifier
        self.__best_models.append(('rf', clf))
        self.__weigths.append(score)

    def logistic_clf(self, x: List[List[float]], y: List[float], params: Dict) -> None:
        """
        Train a logistic classifier on provided examples performing a grid search with specified params

        :param x: training samples
        :param y: labels
        :param params: params to use in classifier (k_fold, grid, ...)
        """

        lr = LogisticRegression(solver='lbfgs', n_jobs=-1)
        k_fold = int(params['k_folds'])

        if params['grid_train'] is True:
            logging.info('Starting grid search with logistic regressor...')

            grid_params = params['grid']

            # Perform grid search
            clf = GridSearchCV(estimator=lr,
                               param_grid=grid_params,
                               cv=k_fold,
                               scoring=self.__scoring_f,
                               n_jobs=-1)
            # Start grid search
            clf.fit(x, y)

            logging.info('Best score for logistic regressor is: {}'.format(clf.best_score_))
            logging.info('Best parameters are: {}'.format(clf.best_params_))
            score = clf.best_score_
            clf = clf.best_estimator_

        else:
            logging.info('Training logistic regressor...')
            # Set parameters
            clf = lr.set_params(**params['params'])

            # Estimate accuracy with cross validation
            metrics = cross_validate(clf, x, y, cv=k_fold, scoring={'accuracy': 'accuracy',
                                                                    'precision': 'precision',
                                                                    'recall': 'recall',
                                                                    'f1': 'f1',
                                                                    'f_beta': self.__scoring_f},
                                     n_jobs=-1)
            avg = {
                'avg_accuracy': mean(metrics['test_accuracy']),
                'avg_precision': mean(metrics['test_precision']),
                'avg_recall': mean(metrics['test_recall']),
                'avg_f1': mean(metrics['test_f1']),
                'avg_f_beta': mean(metrics['test_f_beta'])
            }

            score = avg['avg_f_beta']

            st = pprint.pformat(avg)

            logging.info('PRINTING METRICS FOR LOGISTIC REGRESSOR (AVERAGE):\n {}'.format(st))

        # Save best classifier
        self.__best_models.append(('log', clf))
        self.__weigths.append(score)


def parse_None(el: object) -> object:
    """
    Parse element or list of elements, inserting None when it finds a string 'None'

    :param el: List or single element to parse
    :return: List or single element parsed
    """
    if isinstance(el, list):
        for i in range(0, len(el)):
            if el[i] == 'None':
                el[i] = None
    elif el == 'None':
        el = None
    return el
