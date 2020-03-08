# -*- coding: utf-8 -*-
"""
    Feature selection example.
"""
import argparse
import sys
import typing
import numpy

from sklearn import ensemble
from sklearn import feature_selection
from sklearn import metrics
from sklearn import model_selection
from sklearn import utils


SEED = 42
VARIANCE_MULTIPLIER = 1.0
VARIANCE_MULTIPLIER_BOGUS = 10.0
VERBOSE = False


def generate_data_frame(n: int, m: int, k: int):
    """
    Generate data frame of n objects with m features among which only k features matter.
    :param n: Number of objects.
    :param m: Number of all features.
    :param k: Number of relevant features.
    :return: (Numpy array of shape (n, k), Numpy array of shape (n, m) with redundant features,
    Numpy array of class labels)
    """
    if VERBOSE:
        print("Data frame generating...")
    numpy.random.seed(SEED)
    cov_0 = numpy.random.rand(k, k)
    cov_1 = numpy.random.rand(k, k)
    cov_0 = numpy.dot(cov_0, cov_0.T) + numpy.eye(k, k) * VARIANCE_MULTIPLIER
    cov_1 = numpy.dot(cov_1, cov_1.T) + numpy.eye(k, k) * VARIANCE_MULTIPLIER
    x_0 = numpy.random.multivariate_normal(numpy.ones(k), cov_0, n // 2)
    x_1 = numpy.random.multivariate_normal(-numpy.ones(k), cov_1, n - x_0.shape[0])
    x = numpy.vstack((x_0, x_1))
    y = numpy.array([0] * x_0.shape[0] + [1] * x_1.shape[0])
    x, y = utils.shuffle(x, y, random_state=SEED)
    big_data = numpy.hstack((x, VARIANCE_MULTIPLIER_BOGUS * numpy.random.rand(n, m - k)))
    big_data = utils.shuffle(big_data.T, random_state=SEED).T
    return x, big_data, y


def classify(data: numpy.ndarray, labels: numpy.ndarray):
    """
    Split sample, fit classifier and predict class labels.
    :param data: Data frame with objects and their features.
    :param labels: Class labels array.
    :return: True class labels and predicted class labels for the test sample.
    """
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels)
    classifier = ensemble.RandomForestClassifier(random_state=SEED)
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    return y_test, y_predicted


def select_features(data: numpy.ndarray, labels: numpy.ndarray, k: int):
    """
    Select k best features from the given data frame.
    :param data: Data frame with objects and their features.
    :param labels: Class labels array.
    :param k: Number of features to select.
    :return: Data frame with only best features remain.
    """
    selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif, k=k)
    data = selector.fit_transform(data, labels)
    return data


def feature_selection_example(m: int, k: int):
    """
    Feature selection example.
    :param m: Number of all features.
    :param k: Number of features to select.
    """
    if VERBOSE:
        print("Feature selection example")
    data, big_data, labels = generate_data_frame(m * m, m, k)
    y_true, y_predicted = classify(data, labels)
    print("Relevant features classification report")
    print(metrics.classification_report(y_true, y_predicted))
    y_true, y_predicted = classify(big_data, labels)
    print("All features classification report")
    print(metrics.classification_report(y_true, y_predicted))
    data_selected = select_features(data, labels, k)
    y_true, y_predicted = classify(data_selected, labels)
    print("Selected features classification report")
    print(metrics.classification_report(y_true, y_predicted))


def main(args: typing.List[str]):
    """
    Program entry point.
    :param args: Command line arguments.
    """
    parser = argparse.ArgumentParser(description="Feature selection example.")
    parser.add_argument("-m", "--features-number", type=int, help="Number of all features.", required=True)
    parser.add_argument("-k", "--features-to-select", type=int, help="Number of features to select.", required=True)
    parser.add_argument("-v", "--verbose", help="Verbose output.", type=bool, default=False)
    args = parser.parse_args(args=args[1:])
    global VERBOSE
    VERBOSE = args.verbose
    feature_selection_example(args.features_number, args.features_to_select)


if __name__ == "__main__":
    main(sys.argv)
