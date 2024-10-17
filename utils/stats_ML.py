import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def bootstrap_confidence_interval_two_means(
    obs1: pd.Series, obs2: pd.Series, alpha: float = 0.05, n_bootstrap: int = 1000
) -> tuple:
    """
    Calculate the bootstrap confidence interval for the difference between two means
    :param obs1: dataset number 1
    :type obs1: pd.Series
    :param obs2: dataset number 2
    :type obs2: pd.Series
    :param alpha: significance level (default 0.05)
    :type alpha: float
    :param n_bootstrap: number of bootstrap iterations (default 1000)
    :type n_bootstrap: int
    :return: confidence interval of proportion differences (upper, lower)
    """
    n_obs1, n_obs2 = len(obs1), len(obs2)

    bootstrap_means_diff = []
    for _ in range(n_bootstrap):
        bootstrap_sample1 = np.random.choice(obs1, size=n_obs1, replace=True)
        bootstrap_sample2 = np.random.choice(obs2, size=n_obs2, replace=True)

        mean1 = np.mean(bootstrap_sample1)
        mean2 = np.mean(bootstrap_sample2)

        means_diff = mean1 - mean2
        bootstrap_means_diff.append(means_diff)

    lower_bound = np.percentile(bootstrap_means_diff, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means_diff, 100 * (1 - alpha / 2))

    return lower_bound, upper_bound


def mean_diff_permutation(values: pd.Series, n_obs_a: int, n_obs_b: int) -> float:
    """
    Calculate the mean difference for a single permutation of data from two groups of observations.
    :param values: a Sequence such as a pandas series or list of values for all observations from
                   two independent groups.
    :type values: Sequence
    :param n_obs_a: The number of observations in group A.
    :type n_obs_a: int
    :param n_obs_b: The number of observations in group B.
    :type n_obs_b: int

    :return mean_diff: The mean difference for a single permutation of the data
    from two groups of observations.
    :rtype: float
    """
    total_obs = n_obs_a + n_obs_b
    idx_a = set(random.sample(range(total_obs), n_obs_a))
    idx_b = set(range(total_obs)) - idx_a
    return values.iloc[list(idx_a)].mean() - values.iloc[list(idx_b)].mean()


def model_assessment_series_cv(
    model, X, y, folds=5, model_name="model", random_state=0
):
    """
    A pandas Series for assessment of classification model predictions using k-fold cross validation.

    :param model: the model being used
    :type model: scikit-learn model or similar
    :param X: features
    :type X: array-like
    :param y: outcomes
    :type y: array-like
    :param folds: number of folds
    :type folds: int
    :param model_name: a name for the model to use as a title for the column
    :type model_name: str
    :param random_state: random state to use
    :type random_state: int

    :return: model assessment scores
    :rtype: pandas.Series
    """
    scoring_dict = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    cv_scores = cross_validate(model, X, y, scoring=scoring_dict, cv=cv)
    accuracy = cv_scores["test_accuracy"].mean()
    precision = cv_scores["test_precision"].mean()
    recall = cv_scores["test_recall"].mean()
    f1 = cv_scores["test_f1"].mean()
    roc_auc = cv_scores["test_roc_auc"].mean()

    return pd.Series(
        [accuracy, precision, recall, f1, roc_auc],
        index=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        name=model_name,
    )


def model_assessment_series(observations, predictions, model_name="model"):
    """
    A pandas Series for assessment of classification model predictions.

    :param observations: observations from data
    :type observations: array-like
    :param predictions: predictions from model
    :type predictions: array-like
    :param model_name: a name for the model to use as a title for the column
    :type model_name: str

    :return: model assessment scores
    :rtype: pandas.Series
    """
    return pd.Series(
        [
            accuracy_score(observations, predictions),
            precision_score(observations, predictions),
            recall_score(observations, predictions),
            f1_score(observations, predictions),
            roc_auc_score(observations, predictions),
        ],
        index=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC (for class labels)",
        ],
        name=model_name,
    )
