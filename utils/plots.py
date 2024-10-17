from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    PrecisionRecallDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline


def bar_plots_categorical_features(
    df: pd.DataFrame, categorical_columns: list, target: str = "stroke"
) -> None:
    """
    Bar plots of categorical features in the dataframe

    :param df: A pandas dataframe containing your data to plot
    :type df: pd.DataFrame
    :param categorical_columns: A list of categorical columns in your dataframe
    :type categorical_columns: list
    :param target: name of the target column
    :type target: str
    :return: prints a figure
    None
    """

    columns_not_target = df[categorical_columns].columns[
        df[categorical_columns].columns != target
    ]
    fig, axs = plt.subplots(
        nrows=len(columns_not_target),
        ncols=1,
        figsize=(12, 4 * len(columns_not_target)),
        sharex=True,
    )

    for ax, col in zip(axs, columns_not_target):
        plot_df = (
            df.groupby([col, target], observed=True)
            .size()
            .reset_index()
            .pivot(columns=target, index=col, values=0)
        )
        plot_df.plot(kind="barh", stacked=True, ax=ax)

        container = ax.containers[0]
        for i, (bar, value) in enumerate(zip(container, container.datavalues)):
            width1 = bar.get_width()
            width2 = ax.containers[1][i].get_width() if len(ax.containers) > 1 else 0
            total_width = width1 + width2
            ax.text(
                total_width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{total_width:.0f}",
                va="center",
                ha="left",
                fontsize=10,
                color="black",
            )

        if ax == axs[0]:
            ax.set_title("Distributions of Categorical Features \n With Total Counts")
        else:
            pass

    plt.show()


def plot_heatmap_contingency(
    df: pd.DataFrame,
    row: str,
    col: str,
    title: str,
    figsize: tuple = (8, 6),
    cmap: str = "Blues_r",
    annot: bool = True,
    fmt: str = "d",
    cbar: bool = True,
) -> pd.DataFrame:
    """
    Plots a heatmap for a contingency table.

    :param df: The input DataFrame.
    :param row: The column name for the rows of the contingency table.
    :param col: The column name for the columns of the contingency table.
    :param title: The title of the plot.
    :param figsize: The size of the figure (default is (8, 6)).
    :param cmap: The colormap for the heatmap (default is "Blues_r").
    :param annot: Whether to annotate the cells with the numeric data (default is True).
    :param fmt: String formatting code to use when adding annotations (default is 'd').
    :param cbar: Whether to draw a color bar (default is True).

    :return contingency_table: A table with counts
    :rtype: pd.DataFrame
    """
    contingency_table = pd.crosstab(df[row], df[col])

    plt.figure(figsize=figsize)
    sns.heatmap(contingency_table, annot=annot, fmt=fmt, cmap=cmap, cbar=cbar)

    plt.title(title, fontsize=16)
    plt.xlabel(col.capitalize(), fontsize=14)
    plt.ylabel(row.capitalize(), fontsize=14)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()

    return contingency_table


def plot_numeric_feature_histograms(
    df: pd.DataFrame,
    bins: int = 50,
    figsize: tuple = (12, 8),
    title: str = "Distributions of Numerical Features",
):
    """
    Plots histograms for each numeric column in the DataFrame with appropriate axis labels and a title.

    :param df: The input DataFrame containing numeric columns for which histograms will be plotted.
    :type df: pd.DataFrame
    :param bins: The number of bins to use for each histogram (default is 50).
    :type bins: int
    :param figsize: The size of the figure (default is (12, 8)).
    :type figsize: tuple
    :param title: The title of the plot (default is "Distributions of Numerical Features").
    :type title: str
    """
    axarr = df.hist(bins=bins, figsize=figsize)

    plt.suptitle(title, fontsize=16)

    for ax in axarr.flatten():
        ax.set_xlabel(ax.get_title())
        ax.set_ylabel("Count")
        ax.set_title("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_stroke_class_frequency_by_bmi(
    df: pd.DataFrame, bmi_column: str = "bmi", target_column: str = "stroke"
) -> None:
    """
    Plots the frequency of stroke classes for two groups: rows with missing BMI values and rows
    with non-missing BMI values.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param bmi_column: The name of the column containing BMI values (default is 'bmi').
    :type bmi_column: str
    :param target_column: The name of the column containing the target variable (default is 'stroke').
    :type target_column: str
    """
    bmi_missing = df.loc[df[bmi_column].isna()]
    bmi_not_missing = df.loc[~df[bmi_column].isna()]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)

    bmi_missing[target_column].value_counts(normalize=True).plot(kind="bar", ax=ax1)
    bmi_not_missing[target_column].value_counts(normalize=True).plot(kind="bar", ax=ax2)

    for ax in (ax1, ax2):
        for p in ax.patches:
            ax.annotate(
                str(round(p.get_height(), 2)),
                (p.get_x() + p.get_width() / 2.0, p.get_height() * 1.01),
                ha="center",
            )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel(target_column.capitalize())

    fig.suptitle("Stroke Class Frequency")
    ax1.set_title("BMI Missing")
    ax2.set_title("BMI Not Missing")
    ax1.set_ylabel("Frequency")

    plt.show()


def plot_regressions(df: pd.DataFrame) -> None:
    """
    Plots regression plots with linear and polynomial fits for 'age' vs. 'bmi' and 'age' vs. 'avg_glucose_level',
    along with bar plots showing stroke class frequencies based on missing and non-missing BMI values.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

    sns.regplot(
        x="age", y="bmi", data=df, order=1, line_kws={"color": "r"}, ax=axs[0, 0]
    )
    sns.regplot(
        x="age",
        y="avg_glucose_level",
        data=df,
        order=1,
        line_kws={"color": "r"},
        ax=axs[0, 1],
    )

    axs[0, 0].set_title("Linear Fit: Age vs BMI", fontsize=16, weight="bold")
    axs[0, 0].set_xlabel("Age", fontsize=14)
    axs[0, 0].set_ylabel("BMI", fontsize=14)

    axs[0, 1].set_title(
        "Linear Fit: Age vs Avg Glucose Level", fontsize=16, weight="bold"
    )
    axs[0, 1].set_xlabel("Age", fontsize=14)
    axs[0, 1].set_ylabel("Average Glucose Level", fontsize=14)

    sns.regplot(
        x="age", y="bmi", data=df, order=2, line_kws={"color": "r"}, ax=axs[1, 0]
    )
    sns.regplot(
        x="age",
        y="avg_glucose_level",
        data=df,
        order=2,
        line_kws={"color": "r"},
        ax=axs[1, 1],
    )

    axs[1, 0].set_title(
        "Polynomial Fit: Age vs BMI (Order=2)", fontsize=16, weight="bold"
    )
    axs[1, 0].set_xlabel("Age", fontsize=14)
    axs[1, 0].set_ylabel("BMI", fontsize=14)

    axs[1, 1].set_title(
        "Polynomial Fit: Age vs Avg Glucose Level (Order=2)", fontsize=16, weight="bold"
    )
    axs[1, 1].set_xlabel("Age", fontsize=14)
    axs[1, 1].set_ylabel("Average Glucose Level", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(
        "Regression Analysis and Frequency Distributions", fontsize=20, weight="bold"
    )

    plt.show()


def plot_proportional_distribution(
    df: pd.DataFrame, target_column: str = "stroke"
) -> None:
    """
    Plots the proportional distribution of a target class using a count plot and a pie chart.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param target_column: The name of the target column to analyze (default is 'stroke').
    :type target_column: str
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.countplot(y=df[target_column], stat="proportion", ax=axs[0])
    axs[0].set_xlabel("Proportion", fontsize=14)
    axs[0].set_ylabel(target_column.capitalize(), fontsize=14)
    axs[0].tick_params(axis="both", which="major", labelsize=12)

    colors = sns.color_palette("pastel")
    pie_data = df[target_column].value_counts()
    wedges, texts, autotexts = axs[1].pie(
        pie_data, autopct="%.2f%%", colors=colors, startangle=140
    )

    axs[1].legend(
        wedges,
        pie_data.index,
        title=target_column.capitalize(),
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    axs[1].set_ylabel("")  # Remove y-axis label

    for text in autotexts:
        text.set_fontsize(12)
        text.set_color("black")

    plt.suptitle(
        "Proportional Distribution of Target Class", fontsize=18, weight="bold"
    )

    plt.show()


def plot_permutation_feature_importance(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: List[str],
    n_repeats: int = 20,
    random_state: int = 42,
    n_jobs: int = 2,
) -> None:
    """
    Plots the permutation feature importance for multiple metrics.

    Parameters:
    - model_pipeline (Pipeline): The machine learning pipeline to evaluate.
    - X (pd.DataFrame): The feature data.
    - y (pd.Series): The target data.
    - metrics (List[str]): A list of metric names to evaluate.
    - n_repeats (int): Number of times to permute a feature. Default is 20.
    - random_state (int): Random state for reproducibility. Default is 42.
    - n_jobs (int): Number of jobs to run in parallel. Default is 2.

    Returns:
    - None
    """

    feature_importances: Dict[str, Any] = {}
    for metric in metrics:
        importance = permutation_importance(
            model_pipeline,
            X,
            y,
            scoring=metric,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        feature_importances[metric] = importance

    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(15, 5 * num_metrics))
    if num_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        importances = feature_importances[metric]
        sorted_idx = importances.importances_mean.argsort()
        importances_df = pd.DataFrame(
            importances.importances[sorted_idx].T, columns=X.columns[sorted_idx]
        )

        importances_df.plot.box(vert=False, whis=10, ax=axs[i])
        axs[i].set_title(f"Permutation Importances ({metric})")
        axs[i].axvline(x=0, color="k", linestyle="--")
        axs[i].set_xlabel(f"Decrease in {metric} score")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion " "Matrix",
) -> None:
    """
    Plots an aesthetically pleasing confusion matrix with a title and labels.

    Parameters:
    - y_true (np.ndarray): Array of true labels.
    - y_pred (np.ndarray): Array of predicted labels.
    - class_names (List[str]): List of class names corresponding to the labels.
    - title (str): Title of the plot. Default is 'Confusion Matrix'.

    Returns:
    - None
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))

    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted labels", fontsize=14)
    ax.set_ylabel("True labels", fontsize=14)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax.set_ylim(len(cm) - 0.5, -0.5)

    ax.grid(False)

    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves_with_best_threshold(
    y_true: np.ndarray, y_probs: np.ndarray, metric: str = "recall"
) -> None:
    """
    Plots ROC and Precision-Recall curves side by side with the best threshold marked based on the specified metric.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_probs (np.ndarray): Estimated probabilities or decision function.
    - metric (str): Metric to determine the best threshold. Options are 'precision', 'recall', or 'f1'.
    Default is 'recall'.

    Returns:
    - None
    """

    def best_threshold(y_true, y_probs, metric):
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        if metric == "precision":
            ix = np.argmax(precision)
            best_thresh = thresholds[ix]
            return best_thresh, recall[ix], precision[ix]
        elif metric == "recall":
            ix = np.argmax(recall)
            best_thresh = thresholds[ix]
            return best_thresh, recall[ix], precision[ix]
        elif metric == "f1":
            f1_scores = 2 * precision * recall / (precision + recall)
            ix = np.argmax(f1_scores)
            best_thresh = thresholds[ix]
            return best_thresh, recall[ix], precision[ix]
        else:
            raise ValueError(
                "Metric not recognized. Choose 'precision', 'recall', or 'f1'."
            )

    best_thresh, best_recall, best_precision = best_threshold(y_true, y_probs, metric)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=axs[0])
    axs[0].scatter(best_recall, best_precision, marker="o", color="black", label="Best")
    axs[0].set_title(
        f"Precision-Recall Curve\nBest Threshold ({metric})={best_thresh:.3f}, Recall={best_recall:.3f}, Precision={best_precision:.3f}",
        fontsize=14,
    )
    axs[0].legend()

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    ix = np.argmax(tpr)
    best_thresh_roc = thresholds[ix]

    display = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="log reg"
    )
    display.plot(ax=axs[1])
    axs[1].scatter(fpr[ix], tpr[ix], marker="o", color="black", label="Best")
    axs[1].set_title(
        f"ROC Curve\n Threshold for Max TPR={best_thresh_roc:.3f}, TPR={tpr[ix]:.3f}, FPR={fpr[ix]:.3f}",
        fontsize=14,
    )
    axs[1].legend()

    plt.tight_layout()
    plt.show()
