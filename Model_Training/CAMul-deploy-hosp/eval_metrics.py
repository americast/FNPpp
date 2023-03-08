from typing import Tuple
import numpy as np
import properscoring as ps
from scipy.stats import norm
import matplotlib.pyplot as plt

EPS = 1e-6


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Root Mean Squared Error

    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model

    Returns:
        float: RMSE
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def nrmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Normalized Root Mean Squared Error

    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model

    Returns:
        float: NRMSE
    """
    return rmse(predictions, targets) / (targets.max() - targets.min())


def mape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error

    Args:
        predictions (np.ndarray): Predictions of the model
        targets (np.ndarray): Targets of the model

    Returns:
        float: MAPE
    """
    return np.mean(np.abs((predictions - targets) / targets)) * 100


def crps(mean: np.ndarray, std: np.ndarray, targets: np.ndarray) -> float:
    """
    Quantile-based CRPS

    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)

    Returns:
        float: CRPS
    """
    return ps.crps_gaussian(targets, mean, std).mean()


def crps_samples(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    Quantile-based CRPS

    Args:
        samples (np.ndarray): Samples of the distribution (N, samples)
        targets (np.ndarray): Targets of the model (N)

    Returns:
        float: CRPS
    """
    return ps.crps_ensemble(targets, samples).mean()


def log_score(
    mean: np.ndarray, std: np.ndarray, targets: np.ndarray, window: float = 1.0
) -> float:
    """
    Log Score

    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)

    Returns:
        float: Log Score
    """
    t1 = norm.cdf(targets - window / 2.0, mean, std)
    t2 = norm.cdf(targets + window / 2.0, mean, std)
    return np.log(np.clip(t2 - t1, EPS, 1.0)).mean()


def interval_score(
    mean: np.ndarray, std: np.ndarray, targets: np.ndarray, window: float = 1.0
) -> float:
    """
    Interval Score

    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)

    Returns:
        float: Interval Score
    """
    rd_val = np.round(targets, decimals=1)
    low_val = np.clip(rd_val - window / 2, a_min=0.0, a_max=None)
    high_val = np.clip(rd_val + window / 2, a_min=None, a_max=13)
    t1 = norm.cdf(low_val, loc=mean, scale=std)
    t2 = norm.cdf(high_val, loc=mean, scale=std)
    return np.log(np.clip(t2 - t1, a_min=EPS, a_max=1.0)).mean()


def conf_interval(
    mean: np.ndarray, var: np.ndarray, conf: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Confintance Interval for given confidence level

    Args:
        mean (np.ndarray): Mean of the distribution (N)
        var (np.ndarray): Variance of the distribution (N)
        conf (float): Confidence level

    Returns:
        tuple: (low, high) interval
    """
    out_prob = 1.0 - conf
    high = norm.ppf(1.0 - (out_prob / 2), loc=mean, scale=var**0.5)
    low = norm.ppf((1.0 - conf) / 2, loc=mean, scale=var**0.5)
    return low, high


def pres_recall(
    mean: np.ndarray, var: np.ndarray, target: np.ndarray, conf: float
) -> np.ndarray:
    """
    Fraction of GT points within the confidence interval

    Args:
        mean (np.ndarray): Mean of the distribution (N)
        var (np.ndarray): Variance of the distribution (N)
        target (np.ndarray): Target of the model (N)
        conf (float): Confidence level

    Returns:
        np.ndarray: Fraction of GT points within the confidence interval
    """
    low, high = conf_interval(mean, var, conf)
    truth = ((target > low) & (target < high)).astype("float32")
    return truth.mean(-1)


def get_pr(pred, var, target, color="blue", label="FluFNP"):
    """
    Plot confidence and return Confidence score and AUC

    Args:
        pred (np.ndarray): Predictions of the model (N)
        var (np.ndarray): Variance of the distribution (N)
        target (np.ndarray): Target of the model (N)
        color (str): Color of the line
        label (str): Label of the model

    Returns:
        tuple: (Confidence score, AUC, fraction values)
    """
    x = np.arange(0.05, 1.0, 0.01)
    y = np.array([pres_recall(pred, var, target, c) for c in x])
    # plt.plot(list(x) + [1.0], list(y) + [1.0], label=label, color=color)
    conf_score = np.abs(y - x).sum() * 0.01
    auc = y.sum() * 0.01
    return auc, conf_score, list(y) + [1.0]
