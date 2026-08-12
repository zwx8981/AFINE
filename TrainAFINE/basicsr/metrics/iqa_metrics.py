import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr


def _logistic(x, beta1, beta2, beta3, beta4, beta5):
    return beta1 * (0.5 - 1 / (1 + np.exp(beta2 * (x - beta3)))) + beta4 * x + beta5


def calculate_iqa_metrics(predictions, targets, logistic_mapping=True):
    """Calculate full-dataset IQA metrics; both inputs must be quality-oriented."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.size < 2 or predictions.shape != targets.shape:
        raise ValueError('predictions and targets must have the same shape and contain at least two samples')
    mapped = predictions
    if logistic_mapping and predictions.size >= 6 and np.ptp(predictions) > 0:
        initial = [np.ptp(targets), 1 / max(np.std(predictions), 1e-6), np.mean(predictions), 0, np.mean(targets)]
        try:
            params, _ = curve_fit(_logistic, predictions, targets, p0=initial, maxfev=20000)
            mapped = _logistic(predictions, *params)
        except (RuntimeError, ValueError, OverflowError):
            mapped = predictions
    return {
        # SciPy's correlation result objects are tuple-like across supported
        # versions, while the named ``statistic`` attribute is only available
        # in newer releases.  Indexing keeps validation compatible with the
        # older SciPy commonly installed alongside Python 3.8.
        'SRCC': float(spearmanr(predictions, targets)[0]),
        'PLCC': float(pearsonr(mapped, targets)[0]),
        'KRCC': float(kendalltau(predictions, targets)[0]),
        'RMSE': float(np.sqrt(np.mean((mapped - targets)**2)))
    }
