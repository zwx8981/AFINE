from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .accuracy import calculate_nr_accuracy, calculate_all_accuracy, calculate_all_accuracy_higher_better, calculate_nr_accuracy_higher_better
from .iqa_metrics import calculate_iqa_metrics

__all__ = ['calculate_nr_accuracy', 'calculate_all_accuracy', 'calculate_all_accuracy_higher_better',
           'calculate_nr_accuracy_higher_better', 'calculate_iqa_metrics']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
