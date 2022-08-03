import numpy as np

from replay.metrics.base_metric import NCISMetric


# pylint: disable=too-few-public-methods
class NCISPrecision(NCISMetric):
    """
    Mean percentage of relevant items among top ``K`` recommendations.

    .. math::
        Precision@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij}}}{K}

    .. math::
        Precision@K = \\frac {\sum_{i=1}^{N}Precision@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`"""

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        pred, ground_truth, pred_weights = args
        if len(pred) == 0:
            return 0
        mask = np.isin(pred[:k], ground_truth)
        return sum(pred_weights[mask])/ sum(pred_weights[:k])

