from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from scipy.stats import norm

from replay.models.pop_rec import PopRec
from replay.utils import unionify, unpersist_after


class Wilson(PopRec):
    """
    Calculates lower confidence bound for the confidence interval
    of true fraction of positive ratings.

    ``relevance`` must be converted to binary 0-1 form.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 2], "item_idx": [1, 2], "relevance": [1, 1]})
    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)
    >>> model = Wilson()
    >>> model.fit_predict(data_frame,k=1).toPandas()
       user_idx  item_idx  relevance
    0         1         2   0.206549
    1         2         1   0.206549

    """

    def __init__(self, alpha=0.05):
        """
        :param alpha: significance level, default 0.05
        """
        super().__init__()
        self.alpha = alpha
        self.items_counts_aggr: Optional[DataFrame] = None

    @property
    def _init_args(self):
        return {"alpha": self.alpha}

    @property
    def _dataframes(self):
        return {
            "item_popularity": self.item_popularity,
            "items_counts_aggr": self.items_counts_aggr
        }

    def _fit_partial(self,
                     log: DataFrame,
                     user_features: Optional[DataFrame] = None,
                     item_features: Optional[DataFrame] = None,
                     previous_log: Optional[DataFrame] = None) -> None:
        with unpersist_after(self._dataframes):
            self._check_relevance(log)

            log = log.select("item_idx", sf.col("relevance").alias("pos"), sf.lit(1).alias("total"))

            self.items_counts_aggr = (
                unionify(log, self.items_counts_aggr)
                .groupby("item_idx").agg(
                    sf.sum("pos").alias("pos"),
                    sf.sum("total").alias("total")
                )
            ).cache()

            # https://en.wikipedia.org/w/index.php?title=Binomial_proportion_confidence_interval
            crit = norm.isf(self.alpha / 2.0)
            pos, total = sf.col("pos"), sf.col("total")

            self.item_popularity = self.items_counts_aggr.select(
                "item_idx",
                (
                    (pos + sf.lit(0.5 * crit**2)) / (total + sf.lit(crit**2))
                    - sf.lit(crit) / (total + sf.lit(crit**2)) * sf.sqrt((total - sf.col("pos")) * pos / total
                    + crit**2 / 4)
                ).alias("relevance")
            )

            self.item_popularity.cache().count()
