from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.models.base_rec import NonPersonalizedRecommender
from replay.utils import unionify, unpersist_after


class PopRec(NonPersonalizedRecommender):
    """
    Recommend objects using their popularity.

    Popularity of an item is a probability that random user rated this item.

    .. math::
        Popularity(i) = \\dfrac{N_i}{N}

    :math:`N_i` - number of users who rated item :math:`i`

    :math:`N` - total number of users

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 1, 2, 2, 3, 4], "item_idx": [1, 2, 2, 3, 3, 3], "relevance": [0.5, 1, 0.1, 0.8, 0.7, 1]})
    >>> data_frame
       user_idx  item_idx  relevance
    0         1         1        0.5
    1         1         2        1.0
    2         2         2        0.1
    3         2         3        0.8
    4         3         3        0.7
    5         4         3        1.0

    >>> from replay.utils import convert2spark
    >>> data_frame = convert2spark(data_frame)

    >>> res = PopRec().fit_predict(data_frame, 1)
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         3       0.75
    1         2         1       0.25
    2         3         2       0.50
    3         4         2       0.50

    >>> res = PopRec().fit_predict(data_frame, 1, filter_seen_items=False)
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         3       0.75
    1         2         3       0.75
    2         3         3       0.75
    3         4         3       0.75

    >>> res = PopRec(use_relevance=True).fit_predict(data_frame, 1)
    >>> res.toPandas().sort_values("user_idx", ignore_index=True)
       user_idx  item_idx  relevance
    0         1         3      0.625
    1         2         1      0.125
    2         3         2      0.275
    3         4         2      0.275

    """

    def __init__(self, use_relevance: bool = False):
        """
        :param use_relevance: flag to use relevance values as is or to treat them as 1
        """
        self.use_relevance = use_relevance
        self.all_user_ids: Optional[DataFrame] = None
        self.item_abs_relevances: Optional[DataFrame] = None
        self.item_popularity: Optional[DataFrame] = None

    @property
    def _init_args(self):
        return {"use_relevance": self.use_relevance}

    @property
    def _dataframes(self):
        return {
            "all_user_ids": self.all_user_ids,
            "item_abs_relevances": self.item_abs_relevances,
            "item_popularity": self.item_popularity
        }

    def _clear_cache(self):
        for df in self._dataframes.values():
            if df is not None:
                df.unpersist()

    def _fit_partial(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            previous_log: Optional[DataFrame] = None):
        with unpersist_after(self._dataframes):
            self.all_user_ids = unionify(log.select("user_idx"), self.all_user_ids).distinct().cache()
            self._users_count = self.all_user_ids.count()

            if self.use_relevance:
                # we will save it to update fitted model
                self.item_abs_relevances = (
                    unionify(log.select("item_idx", "relevance"), self.item_abs_relevances)
                    .groupBy("item_idx")
                    .agg(sf.sum("relevance").alias("relevance"))
                ).cache()

                self.item_popularity = (
                    self.item_abs_relevances.withColumn("relevance", sf.col("relevance") / sf.lit(self._users_count))
                )
            else:
                log = unionify(log, previous_log)

                # equal to store a whole old log which may be huge
                item_users = (
                    log
                    .groupBy("item_idx")
                    .agg(sf.collect_set('user_idx').alias('user_idx'))
                )

                self.item_popularity = (
                    item_users
                    .select(
                        "item_idx",
                        (sf.size("user_idx") / sf.lit(self.users_count)).alias("relevance"),
                    )
                )

            self.item_popularity.cache().count()

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        return self._predict_without_sampling(
            log, k, users, items, filter_seen_items
        )

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:

        return pairs.join(self.item_popularity, on="item_idx", how="inner")
