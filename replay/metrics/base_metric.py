"""
Base classes for quality and diversity metrics.
"""
import operator
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from pyspark.sql import Window
from scipy.stats import norm

from replay.constants import AnyDataFrame, IntOrList, NumType
from replay.utils import convert2spark, get_top_k_recs


# pylint: disable=no-member
def sorter(items, index=1):
    """Sorts a list of tuples and chooses unique objects.

    :param items: tuples ``(relevance, item_id, *args)``.
        Sorting is made using relevance values and unique items
        are selected using element at ``index``'s position.
    :param index: index of the element in tuple to be returned
    :return: unique sorted elements
    """
    res = sorted(items, key=operator.itemgetter(0), reverse=True)
    set_res = set()
    list_res = []
    for item in res:
        if item[1] not in set_res:
            set_res.add(item[1])
            list_res.append(item[index])
    return list_res


def get_enriched_recommendations(
    recommendations: AnyDataFrame, ground_truth: AnyDataFrame
) -> DataFrame:
    """
    Merge recommendations and ground truth into a single DataFrame
    and aggregate items into lists so that each user has only one record.

    :param recommendations: recommendation list
    :param ground_truth: test data
    :return:  ``[user_id, pred, ground_truth]``
    """
    recommendations = convert2spark(recommendations)
    ground_truth = convert2spark(ground_truth)
    true_items_by_users = ground_truth.groupby("user_idx").agg(
        sf.collect_set("item_idx").alias("ground_truth")
    )
    sort_udf = sf.udf(
        sorter,
        returnType=st.ArrayType(ground_truth.schema["item_idx"].dataType),
    )
    recommendations = (
        recommendations.groupby("user_idx")
        .agg(sf.collect_list(sf.struct("relevance", "item_idx")).alias("pred"))
        .select("user_idx", sort_udf(sf.col("pred")).alias("pred"))
        .join(true_items_by_users, how="right", on=["user_idx"])
    )

    return recommendations.withColumn(
        "pred",
        sf.coalesce(
            "pred",
            sf.array().cast(
                st.ArrayType(ground_truth.schema["item_idx"].dataType)
            ),
        ),
    )


def process_k(func):
    """Decorator that converts k to list and unpacks result"""

    def wrap(self, recs: DataFrame, k: IntOrList, *args):
        if isinstance(k, int):
            k_list = [k]
        else:
            k_list = k

        res = func(self, recs, k_list, *args)

        if isinstance(k, int):
            return res[k]
        return res

    return wrap


class Metric(ABC):
    """Base metric class"""

    def __str__(self):
        return type(self).__name__

    def __call__(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: model predictions in a
            DataFrame ``[user_id, item_id, relevance]``
        :param ground_truth: test data
            ``[user_id, item_id, timestamp, relevance]``
        :param k: depth cut-off. Truncates recommendation lists to top-k items.
        :return: metric value
        """
        recs = get_enriched_recommendations(recommendations, ground_truth)
        return self._mean(recs, k)

    @process_k
    def _conf_interval(self, recs: DataFrame, k_list: list, alpha: float):
        res = {}
        quantile = norm.ppf((1 + alpha) / 2)
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = (
                distribution.agg(
                    sf.stddev("value").alias("std"),
                    sf.count("value").alias("count"),
                )
                .select(
                    sf.when(
                        sf.isnan(sf.col("std")) | sf.col("std").isNull(),
                        sf.lit(0.0),
                    )
                    .otherwise(sf.col("std"))
                    .cast("float")
                    .alias("std"),
                    "count",
                )
                .first()
            )
            res[k] = quantile * value["std"] / (value["count"] ** 0.5)
        return res

    @process_k
    def _median(self, recs: DataFrame, k_list: list):
        res = {}
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = distribution.agg(
                sf.expr("percentile_approx(value, 0.5)").alias("value")
            ).first()["value"]
            res[k] = value
        return res

    @process_k
    def _mean(self, recs: DataFrame, k_list: list):
        res = {}
        for k in k_list:
            distribution = self._get_metric_distribution(recs, k)
            value = distribution.agg(sf.avg("value").alias("value")).first()[
                "value"
            ]
            res[k] = value
        return res

    def _get_metric_distribution(self, recs: DataFrame, k: int) -> DataFrame:
        """
        :param recs: recommendations
        :param k: depth cut-off
        :return: metric distribution for different cut-offs and users
        """
        cur_class = self.__class__
        distribution = recs.rdd.flatMap(
            # pylint: disable=protected-access
            lambda x: [
                (x[0], float(cur_class._get_metric_value_by_user(k, *x[1:])))
            ]
        ).toDF(
            f"user_idx {recs.schema['user_idx'].dataType.typeName()}, value double"
        )
        return distribution

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        """
        Metric calculation for one user.

        :param k: depth cut-off
        :param pred: recommendations
        :param ground_truth: test data
        :return: metric value for current user
        """

    def user_distribution(
        self,
        log: AnyDataFrame,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> pd.DataFrame:
        """
        Get mean value of metric for all users with the same number of ratings.

        :param log: history DataFrame to calculate number of ratings per user
        :param recommendations: prediction DataFrame
        :param ground_truth: test data
        :param k: depth cut-off
        :return: pandas DataFrame
        """
        log = convert2spark(log)
        count = log.groupBy("user_idx").count()
        if hasattr(self, "_get_enriched_recommendations"):
            recs = self._get_enriched_recommendations(
                recommendations, ground_truth
            )
        else:
            recs = get_enriched_recommendations(recommendations, ground_truth)
        if isinstance(k, int):
            k_list = [k]
        else:
            k_list = k
        res = pd.DataFrame()
        for cut_off in k_list:
            dist = self._get_metric_distribution(recs, cut_off)
            val = count.join(dist, on="user_idx")
            val = (
                val.groupBy("count")
                .agg(sf.avg("value").alias("value"))
                .orderBy(["count"])
                .select("count", "value")
                .toPandas()
            )
            res = res.append(val, ignore_index=True)
        return res


# pylint: disable=too-few-public-methods
class RecOnlyMetric(Metric):
    """Base class for metrics that do not need holdout data"""

    @abstractmethod
    def __init__(self, log: AnyDataFrame, *args, **kwargs):
        pass

    # pylint: disable=no-self-use
    @abstractmethod
    def _get_enriched_recommendations(
        self, recommendations: AnyDataFrame, ground_truth: AnyDataFrame
    ) -> DataFrame:
        pass

    def __call__(  # type: ignore
        self, recommendations: AnyDataFrame, k: IntOrList
    ) -> Union[Dict[int, NumType], NumType]:
        """
        :param recommendations: predictions of a model,
            DataFrame  ``[user_id, item_id, relevance]``
        :param k: depth cut-off
        :return: metric value
        """
        recs = self._get_enriched_recommendations(recommendations, None)
        return self._mean(recs, k)

    @staticmethod
    @abstractmethod
    def _get_metric_value_by_user(k, *args) -> float:
        """
        Metric calculation for one user.

        :param k: depth cut-off
        :param *args: extra parameters, returned by
            '''self._get_enriched_recommendations''' method
        :return: metric value for current user
        """


class NCISMetric(Metric):
    def __init__(
        self,
        prev_policy_weights: AnyDataFrame,
        threshold: float,
        activation: Optional[str] = None

    ):  # pylint: disable=super-init-not-called
        """
        Here we calculate self-information for each item

        :param log: historical data
        """
        self.prev_policy_weights = (convert2spark(prev_policy_weights)
                                    .withColumnRenamed("relevance",
                                                   "prev_relevance"))
        self.threshold = threshold
        if activation is None or activation in ("logit", "sigmoid", "softmax"):
            self.activation = activation
        else:
            raise ValueError(
                "Unexpected `activation` - {}".format(activation))

    def _reweighting(self, recommendations):
        if self.activation == 'softmax':
            min_relevance_by_user = (recommendations
                                     .groupBy("user_idx").agg(
                {"min_prev_rel": sf.min("prev_relevance"),
                 "min_rel": sf.min("relevance")})
            )
            recommendations = (
                recommendations
                .join(min_relevance_by_user, on="user_idx")
                .withColumn("prev_relevance", sf.exp(sf.col(
                    "prev_relevance")-sf.col("min_prev_rel")))
                .withColumn("prev_relevance", sf.col(
                    "prev_relevance")/sf.sum("prev_relevance").over(
                    Window.partitionBy("user_idx")))
                .withColumn("relevance", sf.exp(sf.col(
                    "relevance") - sf.col("min_rel")))
                .withColumn("relevance", sf.col(
                    "relevance") / sf.sum("relevance").over(
                    Window.partitionBy("user_idx")))
            )

        elif self.activation in ['logit', 'sigmoid']:

            recommendations = (
                recommendations
                .withColumn("prev_relevance", 1/ (1+ sf.exp(-sf.col(
                    "prev_relevance"))))
                .withColumn("prev_relevance", 1 / (1 + sf.exp(-sf.col(
                    "prev_relevance"))))
            )

        elif self.activation is None:
            pass

        return (recommendations.withColumn("weight",
                sf.when(sf.col("prev_relevance") == sf.lit(0), sf.lit(
            self.threshold))
                .when(sf.col("relevance")/sf.col("prev_relevance") < sf.lit(
                1.0/self.threshold), sf.lit(1.0/self.threshold))
                .when(sf.col("relevance") / sf.col("prev_relevance") > sf.lit(
                self.threshold), sf.lit(self.threshold))
                .otherwise(sf.col("relevance") / sf.col("prev_relevance"))
                ).select("user_idx", "item_idx", "relevance", "weight"))

    def _get_enriched_recommendations(
            self, recommendations: AnyDataFrame, ground_truth: AnyDataFrame
    ) -> DataFrame:
        """
        Merge recommendations and ground truth into a single DataFrame
        and aggregate items into lists so that each user has only one record.

        :param recommendations: recommendation list
        :param ground_truth: test data
        :return:  ``[user_id, pred, ground_truth]``
        """
        recommendations = convert2spark(recommendations)
        ground_truth = convert2spark(ground_truth)

        true_items_by_users = ground_truth.groupby("user_idx").agg(
            sf.collect_set("item_idx").alias("ground_truth")
        )

        group_on = ["item_idx"]
        if "user_idx" in self.prev_policy_weights.columns:
            group_on.append("user_idx")

        recommendations = recommendations.join(self.prev_policy_weights,
                                                   on=group_on,
                                                   how="left")
        recommendations = recommendations.withColumn(
                "prev_relevance", sf.coalesce("prev_relevance", sf.lit(0.0)))

        recommendations = self._reweighting(recommendations)


        sort_udf = sf.udf(
            sorter,
            returnType=st.ArrayType(ground_truth.schema["item_idx"].dataType),
        )
        weight_udf = sf.udf(
            sorter,
            returnType=st.ArrayType(recommendations.schema["weight"].dataType),
        )

        recommendations = (
            recommendations
                .groupby("user_idx")
                .agg(sf.collect_list(sf.struct("relevance", "item_idx")).alias(
                 "pred"))
                .select("user_idx", sort_udf(sf.col("pred")).alias("pred")
                        , weight_udf(sf.col("pred")).alias("weight"))
                .join(true_items_by_users, how="right", on=["user_idx"])
        )

        return recommendations.withColumn(
            "pred",
            sf.coalesce(
                "pred",
                sf.array().cast(
                    st.ArrayType(ground_truth.schema["item_idx"].dataType)
                ),
            ),
        )