"""
Base classes for quality and diversity metrics.
"""
import operator
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from pyspark.sql import Window
from scipy.stats import norm

from replay.constants import AnyDataFrame, IntOrList, NumType
from replay.utils import convert2spark


# pylint: disable=no-member
def sorter(items, index=1):
    """Sorts a list of tuples and chooses unique objects.

    :param items: tuples ``(relevance, item_id, *args)``.
        Sorting is made using relevance values and unique items
        are selected using element at ``item_idx_index``'s position.
    :param index: item_idx_index of the element in tuple to be returned
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


def sorter_ncis(items, item_idx_index=1, weight_index=2):
    """Sorts a list of tuples and chooses unique objects.

    :param items: tuples ``(relevance, item_id, *args)``.
        Sorting is made using relevance values and unique items
        are selected using element at ``index``'s position.
    :param item_idx_index: index of the element in tuple to be returned
    :return: unique sorted elements
    """
    res = sorted(items, key=operator.itemgetter(0), reverse=True)
    set_res = set()
    item_ids = []
    weights = []
    for item in res:
        if item[1] not in set_res:
            set_res.add(item[1])
            item_ids.append(item[item_idx_index])
            weights.append(item[weight_index])
    return item_ids, weights


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
    """
    Normalized capped importance sampling, where each recommendation is being weighted
    by the ratio of current policy score on previous policy score.
    The weight is also capped by some threshold value.

    Source: arxiv.org/abs/1801.07030
    """

    def __init__(
        self,
        prev_policy_weights: AnyDataFrame,
        threshold: float = 10.0,
        activation: Optional[str] = None,
    ):  # pylint: disable=super-init-not-called
        """
        :param prev_policy_weights: historical item of user-item relevance (previous policy values)
        :threshold: capping threshold, applied after activation,
            relevance values are cropped to interval [1/`threshold`, `threshold`]
        :activation: activation function, applied over relevance values.
            "logit"/"sigmoid", "softmax" or None
        """
        self.prev_policy_weights = convert2spark(
            prev_policy_weights
        ).withColumnRenamed("relevance", "prev_relevance")
        self.threshold = threshold
        if activation is None or activation in ("logit", "sigmoid", "softmax"):
            self.activation = activation
        else:
            raise ValueError(f"Unexpected `activation` - {activation}")
        if threshold <= 0:
            raise ValueError("Threshold should be positive real number")

    @staticmethod
    def _softmax_by_user(df: DataFrame, col_name: str) -> DataFrame:
        """
        Subtract minimal value (relevance) by user from `col_name`
        and apply softmax by user to `col_name`.
        """
        return (
            df.withColumn(
                "_min_rel_user",
                sf.min(col_name).over(Window.partitionBy("user_idx")),
            )
            .withColumn(
                col_name, sf.exp(sf.col(col_name) - sf.col("_min_rel_user"))
            )
            .withColumn(
                col_name,
                sf.col(col_name)
                / sf.sum(col_name).over(Window.partitionBy("user_idx")),
            )
            .drop("_min_rel_user")
        )

    @staticmethod
    def _sigmoid(df: DataFrame, col_name: str) -> DataFrame:
        """
        Apply sigmoid/logistic function to column `col_name`
        """
        return df.withColumn(
            col_name, sf.lit(1.0) / (sf.lit(1.0) + sf.exp(-sf.col(col_name)))
        )

    @staticmethod
    def _weigh_and_clip(
        df: DataFrame,
        threshold: float,
        target_policy_col: str = "relevance",
        prev_policy_col: str = "prev_relevance",
    ):
        """
        Clip weights to fit into interval [1/threshold, threshold].
        """
        lower, upper = 1 / threshold, threshold
        return (
            df.withColumn(
                "weight_unbounded",
                sf.col(target_policy_col) / sf.col(prev_policy_col),
            )
            .withColumn(
                "weight",
                sf.when(sf.col(prev_policy_col) == sf.lit(0.0), sf.lit(upper))
                .when(
                    sf.col("weight_unbounded") < sf.lit(lower), sf.lit(lower)
                )
                .when(
                    sf.col("weight_unbounded") > sf.lit(upper), sf.lit(upper)
                )
                .otherwise(sf.col("weight_unbounded")),
            )
            .select("user_idx", "item_idx", "relevance", "weight")
        )

    def _reweighing(self, recommendations):
        if self.activation == "softmax":
            recommendations = self._softmax_by_user(
                recommendations, col_name="prev_relevance"
            )
            recommendations = self._softmax_by_user(
                recommendations, col_name="relevance"
            )
        elif self.activation in ["logit", "sigmoid"]:
            recommendations = self._sigmoid(
                recommendations, col_name="prev_relevance"
            )
            recommendations = self._sigmoid(
                recommendations, col_name="relevance"
            )

        return self._weigh_and_clip(recommendations, self.threshold)

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

        recommendations = recommendations.join(
            self.prev_policy_weights, on=group_on, how="left"
        ).na.fill(0.0, subset=["prev_relevance"])

        recommendations = self._reweighing(recommendations)
        weight_array_type = st.ArrayType(
            recommendations.schema["weight"].dataType
        )
        item_array_type = st.ArrayType(
            ground_truth.schema["item_idx"].dataType
        )

        top_k_items_and_weights_udf = sf.udf(
            sorter_ncis,
            returnType=st.StructType(
                [
                    st.StructField("pred", item_array_type),
                    st.StructField("weight", weight_array_type),
                ]
            ),
        )

        recommendations = (
            recommendations.groupby("user_idx")
            .agg(
                sf.collect_list(
                    sf.struct("relevance", "item_idx", "weight")
                ).alias("id_pred_weight")
            )
            .withColumn(
                "pred_weight",
                top_k_items_and_weights_udf(sf.col("id_pred_weight")),
            )
            .select(
                "user_idx",
                sf.col("pred_weight.pred"),
                sf.col("pred_weight.weight"),
            )
            .join(true_items_by_users, how="right", on=["user_idx"])
        )

        return recommendations.withColumn(
            "pred",
            sf.coalesce(
                "pred",
                sf.array().cast(item_array_type),
            ),
        ).withColumn(
            "weight",
            sf.coalesce("weight", sf.array().cast(weight_array_type)),
        )
