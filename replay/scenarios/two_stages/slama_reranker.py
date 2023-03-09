from typing import Optional, Dict

from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.functions import expr

from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import WrappingSelectingPipelineModel

from replay.scenarios.two_stages.reranker import ReRanker
from replay.session_handler import State
from replay.utils import get_top_k_recs

import pandas as pd
import numpy as np


class SlamaWrap(ReRanker):
    """
    LightAutoML TabularPipeline binary classification model wrapper for recommendations re-ranking.
    Read more: https://github.com/sberbank-ai-lab/LightAutoML
    """

    def save(self, path: str, overwrite: bool = False, spark: Optional[SparkSession] = None):
        transformer = self.model.transformer()

        if overwrite:
            transformer.write().overwrite().save(path)
        else:
            transformer.write().save(path)

    @classmethod
    def load(cls, path: str, spark: Optional[SparkSession] = None):
        pipeline_model = PipelineModel.load(path)

        return SlamaWrap(transformer=pipeline_model)

    def __init__(
        self,
        params: Optional[Dict] = None,
        config_path: Optional[str] = None,
        transformer: Optional[Transformer] = None
    ):
        """
        Initialize LightAutoML TabularPipeline with passed params/configuration file.

        :param params: dict of model parameters
        :param config_path: path to configuration file
        """
        assert (transformer is not None) != (params is not None or config_path is not None)

        if transformer is not None:
            self.model = None
            self.transformer = transformer
        else:
            self.model = SparkTabularAutoML(
                spark=State().session,
                task=SparkTask("binary"),
                config_path=config_path,
                **(params if params is not None else {}),
            )
            self.transformer = None

    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        """
        Fit the LightAutoML TabularPipeline model with binary classification task.
        Data should include negative and positive user-item pairs.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx, target]``
            columns and features' columns. `Target` column should consist of zeros and ones
            as the model is a binary classification model.
        :param fit_params: dict of parameters to pass to model.fit()
            See LightAutoML TabularPipeline fit_predict parameters.
        """

        if self.transformer is not None:
            raise RuntimeError("The ranker is already fitted")

        params = {
            "roles": {"target": "target"},
            "verbose": 1,
            **({} if fit_params is None else fit_params)
        }
        data = data.drop("user_idx", "item_idx")

        # array_features = [k for k, v in dict(data.dtypes).items() if v == "array<double>"]
        #
        # if len(array_features) > 0:
        #     for array_feature in array_features:
        #         print(f"processing {array_feature}")
        #         array_size = len(data.select(array_feature).head()[0])
        #         data = data.select(
        #             data.columns + [data[array_feature][x].alias(f"{array_feature}_{x}") for x in range(array_size)]
        #         )
        #         data = data.drop(array_feature)
        # TODO: do not forget about persistence manager
        self.model.fit_predict(data, **params)

    def predict(self, data: DataFrame, k: int) -> DataFrame:
        """
        Re-rank data with the model and get top-k recommendations for each user.

        :param data: spark dataframe with obligatory ``[user_idx, item_idx]``
            columns and features' columns
        :param k: number of recommendations for each user
        :return: spark dataframe with top-k recommendations for each user
            the dataframe columns are ``[user_idx, item_idx, relevance]``
        """
        self.logger.info("Starting re-ranking")

        # array_features = [k for k, v in dict(data.dtypes).items() if v == "array<double>"]
        #
        # if len(array_features) > 0:
        #     for array_feature in array_features:
        #         print(f"processing {array_feature}")
        #         array_size = len(data.select(array_feature).head()[0])
        #         data = data.select(
        #             data.columns + [data[array_feature][x].alias(f"{array_feature}_{x}") for x in range(array_size)]
        #         )
        #         data = data.drop(array_feature)

        transformer = self.transformer if self.transformer else self.model.transformer()

        sdf = transformer.transform(data)

        candidates_pred_sdf = sdf.select(
            'user_idx',
            'item_idx',
            vector_to_array('prediction').getItem(1).alias('relevance')
        )

        # size, users_count = candidates_pred_sdf.count(), candidates_pred_sdf.select('user_idx').distinct().count()

        self.logger.info("Re-ranking is finished")

        # TODO: strange, but the further process would hang without maetrialization
        # TODO: probably, it may be related to optimization and lightgbm models
        # TODO: need to dig deeper later
        candidates_pred_sdf = candidates_pred_sdf.cache()
        candidates_pred_sdf.write.mode('overwrite').format('noop').save()

        self.logger.info("top-k")
        top_k_recs = get_top_k_recs(
            recs=candidates_pred_sdf, k=k, id_type="idx"
        )

        # top_k_recs = candidates_pred_sdf.cache()
        # top_k_recs.write.mode('overwrite').format('noop').save()

        # candidates_pred_sdf.write.mode('overwrite').format('noop').save()
        # raise Exception("------")

        return top_k_recs
