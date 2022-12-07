from typing import Optional, Dict

from pyspark.ml import PipelineModel, Transformer
from pyspark.sql import DataFrame, SparkSession
from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.tasks.base import SparkTask

from replay.scenarios.two_stages.reranker import ReRanker
from replay.session_handler import State
from replay.utils import get_top_k_recs


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

        transformer = self.transformer if self.transformer else self.model.transformer()

        sdf = transformer.transform(data)

        # TODO: need to convert predict into the relevance
        candidates_pred_sdf = sdf.select('user_idx', 'item_idx', 'relevance')
        size, users_count = sdf.count(), sdf.select('user_idx').distinct().count()

        self.logger.info(
            "%s candidates rated for %s users",
            size,
            users_count
        )

        self.logger.info("top-k")
        return get_top_k_recs(
            recs=candidates_pred_sdf, k=k, id_type="idx"
        )
