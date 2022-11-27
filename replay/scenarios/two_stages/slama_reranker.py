from typing import Optional, Dict

from pyspark.sql import DataFrame
from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.tasks.base import SparkTask

from replay.scenarios.two_stages.reranker import ReRanker
from replay.utils import get_top_k_recs


class SlamaWrap(ReRanker):
    """
    LightAutoML TabularPipeline binary classification model wrapper for recommendations re-ranking.
    Read more: https://github.com/sberbank-ai-lab/LightAutoML
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize LightAutoML TabularPipeline with passed params/configuration file.

        :param params: dict of model parameters
        :param config_path: path to configuration file
        """
        self.model = SparkTabularAutoML(
            task=SparkTask("binary"),
            config_path=config_path,
            **(params if params is not None else {}),
        )

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

        params = {
            "roles": {"target": "target"},
            "verbose": 1,
            **({} if fit_params is None else fit_params)
        }
        # TODO: add service columns to the dataframe
        # data = data.drop("user_idx", "item_idx")
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

        # TODO: add service columns to the dataframe
        candidates_pred = self.model.predict(data)
        candidates_pred_sdf = candidates_pred.data.select('user_idx', 'item_idx', 'relevance')

        self.logger.info(
            "%s candidates rated for %s users",
            candidates_pred.shape[0],
            candidates_pred.data.select('user_idx').distinct().count()
        )

        self.logger.info("top-k")
        return get_top_k_recs(
            recs=candidates_pred_sdf, k=k, id_type="idx"
        )
