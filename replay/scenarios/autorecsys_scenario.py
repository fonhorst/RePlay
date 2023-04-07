# pylint: disable=too-many-lines
import logging
import os
import importlib

from collections.abc import Iterable
from typing import Dict, Optional, Tuple, List, Union, Any, cast

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame
from replay.experiment import Experiment
from replay.constants import AnyDataFrame
from replay.data_preparator import ToNumericFeatureTransformer
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.metrics import Metric, Precision, NDCG
from replay.models import ALSWrap
from replay.models.base_rec import BaseRecommender, HybridRecommender
from replay.session_handler import State
from replay.splitters import Splitter, UserSplitter
from replay.utils import (
    array_mult,
    cache_if_exists,
    get_log_info,
    get_top_k_recs,
    join_or_return,
    join_with_col_renaming,
    unpersist_if_exists, create_folder, save_transformer, do_path_exists, load_transformer, list_folder, JobGroup,
    cache_and_materialize_if_in_debug, JobGroupWithMetrics,
)
from replay.scenarios import OneStageScenario, TwoStagesScenario
from replay.splitters import Splitter, UserSplitter
from experiments.experiment_utils import (
    get_spark_configs_as_dict,
    check_number_of_allocated_executors,
    get_partition_num,
    get_models,
)
from replay.time import Timer


logger = logging.getLogger("replay")

first_levels_models_params = {
            "replay.models.knn.ItemKNN": {
                "num_neighbours": 100},
            "replay.models.als.ALSWrap": {
                "rank": 100,
                "seed": 42,
                "num_item_blocks": 10,
                "num_user_blocks": 10,
                "hnswlib_params": {
                    "space": "ip",
                    "M": 100,
                    "efS": 2000,
                    "efC": 2000,
                    "post": 0,
                    "index_path": f"file:///tmp/als_hnswlib_index",
                    "build_index_on": "executor",
                },
            },
            "replay.models.word2vec.Word2VecRec": {
                "rank": 100,
                "seed": 42,
                "hnswlib_params": {
                    "space": "ip",
                    "M": 100,
                    "efS": 2000,
                    "efC": 2000,
                    "post": 0,
                    "index_path": f"file:///tmp/word2vec_hnswlib_index",
                    "build_index_on": "executor",
                },
            },
            "replay.models.slim.SLIM": {
                "seed": 42},
        }

second_model_params = {
            "cpu_limit": 20,  # 20
            "memory_limit": int(80 * 0.95),  # 40
            "timeout": 400,
            "general_params": {"use_algos": [["lgb"]]},
            "lgb_params": {
                "use_single_dataset_mode": True,
                "convert_to_onnx": False,
                "mini_batch_size": 1000,
            },
            "linear_l2_params": {"default_params": {"regParam": [1e-5]}},
            "reader_params": {"cv": 2, "advanced_roles": False, "samples": 10_000}
}


class AutoRecSysScenario:

    """

    AutoRecSys scenario which construct training pipeline and return the best model combination: 1stage either two-stage

    """

    def __init__(self, task: str, subtask: str, timeout: int):

        self.scenario = None
        self.task = task
        self.subtask = subtask
        self.timer = Timer(timeout=timeout)
        # self.scenario = self.get_scenario(self,
        #                                   log=log,
        #                                   use_two_stage=True)

    @staticmethod
    def get_scenario(
            self,
            log: DataFrame,
            is_trial: bool = False,
            experiment: Experiment = None) -> Union[OneStageScenario, TwoStagesScenario]:

        # TODO: add choosing models order based on dataset statistics
        # first_level_models_names_default = ["replay.models.als.ALSWrap",
        #                                     "replay.models.slim.SLIM",
        #                                     "replay.models.knn.ItemKNN",
        #                                     "replay.models.word2vec.Word2VecRec"]
        first_level_models_names_default = ["replay.models.als.ALSWrap",
                                            "replay.models.knn.ItemKNN"]

        first_level_models_names_sparse = ["replay.models.knn.ItemKNN",
                                           "replay.models.als.ALSWrap",
                                           "replay.models.word2vec.Word2VecRec",
                                           "replay.models.slim.SLIM"]

        # TODO: refactor this part
        first_level_models = get_models({m: first_levels_models_params[m] for m in first_level_models_names_default})
        # TODO: add heuristics for choosing the most appropriate scenario

        # 0 trial one-stage scenario
        if is_trial:
            scenario = OneStageScenario(
                first_level_models=first_level_models[0],
                user_cat_features_list=None,
                item_cat_features_list=None,
                is_trial=is_trial
            )
            return scenario

        #  heuristics here ==========================================

        logger.info("Choosing the most appropriate scenario")
        logger.info(f"time_left: {self.timer.time_left} sec")
        logger.info(f"time_spent: {self.timer.time_spent} sec")

        if self.timer.time_left >= 100 * self.timer.time_spent:
            logger.info("Two-stage scenario with 1st level models optimization have been chosen (S4)")
            scenario = None
            pass
            #  TODO: add S4 here
        elif self.timer.time_left >= 10 * self.timer.time_spent:
            logger.info("Two-stage scenario with default hyperparameters for 1st level models have been chosen (S3)")

            scenario = TwoStagesScenario(
                    train_splitter=UserSplitter(
                        item_test_size=0.2,
                        shuffle=True,
                        seed=42),
                    first_level_models=get_models(
                        {m: first_levels_models_params[m] for m in first_level_models_names_default}),
                    custom_features_processor=None,
                    num_negatives=10,
                    second_model_type="slama",
                    second_model_params=second_model_params)

        # TODO: add S2 condition
        else:
            logger.info("One scenario with default hyperparameters have been chosen (S1)")

            scenario = OneStageScenario(
                first_level_models=first_level_models,
                user_cat_features_list=None,
                item_cat_features_list=None,
                experiment=experiment
                )

        return scenario

    def fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None
    ):

        logger.info(f"Time left: {self.timer.time_left}")

        # Fit the first model from 1st scenario
        self.scenario = self.get_scenario(self, log=log, is_trial=True)
        self.scenario.fit(log=log, user_features=user_features, item_features=item_features)
        experiment = self.scenario.experiment

        # Determine which scenario will be next
        self.scenario = self.get_scenario(self, log=log, experiment=experiment)
        self.scenario.fit(log=log, user_features=user_features, item_features=item_features)

    def predict(
            self,
            log: DataFrame,
            k: int,
            users: DataFrame,
            items: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            filter_seen_items: bool = True,
    ) -> DataFrame:

        return self.scenario.predict(
            log=log,
            k=k, users=users,
            items=items,
            user_features=user_features,
            item_features=item_features,
            filter_seen_items=filter_seen_items)

    def fit_predict(self):
        pass


