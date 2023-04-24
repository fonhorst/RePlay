# pylint: disable=too-many-lines
import logging
import os
import importlib
import time

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
                # "hnswlib_params": {
                #     "space": "ip",
                #     "M": 100,
                #     "efS": 2000,
                #     "efC": 2000,
                #     "post": 0,
                #     "index_path": f"file:///tmp/als_hnswlib_index_123",
                #     "build_index_on": "executor",
                # },
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
                    "index_path": f"file:///tmp/word2vec_hnswlib_index_123",
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


FIRST_LEVELS_MODELS_PARAMS_BORDERS = {
    "replay.models.als.ALSWrap": {
        "rank": [10, 300]
    },
    "replay.models.knn.ItemKNN": {
        "num_neighbours": [50, 1000],
                                  },
    "replay.models.slim.SLIM": {
        "beta": [1e-6, 1],  # 1e-6, 5 #0.01, 0.1
        "lambda_": [1e-6, 1]  # [1e-6, 2] #0.01, 0.1
    },
    "replay.models.word2vec.Word2VecRec": {
        "rank": [10, 300],
        "window_size": [1, 3],
        "use_idf": [True, False]
    },
}


class AutoRecSysScenario:

    """

    AutoRecSys scenario which construct training pipeline and return the best model combination: 1stage either two-stage

    """

    def __init__(self, task: str, subtask: str, timeout: float):

        self.scenario = None
        self.task = task
        self.subtask = subtask
        self.timer = Timer(timeout=timeout)

    def get_default_two_stage(self):

        first_level_models_names_default = ["replay.models.als.ALSWrap",
                                            "replay.models.slim.SLIM",
                                            ]

        first_level_models_names_sparse = ["replay.models.knn.ItemKNN",
                                           "replay.models.als.ALSWrap",
                                           "replay.models.word2vec.Word2VecRec",
                                           "replay.models.slim.SLIM"]

        first_level_models = get_models({m: first_levels_models_params[m] for m in first_level_models_names_default})

        return TwoStagesScenario(
            train_splitter=UserSplitter(
                item_test_size=0.2,
                shuffle=True,
                seed=42),
            first_level_models=get_models(
                {m: first_levels_models_params[m] for m in first_level_models_names_default}),
            custom_features_processor=None,
            num_negatives=10,
            second_model_type="slama",
            second_model_params=second_model_params,
            second_model_config_path=os.environ.get(
                "PATH_TO_SLAMA_TABULAR_CONFIG", "tabular_config.yml"),
            one_stage_timeout=self.timer.time_left
        )

    def get_default_one_stage(self, experiment=None, is_trial=None):

        first_level_models_names_default = ["replay.models.als.ALSWrap",
                                            "replay.models.slim.SLIM",
                                            ]

        first_level_models_names_sparse = ["replay.models.knn.ItemKNN",
                                           "replay.models.als.ALSWrap",
                                           "replay.models.word2vec.Word2VecRec",
                                           "replay.models.slim.SLIM"]

        first_level_models = get_models({m: first_levels_models_params[m] for m in first_level_models_names_default})
        if is_trial:
            first_level_models = first_level_models[0]

        return OneStageScenario(
                first_level_models=first_level_models,
                user_cat_features_list=None,
                item_cat_features_list=None,
                experiment=experiment,
                timeout=self.timer.time_left,
                set_best_model=True,
                is_trial=is_trial
            )

    @staticmethod
    def get_scenario(
            self,
            log: DataFrame,
            is_trial: bool = False,
            experiment: Experiment = None) -> Tuple[Union[OneStageScenario, TwoStagesScenario], bool]:

        log_size = log.count()

        # TODO: add choosing models order based on dataset statistics

        do_optimization = None

        # 0 trial one-stage scenario
        if is_trial:

            scenario = self.get_default_one_stage(experiment=experiment, is_trial=is_trial)
            do_optimization = False
            return scenario, do_optimization

        #  heuristics here ==========================================

        logger.info("Choosing the most appropriate scenario")
        logger.info(f"time_left: {self.timer.time_left} sec")
        logger.info(f"time_spent: {self.timer.time_spent} sec")

        if log_size > 1_000_000 and self.timer.time_left >= 100 * self.timer.time_spent:
            logger.info(f"log size: {log_size} bigger than 1m")
            logger.info("Two-stage scenario with 1st level models optimization have been chosen (S4)")
            scenario = self.get_default_two_stage()
            do_optimization = True

        elif log_size > 1_000_000 and self.timer.time_left >= 10 * self.timer.time_spent:
            logger.info("Two-stage scenario with default hyperparameters for 1st level models have been chosen (S3)")
            scenario = self.get_default_two_stage()
            do_optimization = False

        elif log_size <= 1_000_000 and self.timer.time_left >= 80 * self.timer.time_spent:
            logger.info(f"log size: {log_size} smaller than 1m")
            logger.info("One scenario with hyperparameters optimization have been chosen (S2)")
            scenario = self.get_default_one_stage(experiment=experiment)
            do_optimization = True

        else:
            logger.info("One scenario with default hyperparameters have been chosen (S1)")

            scenario = self.get_default_one_stage(experiment=experiment)
            do_optimization = False

        # end of heuristics ===============================================================
        return scenario, do_optimization

    def fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None
    ):

        logger.info(f"Time left: {self.timer.time_left}")

        # Fit the first model from 1st scenario
        self.scenario, _ = self.get_scenario(self, log=log, is_trial=True)
        self.scenario.fit(log=log, user_features=user_features, item_features=item_features)
        experiment = self.scenario.experiment

        # Determine which scenario will be next

        self.scenario, do_optimization = self.get_scenario(self, log=log, experiment=experiment)

        if do_optimization:
            spark = State().session
            first_level_train = spark.read.parquet("/tmp/first_level_train.parquet")
            first_level_val = spark.read.parquet("/tmp/first_level_val.parquet")

            # optimize first level models
            # TODO: get it from somewhere
            first_level_models_names_default = ["replay.models.als.ALSWrap",
                                                "replay.models.slim.SLIM",
                                                ]

            param_borders = [
                FIRST_LEVELS_MODELS_PARAMS_BORDERS[model_name] for model_name in first_level_models_names_default
            ]
            logger.debug(f"param borders is: {param_borders}")
            param_found, fallback_params, metrics_values = self.scenario.optimize(
                train=first_level_train,
                test=first_level_val,
                param_borders=[*param_borders, None],
                k=10,  # TODO: get from class
                budget=20,
                criterion=NDCG()
            )

            if type(self.scenario).__name__ == "OneStageScenario":

                logger.debug("choosing the best model inside optimization step: OneStageScenario")
                logger.debug(f"models metric values are: {metrics_values}")
                best_model_index = metrics_values.index(max(metrics_values))
                self.scenario.best_model = self.scenario.first_level_models[best_model_index]
            else:
                logger.debug("Start fit two stage scenario with already optimized hyperparameters")

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


