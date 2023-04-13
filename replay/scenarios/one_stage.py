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
from replay.time import Timer
from replay.model_handler import save, load

logger = logging.getLogger("replay")



# pylint: disable=too-many-locals, too-many-arguments
def get_first_level_model_features(
    model: BaseRecommender,
    pairs: DataFrame,
    user_features: Optional[DataFrame] = None,
    item_features: Optional[DataFrame] = None,
    add_factors_mult: bool = True,
    prefix: str = "",
) -> DataFrame:
    """
    Get user and item embeddings from replay model.
    Can also compute elementwise multiplication between them with ``add_factors_mult`` parameter.
    Zero vectors are returned if a model does not have embeddings for specific users/items.

    :param model: trained model
    :param pairs: user-item pairs to get vectors for `[user_id/user_idx, item_id/item_id]`
    :param user_features: user features `[user_id/user_idx, feature_1, ....]`
    :param item_features: item features `[item_id/item_idx, feature_1, ....]`
    :param add_factors_mult: flag to add elementwise multiplication
    :param prefix: name to add to the columns
    :return: DataFrame
    """
    users = pairs.select("user_idx").distinct()
    items = pairs.select("item_idx").distinct()
    user_factors, user_vector_len = model._get_features_wrap(
        users, user_features
    )
    item_factors, item_vector_len = model._get_features_wrap(
        items, item_features
    )

    pairs_with_features = join_or_return(
        pairs, user_factors, how="left", on="user_idx"
    )
    pairs_with_features = join_or_return(
        pairs_with_features,
        item_factors,
        how="left",
        on="item_idx",
    )

    if user_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "user_factors",
            sf.coalesce(
                sf.col("user_factors"),
                sf.array([sf.lit(0.0)] * user_vector_len),
            ),
        )

    if item_factors is not None:
        pairs_with_features = pairs_with_features.withColumn(
            "item_factors",
            sf.coalesce(
                sf.col("item_factors"),
                sf.array([sf.lit(0.0)] * item_vector_len),
            ),
        )

    if model.__str__() == "LightFMWrap":
        pairs_with_features = (
            pairs_with_features.fillna({"user_bias": 0, "item_bias": 0})
            .withColumnRenamed("user_bias", f"{prefix}_user_bias")
            .withColumnRenamed("item_bias", f"{prefix}_item_bias")
        )

    if (
        add_factors_mult
        and user_factors is not None
        and item_factors is not None
    ):
        pairs_with_features = pairs_with_features.withColumn(
            "factors_mult",
            array_mult(sf.col("item_factors"), sf.col("user_factors")),
        )

    return pairs_with_features


# pylint: disable=too-many-instance-attributes
class OneStageScenario(HybridRecommender):
    """
    *train*:

    1) train ``first_stage_models`` on train dataset
    2) return the best model according to metrics on holdout dataset


    *inference*:

    1) inference of best trained model

    """

    can_predict_cold_users: bool = True
    can_predict_cold_items: bool = True

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        train_val_splitter: Splitter = UserSplitter(
            item_test_size=0.2, shuffle=True, seed=42
        ),
        first_level_models: Union[
            List[BaseRecommender], BaseRecommender
        ] = ALSWrap(rank=128),
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
        custom_features_processor: HistoryBasedFeaturesProcessor = None,
        seed: int = 123,
        is_trial: bool = False,
        experiment: Experiment = None,
        timeout: int = None,
        set_best_model: bool = False
    ) -> None:
        """

        :param first_level_models: model or a list of models
        :param user_cat_features_list: list of user categorical features
        :param item_cat_features_list: list of item categorical features
        :param custom_features_processor: you can pass custom feature processor
        :param seed: random seed

        """
        self.cached_list = []
        self.train_val_splitter = train_val_splitter
        self.first_level_models = (
            first_level_models
            if isinstance(first_level_models, Iterable)
            else [first_level_models]
        )

        self.first_level_item_len = 0
        self.first_level_user_len = 0

        # self.random_model = RandomRec(seed=seed)
        # self.fallback_model = fallback_model
        self.first_level_user_features_transformer = (
            ToNumericFeatureTransformer()
        )
        self.first_level_item_features_transformer = (
            ToNumericFeatureTransformer()
        )

        self.features_processor = (
            custom_features_processor
            if custom_features_processor
            else HistoryBasedFeaturesProcessor(
                user_cat_features_list=user_cat_features_list,
                item_cat_features_list=item_cat_features_list,
            )
        )
        self.seed = seed

        self._job_group_id = ""
        self._experiment = experiment
        self._is_trial = is_trial
        if timeout:
            self.timer = Timer(timeout=timeout)
        self._set_best_model = set_best_model

    @property
    def _init_args(self):
        return {}

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Write statistics"""
        first_level_train, first_level_val = self.train_val_splitter.split(log)
        logger.info("Log info: %s", get_log_info(log))
        logger.info(
            "first_level_train info: %s", get_log_info(first_level_train)
        )
        logger.info(
            "first_level_val info: %s", get_log_info(first_level_val)
        )
        return first_level_train, first_level_val


    def _save_model(self, path: str):
        from replay.model_handler import save
        spark = State().session
        create_folder(path, exists_ok=True)

        # save features
        if self.first_level_user_features_transformer is not None:
            save_transformer(
                self.first_level_user_features_transformer,
                os.path.join(path, "first_level_user_features_transformer")
            )

        if self.first_level_item_features_transformer is not None:
            save_transformer(
                self.first_level_item_features_transformer,
                os.path.join(path, "first_level_item_features_transformer")
            )

        if self.features_processor is not None:
            save_transformer(self.features_processor, os.path.join(path, "features_processor"))

        # Save first level models
        first_level_models_path = os.path.join(path, "first_level_models")
        create_folder(first_level_models_path)
        for i, model in enumerate(self.first_level_models):
            save(model, os.path.join(first_level_models_path, f"model_{i}"))

        # save general data and settings
        data = {
            "first_level_item_len": self.first_level_item_len,
            "first_level_user_len": self.first_level_user_len,
            "seed": self.seed
        }

        spark.createDataFrame([data]).write.parquet(os.path.join(path, "data.parquet"))

    def _load_model(self, path: str):
        from replay.model_handler import load
        spark = State().session

        # load general data and settings
        data = spark.read.parquet(os.path.join(path, "data.parquet")).first().asDict()

        # load transformers for features
        comp_path = os.path.join(path, "first_level_user_features_transformer")
        first_level_user_features_transformer = load_transformer(comp_path) if do_path_exists(comp_path) else None #TODO: check why this dir exists if user_features=None

        comp_path = os.path.join(path, "first_level_item_features_transformer")
        first_level_item_features_transformer = load_transformer(comp_path) if do_path_exists(comp_path) else None #TODO same

        comp_path = os.path.join(path, "features_processor")
        features_processor = load_transformer(comp_path) if do_path_exists(comp_path) else None # TODO same

        # load first level models
        first_level_models_path = os.path.join(path, "first_level_models")
        if do_path_exists(first_level_models_path):
            model_paths = [
                os.path.join(first_level_models_path, model_path)
                for model_path in list_folder(first_level_models_path)
            ]
            first_level_models = [load(model_path) for model_path in model_paths]
        else:
            first_level_models = None

        self.__dict__.update({
            **data,
            "first_level_user_features_transformer": first_level_user_features_transformer,
            "first_level_item_features_transformer": first_level_item_features_transformer,
            "features_processor": features_processor,
            "first_level_models": first_level_models,
        })

    @staticmethod
    def _filter_or_return(dataframe, condition):
        if dataframe is None:
            return dataframe
        return dataframe.filter(condition)

    # pylint: disable=too-many-locals,too-many-statements
    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self._job_group_id = "1stage_fit"

        self.cached_list = []

        # 1. Split train data to train-val parts for optimization and comparison
        log.cache()
        self.cached_list.append(log)

        if self._set_best_model:
            self.logger.info("Data splitting")
            first_level_train, first_level_val = self._split_data(log)

        # self.first_level_item_len = (
        #     first_level_train.select("item_idx").distinct().count()
        # )
        # self.first_level_user_len = (
        #     first_level_train.select("user_idx").distinct().count()
        # )

            first_level_train.cache()
            first_level_val.cache()
            self.cached_list.extend(
                [first_level_train, first_level_val]
            )
        else:
            first_level_train = log

        # 2. Transform user and item features if applicable
        if user_features is not None:
            user_features.cache()
            self.cached_list.append(user_features)

        if item_features is not None:
            item_features.cache()
            self.cached_list.append(item_features)

        with JobGroupWithMetrics(self._job_group_id, "item_features_transformer"):
            if not self.first_level_item_features_transformer.fitted:
                self.first_level_item_features_transformer.fit(item_features)

        with JobGroupWithMetrics(self._job_group_id, "user_features_transformer"):
            if not self.first_level_user_features_transformer.fitted:
                self.first_level_user_features_transformer.fit(user_features)

        first_level_item_features = cache_if_exists(
            self.first_level_item_features_transformer.transform(item_features)
        )
        first_level_user_features = cache_if_exists(
            self.first_level_user_features_transformer.transform(user_features)
        )

        first_level_user_features = first_level_user_features.filter(sf.col("user_idx") < self.first_level_user_len) \
            if first_level_user_features is not None else None

        first_level_item_features = first_level_item_features.filter(sf.col("item_idx") < self.first_level_item_len) \
            if first_level_item_features is not None else None

        self.first_level_user_features = first_level_user_features
        self.first_level_item_features = first_level_item_features


        # 3. Fit first level models
        # logger.info(f"first_level_train: {str(first_level_train.columns)}")
        # print(f"first_level_train: {str(first_level_train.columns)}")
        k = 100

        resume = True if self._experiment else False
        if not self._experiment and self._set_best_model:
            self._experiment = Experiment(
                first_level_val,
                {
                    NDCG(use_scala_udf=True): [k],
                },
            )

        MODEL_NAME_TO_FULL_MODEL_NAME = {
            "ItemKNN": "replay.models.knn.ItemKNN",
            "ALSWrap": "replay.models.als.ALSWrap",
            "Word2VecRec": "replay.models.word2vec.Word2VecRec",
            "SLIM": "replay.models.slim.SLIM"}  # TODO: refactor this part

        if self._is_trial:
            logger.info("Running trial model")
        if resume:
            logger.info("Resuming one-stage scenario")

        models_init_kwargs = {
            MODEL_NAME_TO_FULL_MODEL_NAME[f"{type(base_model).__name__}"]:
                base_model._init_args for base_model in self.first_level_models
        }
        first_level_models = self.first_level_models if not resume else self.first_level_models[1:]

        for base_model in first_level_models:
            # logger.debug("start fitting")
            # logger.debug(base_model.__dict__)

            with JobGroupWithMetrics(self._job_group_id, f"{type(base_model).__name__}._fit_wrap"):
                base_model._fit_wrap(
                    log=first_level_train,
                    user_features=first_level_user_features,
                    item_features=first_level_item_features,
                )

            if not self._set_best_model:  # in the case when one stage is a part of the two stage
                continue

            recs = base_model._predict(
                log=log,
                k=k,
                users=log.select("user_idx").distinct(),
                items=log.select("item_idx").distinct(),
                user_features=first_level_user_features,
                item_features=first_level_item_features,
                filter_seen_items=True
            )

            recs = get_top_k_recs(recs, k)

            recs_cnt = recs.count()
            logger.info(f"recs count: {recs_cnt}")
            mean_recs = recs.groupBy("user_idx").count().select("count").collect()[0][0]
            logger.debug(f"mean recs per user: {mean_recs}")

            # model_params = base_model._init_args

            # logger.info(f"model params : {model_params}")
            logger.debug("calculating metrics")
            self._experiment.add_result(f"{type(base_model).__name__}", recs)
            # saving model
            logger.debug("Saving model...")
            save(base_model, os.path.join("/tmp", f"model_{type(base_model).__name__}"), overwrite=True)
            logger.debug(f"Model saved")
            logger.info(f"Model {type(base_model).__name__} fitted")

            if self._is_trial:
                for dataframe in self.cached_list:
                    unpersist_if_exists(dataframe)
                return

            logger.info(f"Time left: {self.timer.time_left} sec")
            logger.debug(f"time_limit_exceeded: {self.timer.time_limit_exceeded()}")

            if self.timer.time_limit_exceeded():
                logger.info("Time limit exceed")
                if self._set_best_model:
                    logger.info("comparing of fitted models")
                    logger.info(self._experiment.results.sort_values(f"NDCG@{k}"))
                    best_model_name = self._experiment.results.sort_values(f"NDCG@{k}", ascending=False).index[0]
                    best_model_name = MODEL_NAME_TO_FULL_MODEL_NAME[best_model_name]
                    logger.info(f"best_model_name: {best_model_name}")
                    # load best model
                    self.best_model = load(os.path.join("/tmp", f"model_{type(base_model).__name__}"))
                    return
                else:
                    logger.debug("Exit from fitting...")
                    return

        if self._set_best_model:
            # Comparing models
            logger.info(self._experiment.results.sort_values(f"NDCG@{k}"))
            best_model_name = self._experiment.results.sort_values(f"NDCG@{k}", ascending=False).index[0]
            best_model_name = MODEL_NAME_TO_FULL_MODEL_NAME[best_model_name]
            logger.info(f"best_model_name: {best_model_name}")

            def get_model(best_model_name: str) -> BaseRecommender:

                module_name = ".".join(best_model_name.split('.')[:-1])
                class_name = best_model_name.split('.')[-1]
                module = importlib.import_module(module_name)
                clazz = getattr(module, class_name)
                base_model = cast(BaseRecommender, clazz(**models_init_kwargs[best_model_name]))

                return base_model

            best_model = get_model(best_model_name)
            logger.info(f"Fitting the best model: {best_model_name}")
            best_model._fit_wrap(
                log=log,
                user_features=first_level_user_features,
                item_features=first_level_item_features,
            )

            self.best_model = best_model

        unpersist_if_exists(first_level_user_features)
        unpersist_if_exists(first_level_item_features)

        logger.debug(self.first_level_models[0].__dict__)
        for dataframe in self.cached_list:
            unpersist_if_exists(dataframe)

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
        self._job_group_id = "1stage_predict"
        self.cached_list = []

        first_level_user_features = cache_if_exists(
            self.first_level_user_features_transformer.transform(user_features)
        )
        first_level_item_features = cache_if_exists(
            self.first_level_item_features_transformer.transform(item_features)
        )

        unpersist_if_exists(first_level_user_features)
        unpersist_if_exists(first_level_item_features)

        with JobGroupWithMetrics(self._job_group_id, f"{type(self.best_model).__name__}_predict"):
            predictions = self.best_model._predict(
                log=log, k=k, users=users, items=items,
                user_features=first_level_user_features,
                item_features=first_level_item_features,
                filter_seen_items=filter_seen_items)
            predictions = get_top_k_recs(predictions, k=k)

        logger.debug(f"predictions count: {predictions.count()}")
        logger.info(f"predictions.columns: {predictions.columns}")

        return predictions

    def fit_predict(
        self,
        log: AnyDataFrame,
        k: int,
        users: Optional[Union[AnyDataFrame, Iterable]] = None,
        items: Optional[Union[AnyDataFrame, Iterable]] = None,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """
        :param log: input DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param k: length of a recommendation list, must be smaller than the number of ``items``
        :param users: users to get recommendations for
        :param items: items to get recommendations for
        :param user_features: user features``[user_id]`` + feature columns
        :param item_features: item features``[item_id]`` + feature columns
        :param filter_seen_items: flag to removed seen items from recommendations
        :return: DataFrame ``[user_id, item_id, relevance]``
        """
        self.fit(log, user_features, item_features)
        return self.predict(
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            filter_seen_items,
        )

    @property
    def experiment(self):
        return self._experiment

    @staticmethod
    def _optimize_one_model(
        model: BaseRecommender,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[Dict[str, List[Any]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ):
        params = model.optimize(
            train,
            test,
            user_features,
            item_features,
            param_borders,
            criterion,
            k,
            budget,
            new_study,
        )
        return params

    # pylint: disable=too-many-arguments, too-many-locals
    def optimize(
        self,
        train: AnyDataFrame,
        test: AnyDataFrame,
        user_features: Optional[AnyDataFrame] = None,
        item_features: Optional[AnyDataFrame] = None,
        param_borders: Optional[List[Dict[str, List[Any]]]] = None,
        criterion: Metric = Precision(),
        k: int = 10,
        budget: int = 10,
        new_study: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Optimize first level models with optuna.

        :param train: train DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param test: test DataFrame ``[user_id, item_id, timestamp, relevance]``
        :param user_features: user features ``[user_id , timestamp]`` + feature columns
        :param item_features: item features``[item_id]`` + feature columns
        :param param_borders: list with param grids for first level models and a fallback model.
            Empty dict skips optimization for that model.
            Param grid is a dict ``{param: [low, high]}``.
        :param criterion: metric to optimize
        :param k: length of a recommendation list
        :param budget: number of points to train each model
        :param new_study: keep searching with previous study or start a new study
        :return: list of dicts of parameters
        """
        number_of_models = len(self.first_level_models)
        if self.fallback_model is not None:
            number_of_models += 1
        if number_of_models != len(param_borders):
            raise ValueError(
                "Provide search grid or None for every first level model"
            )

        first_level_user_features_tr = ToNumericFeatureTransformer()
        first_level_user_features = first_level_user_features_tr.fit_transform(
            user_features
        )
        first_level_item_features_tr = ToNumericFeatureTransformer()
        first_level_item_features = first_level_item_features_tr.fit_transform(
            item_features
        )

        first_level_user_features = cache_if_exists(first_level_user_features)
        first_level_item_features = cache_if_exists(first_level_item_features)

        params_found = []
        for i, model in enumerate(self.first_level_models):
            if param_borders[i] is None or (
                isinstance(param_borders[i], dict) and param_borders[i]
            ):
                self.logger.info(
                    "Optimizing first level model number %s, %s",
                    i,
                    model.__str__(),
                )
                params_found.append(
                    self._optimize_one_model(
                        model=model,
                        train=train,
                        test=test,
                        user_features=first_level_user_features,
                        item_features=first_level_item_features,
                        param_borders=param_borders[i],
                        criterion=criterion,
                        k=k,
                        budget=budget,
                        new_study=new_study,
                    )
                )
            else:
                params_found.append(None)

        if self.fallback_model is None or (
            isinstance(param_borders[-1], dict) and not param_borders[-1]
        ):
            return params_found, None

        self.logger.info("Optimizing fallback-model")
        fallback_params = self._optimize_one_model(
            model=self.fallback_model,
            train=train,
            test=test,
            user_features=first_level_user_features,
            item_features=first_level_item_features,
            param_borders=param_borders[-1],
            criterion=criterion,
            new_study=new_study,
        )
        unpersist_if_exists(first_level_item_features)
        unpersist_if_exists(first_level_user_features)
        return params_found, fallback_params

    def _get_nearest_items(self, items: DataFrame, metric: Optional[str] = None,
                           candidates: Optional[DataFrame] = None) -> Optional[DataFrame]:
        raise NotImplementedError("Unsupported method")
