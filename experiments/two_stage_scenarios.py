import functools
import importlib
import itertools
import logging
import os
from typing import Dict, cast, Optional, List, Union, Tuple

import mlflow
from pyspark.sql import functions as sf, SparkSession, DataFrame
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator
from replay.experiment import Experiment
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.metrics import MAP, NDCG, HitRate
from replay.model_handler import save, RandomRec, load
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario
from replay.scenarios.two_stages.reranker import LamaWrap, ReRanker
from replay.scenarios.two_stages.two_stages_scenario import get_first_level_model_features
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info, join_with_col_renaming, get_top_k_recs, join_or_return, unpersist_if_exists, \
    cache_if_exists

logger = logging.getLogger(__name__)


def main():
    CORES = 4
    PARTS_NUM = CORES * 3
    K = 5
    SEED = 22

    spark = (
        SparkSession
        .builder
        .master(f"local[{CORES}]")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.driver.memory", "64g")
        .config("spark.executor.memory", "64g")
        .config("spark.default.parallelism", f"{PARTS_NUM}")
        .config("spark.sql.shuffle.partitions", f"{PARTS_NUM}")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.local.dir", "/tmp")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    data = MovieLens("100k")
    preparator = DataPreparator()

    log = preparator.transform(
        columns_mapping={"user_id": "user_id", "item_id": "item_id", "relevance": "rating", "timestamp": "timestamp"},
        data=data.ratings
    ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")

    print(get_log_info(log))

    log = log.repartition(CORES).cache()
    log.write.mode('overwrite').format('noop').save()

    only_positives_log = log.filter(sf.col('relevance') >= 3).withColumn('relevance', sf.lit(1))
    print(get_log_info(only_positives_log))

    # train/test split ml
    train_spl = DateSplitter(
        test_start=0.2,
        drop_cold_items=True,
        drop_cold_users=True,
    )

    train, test = train_spl.split(only_positives_log)
    print('train info:\n', get_log_info(train))
    print('test info:\n', get_log_info(test))

    train = train.cache()
    test = test.cache()
    train.write.mode('overwrite').format('noop').save()
    test.write.mode('overwrite').format('noop').save()

    scenario = TwoStagesScenario(
        train_splitter=UserSplitter(item_test_size=0.5, shuffle=True, seed=42),
        second_model_params={"general_params": {"use_algos": [["lgb", "linear_l2"]]}}
    )
                                 # first_level_models=ALSWrap(),
                                 # fallback_model=PopRec(),
                                 # use_first_level_models_feat=False,
                                 # second_model_params=None,
                                 # second_model_config_path=None,
                                 # num_negatives=100,
                                 # negatives_type='first_level',
                                 # use_generated_features=False,
                                 # user_cat_features_list=None,
                                 # item_cat_features_list=None,
                                 # custom_features_processor=None,
                                 # seed=SEED)

    scenario.fit(log=train, user_features=None, item_features=None)


class EmptyWrap(ReRanker):
    def __init__(self, second_level_train_path: Optional[str] = None):
        self.second_level_train_path = second_level_train_path

    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        if self.second_level_train_path is not None:
            data.write.parquet(self.second_level_train_path)

    def predict(self, data, k) -> DataFrame:
        return data


class CustomTwoStageScenario(TwoStagesScenario):
    def __init__(self,
                 predefined_first_level_train: Optional[DataFrame] = None,
                 predefined_second_level_positives: Optional[DataFrame] = None,
                 empty_second_stage_params: Optional[Dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.predefined_first_level_train = predefined_first_level_train
        self.predefined_second_level_positives = predefined_second_level_positives

        if empty_second_stage_params is not None:
            self.second_stage_model = EmptyWrap(**empty_second_stage_params)

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self.predefined_first_level_train is not None and self.predefined_second_level_positives is not None:
            return self.predefined_first_level_train, self.predefined_second_level_positives
        return super()._split_data(log)


def _get_spark_session() -> SparkSession:
    return SparkSession.builder.master("local[6]").getOrCreate()


def _get_scenario(model_class_name: str,
                  model_kwargs: Dict,
                  first_level_train: DataFrame,
                  second_level_positives: DataFrame,
                  empty_second_stage_params: Optional[Dict] = None,) -> TwoStagesScenario:
    module_name = ".".join(model_class_name.split('.')[:-1])
    class_name = model_class_name.split('.')[-1]
    module = importlib.import_module(module_name)
    clazz = getattr(module, class_name)
    base_model = cast(BaseRecommender, clazz(**model_kwargs))

    scenario = CustomTwoStageScenario(
        predefined_first_level_train=first_level_train,
        predefined_second_level_positives=second_level_positives,
        empty_second_stage_params=empty_second_stage_params,
        train_splitter=UserSplitter(item_test_size=0.5, shuffle=True, seed=42),
        first_level_models=base_model,
        second_model_params={"general_params": {"use_algos": [["lgb", "linear_l2"]]}}
    )

    return scenario


def _estimate_and_report_metrics(model_name: str, test: DataFrame, recs: DataFrame):
    K_list_metrics = [5, 10]

    e = Experiment(
        test,
        {
            MAP(): K_list_metrics,
            NDCG(): K_list_metrics,
            HitRate(): K_list_metrics,
        },
    )
    e.add_result(model_name, recs)

    for k in K_list_metrics:
        mlflow.log_metric(
            "NDCG.{}".format(k), e.results.at[model_name, "NDCG@{}".format(k)]
        )
        mlflow.log_metric(
            "MAP.{}".format(k), e.results.at[model_name, "MAP@{}".format(k)]
        )
        mlflow.log_metric(
            "HitRate.{}".format(k),
            e.results.at[model_name, "HitRate@{}".format(k)],
        )


def _predict_pairs_with_first_level_model(
        self,
        model: BaseRecommender,
        log: DataFrame,
        pairs: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ):
        """
        Get relevance for selected user-item pairs.
        """
        if not model.can_predict_cold_items:
            log, pairs, item_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("item_idx") < self.first_level_item_len,
                )
                for df in [log, pairs, item_features]
            ]
        if not model.can_predict_cold_users:
            log, pairs, user_features = [
                self._filter_or_return(
                    dataframe=df,
                    condition=sf.col("user_idx") < self.first_level_user_len,
                )
                for df in [log, pairs, user_features]
            ]

        return model._predict_pairs(
            pairs=pairs,
            log=log,
            user_features=user_features,
            item_features=item_features,
        )


def dataset_splitting(log_path: str, split_base_path: str, cores: int):
    spark = _get_spark_session()
    data = spark.read.parquet(log_path)
    scenario = _get_scenario()

    # splitting on train and test
    preparator = DataPreparator()

    log = preparator.transform(
        columns_mapping={"user_id": "user_id", "item_id": "item_id", "relevance": "rating", "timestamp": "timestamp"},
        data=data.ratings
    ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")

    print(get_log_info(log))

    log = log.repartition(cores).cache()
    log.write.mode('overwrite').format('noop').save()

    only_positives_log = log.filter(sf.col('relevance') >= 3).withColumn('relevance', sf.lit(1))
    logger.info(get_log_info(only_positives_log))

    # train/test split ml
    train_spl = DateSplitter(
        test_start=0.2,
        drop_cold_items=True,
        drop_cold_users=True,
    )

    train, test = train_spl.split(only_positives_log)
    logger.info('train info:\n', get_log_info(train))
    logger.info('test info:\n', get_log_info(test))

    # split on first and second levels
    first_level_train, second_level_positives = scenario._split_data(train)

    # writing data
    os.makedirs(split_base_path, exist_ok=True)

    train.write.parquet(os.path.join(split_base_path, "train.parquet"))
    test.write.parquet(os.path.join(split_base_path, "test.parquet"))
    first_level_train.write.parquet(os.path.join(split_base_path, "first_level_train.parquet"))
    second_level_positives.write.parquet(os.path.join(split_base_path, "second_level_positives.parquet"))


def first_level_fitting_2(split_base_path: str,
                          model_class_name: str,
                          model_kwargs: Dict,
                          model_path: str,
                          k: int,
                          item_features_path: Optional[str] = None,
                          user_features_path: Optional[str] = None):
    spark = _get_spark_session()

    train = spark.read.parquet(os.path.join(split_base_path, "train.parquet"))
    test = spark.read.parquet(os.path.join(split_base_path, "test.parquet"))
    first_level_train = spark.read.parquet(os.path.join(split_base_path, "first_level_train.parquet"))
    second_level_positives = spark.read.parquet(os.path.join(split_base_path, "second_level_positives.parquet"))

    item_features = spark.read.parquet(item_features_path) if item_features_path is not None else None
    user_features = spark.read.parquet(user_features_path) if user_features_path is not None else None

    # 1. replaces train splitting with pre-splitted data
    # 2. dumps the second level train dataset
    scenario = _get_scenario(
        model_class_name=model_class_name,
        model_kwargs=model_kwargs,
        first_level_train=first_level_train,
        second_level_positives=second_level_positives,
        empty_second_stage_params={
            "second_level_train_path": os.path.join(
                split_base_path,
                f"second_level_train_{model_class_name.replace('.', '__')}.parquet"
            )
        }
    )

    scenario.fit(log=train, user_features=user_features, item_features=item_features)

    # saving the trained first-level model
    model = scenario.first_level_models[0]
    save(model, model_path)

    # because of EmptyWrap this is still predictions from the first level
    # (though with all prepared features required on the second level)
    recs = scenario.predict(
        log=train,
        k=k,
        users=test.select('user_idx').distinct(),
        filter_seen_items=True,
        user_features=user_features,
        item_features=item_features
    )

    recs.write.parquet(
        os.path.join(split_base_path, f"second_level_train_{model_class_name.replace('.', '__')}.parquet")
    )


def first_level_fitting(first_level_train_path: str,
                        test_path: str,
                        model_class_name: str,
                        model_kwargs: Dict,
                        model_path: str,
                        k: int,
                        item_features_path: str,
                        user_features_path: str):
    # get session and read data
    spark = _get_spark_session()
    first_level_train = spark.read.parquet(first_level_train_path)
    test = spark.read.parquet(test_path)

    # instantiate a model
    module_name = ".".join(model_class_name.split('.')[:-1])
    class_name = model_class_name.split('.')[-1]
    module = importlib.import_module(module_name)
    clazz = getattr(module, class_name)
    base_model = cast(BaseRecommender, clazz(**model_kwargs))


    self.first_level_item_features_transformer.fit(item_features)
    self.first_level_user_features_transformer.fit(user_features)


    first_level_user_features = first_level_user_features.filter(sf.col("user_idx") < self.first_level_user_len) \
        if first_level_user_features is not None else None

    first_level_item_features = first_level_item_features.filter(sf.col("item_idx") < self.first_level_item_len) \
        if first_level_item_features is not None else None

    base_model._fit_wrap(
        log=first_level_train,
        user_features=first_level_user_features,
        item_features=first_level_item_features,
    )

    logger.info(f"Saving model to: {model_path}")
    save(base_model, path=model_path, overwrite=True)

    # predict and report metrics
    recs = base_model._predict(
        log=first_level_train,
        k=k,
        users=test.select("user_idx").distinct(),
        user_features=first_level_user_features,
        item_features=first_level_item_features,
        filter_seen_items=True
    )
    _estimate_and_report_metrics(model_class_name, test, recs)


def negative_sampling(
        num_negatives: int,
        first_level_train_path: str,
        second_level_positives_path: str,
        second_level_train_path: str,
        first_level_model_path: Optional[str] = None
):
    logger.info("Generate negative examples")

    spark = _get_spark_session()

    log = spark.read.parquet(first_level_train_path)
    second_level_positive = spark.read.parquet(second_level_positives_path)
    k = num_negatives

    model = (
        load(first_level_model_path)
        if first_level_model_path is not None
        else RandomRec(seed=42)
    )

    # first_level_candidates = self._get_first_level_candidates(
    #     model=negatives_source,
    #     log=first_level_train,
    #     k=self.num_negatives,
    #     users=log.select("user_idx").distinct(),
    #     items=log.select("item_idx").distinct(),
    #     user_features=first_level_user_features,
    #     item_features=first_level_item_features,
    #     log_to_filter=first_level_train,
    # )

    if not model.can_predict_cold_items:
        log, items, item_features = [
            self._filter_or_return(
                dataframe=df,
                condition=sf.col("item_idx") < self.first_level_item_len,
            )
            for df in [log, items, item_features]
        ]
    if not model.can_predict_cold_users:
        log, users, user_features = [
            self._filter_or_return(
                dataframe=df,
                condition=sf.col("user_idx") < self.first_level_user_len,
            )
            for df in [log, users, user_features]
        ]

    log_to_filter_cached = join_with_col_renaming(
        left=log,
        right=users,
        on_col_name="user_idx",
    ).cache()
    max_positives_to_filter = 0

    if log_to_filter_cached.count() > 0:
        max_positives_to_filter = (
            log_to_filter_cached.groupBy("user_idx")
                .agg(sf.count("item_idx").alias("num_positives"))
                .select(sf.max("num_positives"))
                .collect()[0][0]
        )

    pred = model._predict(
        log,
        k=k + max_positives_to_filter,
        users=users,
        items=items,
        user_features=user_features,
        item_features=item_features,
        filter_seen_items=False,
    )

    pred = pred.join(
        log_to_filter_cached.select("user_idx", "item_idx"),
        on=["user_idx", "item_idx"],
        how="anti",
    ).drop("user", "item")

    log_to_filter_cached.unpersist()

    first_level_candidates = get_top_k_recs(pred, k).select("user_idx", "item_idx")

    # here we generate full second_level_train

    second_level_train = (
        first_level_candidates.join(
            second_level_positive.select(
                "user_idx", "item_idx"
            ).withColumn("target", sf.lit(1.0)),
            on=["user_idx", "item_idx"],
            how="left",
        ).fillna(0.0, subset="target")
    ).cache()

    logger.info(
        "Distribution of classes in second-level train dataset:/n %s",
        (
            second_level_train.groupBy("target")
                .agg(sf.count(sf.col("target")).alias("count_for_class"))
                .take(2)
        ),
    )

    second_level_train.write.parquet(second_level_train_path)


def predict_for_second_level_train(
        model_idx: str,
        model_path: str,
        first_level_train_path: str,
        second_level_train_path: str,
        second_level_train_with_features_path: str
):
    logger.info("Adding features to second-level train dataset")

    spark = _get_spark_session()


    # TODO: only partially
    # second_level_train_to_convert = self._add_features_for_second_level(
    #     log_to_add_features=second_level_train,
    #     log_for_first_level_models=first_level_train,
    #     user_features=user_features,
    #     item_features=item_features,
    # ).cache()

    full_second_level_train = spark.read.parquet(second_level_train_path)
    log_for_first_level_models = spark.read.parquet(first_level_train_path)
    model = load(model_path)

    # TODO: may be moved to a separate task
    features_processor = HistoryBasedFeaturesProcessor(
        user_cat_features_list=user_cat_features_list,
        item_cat_features_list=item_cat_features_list,
    )
    features_processor.fit(
        log=log_for_first_level_models,
        user_features=user_features,
        item_features=item_features
    )

    # TODO: move transformers into a separate task?
    first_level_item_features_cached = cache_if_exists(
        self.first_level_item_features_transformer.transform(item_features)
    )
    first_level_user_features_cached = cache_if_exists(
        self.first_level_user_features_transformer.transform(user_features)
    )

    pairs = full_second_level_train.select("user_idx", "item_idx")

    current_pred = _predict_pairs_with_first_level_model(
        model=model,
        log=log_for_first_level_models,
        pairs=pairs,
        user_features=first_level_user_features_cached,
        item_features=first_level_item_features_cached,
    ).withColumnRenamed("relevance", f"rel_{model_idx}_{model}")
    full_second_level_train = full_second_level_train.join(
        sf.broadcast(current_pred),
        on=["user_idx", "item_idx"],
        how="left",
    )

    features = get_first_level_model_features(
        model=model,
        pairs=full_second_level_train.select(
            "user_idx", "item_idx"
        ),
        user_features=first_level_user_features_cached,
        item_features=first_level_item_features_cached,
        prefix=f"m_{model_idx}",
    )
    full_second_level_train = join_with_col_renaming(
        left=full_second_level_train,
        right=features,
        on_col_name=["user_idx", "item_idx"],
        how="left",
    )

    unpersist_if_exists(first_level_user_features_cached)
    unpersist_if_exists(first_level_item_features_cached)

    full_second_level_train_cached = full_second_level_train.fillna(
        0
    ).cache()

    logger.info("Adding features from the dataset")
    full_second_level_train = join_or_return(
        full_second_level_train_cached,
        user_features,
        on="user_idx",
        how="left",
    )
    full_second_level_train = join_or_return(
        full_second_level_train,
        item_features,
        on="item_idx",
        how="left",
    )

    logger.info("Adding generated features")
    full_second_level_train = features_processor.transform(
        log=full_second_level_train
    )

    logger.info(
        "Columns at second level: %s",
        " ".join(full_second_level_train.columns),
    )

    full_second_level_train.write.parquet(second_level_train_with_features_path)

    full_second_level_train_cached.unpersist()


def combine_datasets_for_second_level(second_level_trains_paths: List[str], final_second_level_train: str):
    assert len(second_level_trains_paths) > 0, "Cannot work with empty sequence of paths"

    spark = _get_spark_session()

    dfs = [spark.read.parquet(path) for path in second_level_trains_paths]
    df = functools.reduce(lambda acc, x: acc.join(x, on=["user_idx", "item_idx"]), dfs)

    # TODO: check the resulting dataframe for correctness (no nones in any field)

    df.write.parquet(final_second_level_train)


def second_level_fitting(final_second_level_train: str,
                         second_model_type: str = "lama",
                         second_model_params: Optional[Union[Dict, str]] = None,
                         second_model_config_path: Optional[str] = None):
    spark = _get_spark_session()

    if second_model_type == "lama":
        second_stage_model = LamaWrap(params=second_model_params, config_path=second_model_config_path)
    else:
        raise RuntimeError(f"Currently supported model types: {['lama']}, but received {second_model_type}")

    second_level_train = spark.read.parquet(final_second_level_train)
    second_stage_model.fit(second_level_train)

    save(second_stage_model)

    # predict and report metrics
    recs = second_stage_model.predict(
        log=first_level_train,
        k=k,
        users=test.select("user_idx").distinct(),
        user_features=first_level_user_features,
        item_features=first_level_item_features,
        filter_seen_items=True
    )
    _estimate_and_report_metrics(model_class_name, test, recs)

    pass


def combine_second_level_results():
    # TODO: combine all results (test prediction quality) into a single table
    pass


if __name__ == "__main__":
    main()
