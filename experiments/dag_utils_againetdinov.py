import functools
import importlib
import itertools
import logging
import os
import pickle
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, cast, Optional, List, Union, Tuple

import mlflow
from pyspark.sql import functions as sf, SparkSession, DataFrame, Window

sys.path.insert(0, "/opt/airflow/dags/againetdinov_packages")
import replay
from dag_entities_againetdinov import (
    ArtifactPaths,
    DEFAULT_CPU,
    DEFAULT_MEMORY,
    TASK_CONFIG_FILENAME_ENV_VAR,
)
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.model_handler import save, Splitter, load, ALSWrap
from replay.models import PopRec
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario
from replay.scenarios.two_stages.reranker import LamaWrap, ReRanker
from replay.scenarios.two_stages.slama_reranker import SlamaWrap
from replay.session_handler import State
from replay.splitters import UserSplitter
from replay.utils import (
    get_log_info,
    save_transformer,
    log_exec_timer,
    do_path_exists,
    list_folder,
)

from synapse.ml.lightgbm import LightGBMClassifier
from pyspark.ml.feature import VectorAssembler

logger = logging.getLogger("airflow.task")


def filter_seen_custom(
    recs: DataFrame, log: DataFrame, k: int, users: DataFrame
):
    """
    Filter seen items (presented in log) out of the users' recommendations.
    For each user return from `k` to `k + number of seen by user` recommendations.
    """
    users_log = log.join(users, on="user_idx")

    # filter recommendations presented in interactions log
    recs = recs.join(
        users_log.withColumnRenamed("item_idx", "item")
        .withColumnRenamed("user_idx", "user")
        .select("user", "item"),
        on=(sf.col("user_idx") == sf.col("user"))
        & (sf.col("item_idx") == sf.col("item")),
        how="anti",
    ).drop("user", "item")

    # crop recommendations to first k + max_seen items for each user
    recs = recs.withColumn(
        "temp_rank",
        sf.row_number().over(
            Window.partitionBy("user_idx").orderBy(sf.col("relevance").desc())
        ),
    ).filter(sf.col("temp_rank") <= sf.lit(k))

    return recs


@dataclass(frozen=True)
class FirstLevelModelFiles:
    model_name: str
    train_path: str
    predict_path: str
    model_path: str


def _get_bucketing_key(default: str = "user_idx") -> str:
    key = os.environ.get("REPLAY_BUCKETING_KEY", default)
    assert key in ["user_idx", "item_idx"]
    return key


def _make_bucketing(df: DataFrame, bucketing_key: str, name: str) -> DataFrame:
    spark = State().session
    table_name = f"bucketed_df_{spark.sparkContext.applicationId.replace('-', '__')}_{name}"
    partition_num = spark.sparkContext.defaultParallelism

    logger.info(
        f"Bucketing of the dataset: name={name}, table_name={table_name}"
    )

    (
        df.repartition(partition_num, bucketing_key)
        .write.mode("overwrite")
        .bucketBy(partition_num, bucketing_key)
        .sortBy(bucketing_key)
        .saveAsTable(table_name, format="parquet")
    )

    logger.info(
        f"Bucketing of the dataset is finished: name={name}, table_name={table_name}"
    )

    return spark.table(table_name)


class EmptyWrap(ReRanker):
    @classmethod
    def load(cls, path: str, spark: Optional[SparkSession] = None):
        spark = spark or cls._get_spark_session()
        row = spark.read.parquet(path).first().asDict()
        cls._validate_classname(row["classname"])
        return EmptyWrap()

    def __init__(self, dump_path: Optional[str] = None):
        self.dump_path = dump_path

    def save(
        self,
        path: str,
        overwrite: bool = False,
        spark: Optional[SparkSession] = None,
    ):
        spark = spark or self._get_spark_session()
        spark.createDataFrame(
            [{"data": "", "classname": self.get_classname()}]
        ).write.parquet(path, mode="overwrite" if overwrite else "error")

    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        if self.dump_path is not None:
            data.write.parquet(self.dump_path)

    def predict(self, data, k) -> DataFrame:
        return data


class EmptyRecommender(BaseRecommender):
    @property
    def _init_args(self):
        return dict()

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        pass

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
        return log.select(
            "user_idx", "item_idx", sf.lit(0.0).alias("relevance")
        )

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: Optional[str] = None,
        candidates: Optional[DataFrame] = None,
    ) -> Optional[DataFrame]:
        return None


class PartialTwoStageScenario(TwoStagesScenario):
    def __init__(
        self,
        base_path: str,
        first_level_train_path: str,
        second_level_positives_path: str,
        first_level_train_opt_path: str = None,
        first_level_val_opt_path: str = None,
        second_level_train_path: Optional[str] = None,
        train_splitter: Splitter = UserSplitter(
            item_test_size=0.5, shuffle=True, seed=42
        ),
        first_level_models: Optional[BaseRecommender] = None,
        fallback_model: Optional[BaseRecommender] = PopRec(),
        use_first_level_models_feat: Union[List[bool], bool] = True,
        num_negatives: int = 100,
        negatives_type: str = "first_level",
        use_generated_features: bool = True,
        first_level_user_features_transformer=None,
        first_level_item_features_transformer=None,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
        custom_features_processor: Optional[
            HistoryBasedFeaturesProcessor
        ] = None,
        seed: int = 123,
        presplitted_data: bool = False,
    ):
        if first_level_models is None:
            first_level_models = ALSWrap(rank=100, seed=42)

        super().__init__(
            train_splitter=train_splitter,
            first_level_models=first_level_models,
            fallback_model=fallback_model,
            use_first_level_models_feat=use_first_level_models_feat,
            second_model_params=None,
            second_model_config_path=None,
            num_negatives=num_negatives,
            negatives_type=negatives_type,
            use_generated_features=use_generated_features,
            user_cat_features_list=user_cat_features_list,
            item_cat_features_list=item_cat_features_list,
            custom_features_processor=custom_features_processor,
            seed=seed,
        )
        self.second_stage_model = EmptyWrap(dump_path=second_level_train_path)
        self._base_path = base_path
        self._first_level_train_path = first_level_train_path
        self._second_level_positives_path = second_level_positives_path
        self._first_level_train_opt_path = first_level_train_opt_path
        self._first_level_val_opt_path = first_level_val_opt_path
        self._presplitted_data = presplitted_data
        if first_level_item_features_transformer:
            self.first_level_item_features_transformer = (
                first_level_item_features_transformer
            )
        if first_level_user_features_transformer:
            self.first_level_user_features_transformer = (
                first_level_user_features_transformer
            )

    @property
    def _init_args(self):
        return {
            **super()._init_args,
            "base_path": self._base_path,
            "first_level_train_path": self._first_level_train_path,
            "second_level_positives_path": self._second_level_positives_path,
            "presplitted_data": self._presplitted_data,
        }

    @property
    def first_level_models_relevance_columns(self) -> List[str]:
        return [
            f"rel_{idx}_{model}"
            for idx, model in enumerate(self.first_level_models)
        ]

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self._presplitted_data:
            spark = log.sql_ctx.sparkSession
            first_level_train = _make_bucketing(
                spark.read.parquet(self._first_level_train_path).repartition(
                    spark.sparkContext.defaultParallelism
                ),
                bucketing_key=_get_bucketing_key(default="user_idx"),
                name="first_level_train",
            )
            second_level_positive = _make_bucketing(
                spark.read.parquet(
                    self._second_level_positives_path
                ).repartition(spark.sparkContext.defaultParallelism),
                bucketing_key=_get_bucketing_key(default="user_idx"),
                name="second_level_positive",
            )

            return first_level_train, second_level_positive

        first_level_train, second_level_positive = super()._split_data(log)

        first_level_train.write.parquet(
            self._first_level_train_path, mode="overwrite"
        )
        second_level_positive.write.parquet(
            self._second_level_positives_path, mode="overwrite"
        )

        return first_level_train, second_level_positive

    def _split_optimize(self, first_level_train: DataFrame) -> None:
        first_level_train_opt, first_level_val_opt = self.train_splitter.split(
            first_level_train
        )

        logger.info(
            "first_level_train_opt info: %s",
            get_log_info(first_level_train_opt),
        )
        logger.info(
            "first_level_val_opt info: %s", get_log_info(first_level_val_opt)
        )

        first_level_train_opt.write.parquet(
            self._first_level_train_opt_path, mode="overwrite"
        )
        first_level_val_opt.write.parquet(
            self._first_level_val_opt_path, mode="overwrite"
        )

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: Optional[str] = None,
        candidates: Optional[DataFrame] = None,
    ) -> Optional[DataFrame]:
        raise NotImplementedError("Unsupported method")


class RefitableTwoStageScenario(TwoStagesScenario):
    def __init__(
        self,
        base_path: str,
        train_splitter: Splitter = UserSplitter(
            item_test_size=0.5, shuffle=True, seed=42
        ),
        first_level_models: Optional[
            Union[List[BaseRecommender], BaseRecommender]
        ] = None,
        fallback_model: Optional[BaseRecommender] = PopRec(),
        use_first_level_models_feat: Union[List[bool], bool] = True,
        num_negatives: int = 100,
        negatives_type: str = "first_level",
        use_generated_features: bool = True,
        user_cat_features_list: Optional[List] = None,
        item_cat_features_list: Optional[List] = None,
        custom_features_processor: Optional[
            HistoryBasedFeaturesProcessor
        ] = None,
        seed: int = 123,
    ):
        if first_level_models is None:
            first_level_models = ALSWrap(rank=100, seed=42)

        super().__init__(
            train_splitter=train_splitter,
            first_level_models=first_level_models,
            fallback_model=fallback_model,
            use_first_level_models_feat=use_first_level_models_feat,
            second_model_params=None,
            second_model_config_path=None,
            num_negatives=num_negatives,
            negatives_type=negatives_type,
            use_generated_features=use_generated_features,
            user_cat_features_list=user_cat_features_list,
            item_cat_features_list=item_cat_features_list,
            custom_features_processor=custom_features_processor,
            seed=seed,
        )
        self.second_stage_model = EmptyWrap()
        self._base_path = base_path
        self._first_level_train_path = os.path.join(
            base_path, "first_level_train.parquet"
        )
        self._second_level_positive_path = os.path.join(
            base_path, "second_level_positive.parquet"
        )
        self._first_level_candidates_path = os.path.join(
            base_path, "first_level_candidates.parquet"
        )

        self._are_split_data_dumped = False
        self._are_candidates_dumped = False

        self._return_candidates_with_positives = False

    @property
    def _init_args(self):
        return {**super()._init_args, "base_path": self._base_path}

    @property
    def candidates_with_positives(self) -> bool:
        return self._return_candidates_with_positives

    @property
    def first_level_models_relevance_columns(self) -> List[str]:
        return [
            f"rel_{idx}_{model}"
            for idx, model in enumerate(self.first_level_models)
        ]

    @candidates_with_positives.setter
    def candidates_with_positives(self, val: bool):
        self._return_candidates_with_positives = val

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self._are_split_data_dumped:
            spark = log.sql_ctx.sparkSession
            return (
                spark.read.parquet(self._first_level_train_path),
                spark.read.parquet(self._second_level_positive_path),
            )

        first_level_train, second_level_positive = super()._split_data(log)

        first_level_train.write.parquet(self._first_level_train_path)
        second_level_positive.write.parquet(self._second_level_positive_path)
        self._are_split_data_dumped = True

        return first_level_train, second_level_positive

    def _get_first_level_candidates(
        self,
        model: BaseRecommender,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
        log_to_filter: DataFrame,
    ) -> DataFrame:
        if self._are_candidates_dumped:
            spark = log.sql_ctx.sparkSession
            first_level_candidates = spark.read.parquet(
                self._first_level_candidates_path
            )

            if self.candidates_with_positives:
                assert (
                    self._are_split_data_dumped
                ), "Cannot do that without dumped second level positives"
                second_level_positive = spark.read.parquet(
                    self._second_level_positive_path
                )
                first_level_candidates = first_level_candidates.join(
                    second_level_positive.select(
                        "user_idx", "item_idx"
                    ).withColumn("target", sf.lit(1.0)),
                    on=["user_idx", "item_idx"],
                    how="left",
                ).fillna(0.0, subset="target")

            return first_level_candidates

        first_level_candidates = super()._get_first_level_candidates(
            model,
            log,
            k,
            users,
            items,
            user_features,
            item_features,
            log_to_filter,
        )

        first_level_candidates.write.parquet(self._first_level_candidates_path)
        self._are_candidates_dumped = True

        return first_level_candidates

    def _add_features_for_second_level(
        self,
        log_to_add_features: DataFrame,
        log_for_first_level_models: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ) -> DataFrame:
        candidates_with_features = super()._add_features_for_second_level(
            log_to_add_features,
            log_for_first_level_models,
            user_features,
            item_features,
        )

        if self._are_candidates_dumped and self.candidates_with_positives:
            assert (
                self._are_split_data_dumped
            ), "Cannot do that without dumped second level positives"
            spark = log_to_add_features.sql_ctx.sparkSession
            first_level_candidates = spark.read.parquet(
                self._first_level_candidates_path
            )
            second_level_positive = spark.read.parquet(
                self._second_level_positive_path
            )
            candidates_with_target = (
                first_level_candidates.join(
                    second_level_positive.select(
                        "user_idx", "item_idx"
                    ).withColumn("target", sf.lit(1.0)),
                    on=["user_idx", "item_idx"],
                    how="left",
                )
                .fillna(0.0, subset="target")
                .select("user_idx", "item_idx", "target")
            )

            candidates_with_features = candidates_with_features.join(
                candidates_with_target, on=["user_idx", "item_idx"], how="left"
            )

        return candidates_with_features

    def _get_nearest_items(
        self,
        items: DataFrame,
        metric: Optional[str] = None,
        candidates: Optional[DataFrame] = None,
    ) -> Optional[DataFrame]:
        raise NotImplementedError("Unsupported method")

    def _save_model(self, path: str):
        super()._save_model(path)
        spark = State().session
        spark.createDataFrame(
            [
                {
                    "_are_candidates_dumped": self._are_candidates_dumped,
                    "_are_split_data_dumped": self._are_split_data_dumped,
                }
            ]
        ).write.parquet(os.path.join(path, "data_refittable.parquet"))

    def _load_model(self, path: str):
        super()._load_model(path)
        spark = State().session
        row = (
            spark.read.parquet(os.path.join(path, "data_refittable.parquet"))
            .first()
            .asDict()
        )
        self._are_candidates_dumped = row["_are_candidates_dumped"]
        self._are_split_data_dumped = row["_are_split_data_dumped"]


def load_model(path: str):
    setattr(replay.model_handler, "EmptyRecommender", EmptyRecommender)
    setattr(replay.model_handler, "EmptyWrap", EmptyWrap)
    setattr(
        replay.model_handler,
        "RefitableTwoStageScenario",
        RefitableTwoStageScenario,
    )
    setattr(
        replay.model_handler,
        "PartialTwoStageScenario",
        PartialTwoStageScenario,
    )
    return load(path)


def save_model(model: BaseRecommender, path: str, overwrite: bool = False):
    save(model, path, overwrite)


def get_cluster_session():
    assert os.environ.get("SCRIPT_ENV", "local") == "cluster"
    return SparkSession.builder.getOrCreate()


@contextmanager
def _init_spark_session(
    cpu: int = DEFAULT_CPU,
    memory: int = DEFAULT_MEMORY,
    mem_coeff: float = 0.9,
) -> SparkSession:
    if os.environ.get("SCRIPT_ENV", "local") == "cluster":
        spark = SparkSession.builder.getOrCreate()
    else:
        jars = [
            os.environ.get(
                "REPLAY_JAR_PATH",
                "/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar",
            ),  #'../../scala/target/scala-2.12/replay_2.12-0.1.jar'
            os.environ.get(
                "SLAMA_JAR_PATH",
                "/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar",
            ),  #'../../../LightAutoML/jars/spark-lightautoml_2.12-0.1.1.jar'
        ]

        real_memory = int(memory * mem_coeff)
        max_result_size = max(int(real_memory * 0.8), 1)

        spark = (
            SparkSession.builder.config("spark.jars", ",".join(jars))
            .config(
                "spark.jars.packages",
                "com.microsoft.azure:synapseml_2.12:0.9.5",
            )
            .config(
                "spark.jars.repositories",
                "https://mmlspark.azureedge.net/maven",
            )
            .config(
                "spark.driver.extraJavaOptions",
                "-Dio.netty.tryReflectionSetAccessible=true",
            )
            .config("spark.sql.shuffle.partitions", str(cpu * 3))
            .config("spark.default.parallelism", str(cpu * 3))
            .config("spark.driver.maxResultSize", f"{max_result_size}g")
            .config("spark.driver.memory", f"{real_memory}g")
            .config("spark.executor.memory", f"{real_memory}g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.warehouse.dir", "/tmp/current-spark-warehouse")
            .config("spark.kryoserializer.buffer.max", "256m")
            .master(f"local[{cpu}]")
            .getOrCreate()
        )

    yield spark

    if bool(int(os.environ.get("INIT_SPARK_SESSION_STOP_SESSION", "1"))):
        spark.stop()


def _get_model(
    artifacts: ArtifactPaths, model_class_name: str, model_kwargs: Dict
) -> BaseRecommender:
    module_name = ".".join(model_class_name.split(".")[:-1])
    class_name = model_class_name.split(".")[-1]
    module = importlib.import_module(module_name)
    clazz = getattr(module, class_name)

    # if class_name in ['ALSWrap', 'Word2VecRec'] and 'nmslib_hnsw_params' in model_kwargs:
    #     model_kwargs['nmslib_hnsw_params']["index_path"] = artifacts.hnsw_index_path(model_class_name)

    base_model = cast(BaseRecommender, clazz(**model_kwargs))

    return base_model


# this is @task
def _combine_datasets_for_second_level(
    partial_datasets_paths: List[str],
    combined_dataset_path: str,
    spark: SparkSession,
):
    assert (
        len(partial_datasets_paths) > 0
    ), "Cannot work with empty sequence of paths"

    dfs = [spark.read.parquet(path) for path in partial_datasets_paths]

    # TODO: restore this check later
    # check that all dataframes have the same size
    # lengths = set(df.count() for df in dfs)
    # assert len(lengths) == 1 and next(iter(lengths)) > 0, f"Invalid lengths of datasets: {lengths}"

    # strip all duplicate columns
    no_duplicate_columns_dfs = [
        dfs[0],
        *(
            df.select(
                "user_idx",
                "item_idx",
                *set(df.columns).difference(set(dfs[0].columns)),
            )
            for df in dfs[1:]
        ),
    ]

    feature_columns = set(
        c
        for df in dfs[1:]
        for c in set(df.columns).symmetric_difference(set(dfs[0].columns))
    )

    df = functools.reduce(
        lambda acc, x: acc.join(x, on=["user_idx", "item_idx"]),
        no_duplicate_columns_dfs,
    )
    df = df.cache()

    # TODO: restore it later
    # full_size = df.count()
    # partial_size = dfs[0].count()
    # assert full_size == partial_size, \
    #     f"The resulting dataset's length differs from the input datasets length: {full_size} != {partial_size}"

    # check the resulting dataframe for correctness (no NONEs in any field)
    invalid_values = (
        df.select(
            *[
                sf.sum((sf.isnan(feat) | sf.isnull(feat)).cast("int")).alias(
                    feat
                )
                for feat in feature_columns
            ]
        )
        .first()
        .asDict()
    )

    has_invalid_values = any(
        invalid_values[feat] > 0 for feat in feature_columns
    )

    if has_invalid_values:
        df.unpersist()

    assert (
        not has_invalid_values
    ), f"Found records with invalid values in feature columns: {invalid_values}"

    df.write.parquet(combined_dataset_path)
    df.unpersist()


def _estimate_and_report_metrics(
    model_name: str, test: DataFrame, recs: DataFrame
):
    from replay.experiment import Experiment
    from replay.metrics import MAP, NDCG, HitRate

    K_list_metrics = [5, 10, 25, 50, 100]
    metrics = ["NDCG", "MAP", "HitRate"]

    e = Experiment(
        test,
        {
            MAP(): K_list_metrics,
            NDCG(): K_list_metrics,
            HitRate(): K_list_metrics,
        },
    )
    e.add_result(model_name, recs)

    for metric, k in itertools.product(metrics, K_list_metrics):
        metric_name = f"{metric}.{k}"
        metric_value = e.results.at[model_name, f"{metric}@{k}"]

        print(
            f"Estimated metric {metric_name}={metric_value} for {model_name}"
        )

        mlflow.log_metric(metric_name, metric_value)


def _log_model_settings(
    model_name: str,
    model_type: str,
    k: int,
    artifacts: ArtifactPaths,
    model_params: Dict,
    model_config_path: Optional[str] = None,
):
    dataset_name, _ = os.path.splitext(
        os.path.basename(artifacts.dataset.log_path)
    )
    mlflow.log_param("model", model_name)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("model_config_path", model_config_path)
    mlflow.log_param("k", k)
    mlflow.log_param("experiment_path", artifacts.base_path)
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("dataset_path", artifacts.dataset.log_path)
    mlflow.log_dict(model_params, "model_params.json")


# this is @task (old comment)
def do_fit_predict_second_level(
    artifacts: ArtifactPaths,
    model_name: str,
    k: int,
    train_path: str,
    first_level_predicts_path: str,
    second_model_type: str = "lama",
    second_model_params: Optional[Union[Dict, str]] = None,
    second_model_config_path: Optional[str] = None,
    cpu: int = DEFAULT_CPU,
    memory: int = DEFAULT_MEMORY,
):
    if second_model_type == "lama":
        mem_coeff = 0.3
    else:
        mem_coeff = 0.7

    with _init_spark_session(cpu, memory, mem_coeff=mem_coeff) as spark:
        # checks MLFLOW_EXPERIMENT_ID for the experiment id
        model_name = f"{model_name}_{str(uuid.uuid4()).replace('-', '')}"
        with mlflow.start_run():
            _log_model_settings(
                model_name=model_name,
                model_type=second_model_type,
                k=k,
                artifacts=artifacts,
                model_params=second_model_params,
                model_config_path=second_model_config_path,
            )

            scenario = PartialTwoStageScenario(
                base_path=artifacts.base_path,
                first_level_train_path=artifacts.first_level_train_path,
                second_level_positives_path=artifacts.second_level_positives_path,
                presplitted_data=True,
            )
            if second_model_type == "lama":
                # TODO: make late parametrizing of these param
                second_stage_model = LamaWrap(
                    params=second_model_params,
                    config_path=second_model_config_path,
                )
            elif second_model_type == "slama":
                second_stage_model = SlamaWrap(
                    params=second_model_params,
                    config_path="tabular_config.yml",
                )  # second_model_config_path)
            else:
                raise RuntimeError(
                    f"Currently supported model types: {['lama']}, but received {second_model_type}"
                )

            with log_exec_timer("fit") as timer:
                # train_path = train_path.replace(".parquet", "") + "48.parquet"
                print(train_path)
                tr = spark.read.parquet(train_path)

                # tr.repartition(48).write.mode("overwrite").parquet(os.path.join(train_path))
                # assert False
                # tr = tr.limit(int(tr.count() / 2))  ### Reducing train data for 2nd level model
                logger.info("second lvl model rows")
                logger.info(tr.count())

                second_stage_model.fit(tr)  #

            mlflow.log_metric(timer.name, timer.duration)

            with log_exec_timer("predict") as timer:
                recs = second_stage_model.predict(
                    spark.read.parquet(first_level_predicts_path), k=k
                )
                recs = scenario._filter_seen(
                    recs,
                    artifacts.train,
                    k,
                    artifacts.train.select("user_idx").distinct(),
                )
                recs.write.parquet(
                    artifacts.second_level_predicts_path(model_name)
                )
                # recs.write.mode('overwrite').format('noop').save()

            mlflow.log_metric(timer.name, timer.duration)

            _estimate_and_report_metrics(model_name, artifacts.test, recs)

            with log_exec_timer("model_saving") as timer:
                second_level_model_path = artifacts.second_level_model_path(
                    model_name
                )
                save_transformer(second_stage_model, second_level_model_path)

            mlflow.log_metric(timer.name, timer.duration)


def do_fit_predict_second_level_pure(
    artifacts: ArtifactPaths,
    model_name: str,
    k: int,
    train_path: str,
    first_level_predicts_path: str,
    second_model_type: str = "lama",
    second_model_params: Optional[Union[Dict, str]] = None,
    second_model_config_path: Optional[str] = None,
    cpu: int = DEFAULT_CPU,
    memory: int = DEFAULT_MEMORY,
):
    if second_model_type == "lama":
        mem_coeff = 0.3
    else:
        mem_coeff = 0.7

    with _init_spark_session(cpu, memory, mem_coeff=mem_coeff) as spark:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_ID", "delete"))

        model_name = f"{model_name}_{str(uuid.uuid4()).replace('-', '')}"
        with mlflow.start_run():
            _log_model_settings(
                model_name=model_name,
                model_type=second_model_type,
                k=k,
                artifacts=artifacts,
                model_params=second_model_params,
                model_config_path=second_model_config_path,
            )

            params = {
                "learningRate": 0.01,
                "numLeaves": 32,
                "featureFraction": 0.7,
                "baggingFraction": 0.7,
                "baggingFreq": 1,
                "maxDepth": -1,
                "minGainToSplit": 0.0,
                "maxBin": 255,
                "minDataInLeaf": 5,
                "numIterations": 2000,
                # 'earlyStoppingRound': 200,
                "rawPredictionCol": "raw_prediction",
                "probabilityCol": "LightGBM_prediction_0",
                "predictionCol": "prediction",
                "isUnbalance": True,
                "objective": "binary",
                "metric": "auc",
            }

            lgbm_booster = LightGBMClassifier
            full_data = spark.read.parquet(train_path)  #!
            features = [
                x
                for x in full_data.columns
                if x
                not in [
                    "target",
                    "user_idx",
                    "item_idx",
                    "user_factors",
                    "item_factors",
                    "factors_mult",
                ]
            ]
            feat2dtype = dict(full_data.select(features).dtypes)
            print("features")
            print(feat2dtype)

            assembler = VectorAssembler(
                inputCols=features,
                outputCol=f"LightGBM_vassembler_features",
                handleInvalid="keep",
            )
            lgbm = lgbm_booster(
                **params,
                featuresCol=assembler.getOutputCol(),
                labelCol="target",
                verbosity=1,
                useSingleDatasetMode=True,
                isProvideTrainingMetric=True,
                chunkSize=4_000_000,
                useBarrierExecutionMode=True,
                numTasks=16,
            )
            # transformer = lgbm.fit(full_data) #assembler.transform(full_data))
            transformer = lgbm.fit(assembler.transform(full_data))

            # train_df = train_df.withColumn('is_val', sf.col('reader_fold_num') == fold)
            #
            # valid_df = train_df.where('is_val')
            # train_df = train_df.where(~sf.col('is_val'))
            # full_data = valid_df.unionByName(train_df)
            # full_data = BalancedUnionPartitionsCoalescerTransformer().transform(full_data)
            #
            # # TODO: lgb num tasks should be equal to num cores
            # pref_locs = self._executors[fold * 2: fold * 2 + 2]
            # full_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs).transform(full_data)
            # print(f"Pref lcos for fold {fold}: {pref_locs}")
            # =========================================================
            # train_df, pref_locs = self.get_train(fold)
            #
            # train_df = train_df.withColumn('is_val', sf.col('reader_fold_num') == fold)
            # valid_df = train_df.where('is_val')
            # train_df = train_df.where(~sf.col('is_val'))
            # full_data = valid_df.unionByName(train_df)
            # full_data = BalancedUnionPartitionsCoalescerTransformer().transform(full_data)
            # =======================================================
            # won't work without do_shuffle=True, that means we should move train_df + valid_df merging somewhere upstream
            # full_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs,
            #                                                        do_shuffle=True).transform(full_data)

            # full_data = train_df

            # TODO: lgb num tasks should be equal to num cores
            # fld = fold % 3# max_job_parallelism
            # pref_locs = self._executors[fld * 2: fld * 2 + 2]
            # full_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs).transform(full_data)
            #
            # print(f"Pref lcos for fold {fold}: {pref_locs}")

            # preds_df = transformer.transform(assembler.transform(test_df))

            # score = SparkTask(task_type).get_dataset_metric()
            # metric_value = score(
            #     preds_df.select(
            #         SparkDataset.ID_COLUMN,
            #         sf.col(md['target']).alias('target'),
            #         sf.col(prediction_col).alias('prediction')
            #     )
            # )

            # scenario = PartialTwoStageScenario(
            #     base_path=artifacts.base_path,
            #     first_level_train_path=artifacts.first_level_train_path,
            #     second_level_positives_path=artifacts.second_level_positives_path,
            #     presplitted_data=True
            # )
            # if second_model_type == "lama":
            #     # TODO: make late parametrizing of these param
            #     second_stage_model = LamaWrap(params=second_model_params, config_path=second_model_config_path)
            # elif second_model_type == "slama":
            #     second_stage_model = SlamaWrap(params=second_model_params, config_path="tabular_config.yml") #second_model_config_path)
            # else:
            #     raise RuntimeError(f"Currently supported model types: {['lama']}, but received {second_model_type}")
            #
            # with log_exec_timer("fit") as timer:
            #
            #     tr = spark.read.parquet(train_path)
            #     # tr = tr.limit(int(tr.count() / 2))  ### Reducing train data for 2nd level model
            #     logger.info("second lvl model rows")
            #     logger.info(tr.count())
            #     second_stage_model.fit(tr)  #
            #
            # mlflow.log_metric(timer.name, timer.duration)
            #
            # with log_exec_timer("predict") as timer:
            #     recs = second_stage_model.predict(spark.read.parquet(first_level_predicts_path), k=k)
            #     recs = scenario._filter_seen(recs, artifacts.train, k, artifacts.train.select('user_idx').distinct())
            #     recs.write.parquet(artifacts.second_level_predicts_path(model_name))
            #     # recs.write.mode('overwrite').format('noop').save()
            #
            # mlflow.log_metric(timer.name, timer.duration)
            #
            # _estimate_and_report_metrics(model_name, artifacts.test, recs)
            #
            # with log_exec_timer("model_saving") as timer:
            #     second_level_model_path = artifacts.second_level_model_path(model_name)
            #     save_transformer(second_stage_model, second_level_model_path)
            #
            # mlflow.log_metric(timer.name, timer.duration)


def _infer_trained_models_files(
    artifacts: ArtifactPaths,
) -> List[FirstLevelModelFiles]:
    files = list_folder(artifacts.base_path)

    def model_name(filename: str) -> str:
        return filename.split("_")[-2]

    def get_files(prefix: str) -> Dict[str, str]:
        return {
            model_name(file): artifacts.make_path(file)
            for file in files
            if file.startswith(prefix)
        }

    partial_predicts = get_files("partial_predict")
    partial_trains = get_files("partial_train")
    partial_scenarios = get_files(
        "two_stage_scenario"
    )  # It is required for making missed predictions
    finished_model_names = (
        set(partial_predicts)
        .intersection(partial_trains)
        .intersection(partial_scenarios)
    )

    first_lvl_model_files = [
        FirstLevelModelFiles(
            model_name=mname,
            train_path=partial_trains[mname],
            predict_path=partial_predicts[mname],
            model_path=partial_scenarios[mname],
        )
        for mname in finished_model_names
    ]

    return first_lvl_model_files


class DatasetCombiner:
    @staticmethod
    def _combine(
        artifacts: ArtifactPaths,
        mode: str,
        model_names: List[str],
        models: List[BaseRecommender],
        partial_dfs: List[DataFrame],
        combined_df_path: str,
        sprk_ses: SparkSession,
    ):
        # ======== debug lines ==========
        for model_name in model_names:
            print(model_name)
        for df in partial_dfs:
            df.printSchema()
        # ===============================

        if len(partial_dfs) == 1:
            combined_df = partial_dfs[0]
        else:
            combined_df = partial_dfs[0]
            for right_df in partial_dfs[1:]:
                common_cols = [
                    sf.coalesce(f"left_df.{c}", f"right_df.{c}").alias(c)
                    for c in combined_df.columns
                    if c in right_df.columns
                    and c not in ["user_idx", "item_idx"]
                ]
                columns_from_left = [
                    sf.col(f"left_df.{c}").alias(c)
                    for c in combined_df.columns
                    if (
                        c not in ["user_idx", "item_idx"]
                        and c not in right_df.columns
                    )
                ]
                columns_from_right = [
                    sf.col(f"right_df.{c}").alias(c)
                    for c in right_df.columns
                    if (
                        c not in ["user_idx", "item_idx"]
                        and c not in combined_df.columns
                    )
                ]
                combined_df = combined_df.alias("left_df")
                right_df = right_df.alias("right_df")
                combined_df = combined_df.join(
                    right_df, on=["user_idx", "item_idx"], how="outer"
                ).select(
                    "user_idx",
                    "item_idx",
                    *common_cols,
                    *columns_from_left,
                    *columns_from_right,
                )

        for model in models:
            if isinstance(model, ALSWrap):
                print("ALSWrap in models list. Will join item_factors and user_factors to combined_df.")
                item_factors: DataFrame = model.model.itemFactors.select(
                    sf.col("id").alias("item_idx"),
                    sf.col("features").alias("item_factors")
                )
                user_factors: DataFrame = model.model.userFactors.select(
                    sf.col("id").alias("user_idx"),
                    sf.col("features").alias("user_factors")
                )
                item_factors.printSchema()
                user_factors.printSchema()

                combined_df = combined_df.drop("item_factors", "user_factors")
                combined_df = combined_df.join(item_factors, on="item_idx", how="left")
                combined_df = combined_df.join(user_factors, on="user_idx", how="left")

                break

        # DEBUG: checking number of lines where user_factors/item_factors is null
        count_null_user_factors = combined_df.where(
            sf.col("user_factors").isNull()
        ).count()
        count_null_item_factors = combined_df.where(
            sf.col("item_factors").isNull()
        ).count()
        print(f"count_null_user_factors: {count_null_user_factors}")
        print(f"count_null_item_factors: {count_null_item_factors}")

        # processing "rel_*" columns
        rel_cols = []
        for c in combined_df.columns:
            if c.startswith("rel_"):
                # adding "rel_*_is_null" features
                rel_cols.append(
                    sf.when(sf.col(c).isNull(), True)
                    .otherwise(False)
                    .alias(c + "_is_null")
                )
                # null -> NaN
                rel_cols.append(
                    sf.when(sf.col(c).isNull(), float("nan"))
                    .otherwise(sf.col(c))
                    .alias(c)
                )
        other_cols = [
            c for c in combined_df.columns if not c.startswith("rel_")
        ]
        combined_df = combined_df.select(*other_cols, *rel_cols)

        # DEBUG: final schema
        combined_df.printSchema()

        # DEBUG: final rows number
        print(f"final rows number: {combined_df.count()}")

        print("Saving new parquet in ", combined_df_path)
        combined_df.write.parquet(combined_df_path)

        print("Saved")

    @staticmethod
    def do_combine_datasets(
        artifacts: ArtifactPaths,
        combined_train_path: str,
        combined_predicts_path: str,
        desired_models: Optional[List[str]] = None,
        mode: str = "union",
    ):
        with _init_spark_session() as spark, mlflow.start_run():
            train_exists = do_path_exists(combined_train_path)
            predicts_exists = do_path_exists(combined_predicts_path)

            assert train_exists == predicts_exists, (
                f"The both datasets should either exist or be absent. "
                f"Train {combined_train_path}, Predicts {combined_predicts_path}"
            )

            if train_exists and predicts_exists:
                logger.info(
                    f"Datasets {combined_train_path} and {combined_predicts_path} exist. Nothing to do."
                )
                return

            logger.info("Inferring trained models and their files")
            logger.info("Base path", artifacts.base_path)
            model_files = _infer_trained_models_files(artifacts)

            logger.info(
                f"Found the following models that have all required files: "
                f"{[mfiles.model_name for mfiles in model_files]}"
            )
            found_mpaths = "\n".join(
                [mfiles.model_path for mfiles in model_files]
            )
            logger.info(f"Found models paths:\n {found_mpaths}")

            if desired_models is not None:
                logger.info(
                    f"Checking availability of the desired models: {desired_models}"
                )
                model_files = [
                    mfiles
                    for mfiles in model_files
                    if mfiles.model_name.lower() in desired_models
                ]
                not_available_models = set(desired_models).difference(
                    mfiles.model_name.lower() for mfiles in model_files
                )
                assert (
                    len(not_available_models) == 0
                ), f"Not all desired models available: {not_available_models}"

            used_mpaths = "\n".join(
                [mfiles.model_path for mfiles in model_files]
            )
            logger.info(f"Continue with models:\n {used_mpaths}")

            # creating combined train
            logger.info("Creating combined train")
            model_names = [mfiles.model_name.lower() for mfiles in model_files]
            partial_train_dfs = [
                spark.read.parquet(mfiles.train_path) for mfiles in model_files
            ]  #  train files for 1st lvl models
            partial_predicts_dfs = [
                spark.read.parquet(mfiles.predict_path)
                for mfiles in model_files
            ]
            logger.info("Loading models")
            models = [
                cast(
                    BaseRecommender,
                    cast(
                        PartialTwoStageScenario, load_model(mfiles.model_path)
                    ).first_level_models[0],
                )
                for mfiles in model_files
            ]
            logger.info("Models loaded")

            print("Combiner train parts is started")
            with log_exec_timer("DatasetCombiner partial_train_dfs") as timer:
                DatasetCombiner._combine(
                    artifacts=artifacts,
                    mode=mode,
                    model_names=model_names,
                    models=models,
                    partial_dfs=partial_train_dfs,
                    combined_df_path=combined_train_path,
                    sprk_ses=spark,
                )
            mlflow.log_metric("combiner_train_sec", timer.duration)

            # combine predicts
            logger.info("Creating combined predicts")

            print("Combiner predict parts is started")
            with log_exec_timer(
                "DatasetCombiner partial_predicts_dfs"
            ) as timer:
                DatasetCombiner._combine(
                    artifacts=artifacts,
                    mode=mode,
                    model_names=model_names,
                    models=models,
                    partial_dfs=partial_predicts_dfs,
                    combined_df_path=combined_predicts_path,
                    sprk_ses=spark,
                )
            mlflow.log_metric("combiner_predicts_sec", timer.duration)

            # common_cols.remove('target')
            # partial_predicts_dfs = [spark.read.parquet(mfiles.predict_path) for mfiles in model_files]
            # full_predicts_df = functools.reduce(
            #     lambda acc, x: acc.join(x.drop(*common_cols), on=['user_idx', 'item_idx']),
            #     partial_predicts_dfs,
            #     partial_predicts_dfs[0].select('user_idx', 'item_idx', *common_cols)
            # )
            # full_predicts_df.write.parquet(combined_predicts_path)

            logger.info("Combining finished")


if __name__ == "__main__":
    spark = get_cluster_session()

    config_filename = os.environ.get(
        TASK_CONFIG_FILENAME_ENV_VAR, "task_config.pickle"
    )

    with open(config_filename, "rb") as f:
        task_config = pickle.load(f)

    print("Task configs:")
    for k, v in task_config.items():
        print(k, v)
    print()

    print(f"config filename: {config_filename}")
    print()

    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8822"
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(
        os.environ.get("EXPERIMENT", config_filename.split("_")[2])
    )

    if config_filename.split("_")[2] == "2lvl":  # TODO: refactor
        do_fit_predict_second_level(**task_config)
    elif config_filename.split("_")[2] == "combiner":  # TODO refactor
        DatasetCombiner.do_combine_datasets(**task_config)
    elif config_filename.split("_")[2] == "pure":
        do_fit_predict_second_level_pure(**task_config)
    else:
        raise ValueError(f"Unknown name format of {config_filename}")
