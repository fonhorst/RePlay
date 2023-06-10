import random
import sys
import functools
import importlib
import itertools
import logging
import os
import pickle
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, cast, Optional, List, Union, Tuple
import inspect

import mlflow
from pyspark.sql import functions as sf, SparkSession, DataFrame, Window

sys.path.insert(0, "/opt/airflow/dags/dalexandrov_packages")
import replay
from dag_entities_dalexandrov import ArtifactPaths, DEFAULT_CPU, DEFAULT_MEMORY, TASK_CONFIG_FILENAME_ENV_VAR
from dag_entities_dalexandrov import FIRST_LEVELS_MODELS_PARAMS_BORDERS, FIRST_LEVELS_MODELS_PARAMS
from replay.data_preparator import Indexer
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.model_handler import save, Splitter, load, ALSWrap
from replay.models import PopRec
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario, OneStageUser2ItemScenario
from replay.scenarios.two_stages.reranker import LamaWrap, ReRanker
from replay.scenarios.two_stages.slama_reranker import SlamaWrap
from replay.scenarios.two_stages.two_stages_scenario import get_first_level_model_features
from replay.session_handler import State
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import log_exec_timer, JobGroup, join_with_col_renaming, save_transformer, do_path_exists,\
    load_transformer, list_folder, get_log_info
from replay.metrics import NDCG
from pyspark.ml import PipelineModel
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.validation.iterators import SparkFoldsIterator
from sparklightautoml.dataset import persistence

# logger = logging.getLogger('airflow.task')
StreamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
StreamHandler.setFormatter(formatter)

logging.basicConfig(
    level=logging.ERROR,
    handlers=[StreamHandler])

logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)

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
            Window.partitionBy("user_idx").orderBy(
                sf.col("relevance").desc()
            )
        ),
    ).filter(sf.col("temp_rank") <= sf.lit(k))

    return recs


@dataclass(frozen=True)
class FirstLevelModelFiles:
    model_name: str
    train_path: str
    predict_path: str
    model_path: str


def _get_bucketing_key(default: str = 'user_idx') -> str:
    key = os.environ.get("REPLAY_BUCKETING_KEY", default)
    assert key in ['user_idx', 'item_idx']
    return key


def _make_bucketing(df: DataFrame, bucketing_key: str, name: str) -> DataFrame:
    spark = State().session
    table_name = f"bucketed_df_{spark.sparkContext.applicationId.replace('-', '__')}_{name}"
    partition_num = spark.sparkContext.defaultParallelism

    logger.info(f"Bucketing of the dataset: name={name}, table_name={table_name}")

    (
        df.repartition(partition_num, bucketing_key)
        .write.mode("overwrite")
        .bucketBy(partition_num, bucketing_key)
        .sortBy(bucketing_key)
        .saveAsTable(table_name, format="parquet")
    )

    logger.info(f"Bucketing of the dataset is finished: name={name}, table_name={table_name}")

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

    def save(self, path: str, overwrite: bool = False, spark: Optional[SparkSession] = None):
        spark = spark or self._get_spark_session()
        spark.createDataFrame([{"data": '', "classname": self.get_classname()}]).write.parquet(
            path,
            mode='overwrite' if overwrite else 'error'
        )

    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        if self.dump_path is not None:
            data.write.parquet(self.dump_path)

    def predict(self, data, k) -> DataFrame:
        return data


class EmptyRecommender(BaseRecommender):
    @property
    def _init_args(self):
        return dict()

    def _fit(self, log: DataFrame, user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        pass

    def _predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                 user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        return log.select("user_idx", "item_idx", sf.lit(0.0).alias("relevance"))

    def _get_nearest_items(self, items: DataFrame, metric: Optional[str] = None,
                           candidates: Optional[DataFrame] = None) -> Optional[DataFrame]:
        return None


class PartialTwoStageScenario(TwoStagesScenario):
    def __init__(self,
                 base_path: str,
                 first_level_train_path: str,
                 second_level_positives_path: str,
                 first_level_train_opt_path: str = None,
                 first_level_val_opt_path: str = None,
                 second_level_train_path: Optional[str] = None,
                 train_splitter: Splitter = UserSplitter(
                     item_test_size=0.5, shuffle=True, seed=42
                 ),
                 splitter_opt: Splitter = UserSplitter(item_test_size=0.2, shuffle=True, seed=42
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
                 custom_features_processor: Optional[HistoryBasedFeaturesProcessor] = None,
                 seed: int = 123,
                 presplitted_data: bool = False
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
            seed=seed
        )
        self.second_stage_model = EmptyWrap(dump_path=second_level_train_path)
        self._base_path = base_path
        self._first_level_train_path = first_level_train_path
        self._second_level_positives_path = second_level_positives_path
        self._first_level_train_opt_path = first_level_train_opt_path
        self._first_level_val_opt_path = first_level_val_opt_path
        self._presplitted_data = presplitted_data
        if first_level_item_features_transformer:
            self.first_level_item_features_transformer = first_level_item_features_transformer
        if first_level_user_features_transformer:
            self.first_level_user_features_transformer = first_level_user_features_transformer
        self.splitter_opt = splitter_opt

    @property
    def _init_args(self):
        return {
            **super()._init_args,
            "base_path": self._base_path,
            "first_level_train_path": self._first_level_train_path,
            "second_level_positives_path": self._second_level_positives_path,
            "presplitted_data": self._presplitted_data
        }

    @property
    def first_level_models_relevance_columns(self) -> List[str]:
        return [f"rel_{idx}_{model}" for idx, model in enumerate(self.first_level_models)]

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self._presplitted_data:
            spark = log.sql_ctx.sparkSession
            first_level_train = _make_bucketing(
                spark.read.parquet(self._first_level_train_path).repartition(spark.sparkContext.defaultParallelism),
                bucketing_key=_get_bucketing_key(default='user_idx'),
                name="first_level_train"
            )
            second_level_positive = _make_bucketing(
                spark.read.parquet(self._second_level_positives_path).repartition(
                    spark.sparkContext.defaultParallelism),
                bucketing_key=_get_bucketing_key(default='user_idx'),
                name="second_level_positive"
            )

            return first_level_train, second_level_positive

        first_level_train, second_level_positive = super()._split_data(log)

        first_level_train.write.parquet(self._first_level_train_path, mode='overwrite')
        second_level_positive.write.parquet(self._second_level_positives_path, mode='overwrite')

        return first_level_train, second_level_positive

    def _split_optimize(self, first_level_train: DataFrame) -> None:

        first_level_train_opt, first_level_val_opt = self.splitter_opt.split(first_level_train)

        logger.info(
            "first_level_train_opt info: %s", get_log_info(first_level_train_opt)
        )
        logger.info(
            "first_level_val_opt info: %s", get_log_info(first_level_val_opt)
        )

        first_level_train_opt.write.parquet(self._first_level_train_opt_path, mode='overwrite')
        first_level_val_opt.write.parquet(self._first_level_val_opt_path, mode='overwrite')

    def _get_nearest_items(self, items: DataFrame, metric: Optional[str] = None,
                           candidates: Optional[DataFrame] = None) -> Optional[DataFrame]:
        raise NotImplementedError("Unsupported method")


class RefitableTwoStageScenario(TwoStagesScenario):
    def __init__(self,
                 base_path: str,
                 train_splitter: Splitter = UserSplitter(
                     item_test_size=0.5, shuffle=True, seed=42
                 ),
                 first_level_models: Optional[Union[List[BaseRecommender], BaseRecommender]] = None,
                 fallback_model: Optional[BaseRecommender] = PopRec(),
                 use_first_level_models_feat: Union[List[bool], bool] = True,
                 num_negatives: int = 100,
                 negatives_type: str = "first_level",
                 use_generated_features: bool = True,
                 user_cat_features_list: Optional[List] = None,
                 item_cat_features_list: Optional[List] = None,
                 custom_features_processor: Optional[HistoryBasedFeaturesProcessor] = None,
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
            seed=seed
        )
        self.second_stage_model = EmptyWrap()
        self._base_path = base_path
        self._first_level_train_path = os.path.join(base_path, "first_level_train.parquet")
        self._second_level_positive_path = os.path.join(base_path, "second_level_positive.parquet")
        self._first_level_candidates_path = os.path.join(base_path, "first_level_candidates.parquet")

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
        return [f"rel_{idx}_{model}" for idx, model in enumerate(self.first_level_models)]

    @candidates_with_positives.setter
    def candidates_with_positives(self, val: bool):
        self._return_candidates_with_positives = val

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self._are_split_data_dumped:
            spark = log.sql_ctx.sparkSession
            return (
                spark.read.parquet(self._first_level_train_path),
                spark.read.parquet(self._second_level_positive_path)
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
            first_level_candidates = spark.read.parquet(self._first_level_candidates_path)

            if self.candidates_with_positives:
                assert self._are_split_data_dumped, "Cannot do that without dumped second level positives"
                second_level_positive = spark.read.parquet(self._second_level_positive_path)
                first_level_candidates = (
                    first_level_candidates.join(
                        second_level_positive.select(
                            "user_idx", "item_idx"
                        ).withColumn("target", sf.lit(1.0)),
                        on=["user_idx", "item_idx"],
                        how="left",
                    ).fillna(0.0, subset="target")
                )

            return first_level_candidates

        first_level_candidates = super()._get_first_level_candidates(model, log, k, users,
                                                                     items, user_features, item_features, log_to_filter)

        first_level_candidates.write.parquet(self._first_level_candidates_path)
        self._are_candidates_dumped = True

        return first_level_candidates

    def _add_features_for_second_level(self, log_to_add_features: DataFrame, log_for_first_level_models: DataFrame,
                                       user_features: DataFrame, item_features: DataFrame) -> DataFrame:
        candidates_with_features = super()._add_features_for_second_level(
            log_to_add_features, log_for_first_level_models, user_features, item_features
        )

        if self._are_candidates_dumped and self.candidates_with_positives:
            assert self._are_split_data_dumped, "Cannot do that without dumped second level positives"
            spark = log_to_add_features.sql_ctx.sparkSession
            first_level_candidates = spark.read.parquet(self._first_level_candidates_path)
            second_level_positive = spark.read.parquet(self._second_level_positive_path)
            candidates_with_target = (
                first_level_candidates.join(
                    second_level_positive.select(
                        "user_idx", "item_idx"
                    ).withColumn("target", sf.lit(1.0)),
                    on=["user_idx", "item_idx"],
                    how="left",
                ).fillna(0.0, subset="target").select("user_idx", "item_idx", "target")
            )

            candidates_with_features = candidates_with_features \
                .join(candidates_with_target, on=["user_idx", "item_idx"], how='left')

        return candidates_with_features

    def _get_nearest_items(self, items: DataFrame, metric: Optional[str] = None,
                           candidates: Optional[DataFrame] = None) -> Optional[DataFrame]:
        raise NotImplementedError("Unsupported method")

    def _save_model(self, path: str):
        super()._save_model(path)
        spark = State().session
        spark.createDataFrame([{
            "_are_candidates_dumped": self._are_candidates_dumped,
            "_are_split_data_dumped": self._are_split_data_dumped
        }]).write.parquet(os.path.join(path, "data_refittable.parquet"))

    def _load_model(self, path: str):
        super()._load_model(path)
        spark = State().session
        row = spark.read.parquet(os.path.join(path, "data_refittable.parquet")).first().asDict()
        self._are_candidates_dumped = row["_are_candidates_dumped"]
        self._are_split_data_dumped = row["_are_split_data_dumped"]


class PartialOneStageScenario(OneStageUser2ItemScenario):
    def __init__(
            self,
            train_val_splitter: Splitter = UserSplitter(
                item_test_size=0.2, shuffle=True, seed=42
            ),
            first_level_models: Union[
                List[BaseRecommender], BaseRecommender
            ] = ALSWrap(rank=128),
            fallback_model: Optional[BaseRecommender] = PopRec(),
            user_cat_features_list: Optional[List] = None,
            item_cat_features_list: Optional[List] = None,
            custom_features_processor: HistoryBasedFeaturesProcessor = None,
            seed: int = 123,
            set_best_model: bool = False,
            k: int = 10,
            first_level_train_path: str = None,
            first_level_val_path: str = None
    ):

        super().__init__(
            train_val_splitter=train_val_splitter,
            first_level_models=first_level_models,
            fallback_model=fallback_model,
            user_cat_features_list=user_cat_features_list,
            item_cat_features_list=item_cat_features_list,
            custom_features_processor=custom_features_processor,
            seed=seed,
            set_best_model=set_best_model,
            k=k,
            )

        self._first_level_train_path = first_level_train_path
        self._first_level_val_path = first_level_val_path

    def _split_wrap(self, log):

        output_dict = {}

        first_level_train = spark.read.parquet(self._first_level_train_path).repartition(
            spark.sparkContext.defaultParallelism)

        first_level_val = spark.read.parquet(self._first_level_val_path).repartition(
            spark.sparkContext.defaultParallelism)

        output_dict["first_level_train"] = first_level_train
        output_dict["first_level_val"] = first_level_val

        return output_dict


def load_model(path: str):
    setattr(replay.model_handler, 'EmptyRecommender', EmptyRecommender)
    setattr(replay.model_handler, 'EmptyWrap', EmptyWrap)
    setattr(replay.model_handler, 'RefitableTwoStageScenario', RefitableTwoStageScenario)
    setattr(replay.model_handler, 'PartialTwoStageScenario', PartialTwoStageScenario)
    return load(path)


def save_model(model: BaseRecommender, path: str, overwrite: bool = False):
    save(model, path, overwrite)


def get_cluster_session():
    assert os.environ.get('SCRIPT_ENV', 'local') == 'cluster'
    return SparkSession.builder.getOrCreate()


@contextmanager
def _init_spark_session(cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY, mem_coeff: float = 0.9) -> SparkSession:
    if os.environ.get('SCRIPT_ENV', 'local') == 'cluster':
        spark = SparkSession.builder.getOrCreate()
    else:

        jars = [
            os.environ.get("REPLAY_JAR_PATH", "/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar"),
            # '../../scala/target/scala-2.12/replay_2.12-0.1.jar'
            os.environ.get("SLAMA_JAR_PATH", '/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar')
            # '../../../LightAutoML/jars/spark-lightautoml_2.12-0.1.1.jar'
        ]

        real_memory = int(memory * mem_coeff)
        max_result_size = max(int(real_memory * 0.8), 1)

        spark = (
            SparkSession
            .builder
            .config("spark.jars", ",".join(jars))
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
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


def _get_model(artifacts: ArtifactPaths, model_class_name: str, model_kwargs: Dict) -> BaseRecommender:
    module_name = ".".join(model_class_name.split('.')[:-1])
    class_name = model_class_name.split('.')[-1]
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
        spark: SparkSession):
    assert len(partial_datasets_paths) > 0, "Cannot work with empty sequence of paths"

    dfs = [spark.read.parquet(path) for path in partial_datasets_paths]

    # TODO: restore this check later
    # check that all dataframes have the same size
    # lengths = set(df.count() for df in dfs)
    # assert len(lengths) == 1 and next(iter(lengths)) > 0, f"Invalid lengths of datasets: {lengths}"

    # strip all duplicate columns
    no_duplicate_columns_dfs = [
        dfs[0],
        *(
            df.select("user_idx", "item_idx", *set(df.columns).difference(set(dfs[0].columns)))
            for df in dfs[1:]
        )
    ]

    feature_columns = set(c for df in dfs[1:] for c in set(df.columns).symmetric_difference(set(dfs[0].columns)))

    df = functools.reduce(lambda acc, x: acc.join(x, on=["user_idx", "item_idx"]), no_duplicate_columns_dfs)
    df = df.cache()

    # TODO: restore it later
    # full_size = df.count()
    # partial_size = dfs[0].count()
    # assert full_size == partial_size, \
    #     f"The resulting dataset's length differs from the input datasets length: {full_size} != {partial_size}"

    # check the resulting dataframe for correctness (no NONEs in any field)
    invalid_values = df.select(*[
        sf.sum((sf.isnan(feat) | sf.isnull(feat)).cast('int')).alias(feat)
        for feat in feature_columns
    ]).first().asDict()

    has_invalid_values = any(invalid_values[feat] > 0 for feat in feature_columns)

    if has_invalid_values:
        df.unpersist()

    assert not has_invalid_values, \
        f"Found records with invalid values in feature columns: {invalid_values}"

    df.write.parquet(combined_dataset_path)
    df.unpersist()


def _estimate_and_report_metrics(model_name: str, test: DataFrame, recs: DataFrame):
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
        print(f"Estimated metric {metric_name}={metric_value} for {model_name}")
        mlflow.log_metric(metric_name, metric_value)


def _estimate_and_report_budget_metrics(model_name: str, test: DataFrame, recs: DataFrame, b: int):
    from replay.experiment import Experiment
    from replay.metrics import MAP, NDCG, HitRate

    e = Experiment(
        test,
        {
            NDCG(): 100,
        },
    )
    e.add_result(model_name, recs)
    mlflow.log_metric(f"NDCG.100-b{b}", e.results.iloc[0]["NDCG@100"])


def _log_model_settings(model_name: str,
                        model_type: str,
                        k: int,
                        artifacts: ArtifactPaths,
                        model_params: Dict,
                        model_config_path: Optional[str] = None):
    dataset_name, _ = os.path.splitext(os.path.basename(artifacts.dataset.log_path))
    mlflow.log_param("model", model_name)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("model_config_path", model_config_path)
    mlflow.log_param("k", k)
    mlflow.log_param("experiment_path", artifacts.base_path)
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("dataset_path", artifacts.dataset.log_path)
    mlflow.log_dict(model_params, "model_params.json")


# def do_collect_best_params(artifacts: ArtifactPaths, *model_class_names: str):
#
#     best_params_dict = {}
#     for model_class_name in model_class_names:
#         with open(os.path.join("file://", artifacts.base_path,
#                                f"best_params_{MODELNAME2FULLNAME[model_class_name]}.pickle"), "rb") as f:
#             best_params = pickle.load(f)
#             best_params_dict[model_class_name] = best_params
#     return best_params_dict


def do_dataset_splitting(artifacts: ArtifactPaths, partitions_num: int):
    from replay.data_preparator import DataPreparator

    train_exists = do_path_exists(artifacts.train_path)
    test_exists = do_path_exists(artifacts.test_path)

    if train_exists and test_exists:
        logger.info(f"Datasets {artifacts.train_path} and {artifacts.test_path} exist. Nothing to do.")

        return

    with _init_spark_session(DEFAULT_CPU, DEFAULT_MEMORY):
        data = artifacts.log

        if 'timestamp' in data.columns:
            data = data.withColumn('timestamp', sf.col('timestamp').cast('long'))

        # splitting on train and test
        preparator = DataPreparator()

        if artifacts.dataset.name == 'netflix_small':
            # data = data.withColumn('timestamp', sf.col('timestamp').cast('long'))
            # log = preparator.transform(
            #     columns_mapping={"user_id": "user_idx", "item_id": "item_idx",
            #                      "relevance": "relevance", "timestamp": "timestamp"},
            #     data=data

            print("data columns")
            print(data.columns)
            preparator.setColumnsMapping({"user_id": "user_idx", "item_id": "item_idx",
                                          "relevance": "relevance", "timestamp": "timestamp"})

            log = preparator.transform(data
                                       ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id",
                                                                                                    "item_idx")

            # train/test split ml
            train_spl = DateSplitter(
                test_start=0.2,
                drop_cold_items=True,
                drop_cold_users=True,
            )
        elif artifacts.dataset.name.startswith('ml') or artifacts.dataset.name.startswith('netflix'):
            data = (
                data
                .withColumn('user_id', sf.col('user_id').cast('int'))
                .withColumn('item_id', sf.col('item_id').cast('int'))
                .withColumn('timestamp', sf.col('timestamp').cast('long'))
            )

            print("data columns")
            print(data.columns)
            # preparator.setColumnsMapping({"user_id": "user_id", "item_id": "item_id",
            #                               "relevance": "rating", "timestamp": "timestamp"})

            log = preparator.transform(data=data,
                                       columns_mapping={
                                           "user_id": "user_id",
                                           "item_id": "item_id",
                                           "relevance": "rating",
                                           "timestamp": "timestamp"})\
                .withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id","item_idx")

            # log = preparator.transform(
            # columns_mapping = {"user_id": "user_id", "item_id": "item_id",
            #                    "relevance": "rating", "timestamp": "timestamp"},
            # data = data)

            # train/test split ml
            train_spl = DateSplitter(
                test_start=0.2,
                drop_cold_items=True,
                drop_cold_users=True,
            )
        elif artifacts.dataset.name.startswith('msd'):
            # log = preparator.transform(
            #     columns_mapping={"user_id": "user_id", "item_id": "item_id", "relevance": "play_count"},
            #     data=data
            # ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")

            preparator.setColumnsMapping({"user_id": "user_id", "item_id": "item_id",
                                          "relevance": "play_count"})  # , "timestamp": "timestamp"})

            log = preparator.transform(data
                                       ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id",
                                                                                                    "item_idx")

            indexer = Indexer(user_col="user_idx", item_col="item_idx")
            indexer.fit(
                users=log.select("user_idx"), items=log.select("item_idx")
            )

            log = indexer.transform(df=log)

            # train/test split ml
            train_spl = UserSplitter(
                item_test_size=0.2,
                shuffle=True,
                drop_cold_items=True,
                drop_cold_users=True,
                seed=42
            )
        else:
            raise Exception(f"Unsupported dataset name: {artifacts.dataset.name}")

        print(get_log_info(log))

        log = log.repartition(partitions_num).cache()
        log.write.mode('overwrite').format('noop').save()

        only_positives_log = log.filter(sf.col('relevance') >= 3).withColumn('relevance', sf.lit(1))
        logger.info(get_log_info(only_positives_log))

        train, test = train_spl.split(only_positives_log)
        logger.info(f'train info: \n{get_log_info(train)}')
        logger.info(f'test info:\n{get_log_info(test)}')

        # writing data
        os.makedirs(artifacts.base_path, exist_ok=True)

        assert train.count() > 0
        assert test.count() > 0

        train.drop('_c0').write.parquet(artifacts.train_path, mode='overwrite')
        test.drop('_c0').write.parquet(artifacts.test_path, mode='overwrite')


def do_init_refitable_two_stage_scenario(artifacts: ArtifactPaths):
    with _init_spark_session(DEFAULT_CPU, DEFAULT_MEMORY):
        scenario = RefitableTwoStageScenario(base_path=artifacts.base_path)

        scenario.fit(log=artifacts.train, user_features=artifacts.user_features, item_features=artifacts.item_features)

        save(scenario, artifacts.two_stage_scenario_path)


def do_fit_feature_transformers(artifacts: ArtifactPaths, cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    from replay.data_preparator import ToNumericFeatureTransformer

    logger.info("Starting do_fit_feature_transformers")

    with _init_spark_session(cpu, memory) as spark:
        if artifacts.user_features is not None or artifacts.item_features is not None:
            assert \
                (artifacts.user_features is not None and artifacts.item_features is not None) \
                or (artifacts.user_features is None and artifacts.item_features is not None), \
                "Cannot handle when only user or item features is defined"

            ift_exists = do_path_exists(artifacts.item_features_transformer_path)
            uft_exists = do_path_exists(artifacts.user_features_transformer_path)

            if any([ift_exists, uft_exists]) and not all([ift_exists, uft_exists]):
                raise Exception(
                    f"The paths should be all either existing or absent. "
                    f"But: {artifacts.item_features_transformer_path} exists == {ift_exists}, "
                    f"{artifacts.user_features_transformer_path} exists == {uft_exists}."
                )

            if not ift_exists:
                first_level_user_features_transformer = ToNumericFeatureTransformer()
                first_level_item_features_transformer = ToNumericFeatureTransformer()

                logger.info("Fitting user features transformer...")
                first_level_user_features_transformer.fit(artifacts.user_features)
                logger.info("Fitting item features transformer...")
                first_level_item_features_transformer.fit(artifacts.item_features)

                logger.info("Saving fitted user and items transformers")
                save_transformer(first_level_item_features_transformer, artifacts.item_features_transformer_path)
                save_transformer(first_level_user_features_transformer, artifacts.user_features_transformer_path)

        hbt_exists = do_path_exists(artifacts.history_based_transformer_path)

        if hbt_exists:
            return

        hbt_transformer = HistoryBasedFeaturesProcessor(
            user_cat_features_list=artifacts.dataset.user_cat_features,
            item_cat_features_list=artifacts.dataset.item_cat_features,
        )

        logger.info("Fitting history-based features transformer...")
        hbt_transformer.fit(
            log=spark.read.parquet(artifacts.first_level_train_path),
            user_features=artifacts.user_features,
            item_features=artifacts.item_features
        )

        save_transformer(hbt_transformer, artifacts.history_based_transformer_path, overwrite=True)


def do_presplit_data(artifacts: ArtifactPaths, item_test_size_second_level: float, item_test_size_opt: float,
                     cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    with _init_spark_session(cpu, memory):
        flt_exists = do_path_exists(artifacts.first_level_train_path)
        slp_exists = do_path_exists(artifacts.second_level_positives_path)
        opt_train_exists = do_path_exists(artifacts.first_level_train_opt_path)
        opt_val_exist = do_path_exists(artifacts.first_level_val_opt_path)

        if flt_exists != slp_exists:
            raise Exception(
                f"The paths should be both either existing or absent. "
                f"But: {artifacts.first_level_train_path} exists == {flt_exists}, "
                f"{artifacts.second_level_positives_path} exists == {slp_exists}."
            )

        if flt_exists and slp_exists and opt_train_exists and opt_val_exist:
            logger.info(f"Datasets {artifacts.first_level_train_path}, {artifacts.second_level_positives_path},"
                        f" {artifacts.first_level_train_opt_path} and {artifacts.first_level_val_opt_path} exist. "
                        f"Nothing to do.")

            return

        scenario = PartialTwoStageScenario(
            base_path=artifacts.base_path,
            first_level_train_path=artifacts.first_level_train_path,
            second_level_positives_path=artifacts.second_level_positives_path,
            first_level_train_opt_path=artifacts.first_level_train_opt_path,
            first_level_val_opt_path=artifacts.first_level_val_opt_path,
            train_splitter=UserSplitter(item_test_size=item_test_size_second_level, shuffle=True, seed=42),
            splitter_opt=UserSplitter(item_test_size=item_test_size_opt, shuffle=True, seed=42),
            presplitted_data=False
        )

        scenario._split_data(artifacts.train)
        scenario._split_optimize(artifacts.first_level_train)


def do_fit_predict_one_stage(
        artifacts: ArtifactPaths,
        model_class_name: str,
        k: int,
        cpu: int = DEFAULT_CPU,
        memory: int = DEFAULT_MEMORY,
        get_optimized_params: bool = False,
        do_optimization: bool = False,
        mlflow_experiments_id: str = "delete"):

    with _init_spark_session(cpu, memory):
        with _init_spark_session(cpu, memory):

            mlflow.set_tracking_uri("http://node2.bdcl:8822")
            mlflow.set_experiment(mlflow_experiments_id)

            # if get_optimized_params:
            #     with open(os.path.join("file://", artifacts.base_path,
            #                            f"best_params_{model_class_name}.pickle"), "rb") as f:
            #         model_kwargs = pickle.load(f)[0][0]
            # else:
            model_kwargs = FIRST_LEVELS_MODELS_PARAMS[model_class_name]

            logger.debug(f"model params: {model_kwargs}")

            with mlflow.start_run():

                _log_model_settings(
                    model_name=model_class_name,
                    model_type=model_class_name,
                    k=k,
                    artifacts=artifacts,
                    model_params=model_kwargs,
                    model_config_path=None
                )

                scenario_name_postfix = 'optimized' if do_optimization else 'default'
                mlflow.log_param("scenario", f"one-stage_{scenario_name_postfix}")

                if get_optimized_params:
                    mlflow.log_param("optimized", "True")
                    optimized_postfix = "optimized"
                else:
                    mlflow.log_param("optimized", "False")
                    optimized_postfix = "default"

                base_path = artifacts.base_path
                logger.debug(f"base_path: {base_path}")

                mlflow.log_param("opt_split", base_path.split("_")[-1])
                mlflow.log_param("2lvl_split", base_path.split("_")[-3])

                first_level_model = _get_model(artifacts, model_class_name, model_kwargs)

                if artifacts.user_features is not None:
                    user_feature_transformer = load_transformer(artifacts.user_features_transformer_path)
                else:
                    user_feature_transformer = None
                if artifacts.item_features is not None:
                    item_feature_transformer = load_transformer(artifacts.item_features_transformer_path)
                else:
                    item_feature_transformer = None
                # history_based_transformer = load_transformer(artifacts.history_based_transformer_path)

                scenario = PartialOneStageScenario(
                    first_level_models=first_level_model,
                    user_cat_features_list=None,
                    item_cat_features_list=None,
                    set_best_model=True,
                    first_level_train_path=artifacts.first_level_train_opt_path,
                    first_level_val_path=artifacts.first_level_val_opt_path,
                    k=100

                )

                user_features = artifacts.user_features
                item_features = artifacts.item_features


                if do_optimization:
                    budget_list = [5, 10, 20, 50]
                    budget_time_list = []
                    with JobGroup("optimize", "optimization of first lvl models"):
                        for b_idx, b in enumerate(budget_list):
                            logger.debug(f"optimization budget: {b}")
                            if b_idx != 0:
                                delta_b = b - budget_list[b_idx-1]
                                new_study = False  # continue study
                            else:
                                delta_b = b
                                new_study = True

                            with log_exec_timer(
                                    f"budget_{b}_optimization"
                            ) as timer:
                                params_found, fallback_params, metrics_values = scenario.optimize(
                                    train=artifacts.first_level_train_opt,
                                    test=artifacts.first_level_val_opt,
                                    param_borders=[
                                        FIRST_LEVELS_MODELS_PARAMS_BORDERS[model_class_name],
                                        None],
                                    k=k,
                                    budget=delta_b,
                                    criterion=NDCG(),
                                    new_study=new_study
                                )
                                mlflow.log_metric(f"NDCG.{k}_opt_b{b}", metrics_values[0])
                            budget_time_list.append(timer.duration)

                            with log_exec_timer(
                                    f"budget_{b}_fit_predict_calc_metrics"
                            ) as fit_predict_timer:
                                scenario.fit(
                                    log=artifacts.train,
                                    user_features=user_features,
                                    item_features=item_features)

                                test_recs = scenario.predict(
                                    log=artifacts.train,
                                    k=k,
                                    items=artifacts.train.select("item_idx").distinct(),
                                    users=artifacts.test.select("user_idx").distinct(),
                                    user_features=artifacts.user_features,
                                    item_features=artifacts.item_features,
                                    filter_seen_items=True
                                ).cache()

                                logger.info("Estimating metrics...")
                                _estimate_and_report_budget_metrics(model_class_name, artifacts.test, test_recs, b=b)

                                if b_idx == 0:
                                    b_time = budget_time_list[-1]
                                else:
                                    b_time += budget_time_list[-1]

                            mlflow.log_metric(f"budget_{b}_duration", b_time+fit_predict_timer.duration)



                        # best_params = scenario.optimize(train=artifacts.first_level_train_opt,
                        #                                 test=artifacts.first_level_val_opt,
                        #                                 param_borders=[
                        #                                     FIRST_LEVELS_MODELS_PARAMS_BORDERS[model_class_name],
                        #                                     None],
                        #                                 k=k,
                        #                                 budget=20,
                        #                                 criterion=NDCG(),
                        #                                 )
                        # # for ALS only + fallback {"rank": [50, 200]}
                        # print(best_params)
                        # logger.info(best_params)
                        # print("cwd")
                        # print(os.getcwd())
                        # print(os.path.join(artifacts.base_path, f"best_params_{model_class_name}.pickle"))
                        # with open(os.path.join(artifacts.base_path, f"best_params_{model_class_name}.pickle"),
                        #           "wb") as f:
                        #     pickle.dump(best_params, f)
                        return 0

                else:

                    with JobGroup("fit", "fitting of one stage"):
                        scenario.fit(
                            log=artifacts.train,
                            user_features=user_features,
                            item_features=item_features)

                    logger.info("Fit is ended. Predicting...")

                    with JobGroup("predict", "predicting the test"):
                        test_recs = scenario.predict(
                            log=artifacts.train,
                            k=k,
                            items=artifacts.train.select("item_idx").distinct(),
                            users=artifacts.test.select("user_idx").distinct(),
                            user_features=artifacts.user_features,
                            item_features=artifacts.item_features,
                            filter_seen_items=True
                        ).cache()

                # rel_cols = [c for c in test_recs.columns if c.startswith('rel_')]
                # assert len(rel_cols) == 1
                #
                # test_recs = test_recs.withColumnRenamed(rel_cols[0], 'relevance')

                # test_recs = filter_seen_custom(recs=test_recs, log=artifacts.train, k=k,
                #                                users=artifacts.train.select('user_idx').distinct())



                    mlflow.log_metric("NDCG.100-val", scenario._experiment.results.iloc[0]["NDCG@100"])
                    logger.info("Estimating metrics...")

                    _estimate_and_report_metrics(model_class_name, artifacts.test, test_recs)

                    test_recs = test_recs.withColumnRenamed('relevance',
                                                            f"rel_type{type(scenario.best_model).__name__}")
                    test_recs.write.parquet(artifacts.partial_predicts_path(model_class_name, optimized_postfix))
                    test_recs.unpersist()

                    logger.info("Saving the model")
                    with JobGroup("save", "saving the model"):

                        save(scenario, artifacts.partial_two_stage_scenario_path(model_class_name, optimized_postfix),
                             overwrite=True)



def do_fit_predict_first_level_model(artifacts: ArtifactPaths,
                                     model_class_name: str,
                                     # model_kwargs: Dict,
                                     k: int,
                                     cpu: int = DEFAULT_CPU,
                                     memory: int = DEFAULT_MEMORY,
                                     get_optimized_params: bool = False,
                                     do_optimization: bool = False,
                                     mlflow_experiments_id: str = "delete"):
    with _init_spark_session(cpu, memory):

        mlflow.set_tracking_uri("http://node2.bdcl:8822")
        mlflow.set_experiment(mlflow_experiments_id)

        if get_optimized_params:
            with open(os.path.join("file://", artifacts.base_path,
                                   f"best_params_{model_class_name}.pickle"), "rb") as f:
                model_kwargs = pickle.load(f)[0][0]
        else:
            model_kwargs = FIRST_LEVELS_MODELS_PARAMS[model_class_name]

        print("model params")
        print(model_kwargs)

        with mlflow.start_run():

            _log_model_settings(
                model_name=model_class_name,
                model_type=model_class_name,
                k=k,
                artifacts=artifacts,
                model_params=model_kwargs,
                model_config_path=None
            )

            # if get_optimized_params:
            #     mlflow.log_param("optimized", "True")
            #     optimized_postfix = "optimized"
            # else:
            #     mlflow.log_param("optimized", "False")
            #     optimized_postfix = "default"

            first_level_model = _get_model(artifacts, model_class_name, model_kwargs)

            if artifacts.user_features is not None:
                user_feature_transformer = load_transformer(artifacts.user_features_transformer_path)
            else:
                user_feature_transformer = None
            if artifacts.item_features is not None:
                item_feature_transformer = load_transformer(artifacts.item_features_transformer_path)
            else:
                item_feature_transformer = None
            # history_based_transformer = load_transformer(artifacts.history_based_transformer_path)
            optimized_postfix = "pad"

            scenario = PartialTwoStageScenario(
                train_splitter=UserSplitter(
                    item_test_size=0.2, shuffle=True, seed=42),  # for fair comparison
                base_path=artifacts.base_path,
                first_level_train_path=artifacts.first_level_train_path,
                second_level_positives_path=artifacts.second_level_positives_path,
                second_level_train_path=artifacts.partial_train_path(model_class_name, optimized_postfix),
                first_level_models=first_level_model,
                first_level_item_features_transformer=None,  # item_feature_transformer,
                first_level_user_features_transformer=None,  # user_feature_transformer,
                custom_features_processor=None,  # history_based_transformer,
                presplitted_data=True,
                use_generated_features=False,
                use_first_level_models_feat=False,
                num_negatives=10 #todo
            )

            bucketing_key = _get_bucketing_key(default='user_idx')

            if bucketing_key == "user_idx" and artifacts.user_features is not None:
                user_features = _make_bucketing(
                    artifacts.user_features,
                    bucketing_key=bucketing_key,
                    name="user_features"
                )
                item_features = artifacts.item_features
            elif bucketing_key == "item_idx" and artifacts.item_features is not None:
                user_features = artifacts.user_features
                item_features = _make_bucketing(
                    artifacts.item_features,
                    bucketing_key=bucketing_key,
                    name="item_features"
                )
            else:
                user_features = None  # artifacts.user_features ###!!!
                item_features = None  # artifacts.item_features ###!!!

            if do_optimization:
                if do_optimization:
                    budget_list = [5, 10] # todo#, 20, 50]
                    budget_time_list = []
                    with JobGroup("optimize", "optimization of first lvl models"):
                        for b_idx, b in enumerate(budget_list):
                            scenario.second_stage_model = EmptyWrap(
                                dump_path=artifacts.partial_train_path(model_class_name, str(b)))
                            logger.debug(f"optimization budget: {b}")
                            if b_idx != 0:
                                delta_b = b - budget_list[b_idx-1]
                                new_study = False  # continue study
                            else:
                                delta_b = b
                                new_study = True

                            with log_exec_timer(
                                    f"budget_{b}_optimization"
                            ) as timer:
                                params_found, fallback_params, metrics_values = scenario.optimize(
                                    train=artifacts.first_level_train_opt,
                                    test=artifacts.first_level_val_opt,
                                    param_borders=[
                                        FIRST_LEVELS_MODELS_PARAMS_BORDERS[model_class_name],
                                        None],
                                    k=k,
                                    budget=delta_b,
                                    criterion=NDCG(),
                                    new_study=new_study
                                )
                                mlflow.log_metric(f"NDCG.{k}_opt_b{b}", metrics_values[0])
                            budget_time_list.append(timer.duration)

                            with log_exec_timer(
                                    f"budget_{b}_fit_predict_calc_metrics"
                            ) as fit_predict_timer:
                                scenario.fit(
                                    log=artifacts.train,
                                    user_features=user_features,
                                    item_features=item_features)

                                test_recs = scenario.predict(
                                    log=artifacts.train,
                                    k=k,
                                    items=artifacts.train.select("item_idx").distinct(),
                                    users=artifacts.test.select("user_idx").distinct(),
                                    user_features=artifacts.user_features,
                                    item_features=artifacts.item_features,
                                    filter_seen_items=True
                                ).cache()

                                rel_cols = [c for c in test_recs.columns if c.startswith('rel_')]
                                assert len(rel_cols) == 1
                                test_recs = test_recs.withColumnRenamed(rel_cols[0], 'relevance')

                                logger.info("Estimating metrics...")
                                _estimate_and_report_budget_metrics(model_class_name, artifacts.test, test_recs, b=b)
                                test_recs = test_recs.withColumnRenamed('relevance', rel_cols[0])

                                test_recs.write.parquet(
                                    artifacts.partial_predicts_path(model_class_name, str(b)))

                                test_recs.unpersist()

                                if b_idx == 0:
                                    b_time = budget_time_list[-1]
                                else:
                                    b_time += budget_time_list[-1]

                            mlflow.log_metric(f"budget_{b}_duration", b_time+fit_predict_timer.duration)

                            logger.info("Saving the model")
                            with JobGroup("save", "saving the model"):

                                save(scenario,
                                     artifacts.partial_two_stage_scenario_path(model_class_name, str(b)),
                                     overwrite=True)

                # with JobGroup("optimize", "optimization of first lvl models"):
                #     best_params = scenario.optimize(train=artifacts.first_level_train_opt,
                #                                     test=artifacts.first_level_val_opt,
                #                                     param_borders=[FIRST_LEVELS_MODELS_PARAMS_BORDERS[model_class_name],
                #                                                    None],
                #                                     k=k,
                #                                     budget=20,
                #                                     criterion=NDCG(),
                #                                     )
                #     # for ALS only + fallback {"rank": [50, 200]}
                #     print(best_params)
                #     logger.info(best_params)
                #     print("cwd")
                #     print(os.getcwd())
                #     print(os.path.join(artifacts.base_path, f"best_params_{model_class_name}.pickle"))
                #     with open(os.path.join(artifacts.base_path, f"best_params_{model_class_name}.pickle"), "wb") as f:
                #         pickle.dump(best_params, f)
                #     return 0

            else:

                with JobGroup("fit", "fitting of two stage :empty wrap for second stage"):
                    scenario.fit(log=artifacts.train.limit(10_000), user_features=user_features, #todo
                                 item_features=item_features)
                    # print("scenario")
                    # print(scenario.first_level_models[0].__dict__)
                    # sim_m = scenario.first_level_models[0].similarity
                    # print("similarity")
                    # print(sim_m.count())
                    # print(sim_m.show(3))

                logger.info("Fit is ended. Predicting...")

                with JobGroup("predict", "predicting the test"):
                    test_recs = scenario.predict(
                        log=artifacts.train,
                        k=100,
                        items=artifacts.train.select("item_idx").distinct(),
                        users=artifacts.test.select("user_idx").distinct().limit(100), #todo
                        user_features=artifacts.user_features,
                        item_features=artifacts.item_features,
                        filter_seen_items=True
                    ).cache()

                logger.info(f"test_recs columns: {test_recs.columns}")
                rel_cols = [c for c in test_recs.columns if c.startswith('rel_')]
                assert len(rel_cols) == 1

                test_recs = test_recs.withColumnRenamed(rel_cols[0], 'relevance')
                test_recs = filter_seen_custom(recs=test_recs, log=artifacts.train, k=k,
                                               users=artifacts.train.select('user_idx').distinct())

                logger.info("Estimating metrics...")

                _estimate_and_report_metrics(model_class_name, artifacts.test, test_recs)

                test_recs = test_recs.withColumnRenamed('relevance', rel_cols[0])

                logger.info(f"partial predict save: {test_recs}")
                test_recs.write.parquet(artifacts.partial_predicts_path(model_class_name, optimized_postfix))

                test_recs.unpersist()

                logger.info("Saving the model")
                with JobGroup("save", "saving the model"):

                    save(scenario, artifacts.partial_two_stage_scenario_path(model_class_name, optimized_postfix),
                         overwrite=True)


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
        memory: int = DEFAULT_MEMORY):
    if second_model_type == "lama":
        mem_coeff = 0.3
    else:
        mem_coeff = 0.7

    with _init_spark_session(cpu, memory, mem_coeff=mem_coeff) as spark:

        mlflow.set_tracking_uri("http://node2.bdcl:8822")
        mlflow.set_experiment("paper_recsys")

        # checks MLFLOW_EXPERIMENT_ID for the experiment id
        model_name = f"{model_name}_{str(uuid.uuid4()).replace('-', '')}"
        with mlflow.start_run():
            _log_model_settings(
                model_name=model_name,
                model_type=second_model_type,
                k=k,
                artifacts=artifacts,
                model_params=second_model_params,
                model_config_path=second_model_config_path
            )

            scenario = PartialTwoStageScenario(
                base_path=artifacts.base_path,
                first_level_train_path=artifacts.first_level_train_path,
                second_level_positives_path=artifacts.second_level_positives_path,
                presplitted_data=True
            )
            if second_model_type == "lama":
                # TODO: make late parametrizing of these param
                second_stage_model = LamaWrap(params=second_model_params, config_path=second_model_config_path)
            elif second_model_type == "slama":
                second_stage_model = SlamaWrap(params=second_model_params,
                                               config_path="tabular_config.yml")  # second_model_config_path)
            else:
                raise RuntimeError(f"Currently supported model types: {['lama']}, but received {second_model_type}")

            with log_exec_timer("fit") as timer:

                # train_path = train_path.replace(".parquet", "") + "48.parquet"
                # train_path = train_path.replace(".parquet", "") + "_10neg.parquet"

                print(train_path)
                tr = spark.read.parquet(train_path)
                tr = tr.repartition(1192 * 3)

                # tr.repartition(48).write.mode("overwrite").parquet(os.path.join(train_path))
                # assert False
                # tr = tr.limit(int(tr.count() / 2))  ### Reducing train data for 2nd level model
                logger.info("second lvl model rows")
                logger.info(tr.count())

                second_stage_model.fit(tr)  #

            mlflow.log_metric(timer.name, timer.duration)

            with log_exec_timer("predict") as timer:
                recs = second_stage_model.predict(spark.read.parquet(first_level_predicts_path), k=k)
                recs = scenario._filter_seen(recs, artifacts.train, k, artifacts.train.select('user_idx').distinct())
                recs.write.parquet(artifacts.second_level_predicts_path(model_name))
                # recs.write.mode('overwrite').format('noop').save()

            mlflow.log_metric(timer.name, timer.duration)

            _estimate_and_report_metrics(model_name, artifacts.test, recs)

            with log_exec_timer("model_saving") as timer:
                second_level_model_path = artifacts.second_level_model_path(model_name)
                save_transformer(second_stage_model, second_level_model_path)

            mlflow.log_metric(timer.name, timer.duration)


def get_persistence_manager(name: Optional[str] = None):
    BUCKET_NUMS = 16
    PERSISTENCE_MANAGER_ENV_VAR = "PERSISTENCE_MANAGER"

    arg_vals = {
        "bucketed_datasets_folder": "/tmp",
        "bucket_nums": BUCKET_NUMS
    }

    class_name = name or os.environ.get(PERSISTENCE_MANAGER_ENV_VAR, None) or "CompositeBucketedPersistenceManager"
    clazz = getattr(persistence, class_name)
    sig = inspect.signature(getattr(clazz, "__init__"))

    ctr_arg_vals = {
        name: arg_vals.get(name, None if p.default is p.empty else p.default)
        for name, p in sig.parameters.items() if name != 'self'
    }

    none_val_args = [name for name, val in ctr_arg_vals.items() if val is None]
    assert len(none_val_args) == 0, f"Cannot instantiate class {class_name}. " \
                                    f"Values for the following arguments have not been found: {none_val_args}"

    return clazz(**ctr_arg_vals)


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
        mlflow_experiments_id: str = "pure",
        sampling: int = 0):
    if second_model_type == "lama":
        mem_coeff = 0.3
    else:
        mem_coeff = 0.7

    with _init_spark_session(cpu, memory, mem_coeff=mem_coeff) as spark:

        mlflow_tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI"
        )
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        os.environ["MLFLOW_EXPERIMENT_ID"] = mlflow_experiments_id
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_ID", "delete"))

        model_name = f"{model_name}_{str(uuid.uuid4()).replace('-', '')}"

        # run_name = "full_outer_join"
        run_name = "original"

        with mlflow.start_run(run_name=run_name):
            _log_model_settings(
                model_name=model_name,
                model_type=second_model_type,
                k=k,
                artifacts=artifacts,
                model_params=second_model_params,
                model_config_path=second_model_config_path
            )

            mlflow.log_param("lgbm_val_size", "100_000")

            do_fit = True
            if run_name == "original":
                outer_join = False
            else:
                outer_join = True

            roles = {"target": "target", "drop": ["user_idx", "item_idx", "na_i_log_features"]}

            if "msd" in train_path:
                # get dataset with negative sampling
                print("get dataset with negative sampling")
                train_path = train_path.replace(".parquet", "") + "_10neg.parquet"
            if outer_join:
                print("outer_join")
                train_path = train_path.replace(".parquet", "") + "_outer_join.parquet"
                first_level_predicts_path = first_level_predicts_path.replace(".parquet", "") + "_outer_join.parquet"

            print("train_path")
            print(train_path)
            train_df = spark.read.parquet(train_path)

            print("predict_path")
            print(first_level_predicts_path)
            test_df = spark.read.parquet(first_level_predicts_path)
            print("train partitions")
            print(train_df.rdd.getNumPartitions())
            print("test partitions")
            print(test_df.rdd.getNumPartitions())
            print("repartitioning")
            train_df = train_df.repartition(1192 * 3)
            print("train partitions")
            print(train_df.rdd.getNumPartitions())

            if sampling != 0:
                print("making negative sampling")
                neg = train_df.filter(train_df.target == 0)
                pos = train_df.filter(train_df.target == 1)
                neg_new = neg.sample(fraction=sampling * pos.count() / neg.count())  # with random seed
                train_df = pos.union(neg_new)
                train_df.cache()
                train_df.count()

            mlflow.log_param("neg_sampling", sampling)
            mlflow.log_param("train_count", train_df.count())

            cv = 5
            task = SparkTask("binary")
            score = task.get_dataset_metric()
            print("Starting reader")
            print(train_df.columns)
            print(train_df.count())
            sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False, n_jobs=-1,
                                         random_state=random.randint(0, 100))  # seed =42 by default
            sdataset = sreader.fit_read(train_df, roles=roles)
            print("Reader finished")
            iterator = SparkFoldsIterator(sdataset).convert_to_holdout_iterator()

            spark_ml_algo = SparkBoostLGBM(freeze_defaults=True, use_single_dataset_mode=True)
            spark_features_pipeline = SparkLGBSimpleFeatures()

            ml_pipe = SparkMLPipeline(
                ml_algos=[spark_ml_algo],
                pre_selection=None,
                features_pipeline=spark_features_pipeline,
                post_selection=None
            )
            if do_fit:
                print("starting fit_predict")
                with log_exec_timer("fit_predict") as timer:
                    oof_preds_ds = ml_pipe.fit_predict(iterator)
                mlflow.log_metric(timer.name, timer.duration)
                oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
                logger.info(f"OOF score: {oof_score}")
                print("OOF score")
                print(oof_score)
                print("oof_preds_ds_type")
                print(type(oof_preds_ds))
                print(oof_preds_ds.features)
                print(oof_preds_ds.data)

            # 1. first way (LAMA API)
            print("train df columns")
            print(train_df.columns)

            print("test df columns")
            print(test_df.columns)
            # test_sds = sreader.read(test_df, add_array_attrs=True)
            # test_preds_ds = ml_pipe.predict(test_sds)
            # test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
            # print(f"Test score (#1 way):")
            # print(test_score)

            if do_fit:
                rand_int = random.randint(0, 100)
                transformer = PipelineModel(stages=[sreader.transformer(add_array_attrs=True), ml_pipe.transformer()])
                transformer.write().overwrite().save(f"/tmp/reader_and_spark_ml_pipe_lgb_{rand_int}")

            pipeline_model = PipelineModel.load(f"/tmp/reader_and_spark_ml_pipe_lgb_{rand_int}")
            test_pred_df = pipeline_model.transform(test_df)
            # test_pred_df = transformer.transform(test_df)
            # test_pred_df = transformer.transform(test_df)
            # test_pred_df = test_pred_df.select(
            #     SparkDataset.ID_COLUMN,
            #     F.col(roles['target']).alias('target'),
            #     F.col(spark_ml_algo.prediction_feature).alias('prediction')
            # )
            print("spark_ml_algo.prediction_feature")
            print(spark_ml_algo.prediction_feature)
            test_pred_df = test_pred_df.withColumnRenamed(spark_ml_algo.prediction_feature, "relevance")
            # test_score = score(test_pred_df)
            # logger.info(f"Test score (#3 way): {test_score}")

            # top_k_recs = get_top_k_recs(
            #     recs=test_pred_df, k=k, id_type="idx"
            # ) #TODO remove

            top_k_recs = test_pred_df
            print("top-k-recs")

            # top_k_recs.write.mode("overwrite").parquet(os.path.join(artifacts.base_path, "pred_before.parquet"))

            firstelement = udf(lambda v: float(v[1]), DoubleType())
            top_k_recs = top_k_recs.withColumn("relevance", firstelement('relevance'))

            # recs = scenario._filter_seen(top_k_recs, artifacts.train, k, artifacts.train.select('user_idx').distinct())
            recs = filter_seen_custom(recs=top_k_recs, log=artifacts.train, k=100,
                                      users=artifacts.train.select('user_idx').distinct())

            # recs.write.parquet(artifacts.second_level_predicts_path(model_name))
            # recs.write.mode('overwrite').format('noop').save()
            print("recs columns")
            print(recs.columns)
            recs = recs.select("user_idx", "item_idx", "relevance", "temp_rank")

            # recs.write.mode("overwrite").parquet(os.path.join(artifacts.base_path, "recs_test.parquet"))
            # recs.write.mode("overwrite").parquet(os.path.join(artifacts.base_path, "predicts_pure2.parquet"))

            _estimate_and_report_metrics(model_name, artifacts.test, recs)

            print("feature importance")
            fi = spark_ml_algo.get_features_score()
            print(fi)


def _infer_trained_models_files(artifacts: ArtifactPaths, model_type: str = "", b='') -> List[FirstLevelModelFiles]: #model_type: str = "default"
    files = list_folder(artifacts.base_path)
    MODEL_NAMES = ["Word2VecRec", "SLIM", "ItemKNN", "ALSWrap"]

    def model_name(filename: str) -> str:
        mname = filename.split('_')[-2]
        if mname in MODEL_NAMES:
            return mname
        else:
            mname = filename.split('_')[-3]
            if mname in MODEL_NAMES:
                return mname
            else:
                print(f"model name {mname} does not recognised as one of {MODEL_NAMES} models")
                assert False

    def get_files(prefix: str, model_type=model_type, b='') -> Dict[str, str]:
        print(f"b in get_files: {b}")

        if b != "":
            return {
                model_name(file):
                artifacts.make_path(file) for file in files if file.startswith(prefix) and model_type in file and f"_{b}" in file
            }
        else:
            return {
                model_name(file):
                    artifacts.make_path(file) for file in files if
                file.startswith(prefix) and model_type in file
            }

    partial_predicts = get_files('partial_predict',b=b)
    partial_trains = get_files('partial_train', b=b)
    partial_scenarios = get_files('two_stage_scenario', b=b)  # It is required for making missed predictions
    finished_model_names = set(partial_predicts).intersection(partial_trains).intersection(partial_scenarios)

    first_lvl_model_files = [
        FirstLevelModelFiles(
            model_name=mname,
            train_path=partial_trains[mname],
            predict_path=partial_predicts[mname],
            model_path=partial_scenarios[mname]
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
            full_outer_join: bool = False
    ):
        # ======== debug lines ==========
        for model_name in model_names:
            print(model_name)
        for df in partial_dfs:
            df.printSchema()
        # ===============================

        if mode == 'union':
            required_pairs = (
                functools.reduce(
                    lambda acc, x: acc.unionByName(x),
                    (df.select('user_idx', 'item_idx') for df in partial_dfs)
                ).distinct()
            )
        else:
            # "leading_<model_name>"
            leading_model_name = mode.split('_')[-1]
            required_pairs = (
                partial_dfs[model_names.index(leading_model_name)]
                .select('user_idx', 'item_idx')
                .distinct()
            )

        logger.info("Selecting missing pairs")

        missing_pairs = [
            required_pairs.join(df, on=['user_idx', 'item_idx'], how='anti').select('user_idx',
                                                                                    'item_idx').distinct()
            for df in partial_dfs
        ]

        def get_rel_col(df: DataFrame) -> str:

            logger.info(f"get_rel_col func columns: {df.columns}")
            rel_col = [c for c in df.columns if c.startswith('rel_')][0]

            # try:
            # except IndexError:
            #     rel_col = "relevance"
            return rel_col

        def make_missing_predictions(model, mpairs: DataFrame, partial_df: DataFrame) -> DataFrame:

            # with _init_spark_session(DEFAULT_CPU, DEFAULT_MEMORY) as spark:

            mpairs = mpairs.cache()

            if mpairs.count() == 0:
                return partial_df

            current_pred = model._predict_pairs(
                mpairs,
                log=artifacts.train,
                user_features=artifacts.user_features,
                item_features=artifacts.item_features
            ).withColumnRenamed('relevance', get_rel_col(partial_df))

            features = get_first_level_model_features(
                model=model,
                pairs=current_pred.select(
                    "user_idx", "item_idx"
                ),
                user_features=artifacts.user_features.cache() if artifacts.user_features is not None else None,
                item_features=artifacts.item_features.cache() if artifacts.item_features is not None else None,
                prefix=f"m_0",
            )
            current_pred_with_features = join_with_col_renaming(
                left=current_pred,
                right=features,
                on_col_name=["user_idx", "item_idx"],
                how="left",
            )
            current_pred_with_features.write.mode("overwrite").parquet(os.path.join("hdfs://node21.bdcl:9000",
                                                                                    artifacts.base_path,
                                                                                    f"current_pred_with_features_{model}.parquet"))
            current_pred_with_features = sprk_ses.read.parquet(os.path.join("hdfs://node21.bdcl:9000",
                                                                            artifacts.base_path,
                                                                            f"current_pred_with_features_{model}.parquet"))

            # return current_pred_with_features
            return partial_df.unionByName(current_pred_with_features.select(*partial_df.columns))

        common_cols = functools.reduce(
            lambda acc, cols: acc.intersection(cols),
            [set(df.columns) for df in partial_dfs]
        )
        common_cols.remove('user_idx')
        common_cols.remove('item_idx')

        logger.info(f"common cols: {common_cols}")
        logger.info("Making missing predictions")
        logger.info(f"partial df cols before drop: {partial_dfs[0].columns}", )
        logger.info(f"partial df cols after drop: {partial_dfs[0].drop(*common_cols)}")
        extended_train_dfs = [
            make_missing_predictions(model, mpairs, partial_df.drop(*common_cols))
            for model, mpairs, partial_df in zip(models, missing_pairs, partial_dfs)
        ]

        features_for_required_pairs_df = [
            required_pairs.join(df.select('user_idx', 'item_idx', *common_cols), on=['user_idx', 'item_idx'])
            for df in partial_dfs
        ]
        logger.info("Collecting required pairs with features")

        required_pairs_with_features = functools.reduce(
            lambda acc, df: acc.unionByName(df),  # !!
            features_for_required_pairs_df
        ).drop_duplicates(['user_idx', 'item_idx'])

        # we apply left here because some algorithms like itemknn cannot predict beyond their inbuilt top
        new_train_df = functools.reduce(
            lambda acc, x: acc.join(x, on=['user_idx', 'item_idx'], how='left'),
            extended_train_dfs,
            required_pairs_with_features
        )

        logger.info(f"Saving new parquet in {combined_df_path}")
        logger.info(f"combiner columns is {new_train_df.columns}")

        new_train_df.write.parquet(combined_df_path)  # mode("overwrite")

        logger.info("Saved")

    @staticmethod
    def do_combine_datasets(
            artifacts: ArtifactPaths,
            combined_train_path: str,
            combined_predicts_path: str,
            desired_models: Optional[List[str]] = None,
            mode: str = "union",
            model_type: str = "",
            b: str = ""
    ):
        with _init_spark_session() as spark:

            mlflow.set_tracking_uri("http://node2.bdcl:8822")
            mlflow.set_experiment("paper_recsys")

            with mlflow.start_run():

                _log_model_settings(
                    model_name="",
                    model_type="",
                    model_params={},
                    k=k,
                    artifacts=artifacts,

                )

                train_exists = do_path_exists(combined_train_path)
                predicts_exists = do_path_exists(combined_predicts_path)

                assert train_exists == predicts_exists, \
                    f"The both datasets should either exist or be absent. " \
                    f"Train {combined_train_path}, Predicts {combined_predicts_path}"

                if train_exists and predicts_exists:
                    logger.info(f"Datasets {combined_train_path} and {combined_predicts_path} exist. Nothing to do.")

                    return

                logger.info("Inferring trained models and their files")
                logger.info(f"Base path: {artifacts.base_path}")
                model_files = _infer_trained_models_files(artifacts, model_type, b)

                logger.info(f"Found the following models that have all required files: "
                            f"{[mfiles.model_name for mfiles in model_files]}")
                found_mpaths = '\n'.join([mfiles.model_path for mfiles in model_files])
                logger.info(f"Found models paths:\n {found_mpaths}")
                if desired_models is not None:
                    logger.info(f"Checking availability of the desired models: {desired_models}")
                    model_files = [mfiles for mfiles in model_files if mfiles.model_name.lower() in desired_models]
                    not_available_models = set(desired_models).difference(
                        mfiles.model_name.lower() for mfiles in model_files
                    )
                    assert len(not_available_models) == 0, f"Not all desired models available: {not_available_models}"

                used_mpaths = '\n'.join([mfiles.model_path for mfiles in model_files])
                logger.info(f"Continue with models:\n {used_mpaths}")

                # creating combined train
                logger.info("Creating combined train")
                model_names = [mfiles.model_name.lower() for mfiles in model_files]
                partial_train_dfs = [spark.read.parquet(mfiles.train_path) for mfiles in
                                     model_files]  # train files for 1st lvl models
                partial_predicts_dfs = [spark.read.parquet(mfiles.predict_path) for mfiles in model_files]
                logger.info("Loading models")
                models = [
                    cast(BaseRecommender, cast(PartialTwoStageScenario, load_model(mfiles.model_path))
                         .one_stage_scenario.first_level_models[0])
                    for mfiles in model_files
                ]
                logger.info("Models loaded")

                with log_exec_timer(
                        "train_combiner"
                ) as train_combiner_timer:
                    DatasetCombiner._combine(
                        artifacts=artifacts,
                        mode=mode,
                        model_names=model_names,
                        models=models,
                        partial_dfs=partial_train_dfs,
                        combined_df_path=combined_train_path,
                        sprk_ses=spark
                    )

                with log_exec_timer(
                        "predicts_combiner"
                ) as predicts_combiner_timer:
                    logger.info("Creating combined predicts")

                    DatasetCombiner._combine(
                        artifacts=artifacts,
                        mode=mode,
                        model_names=model_names,
                        models=models,
                        partial_dfs=partial_predicts_dfs,
                        combined_df_path=combined_predicts_path,
                        sprk_ses=spark
                    )
                # common_cols.remove('target')
                # partial_predicts_dfs = [spark.read.parquet(mfiles.predict_path) for mfiles in model_files]
                # full_predicts_df = functools.reduce(
                #     lambda acc, x: acc.join(x.drop(*common_cols), on=['user_idx', 'item_idx']),
                #     partial_predicts_dfs,
                #     partial_predicts_dfs[0].select('user_idx', 'item_idx', *common_cols)
                # )
                # full_predicts_df.write.parquet(combined_predicts_path)
                mlflow.log_metric("train_combiner_time", train_combiner_timer.duration)
                mlflow.log_metric("pred_combiner_time", predicts_combiner_timer.duration)
                mlflow.log_param("combined_train_path", combined_train_path)
                mlflow.log_param("combined_predicts_path", combined_predicts_path)
                mlflow.log_param("budget", b)

                logger.info("Combining finished")


if __name__ == "__main__":

    spark = get_cluster_session()

    config_filename = os.environ.get(TASK_CONFIG_FILENAME_ENV_VAR, "task_config.pickle")

    with open(config_filename, "rb") as f:
        task_config = pickle.load(f)

    print("Task configs")
    for k, v in task_config.items():
        print(k, v)

    print("config filename")
    print(config_filename)

    if config_filename.split('_')[2] == "2lvl":  # TODO: refactor
        do_fit_predict_second_level(**task_config)
    elif config_filename.split('_')[2] == "combiner":  # TODO refactor
        DatasetCombiner.do_combine_datasets(**task_config)
    elif config_filename.split('_')[2] == "pure":
        do_fit_predict_second_level_pure(**task_config)
    else:
        do_fit_predict_first_level_model(**task_config)
        # do_fit_predict_one_stage(**task_config)
