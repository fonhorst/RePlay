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

import mlflow
from pyspark.sql import functions as sf, SparkSession, DataFrame

import replay
from dag_entities import ArtifactPaths, DEFAULT_CPU, DEFAULT_MEMORY, TASK_CONFIG_FILENAME_ENV_VAR
from replay.data_preparator import Indexer
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.model_handler import save, Splitter, load, ALSWrap
from replay.models import PopRec
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario
from replay.scenarios.two_stages.reranker import LamaWrap, ReRanker
from replay.scenarios.two_stages.slama_reranker import SlamaWrap
from replay.scenarios.two_stages.two_stages_scenario import get_first_level_model_features
from replay.session_handler import State
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info, save_transformer, log_exec_timer, do_path_exists, JobGroup, load_transformer, \
    list_folder, join_with_col_renaming

logger = logging.getLogger('airflow.task')


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
        self._presplitted_data = presplitted_data
        if first_level_item_features_transformer:
            self.first_level_item_features_transformer = first_level_item_features_transformer
        if first_level_user_features_transformer:
            self.first_level_user_features_transformer = first_level_user_features_transformer

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
                spark.read.parquet(self._second_level_positives_path).repartition(spark.sparkContext.defaultParallelism),
                bucketing_key=_get_bucketing_key(default='user_idx'),
                name="second_level_positive"
            )

            return first_level_train, second_level_positive

        first_level_train, second_level_positive = super()._split_data(log)

        first_level_train.write.parquet(self._first_level_train_path, mode='overwrite')
        second_level_positive.write.parquet(self._second_level_positives_path, mode='overwrite')

        return first_level_train, second_level_positive

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

            candidates_with_features = candidates_with_features\
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
            os.environ.get("REPLAY_JAR_PATH", '../../scala/target/scala-2.12/replay_2.12-0.1.jar'),
            os.environ.get("SLAMA_JAR_PATH", '../../../LightAutoML/jars/spark-lightautoml_2.12-0.1.1.jar')
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


def do_dataset_splitting(artifacts: ArtifactPaths, partitions_num: int):
    from replay.data_preparator import DataPreparator
    with _init_spark_session(DEFAULT_CPU, DEFAULT_MEMORY):
        data = artifacts.log

        if 'timestamp' in data.columns:
            data = data.withColumn('timestamp', sf.col('timestamp').cast('long'))

        # splitting on train and test
        preparator = DataPreparator()

        if artifacts.dataset.name == 'netflix_small':
            data = data.withColumn('timestamp', sf.col('timestamp').cast('long'))
            preparator.setColumnsMapping({"user_id": "user_idx", "item_id": "item_idx",
                                 "relevance": "relevance", "timestamp": "timestamp"})
            log = preparator.transform(
                data
            ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")

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
            preparator.setColumnsMapping({"user_id": "user_id", "item_id": "item_id",
                                    "relevance": "rating", "timestamp": "timestamp"})
            log = preparator.transform(
                data
            ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")

            # train/test split ml
            train_spl = DateSplitter(
                test_start=0.2,
                drop_cold_items=True,
                drop_cold_users=True,
            )
        elif artifacts.dataset.name.startswith('msd'):
            preparator.setColumnsMapping({"user_id": "user_id", "item_id": "item_id", "relevance": "play_count"})
            log = preparator.transform(
                data
            ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")

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
                or (artifacts.user_features is None and artifacts.item_features is not None),\
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


def do_presplit_data(artifacts: ArtifactPaths, cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    with _init_spark_session(cpu, memory):
        flt_exists = do_path_exists(artifacts.first_level_train_path)
        slp_exists = do_path_exists(artifacts.second_level_positives_path)

        if flt_exists != slp_exists:
            raise Exception(
                f"The paths should be both either existing or absent. "
                f"But: {artifacts.first_level_train_path} exists == {flt_exists}, "
                f"{artifacts.second_level_positives_path} exists == {slp_exists}."
            )

        if flt_exists and slp_exists:
            return

        scenario = PartialTwoStageScenario(
            base_path=artifacts.base_path,
            first_level_train_path=artifacts.first_level_train_path,
            second_level_positives_path=artifacts.second_level_positives_path,
            presplitted_data=False
        )

        scenario._split_data(artifacts.train)


def do_fit_predict_first_level_model(artifacts: ArtifactPaths,
                                     model_class_name: str,
                                     model_kwargs: Dict,
                                     k: int,
                                     cpu: int = DEFAULT_CPU,
                                     memory: int = DEFAULT_MEMORY):
    with _init_spark_session(cpu, memory):
        _log_model_settings(
            model_name=model_class_name,
            model_type=model_class_name,
            k=k,
            artifacts=artifacts,
            model_params=model_kwargs,
            model_config_path=None
        )

        first_level_model = _get_model(artifacts, model_class_name, model_kwargs)

        if artifacts.user_features is not None:
            user_feature_transformer = load_transformer(artifacts.user_features_transformer_path)
        else:
            user_feature_transformer = None
        if artifacts.item_features is not None:
            item_feature_transformer = load_transformer(artifacts.item_features_transformer_path)
        else:
            item_feature_transformer = None
        history_based_transformer = load_transformer(artifacts.history_based_transformer_path)

        scenario = PartialTwoStageScenario(
            base_path=artifacts.base_path,
            first_level_train_path=artifacts.first_level_train_path,
            second_level_positives_path=artifacts.second_level_positives_path,
            second_level_train_path=artifacts.partial_train_path(model_class_name),
            first_level_models=first_level_model,
            first_level_item_features_transformer=item_feature_transformer,
            first_level_user_features_transformer=user_feature_transformer,
            custom_features_processor=history_based_transformer,
            presplitted_data=True
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
            user_features = artifacts.user_features
            item_features = artifacts.item_features

        with JobGroup("fit", "fitting of two stage"):
            scenario.fit(log=artifacts.train, user_features=user_features, item_features=item_features)

        logger.info("Fit is ended. Predicting...")

        with JobGroup("predict", "predicting the test"):
            test_recs = scenario.predict(
                log=artifacts.train,
                k=k,
                user_features=artifacts.user_features,
                item_features=artifacts.item_features,
                filter_seen_items=True
            ).cache()

        test_recs.write.parquet(artifacts.partial_predicts_path(model_class_name))

        logger.info("Estimating metrics...")

        rel_cols = [c for c in test_recs.columns if c.startswith('rel_')]
        assert len(rel_cols) == 1

        test_recs = test_recs.withColumnRenamed(rel_cols[0], 'relevance')

        _estimate_and_report_metrics(model_class_name, artifacts.test, test_recs)

        test_recs.unpersist()

        logger.info("Saving the model")
        with JobGroup("save", "saving the model"):
            save(scenario, artifacts.partial_two_stage_scenario_path(model_class_name))


# this is @task
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
                second_stage_model = SlamaWrap(params=second_model_params, config_path=second_model_config_path)
            else:
                raise RuntimeError(f"Currently supported model types: {['lama']}, but received {second_model_type}")

            with log_exec_timer("fit") as timer:
                second_stage_model.fit(spark.read.parquet(train_path))

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


def _infer_trained_models_files(artifacts: ArtifactPaths) -> List[FirstLevelModelFiles]:
    files = list_folder(artifacts.base_path)

    def model_name(filename: str) -> str:
        return filename.split('_')[-2]

    def get_files(prefix: str) -> Dict[str, str]:
        return {model_name(file): artifacts.make_path(file) for file in files if file.startswith(prefix)}

    partial_predicts = get_files('partial_predict')
    partial_trains = get_files('partial_train')
    partial_scenarios = get_files('two_stage_scenario')

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
            combined_df_path: str
    ):
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

        # TODO: probably, we should cache required_pairs here beforehand
        missing_pairs = [
            # TODO: unneccessary distinct in the end?
            required_pairs.join(df, on=['user_idx', 'item_idx'], how='anti').select('user_idx', 'item_idx').distinct()
            for df in partial_dfs
        ]

        def get_rel_col(df: DataFrame) -> str:
            return [c for c in df.columns if c.startswith('rel_')][0]

        def make_missing_predictions(model, mpairs: DataFrame, partial_df: DataFrame) -> DataFrame:
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
            return partial_df.unionByName(current_pred_with_features.select(*partial_df.columns))

        # TODO: am i correct on calculating this features?
        common_cols = functools.reduce(
            lambda acc, cols: acc.intersection(cols),
            [set(df.columns) for df in partial_dfs]
        )
        common_cols.remove('user_idx')
        common_cols.remove('item_idx')

        features_for_required_pairs_df = [
            required_pairs.join(df.select('user_idx', 'item_idx', *common_cols), on=['user_idx', 'item_idx'])
            for df in partial_dfs
        ]

        required_pairs_with_features = functools.reduce(
            lambda acc, df: acc.unionByName(df),
            features_for_required_pairs_df
        ).drop_duplicates(['user_idx', 'item_idx'])

        # predict features for missing pairs in each model's predict
        extended_train_dfs = [
            make_missing_predictions(model, mpairs, partial_df.drop(*common_cols))
            for model, mpairs, partial_df in zip(models, missing_pairs, partial_dfs)
        ]

        # we apply left here because some algorithms like itemknn cannot predict beyond their inbuilt top
        new_train_df = functools.reduce(
            lambda acc, x: acc.join(x, on=['user_idx', 'item_idx'], how='left'),
            extended_train_dfs,
            required_pairs_with_features
        )

        new_train_df.write.parquet(combined_df_path)

    @staticmethod
    def do_combine_datasets(
            artifacts: ArtifactPaths,
            combined_train_path: str,
            combined_predicts_path: str,
            desired_models: Optional[List[str]] = None,
            mode: str = 'union'):
        with _init_spark_session() as spark:

            train_exists = do_path_exists(combined_train_path)
            predicts_exists = do_path_exists(combined_predicts_path)

            assert train_exists == predicts_exists, \
                f"The both datasets should either exist or be absent. " \
                f"Train {combined_train_path}, Predicts {combined_predicts_path}"

            if train_exists and predicts_exists:
                logger.info(f"Datasets {combined_train_path} and {combined_predicts_path} exist. Nothing to do.")
                return

            logger.info("Inferring traine models and their files")
            model_files = _infer_trained_models_files(artifacts)

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
            partial_train_dfs = [spark.read.parquet(mfiles.train_path) for mfiles in model_files]
            partial_predicts_dfs = [spark.read.parquet(mfiles.predict_path) for mfiles in model_files]

            models = [
                cast(BaseRecommender, cast(PartialTwoStageScenario, load_model(mfiles.model_path))
                     .first_level_models[0])
                for mfiles in model_files
            ]

            DatasetCombiner._combine(
                artifacts=artifacts,
                mode=mode,
                model_names=model_names,
                models=models,
                partial_dfs=partial_train_dfs,
                combined_df_path=combined_train_path
            )

            # combine predicts
            logger.info("Creating combined predicts")

            DatasetCombiner._combine(
                artifacts=artifacts,
                mode=mode,
                model_names=model_names,
                models=models,
                partial_dfs=partial_predicts_dfs,
                combined_df_path=combined_predicts_path
            )
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

    config_filename = os.environ.get(TASK_CONFIG_FILENAME_ENV_VAR, "task_config.pickle")

    with open(config_filename, "rb") as f:
        task_config = pickle.load(f)

    print("Task config artifacts:")
    print(f"{task_config['artifacts']}")

    do_fit_predict_first_level_model(**task_config)
