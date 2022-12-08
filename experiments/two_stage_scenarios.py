import functools
import importlib
import itertools
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, cast, Optional, List, Union, Tuple, Sequence, Any

import mlflow
import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.utils.helpers import chain, cross_downstream
from kubernetes.client import models as k8s
from pyspark.sql import functions as sf, SparkSession, DataFrame

import replay
from replay.data_preparator import DataPreparator
from replay.experiment import Experiment
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.metrics import MAP, NDCG, HitRate
from replay.model_handler import save, Splitter, load, ALSWrap
from replay.models import PopRec
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario
from replay.scenarios.two_stages.reranker import LamaWrap, ReRanker
from replay.scenarios.two_stages.slama_reranker import SlamaWrap
from replay.session_handler import State
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info, save_transformer, log_exec_timer, do_path_exists

logger = logging.getLogger(__name__)

DEFAULT_CPU = 6
BIG_CPU = 12
EXTRA_BIG_CPU = 20

DEFAULT_MEMORY = 30
BIG_MEMORY = 40
EXTRA_BIG_MEMORY = 80


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    log_path: str
    user_features_path: Optional[str] = None
    item_features_path: Optional[str] = None


dense_hnsw_params = {
    "method": "hnsw",
    "space": "negdotprod",
    "M": 100,
    "efS": 2000,
    "efC": 2000,
    "post": 0,
    # hdfs://node21.bdcl:9000
    # "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark.sparkContext.applicationId}",
    "build_index_on": "executor"
}


FIRST_LEVELS_MODELS_PARAMS = {
    "replay.models.als.ALSWrap": {"rank": 100, "seed": 42, "nmslib_hnsw_params": dense_hnsw_params},
    "replay.models.knn.ItemKNN": {"num_neighbours": 1000},
    "replay.models.cluster.ClusterRec": {"num_clusters": 100},
    "replay.models.slim.SLIM": {"seed": 42},
    "replay.models.word2vec.Word2VecRec": {"rank": 100, "seed": 42, "nmslib_hnsw_params": dense_hnsw_params},
    "replay.models.ucb.UCB": {"seed": 42}
}

MODELNAME2FULLNAME = {
    "als": "replay.models.als.ALSWrap",
    "itemknn": "replay.models.knn.ItemKNN",
    "cluster": "replay.models.cluster.ClusterRec",
    "slim": "replay.models.slim.SLIM",
    "word2vec": "replay.models.word2vec.Word2VecRec",
    "ucb": "replay.models.ucb.UCB"
}

DATASETS = {
    dataset.name: dataset  for dataset in [
        DatasetInfo(
            name="ml100k",
            log_path="/opt/spark_data/replay/ml100k_ratings.csv",
            user_features_path="/opt/spark_data/replay/ml100k_users.csv",
            item_features_path="/opt/spark_data/replay/ml100k_items.csv"
        ),

        DatasetInfo(
            name="ml1m",
            log_path="/opt/spark_data/replay/ml1m_ratings.csv",
            user_features_path="/opt/spark_data/replay/ml1m_users.csv",
            item_features_path="/opt/spark_data/replay/ml1m_items.csv"
        ),

        DatasetInfo(
            name="ml25m",
            log_path="/opt/spark_data/replay/ml25m_ratings.csv"
        ),

        DatasetInfo(
            name="msd",
            log_path="/opt/spark_data/replay_datasets/MillionSongDataset/original.parquet"
        )
    ]
}


def _get_models_params(*model_names: str) -> Dict[str, Any]:
    return {
        MODELNAME2FULLNAME[model_name]: FIRST_LEVELS_MODELS_PARAMS[MODELNAME2FULLNAME[model_name]]
        for model_name in model_names
    }


big_executor_config = {
    "pod_override": k8s.V1Pod(
        spec=k8s.V1PodSpec(
            containers=[
                k8s.V1Container(
                    name="base",
                    resources=k8s.V1ResourceRequirements(
                        requests={"cpu": BIG_CPU, "memory": f"{BIG_MEMORY}Gi"},
                        limits={"cpu": BIG_CPU, "memory": f"{BIG_MEMORY}Gi"})
                )
            ],
        )
    ),
}


extra_big_executor_config = {
    "pod_override": k8s.V1Pod(
        spec=k8s.V1PodSpec(
            containers=[
                k8s.V1Container(
                    name="base",
                    resources=k8s.V1ResourceRequirements(
                        requests={"cpu": EXTRA_BIG_CPU, "memory": f"{EXTRA_BIG_MEMORY}Gi"},
                        limits={"cpu": EXTRA_BIG_CPU, "memory": f"{EXTRA_BIG_MEMORY}Gi"})
                )
            ],
        )
    ),
}


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
            return (
                spark.read.parquet(self._first_level_train_path).repartition(spark.sparkContext.defaultParallelism),
                spark.read.parquet(self._second_level_positives_path).repartition(spark.sparkContext.defaultParallelism)
            )

        first_level_train, second_level_positive = super()._split_data(log)

        first_level_train.write.parquet(self._first_level_train_path)
        second_level_positive.write.parquet(self._second_level_positives_path)

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


@dataclass
class ArtifactPaths:
    base_path: str
    log_path: str
    user_features_path: Optional[str] = None
    item_features_path: Optional[str] = None
    uid: str = f"{uuid.uuid4()}".replace('-', '')
    partial_train_prefix: str = "partial_train"
    partial_predict_prefix: str = "partial_predict"
    second_level_model_prefix: str = "second_level_model"
    second_level_predicts_prefix: str = "second_level_predicts"
    first_level_train_predefined_path: Optional[str] = None
    second_level_positives_predefined_path: Optional[str] = None

    template_fields: Sequence[str] = ("base_path",)

    @property
    def train_path(self) -> str:
        return os.path.join(self.base_path, "train.parquet")

    @property
    def test_path(self) -> str:
        return os.path.join(self.base_path, "test.parquet")

    @property
    def two_stage_scenario_path(self) -> str:
        return os.path.join(self.base_path, "two_stage_scenario")

    @property
    def partial_train_paths(self) -> List[str]:
        return [
            os.path.join(self.base_path, path)
            for path in os.listdir(self.base_path)
            if path.startswith(self.partial_train_prefix)
        ]

    @property
    def partial_predicts_paths(self) -> List[str]:
        return [
            os.path.join(self.base_path, path)
            for path in os.listdir(self.base_path)
            if path.startswith(self.partial_predict_prefix)
        ]

    @property
    def full_second_level_train_path(self) -> str:
        return os.path.join(self.base_path, "full_second_level_train.parquet")

    @property
    def full_second_level_predicts_path(self) -> str:
        return os.path.join(self.base_path, "full_second_level_predicts.parquet")

    @property
    def log(self) -> DataFrame:
        if self.log_path.endswith('.csv'):
            return self._get_session().read.csv(self.log_path, header=True)

        if self.log_path.endswith('.parquet'):
            return self._get_session().read.parquet(self.log_path)

        raise Exception("Unsupported format of the file, only csv and parquet are supported")

    @property
    def user_features(self) -> Optional[DataFrame]:
        if self.user_features_path is None:
            return None

        return (
            self._get_session().read.csv(self.user_features_path, header=True)
            .withColumnRenamed('user_id', 'user_idx')
            .withColumn('user_idx', sf.col('user_idx').cast('int'))
            .drop('_c0')
        )

    @property
    def item_features(self) -> Optional[DataFrame]:
        if self.item_features_path is None:
            return None

        return (
            self._get_session().read.csv(self.item_features_path, header=True)
            .withColumnRenamed('item_id', 'item_idx')
            .withColumn('item_idx', sf.col('item_idx').cast('int'))
            .drop('_c0')
        )

    @property
    def train(self) -> DataFrame:
        return self._get_session().read.parquet(self.train_path)

    @property
    def test(self) -> DataFrame:
        return self._get_session().read.parquet(self.test_path)

    @property
    def full_second_level_train(self) -> DataFrame:
        return self._get_session().read.parquet(self.full_second_level_train_path)

    @property
    def full_second_level_predicts(self) -> DataFrame:
        return self._get_session().read.parquet(self.full_second_level_predicts_path)

    @property
    def first_level_train_path(self) -> str:
        return self.first_level_train_predefined_path if self.first_level_train_predefined_path is not None \
            else os.path.join(self.base_path, "first_level_train.parquet")

    @property
    def second_level_positives_path(self) -> str:
        return self.second_level_positives_predefined_path if self.second_level_positives_predefined_path is not None \
            else os.path.join(self.base_path, "second_level_positives.parquet")

    def partial_two_stage_scenario_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path, f"two_stage_scenario_{model_cls_name.split('.')[-1]}_{self.uid}")

    def model_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path, f"model_{model_cls_name.replace('.', '__')}_{self.uid}")

    def hnsw_index_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path, f"hnsw_model_index_{model_cls_name.replace('.', '__')}_{self.uid}")

    def partial_train_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path,
                            f"{self.partial_train_prefix}_{model_cls_name.replace('.', '__')}_{self.uid}.parquet")

    def partial_predicts_path(self, model_cls_name: str):
        return os.path.join(self.base_path,
                            f"{self.partial_predict_prefix}_{model_cls_name.replace('.', '__')}_{self.uid}.parquet")

    def second_level_model_path(self, model_name: str) -> str:
        return os.path.join(self.base_path, f"{self.second_level_model_prefix}_{model_name}")

    def second_level_predicts_path(self, model_name: str) -> str:
        return os.path.join(self.base_path, f"{self.second_level_predicts_prefix}_{model_name}.parquet")

    def _get_session(self) -> SparkSession:
        return SparkSession.getActiveSession()


@contextmanager
def _init_spark_session(cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY) -> SparkSession:
    if os.environ.get('SCRIPT_ENV', 'local') == 'cluster':
        return SparkSession.builder.getOrCreate()

    jars = [
        os.environ.get("REPLAY_JAR_PATH", 'scala/target/scala-2.12/replay_2.12-0.1.jar'),
        os.environ.get("SLAMA_JAR_PATH", '../LightAutoML/jars/spark-lightautoml_2.12-0.1.1.jar')
    ]
    spark = (
        SparkSession
        .builder
        .config("spark.jars", ",".join(jars))
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
        .config("spark.sql.shuffle.partitions", str(cpu * 3))
        .config("spark.default.parallelism", str(cpu * 3))
        .config("spark.driver.maxResultSize", "6g")
        .config("spark.driver.memory", f"{memory}g")
        .config("spark.executor.memory", f"{memory}g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
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

    if class_name in ['ALSWrap', 'Word2VecRec'] and 'nmslib_hnsw_params' in model_kwargs:
        model_kwargs['nmslib_hnsw_params']["index_path"] = artifacts.hnsw_index_path(model_class_name)

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
    dataset_name, _ = os.path.splitext(os.path.basename(artifacts.log_path))
    mlflow.log_param("model", model_name)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("model_config_path", model_config_path)
    mlflow.log_param("k", k)
    mlflow.log_param("experiment_path", artifacts.base_path)
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("dataset_path", artifacts.log_path)
    mlflow.log_dict(model_params, "model_params.json")


@task
def dataset_splitting(artifacts: ArtifactPaths, partitions_num: int, dataset_name: str):
    with _init_spark_session(DEFAULT_CPU, DEFAULT_MEMORY):
        data = (
            artifacts.log
            .withColumn('user_id', sf.col('user_id').cast('int'))
            .withColumn('item_id', sf.col('item_id').cast('int'))
            .withColumn('timestamp', sf.col('timestamp').cast('int'))
        )

        # splitting on train and test
        preparator = DataPreparator()

        if dataset_name.startswith('ml'):
            log = preparator.transform(
                columns_mapping={"user_id": "user_id", "item_id": "item_id",
                                 "relevance": "rating", "timestamp": "timestamp"},
                data=data
            ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")
        else:
            raise Exception(f"Unsupported dataset name: {dataset_name}")

        print(get_log_info(log))

        log = log.repartition(partitions_num).cache()
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
        logger.info(f'train info: \n{get_log_info(train)}')
        logger.info(f'test info:\n{get_log_info(test)}')

        # writing data
        os.makedirs(artifacts.base_path, exist_ok=True)

        assert train.count() > 0
        assert test.count() > 0

        train.drop('_c0').write.parquet(artifacts.train_path)
        test.drop('_c0').write.parquet(artifacts.test_path)


@task
def init_refitable_two_stage_scenario(artifacts: ArtifactPaths):
    with _init_spark_session(DEFAULT_CPU, DEFAULT_MEMORY):
        scenario = RefitableTwoStageScenario(base_path=artifacts.base_path)

        scenario.fit(log=artifacts.train, user_features=artifacts.user_features, item_features=artifacts.item_features)

        save(scenario, artifacts.two_stage_scenario_path)


@task
def presplit_data(artifacts: ArtifactPaths, cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
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


def fit_predict_first_level_model(artifacts: ArtifactPaths,
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

        scenario = PartialTwoStageScenario(
            base_path=artifacts.base_path,
            first_level_train_path=artifacts.first_level_train_path,
            second_level_positives_path=artifacts.second_level_positives_path,
            second_level_train_path=artifacts.partial_train_path(model_class_name),
            first_level_models=first_level_model,
            presplitted_data=True
        )

        scenario.fit(log=artifacts.train, user_features=artifacts.user_features, item_features=artifacts.item_features)

        save(scenario, artifacts.partial_two_stage_scenario_path(model_class_name))

        test_recs = scenario.predict(
            log=artifacts.test,
            k=k,
            user_features=artifacts.user_features,
            item_features=artifacts.item_features,
            filter_seen_items=True
        )

        test_recs.write.parquet(artifacts.partial_predicts_path(model_class_name))


# this is @task
def first_level_fitting(artifacts: ArtifactPaths,
                        model_class_name: str, model_kwargs: Dict, k: int,
                        cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    with _init_spark_session(cpu, memory):
        # checks MLFLOW_EXPERIMENT_ID for the experiment id
        with mlflow.start_run():
            _log_model_settings(
                model_name=model_class_name,
                model_type=model_class_name,
                k=k,
                artifacts=artifacts,
                model_params=model_kwargs,
                model_config_path=None
            )

            setattr(replay.model_handler, 'EmptyRecommender', EmptyRecommender)
            setattr(replay.model_handler, 'RefitableTwoStageScenario', RefitableTwoStageScenario)

            with log_exec_timer("scenario_loading") as timer:
                scenario = cast(RefitableTwoStageScenario, load(artifacts.two_stage_scenario_path))

            mlflow.log_metric(timer.name, timer.duration)
            scenario.first_level_models = [_get_model(artifacts, model_class_name, model_kwargs)]

            train = artifacts.train.cache()
            user_features = artifacts.user_features.cache() if artifacts.user_features is not None else None
            item_features = artifacts.item_features.cache() if artifacts.item_features is not None else None

            with log_exec_timer("fit") as timer:
                scenario.fit(log=train, user_features=user_features, item_features=item_features)

            mlflow.log_metric(timer.name, timer.duration)
            assert len(scenario.first_level_models_relevance_columns) == 1

            with log_exec_timer("model_saving") as timer:
                save(scenario.first_level_models[0], artifacts.model_path(model_class_name))

            mlflow.log_metric(timer.name, timer.duration)

            # because of EmptyWrap this is still predictions from the first level
            # (though with all prepared features required on the second level)
            scenario.candidates_with_positives = True
            recs = scenario.predict(log=train, k=k, users=train.select('user_idx').distinct(),
                                    filter_seen_items=False, user_features=user_features, item_features=item_features)
            recs = recs.cache()
            assert scenario.first_level_models_relevance_columns[0] in recs.columns
            assert "target" and recs.columns
            assert recs.select('target').distinct().count() == 2
            recs.write.parquet(artifacts.partial_train_path(model_class_name))
            recs.unpersist()

            # getting first level predictions that can be re-evaluated by the second model
            with log_exec_timer("predict") as timer:
                scenario.candidates_with_positives = False
                recs = scenario.predict(log=train, k=k, users=train.select('user_idx').distinct(),
                                        filter_seen_items=True,
                                        user_features=user_features, item_features=item_features)
                assert scenario.first_level_models_relevance_columns[0] in recs.columns
                recs.write.parquet(artifacts.partial_predicts_path(model_class_name))

            mlflow.log_metric(timer.name, timer.duration)

            rel_col_name = scenario.first_level_models_relevance_columns[0]
            _estimate_and_report_metrics(
                model_class_name,
                artifacts.test,
                recs.withColumnRenamed(rel_col_name, 'relevance')
            )

            train.unpersist()
            user_features.unpersist()
            item_features.unpersist()


@task
def combine_train_predicts_for_second_level(artifacts: ArtifactPaths):
    with _init_spark_session(DEFAULT_CPU, DEFAULT_MEMORY) as spark:
        _combine_datasets_for_second_level(
            artifacts.partial_train_paths,
            artifacts.full_second_level_train_path,
            spark
        )
        _combine_datasets_for_second_level(
            artifacts.partial_predicts_paths,
            artifacts.full_second_level_predicts_path,
            spark
        )


# this is @task
def second_level_fitting(
        artifacts: ArtifactPaths,
        model_name: str,
        k: int,
        second_model_type: str = "lama",
        second_model_params: Optional[Union[Dict, str]] = None,
        second_model_config_path: Optional[str] = None,
        cpu: int = DEFAULT_CPU,
        memory: int = DEFAULT_MEMORY):
    with _init_spark_session(cpu, memory):
        # checks MLFLOW_EXPERIMENT_ID for the experiment id
        with mlflow.start_run():
            _log_model_settings(
                model_name=model_name,
                model_type=second_model_type,
                k=k,
                artifacts=artifacts,
                model_params=second_model_params,
                model_config_path=second_model_config_path
            )

            with log_exec_timer("scenario_loading") as timer:
                setattr(replay.model_handler, 'EmptyRecommender', EmptyRecommender)
                setattr(replay.model_handler, 'RefitableTwoStageScenario', RefitableTwoStageScenario)
                scenario = load(artifacts.two_stage_scenario_path)

            mlflow.log_metric(timer.name, timer.duration)

            if second_model_type == "lama":
                second_stage_model = LamaWrap(params=second_model_params, config_path=second_model_config_path)
            elif second_model_type == "slama":
                second_stage_model = SlamaWrap(params=second_model_params, config_path=second_model_config_path)
            else:
                raise RuntimeError(f"Currently supported model types: {['lama']}, but received {second_model_type}")

            with log_exec_timer("fit") as timer:
                second_stage_model.fit(artifacts.full_second_level_train)

            mlflow.log_metric(timer.name, timer.duration)

            with log_exec_timer("predict") as timer:
                recs = second_stage_model.predict(artifacts.full_second_level_predicts, k=k)
                recs = scenario._filter_seen(recs, artifacts.train, k, artifacts.train.select('user_idx').distinct())
                recs.write.parquet(artifacts.second_level_predicts_path(model_name))

            mlflow.log_metric(timer.name, timer.duration)

            with log_exec_timer("model_saving") as timer:
                second_level_model_path = artifacts.second_level_model_path(model_name)
                save_transformer(second_stage_model, second_level_model_path)

            mlflow.log_metric(timer.name, timer.duration)

            _estimate_and_report_metrics(model_name, artifacts.test, recs)


def build_two_stage_scenario_dag(
        dag_id: str,
        first_level_models: Dict[str, Dict],
        second_level_models: Dict[str, Dict],
        log_path: str,
        user_features_path: Optional[str] = None,
        item_features_path: Optional[str] = None,
        k: int = 100,
        mlflow_exp_id: str = "107",
        use_big_exec_config_for_first_level: bool = False,
        use_extra_big_exec_config_for_second_level: bool = False
) -> DAG:
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
            tags=['two_stage', 'replay', 'slama']
    ) as dag:
        os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
        os.environ["MLFLOW_EXPERIMENT_ID"] = os.environ.get("MLFLOW_EXPERIMENT_ID", mlflow_exp_id)

        artifacts = ArtifactPaths(
            # base_path="/opt/spark_data/replay/experiments/two_stage_{{ ds }}_{{ run_id | replace(':', '__') | replace('+', '__') }}",
            base_path="/opt/spark_data/replay/experiments/two_stage_2022-12-07_manual__2022-12-07T11__55__33.459452__00__00",
            log_path=log_path,
            user_features_path=user_features_path,
            item_features_path=item_features_path
        )

        splitting = dataset_splitting(artifacts, partitions_num=6, dataset_name="ml1m")
        create_scenario_datasets = init_refitable_two_stage_scenario(artifacts)
        fit_first_level_models = [
            task(
                task_id=f"first_level_{model_class_name.split('.')[-1]}",
                executor_config=big_executor_config if use_big_exec_config_for_first_level else None
            )(first_level_fitting)(
                artifacts,
                model_class_name,
                model_kwargs,
                k,
                cpu=BIG_CPU if use_big_exec_config_for_first_level else DEFAULT_CPU,
                memory=BIG_MEMORY if use_big_exec_config_for_first_level else DEFAULT_MEMORY
            )
            for model_class_name, model_kwargs in first_level_models.items()
        ]
        combining = combine_train_predicts_for_second_level(artifacts)
        fit_second_level_models = [
            task(
                task_id=f"second_level_{model_name}",
                executor_config=extra_big_executor_config if use_extra_big_exec_config_for_second_level else None
            )(second_level_fitting)(
                artifacts,
                model_name,
                k,
                cpu=EXTRA_BIG_CPU if use_extra_big_exec_config_for_second_level else DEFAULT_CPU,
                memory=EXTRA_BIG_MEMORY if use_extra_big_exec_config_for_second_level else DEFAULT_MEMORY,
                **model_kwargs
            )
            for model_name, model_kwargs in second_level_models.items()
        ]

        chain(splitting, create_scenario_datasets, fit_first_level_models)
        cross_downstream(fit_first_level_models, combining)
        chain(combining, fit_second_level_models)

    return dag


def build_2stage_integration_test_dag() -> DAG:
    first_level_models = {
        # "replay.models.als.ALSWrap": {"rank": 10, "nmslib_hnsw_params": dense_hnsw_params},
        "replay.models.als.ALSWrap": {"rank": 10},
        "replay.models.knn.ItemKNN": {"num_neighbours": 10}
    }

    second_level_models = {
        "default_lama": {
            "second_model_type": "lama",
            "second_model_params": {
                "general_params": {"use_algos": [["lgb", "linear_l2"]]},
                "reader_params": {"cv": 3, "advanced_roles": False}
            }
        },
        "default_slama_2": {
            "second_model_type": "lama",
            "second_model_params": {
                "general_params": {"use_algos": [["lgb"]]},
                "reader_params": {"cv": 3, "advanced_roles": False}
            }
        },
        "slama_lgb": {
            "second_model_type": "slama",
            "second_model_params": {
                "general_params": {"use_algos": [["lgb"]]},
                "reader_params": {"cv": 3, "advanced_roles": False}
            }
        }
    }

    return build_two_stage_scenario_dag(
        dag_id="2stage_integration_test",
        first_level_models=first_level_models,
        second_level_models=second_level_models,
        log_path="/opt/spark_data/replay/ml100k_ratings.csv",
        user_features_path="/opt/spark_data/replay/ml100k_users.csv",
        item_features_path="/opt/spark_data/replay/ml100k_items.csv"
    )


def build_2stage_ml1m_dag() -> DAG:
    first_level_models = {
        "replay.models.als.ALSWrap": {"rank": 100, "seed": 42},
        "replay.models.knn.ItemKNN": {"num_neighbours": 1000},
        "replay.models.cluster.ClusterRec": {"num_clusters": 100},
        "replay.models.slim.SLIM": {"seed": 42},
        # "replay.models.word2vec.Word2VecRec": {"rank": 100, "seed": 42},
        "replay.models.ucb.UCB": {"seed": 42}
    }

    second_level_models = {
        "default_lama": {
            "second_model_type": "lama",
            "second_model_params": {
                "cpu_limit": EXTRA_BIG_CPU,
                "memory_limit": int(EXTRA_BIG_MEMORY * 0.95),
                "timeout": 10800,
                "general_params": {"use_algos": [["lgb_tuned"]]},
                "reader_params": {"cv": 5, "advanced_roles": True},
                "tuning_params": {'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
            }
        }
    }

    return build_two_stage_scenario_dag(
        dag_id="2stage_ml1m",
        first_level_models=first_level_models,
        second_level_models=second_level_models,
        log_path="/opt/spark_data/replay/ml1m_ratings.csv",
        user_features_path="/opt/spark_data/replay/ml1m_users.csv",
        item_features_path="/opt/spark_data/replay/ml1m_items.csv",
        use_big_exec_config_for_first_level=True,
        use_extra_big_exec_config_for_second_level=True
    )


def build_2stage_ml1m_itemknn_dag() -> DAG:
    first_level_models = {
        # "replay.models.als.ALSWrap": {"rank": 100, "seed": 42},
        "replay.models.knn.ItemKNN": {"num_neighbours": 1000},
        # "replay.models.cluster.ClusterRec": {"num_clusters": 100},
        # "replay.models.slim.SLIM": {"seed": 42},
        # "replay.models.word2vec.Word2VecRec": {"rank": 100, "seed": 42},
        # "replay.models.ucb.UCB": {"seed": 42}
    }

    second_level_models = {
        "default_lama": {
            "second_model_type": "lama",
            "second_model_params": {
                "cpu_limit": EXTRA_BIG_CPU,
                "memory_limit": int(EXTRA_BIG_MEMORY * 0.95),
                "timeout": 10800,
                "general_params": {"use_algos": [["lgb_tuned"]]},
                "reader_params": {"cv": 5, "advanced_roles": False},
                "tuning_params": {'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
            }
        }
    }

    return build_two_stage_scenario_dag(
        dag_id="2stage_ml1m_itemknn",
        first_level_models=first_level_models,
        second_level_models=second_level_models,
        log_path="/opt/spark_data/replay/ml1m_ratings.csv",
        user_features_path="/opt/spark_data/replay/ml1m_users.csv",
        item_features_path="/opt/spark_data/replay/ml1m_items.csv",
        use_big_exec_config_for_first_level=True,
        use_extra_big_exec_config_for_second_level=True
    )


def build_2stage_ml1m_alswrap_dag() -> DAG:
    first_level_models = {
        "replay.models.als.ALSWrap": {"rank": 100, "seed": 42},
        # "replay.models.knn.ItemKNN": {"num_neighbours": 1000},
        # "replay.models.cluster.ClusterRec": {"num_clusters": 100},
        # "replay.models.slim.SLIM": {"seed": 42},
        # "replay.models.word2vec.Word2VecRec": {"rank": 100, "seed": 42},
        # "replay.models.ucb.UCB": {"seed": 42}
    }

    second_level_models = {
        "default_lama": {
            "second_model_type": "lama",
            "second_model_params": {
                "cpu_limit": EXTRA_BIG_CPU,
                "memory_limit": int(EXTRA_BIG_MEMORY * 0.95),
                "timeout": 10800,
                "general_params": {"use_algos": [["lgb_tuned"]]},
                "reader_params": {"cv": 5, "advanced_roles": False},
                "tuning_params": {'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
            }
        },

        "lama_single_lgb": {
            "second_model_type": "lama",
            "second_model_params": {
                "cpu_limit": EXTRA_BIG_CPU,
                "memory_limit": int(EXTRA_BIG_MEMORY * 0.95),
                "timeout": 10800,
                "general_params": {"use_algos": [["lgb"]]},
                "reader_params": {"cv": 5, "advanced_roles": False},
                "tuning_params": {'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
            }
        },

    }

    return build_two_stage_scenario_dag(
        dag_id="2stage_ml1m_alswrap",
        first_level_models=first_level_models,
        second_level_models=second_level_models,
        log_path="/opt/spark_data/replay/ml1m_ratings.csv",
        user_features_path="/opt/spark_data/replay/ml1m_users.csv",
        item_features_path="/opt/spark_data/replay/ml1m_items.csv",
        use_big_exec_config_for_first_level=True,
        use_extra_big_exec_config_for_second_level=True
    )


def build_2stage_ml25m_dag() -> DAG:
    first_level_models = {
        "replay.models.als.ALSWrap": {"rank": 100, "seed": 42},
        "replay.models.knn.ItemKNN": {"num_neighbours": 1000},
        "replay.models.cluster.ClusterRec": {"num_clusters": 100},
        "replay.models.slim.SLIM": {"seed": 42},
        "replay.models.word2vec.Word2VecRec": {"rank": 100, "seed": 42},
        "replay.models.ucb.UCB": {"seed": 42}
    }

    second_level_models = {
        "default_lama": {
            "second_model_type": "lama",
            "second_model_params": {
                "cpu_limit": EXTRA_BIG_CPU,
                "memory_limit": int(EXTRA_BIG_MEMORY * 0.95),
                "timeout": 10800,
                "general_params": {"use_algos": [["tuned_lgb"]]},
                "reader_params": {"cv": 5, "advanced_roles": True},
                "tuning_params": {'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
            }
        }
    }

    return build_two_stage_scenario_dag(
        dag_id="2stage_ml25m",
        first_level_models=first_level_models,
        second_level_models=second_level_models,
        log_path="/opt/spark_data/replay/ml25m_ratings.csv",
        user_features_path=None,
        item_features_path=None,
        use_big_exec_config_for_first_level=True,
        use_extra_big_exec_config_for_second_level=True
    )


def build_fit_predict_first_level_models_dag(
        dag_id: str,
        mlflow_exp_id: str,
        model_params_map: Dict[str, Dict[str, Any]],
        dataset: DatasetInfo
):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
            tags=['two_stage', 'replay', 'first level']
    ) as dag:
        os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
        os.environ["MLFLOW_EXPERIMENT_ID"] = os.environ.get("MLFLOW_EXPERIMENT_ID", mlflow_exp_id)

        # TODO: fix it later
        # path_suffix = Variable.get(f'{dataset.name}_artefacts_dir_suffix', 'default')
        path_suffix = 'default'
        artifacts = ArtifactPaths(
            base_path=f"/opt/spark_data/replay/experiments/{dataset.name}_first_level_{path_suffix}",
            log_path=dataset.log_path,
            user_features_path=dataset.user_features_path,
            item_features_path=dataset.item_features_path
        )
        k = 100

        first_level_models = [
            task(
                task_id=f"fit_predict_first_level_model_{model_class_name.split('.')[-1]}",
                executor_config=big_executor_config
            )(fit_predict_first_level_model)(
                artifacts=artifacts,
                model_class_name=model_class_name,
                model_kwargs=model_kwargs,
                k=k,
                cpu=BIG_CPU,
                memory=BIG_MEMORY
            )
            for model_class_name, model_kwargs in model_params_map.items()
        ]

        dataset_splitting(artifacts, partitions_num=100, dataset_name=dataset.name) \
            >> presplit_data(artifacts) \
            >> first_level_models

    return dag


integration_dag = build_2stage_integration_test_dag()

ml1m_dag = build_2stage_ml1m_dag()

ml1m_itemknn_dag = build_2stage_ml1m_itemknn_dag()

ml1m_alswrap_dag = build_2stage_ml1m_alswrap_dag()

ml25m_dag = build_2stage_ml25m_dag()

ml1m_first_level_dag = build_fit_predict_first_level_models_dag(
    dag_id="ml1m_first_level_dag",
    mlflow_exp_id="107",
    model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "cluster"),
    dataset=DATASETS["ml1m"]
)
