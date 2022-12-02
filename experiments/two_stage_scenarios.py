import functools
import importlib
import itertools
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, cast, Optional, List, Union, Tuple

import mlflow
import pendulum
from airflow.decorators import task, dag
from airflow.utils.helpers import chain, cross_downstream
from pyspark.sql import functions as sf, SparkSession, DataFrame
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator
from replay.experiment import Experiment
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.metrics import MAP, NDCG, HitRate
from replay.model_handler import save, PopRec, Splitter, load
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario
from replay.scenarios.two_stages.reranker import LamaWrap, ReRanker
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info

logger = logging.getLogger(__name__)


class EmptyWrap(ReRanker):
    def fit(self, data: DataFrame, fit_params: Optional[Dict] = None) -> None:
        pass

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


class RefitableTwoStageScenario(TwoStagesScenario):
    def __init__(self,
                 base_path: str,
                 train_splitter: Splitter = UserSplitter(
                     item_test_size=0.5, shuffle=True, seed=42
                 ),
                 fallback_model: Optional[BaseRecommender] = PopRec(),
                 use_first_level_models_feat: Union[List[bool], bool] = True,
                 num_negatives: int = 100,
                 negatives_type: str = "first_level",
                 use_generated_features: bool = True,
                 user_cat_features_list: Optional[List] = None,
                 item_cat_features_list: Optional[List] = None,
                 custom_features_processor: HistoryBasedFeaturesProcessor = None,
                 seed: int = 123,
                 ):
        super().__init__(
            train_splitter=train_splitter,
            first_level_models=[EmptyRecommender()],
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
        self._first_level_train_path = os.path.join(base_path, "first_level_train.parquet")
        self._second_level_positive_path = os.path.join(base_path, "second_level_positive.parquet")
        self._first_level_candidates_path = os.path.join(base_path, "first_level_candidates.parquet")

        self._are_split_data_dumped = False
        self._are_candidates_dumped = False

        self._return_candidates_with_positives = False

    @property
    def candidates_with_positives(self) -> bool:
        return self._return_candidates_with_positives

    @candidates_with_positives.setter
    def candidates_with_positives(self, val: bool):
        self._return_candidates_with_positives = val

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self._are_split_data_dumped:
            spark = log.sql_ctx.sparkSession
            return spark.read.parquet(self._first_level_train_path), \
                   spark.read.parquet(self._second_level_positive_path)

        first_level_train, second_level_positive = super()._split_data(log)

        first_level_train.write.parquet(self._first_level_train_path)
        second_level_positive.write.parquet(self._second_level_positive_path)

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

        return first_level_candidates


# def _get_scenario(
#         model_class_name: str,
#         model_kwargs: Dict,
#         predefined_train_and_positives: Optional[Union[Tuple[DataFrame, DataFrame], Tuple[str, str]]] = None,
#         predefined_negatives: Optional[Union[DataFrame, str]] = None,
#         empty_second_stage_params: Optional[Dict] = None,
# ) -> TwoStagesScenario:
#     module_name = ".".join(model_class_name.split('.')[:-1])
#     class_name = model_class_name.split('.')[-1]
#     module = importlib.import_module(module_name)
#     clazz = getattr(module, class_name)
#     base_model = cast(BaseRecommender, clazz(**model_kwargs))
#
#     scenario = CustomTwoStageScenario(
#         predefined_train_and_positives=predefined_train_and_positives,
#         predefined_negatives=predefined_negatives,
#         empty_second_stage_params=empty_second_stage_params,
#         train_splitter=UserSplitter(item_test_size=0.5, shuffle=True, seed=42),
#         first_level_models=base_model,
#         second_model_params={"general_params": {"use_algos": [["lgb", "linear_l2"]]}}
#     )
#
#     return scenario
@dataclass(frozen=True)
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
        return [path for path in os.listdir(self.base_path) if path.startswith(self.partial_train_prefix)]

    @property
    def partial_predicts_paths(self) -> List[str]:
        return [path for path in os.listdir(self.base_path) if path.startswith(self.partial_predict_prefix)]

    @property
    def full_second_level_train_path(self) -> str:
        return os.path.join(self.base_path, "full_second_level_train.parquet")

    @property
    def full_second_level_predicts_path(self) -> str:
        return os.path.join(self.base_path, "full_second_level_predicts.parquet")

    @property
    def log(self) -> DataFrame:
        return (
            self._get_session().read.csv(self.log_path, header=True)
        )

    @property
    def user_features(self) -> Optional[DataFrame]:
        return (
            self._get_session().read.csv(self.user_features_path, header=True)
            .withColumnRenamed('user_id', 'user_idx')
            .withColumn('user_idx', sf.col('user_idx').cast('int'))
            .drop('_c0')
        )

    @property
    def item_features(self) -> Optional[DataFrame]:
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

    def model_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path, f"model_{model_cls_name.replace('.', '__')}_{self.uid}")

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
def _init_spark_session() -> SparkSession:
    spark = SparkSession.builder.master("local[4]").getOrCreate()

    yield spark

    spark.stop()


def _get_model(model_class_name: str, model_kwargs: Dict) -> BaseRecommender:
    module_name = ".".join(model_class_name.split('.')[:-1])
    class_name = model_class_name.split('.')[-1]
    module = importlib.import_module(module_name)
    clazz = getattr(module, class_name)
    base_model = cast(BaseRecommender, clazz(**model_kwargs))

    return base_model


# this is @task
def _combine_datasets_for_second_level(partial_datasets_paths: List[str], combined_dataset_path: str):
    assert len(partial_datasets_paths) > 0, "Cannot work with empty sequence of paths"

    spark = _init_spark_session()

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
    K_list_metrics = [5, 10]
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


@task
def dataset_splitting(artifacts: ArtifactPaths, partitions_num: int):
    with _init_spark_session():
        data = (
            artifacts.log
            .withColumn('user_id', sf.col('user_id').cast('int'))
            .withColumn('item_id', sf.col('item_id').cast('int'))
            .withColumn('timestamp', sf.col('timestamp').cast('int'))
        )

        # splitting on train and test
        preparator = DataPreparator()

        log = preparator.transform(
            columns_mapping={"user_id": "user_id", "item_id": "item_id", "relevance": "rating", "timestamp": "timestamp"},
            data=data
        ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")

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
    with _init_spark_session():
        scenario = RefitableTwoStageScenario(base_path=artifacts.base_path)

        scenario.fit(log=artifacts.train, user_features=artifacts.user_features, item_features=artifacts.item_features)

        save(scenario, artifacts.two_stage_scenario_path)


# this is @task
def first_level_fitting(artifacts: ArtifactPaths, model_class_name: str, model_kwargs: Dict, k: int):
    with _init_spark_session():
        scenario = cast(RefitableTwoStageScenario, load(artifacts.two_stage_scenario_path))
        scenario.first_level_models = [_get_model(model_class_name, model_kwargs)]

        train = artifacts.train.cache()
        user_features = artifacts.user_features.cache()
        item_features = artifacts.item_features.cache()

        scenario.fit(log=train, user_features=user_features, item_features=item_features)

        save(scenario.first_level_models[0], artifacts.model_path(model_class_name))

        # because of EmptyWrap this is still predictions from the first level
        # (though with all prepared features required on the second level)
        scenario.candidates_with_positives = True
        recs = scenario.predict(log=train, k=k, users=train.select('user_idx').distinct(),
                                filter_seen_items=True, user_features=user_features, item_features=item_features)
        recs.write.parquet(artifacts.partial_train_path(model_class_name))

        # getting first level predictions that can be re-evaluated by the second model
        scenario.candidates_with_positives = False
        recs = scenario.predict(log=train, k=k, users=train.select('user_idx').distinct(),
                                filter_seen_items=True, user_features=user_features, item_features=item_features)
        recs.write.parquet(artifacts.partial_predicts_path(model_class_name))

        _estimate_and_report_metrics(model_class_name, artifacts.test, recs)

        train.unpersist()
        user_features.unpersist()
        item_features.unpersist()


@task
def combine_train_predicts_for_second_level(artifacts: ArtifactPaths):
    with _init_spark_session():
        _combine_datasets_for_second_level(artifacts.partial_train_paths, artifacts.full_second_level_train_path)
        _combine_datasets_for_second_level(artifacts.partial_predicts_paths, artifacts.full_second_level_predicts_path)


# this is @task
def second_level_fitting(
        artifacts: ArtifactPaths,
        model_name: str,
        k: int,
        second_model_type: str = "lama",
        second_model_params: Optional[Union[Dict, str]] = None,
        second_model_config_path: Optional[str] = None):
    with _init_spark_session():
        scenario = load(artifacts.two_stage_scenario_path)

        if second_model_type == "lama":
            second_stage_model = LamaWrap(params=second_model_params, config_path=second_model_config_path)
        else:
            raise RuntimeError(f"Currently supported model types: {['lama']}, but received {second_model_type}")

        second_stage_model.fit(artifacts.full_second_level_train)

        # TODO: save the second_model
        # artifacts.second_level_model_path(model_name)

        recs = second_stage_model.predict(artifacts.full_second_level_predicts, k=k)
        recs = scenario._filter_seen(recs, artifacts.train, k, artifacts.train.select('user_idx').distinct())
        recs.write.parquet(artifacts.second_level_predicts_path(model_name))

        _estimate_and_report_metrics(model_name, artifacts.test, recs)


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=['example'],
)
def build_full_dag():
    k = 10

    artifacts = ArtifactPaths(
        base_path="/opt/experiments/two_stage_{{ ds }}_{{ run_id }}",
        log_path = "/opt/data/ml100k_ratings.parquet",
        item_features_path = "/opt/data/ml100k_items.parquet",
        user_features_path = "/opt/data/ml100k_users.parquet"
    )

    first_model_class_name = "replay.models.als.ALSWrap"
    models = {
        first_model_class_name: {"rank": 128}
    }

    second_level_models = {
        "default_lama": {
            "second_model_type": "lama",
            "second_model_params": {"general_params": {"use_algos": [["lgb", "linear_l2"]]}}
        }
    }

    splitting = dataset_splitting(artifacts, partitions_num=4)
    create_scenario_datasets = init_refitable_two_stage_scenario(artifacts)
    fit_first_level_models = [
        task(task_id=f"first_level_{model_class_name.split('.')[-1]}")(first_level_fitting)(
            artifacts,
            model_class_name,
            model_kwargs,
            k
        )
        for model_class_name, model_kwargs in models.items()
    ]
    combining = combine_train_predicts_for_second_level(artifacts)
    fit_second_level_models = [
        task(task_id=f"second_level_{model_name}")(second_level_fitting)(
            artifacts,
            model_name,
            k,
            **model_kwargs
        )
        for model_name, model_kwargs in second_level_models.items()
    ]

    chain(splitting, create_scenario_datasets, fit_first_level_models)
    cross_downstream(fit_first_level_models, combining)
    chain(combining, fit_second_level_models)


dag = build_full_dag()

# if __name__ == "__main__":
#     main()
