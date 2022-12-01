import functools
import importlib
import logging
import os
import uuid
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
from replay.metrics import MAP, NDCG, HitRate
from replay.model_handler import save
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario
from replay.scenarios.two_stages.reranker import LamaWrap, ReRanker
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info

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
                 predefined_train_and_positives: Optional[Union[Tuple[DataFrame, DataFrame], Tuple[str, str]]] = None,
                 predefined_negatives: Optional[Union[DataFrame, str]] = None,
                 predefined_test_candidate_features: Optional[DataFrame] = None,
                 empty_second_stage_params: Optional[Dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.predefined_train_and_positives = predefined_train_and_positives
        self.predefined_negatives = predefined_negatives
        self.predefined_test_candidate_features = predefined_test_candidate_features
        self._is_already_dumped = False

        if empty_second_stage_params is not None:
            self.second_stage_model = EmptyWrap(**empty_second_stage_params)

    def _split_data(self, log: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if self.predefined_train_and_positives is not None and isinstance(self.predefined_train_and_positives[0], DataFrame):
            return self.predefined_train_and_positives

        first_level_train, second_level_positives = super()._split_data(log)

        if self.predefined_train_and_positives is not None and isinstance(self.predefined_train_and_positives[0], str):
            first_level_train_path, second_level_positives_path = self.predefined_train_and_positives
            first_level_train.write.parquet(first_level_train_path)
            second_level_positives.write.parquet(second_level_positives_path)

        return first_level_train, second_level_positives

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
        if self.predefined_negatives is not None and isinstance(self.predefined_negatives, DataFrame):
            return self.predefined_negatives

        negative_candidates = super()._get_first_level_candidates(model, log, k, users,
                                                                  items, user_features, item_features, log_to_filter)

        if self.predefined_negatives is not None \
                and isinstance(self.predefined_negatives, str) \
                and not self._is_already_dumped:
            negative_candidates.write.parquet(self.predefined_negatives)
            self._is_already_dumped = True

        return negative_candidates

    def _predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                 user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:

        if self.predefined_test_candidate_features is not None:
            return self.second_stage_model.predict(data=self.predefined_test_candidate_features, k=k)

        return super()._predict(log, k, users, items, user_features, item_features, filter_seen_items)


def _get_scenario(
        model_class_name: str,
        model_kwargs: Dict,
        predefined_train_and_positives: Optional[Union[Tuple[DataFrame, DataFrame], Tuple[str, str]]] = None,
        predefined_negatives: Optional[Union[DataFrame, str]] = None,
        empty_second_stage_params: Optional[Dict] = None,
) -> TwoStagesScenario:
    module_name = ".".join(model_class_name.split('.')[:-1])
    class_name = model_class_name.split('.')[-1]
    module = importlib.import_module(module_name)
    clazz = getattr(module, class_name)
    base_model = cast(BaseRecommender, clazz(**model_kwargs))

    scenario = CustomTwoStageScenario(
        predefined_train_and_positives=predefined_train_and_positives,
        predefined_negatives=predefined_negatives,
        empty_second_stage_params=empty_second_stage_params,
        train_splitter=UserSplitter(item_test_size=0.5, shuffle=True, seed=42),
        first_level_models=base_model,
        second_model_params={"general_params": {"use_algos": [["lgb", "linear_l2"]]}}
    )

    return scenario


def _get_spark_session() -> SparkSession:
    return SparkSession.builder.master("local[6]").getOrCreate()


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


@task
def dataset_splitting(log_path: str, base_path: str, train_path: str, test_path: str, cores: int):
    spark = _get_spark_session()
    data = spark.read.csv(log_path, header=True)
    data = (
        data
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
    logger.info(f'train info: \n{get_log_info(train)}')
    logger.info(f'test info:\n{get_log_info(test)}')

    # writing data
    os.makedirs(base_path, exist_ok=True)

    assert train.count() > 0
    assert test.count() > 0

    train.write.parquet(train_path)
    test.write.parquet(test_path)


# this is @task
def first_level_fitting(
        train_path: str,
        test_path: str,
        model_class_name: str,
        model_kwargs: Dict,
        model_path: str,
        second_level_partial_train_path: str,
        first_level_model_predictions_path: str,
        k: int,
        intermediate_datasets_mode: str = "use",
        predefined_train_and_positives_path: Optional[Tuple[str, str]] = None,
        predefined_negatives_path: Optional[str] = None,
        item_features_path: Optional[str] = None,
        user_features_path: Optional[str] = None):
    spark = _get_spark_session()

    train = spark.read.parquet(train_path).drop('_c0')
    test = spark.read.parquet(test_path).drop('_c0')

    if intermediate_datasets_mode == "use":
        first_level_train_path, second_level_positives_path = predefined_train_and_positives_path
        first_level_train = spark.read.parquet(first_level_train_path)
        second_level_positives = spark.read.parquet(second_level_positives_path)

        predefined_train_and_positives = first_level_train, second_level_positives
        predefined_negatives = spark.read.parquet(predefined_negatives_path)
    else:
        predefined_train_and_positives = predefined_train_and_positives_path
        predefined_negatives = predefined_negatives_path

    item_features = spark.read.csv(item_features_path, header=True).withColumnRenamed('item_id', 'item_idx') if item_features_path is not None else None
    user_features = spark.read.csv(user_features_path, header=True).withColumnRenamed('user_id', 'user_idx') if user_features_path is not None else None

    item_features = item_features.withColumn('item_idx', sf.col('item_idx').cast('int')).drop('_c0')
    user_features = user_features.withColumn('user_idx', sf.col('user_idx').cast('int')).drop('_c0')

    # 1. replaces train splitting with pre-splitted data
    # 2. dumps the second level train dataset
    # 3. dumps or uses negative_samples
    scenario = _get_scenario(
        model_class_name=model_class_name,
        model_kwargs=model_kwargs,
        predefined_train_and_positives=predefined_train_and_positives,
        predefined_negatives=predefined_negatives,
        empty_second_stage_params={
            "second_level_train_path": second_level_partial_train_path
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

    recs.write.parquet(first_level_model_predictions_path)


# this is @task
def second_level_fitting(
        model_name: str,
        train_path: str,
        test_path: str,
        final_second_level_train_path: str,
        test_candidate_features_path: str,
        second_level_model_path: str,
        second_level_predictions_path: str,
        k: int,
        second_model_type: str = "lama",
        second_model_params: Optional[Union[Dict, str]] = None,
        second_model_config_path: Optional[str] = None):
    spark = _get_spark_session()

    scenario = CustomTwoStageScenario(
        train_splitter=UserSplitter(item_test_size=0.5, shuffle=True, seed=42),
        second_model_params={"general_params": {"use_algos": [["lgb", "linear_l2"]]}}
    )

    if second_model_type == "lama":
        second_stage_model = LamaWrap(params=second_model_params, config_path=second_model_config_path)
    else:
        raise RuntimeError(f"Currently supported model types: {['lama']}, but received {second_model_type}")

    train = spark.read.parquet(train_path)
    test = spark.read.parquet(test_path)
    second_level_train = spark.read.parquet(final_second_level_train_path)
    test_candidate_features = spark.read.parquet(test_candidate_features_path)

    second_stage_model.fit(second_level_train)
    scenario.second_stage_model = second_stage_model

    # TODO: save the second_model

    recs = scenario.predict(
        log=train,
        k=k,
        users=test_candidate_features.select('user_idx').distinct(),
        filter_seen_items=True,
        user_features=None,
        item_features=None
    )

    recs.write.parquet(second_level_predictions_path)

    _estimate_and_report_metrics(model_name, test, recs)


# this is @task
def combine_datasets_for_second_level(partial_datasets_paths: List[str], full_dataset_path: str):
    assert len(partial_datasets_paths) > 0, "Cannot work with empty sequence of paths"

    spark = _get_spark_session()

    dfs = [spark.read.parquet(path) for path in partial_datasets_paths]
    df = functools.reduce(lambda acc, x: acc.join(x, on=["user_idx", "item_idx"]), dfs)

    # TODO: check the resulting dataframe for correctness (no NONEs in any field)

    df.write.parquet(full_dataset_path)


@dataclass(frozen=True)
class ArtifactPaths:
    base_path: str
    uid: str = f"{uuid.uuid4()}".replace('-', '')

    @property
    def train_path(self) -> str:
        return os.path.join(self.base_path, "train.parquet")

    @property
    def test_path(self) -> str:
        return os.path.join(self.base_path, "test.parquet")

    @property
    def first_level_train_path(self) -> str:
        return os.path.join(self.base_path, "first_level_train.parquet")

    @property
    def second_level_positives_path(self) -> str:
        return os.path.join(self.base_path, "second_level_positives.parquet")

    @property
    def negatives_path(self) -> str:
        return os.path.join(self.base_path, "second_level_negatives.parquet")

    def model_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path, f"model_{model_cls_name.replace('.', '__')}_{self.uid}")

    def partial_train_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path, f"partial_train_{model_cls_name.replace('.', '__')}_{self.uid}.parquet")

    def predictions_path(self, model_cls_name: str) -> str:
        return os.path.join(self.base_path, f"predictions_{model_cls_name.replace('.', '__')}_{self.uid}.parquet")

    @property
    def full_first_level_train_path(self) -> str:
        return os.path.join(self.base_path, "full_first_to_second_train.parquet")

    @property
    def full_first_level_predictions_path(self) -> str:
        return os.path.join(self.base_path, "full_first_to_second_predictions.parquet")

    @property
    def second_level_model_path(self) -> str:
        return os.path.join(self.base_path, "second_level_model")

    @property
    def second_level_predictions_path(self) -> str:
        return os.path.join(self.base_path, "second_level_predictions.parquet")




@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=['example'],
)
def build_full_dag():
    cores = 6
    k = 10

    # external paths
    log_path = "/opt/data/ml100k_ratings.parquet"
    item_features_path = "/opt/data/ml100k_items.parquet"
    user_features_path = "/opt/data/ml100k_users.parquet"

    # the base path for all intermeidate and final datasets
    base_path = "/opt/experiments/two_stage_{{ ds }}_{{ run_id }}"

    # intermediate and final datasets
    artifacts = ArtifactPaths(base_path)

    first_model_class_name = "replay.models.als.ALSWrap"
    models = {
        first_model_class_name: {"rank": 128}
    }

    second_level_models = {
        "default_lama": {
            "second_model_params": {"general_params": {"use_algos": [["lgb", "linear_l2"]]}}
        }
    }

    splitting = dataset_splitting(log_path, base_path, artifacts.train_path, artifacts.test_path, cores)

    fit_initial_first_level_model = \
        task(task_id=f"initial_level_model_{first_model_class_name.split('.')[-1]}")(first_level_fitting)(
            train_path=artifacts.train_path,
            test_path=artifacts.test_path,
            model_class_name=first_model_class_name,
            model_kwargs=models[first_model_class_name],
            model_path=artifacts.model_path(first_model_class_name),
            second_level_partial_train_path=artifacts.partial_train_path(first_model_class_name),
            first_level_model_predictions_path=artifacts.predictions_path(first_model_class_name),
            k=k,
            intermediate_datasets_mode="dump",
            predefined_train_and_positives_path=(
                artifacts.first_level_train_path,
                artifacts.second_level_positives_path
            ),
            predefined_negatives_path=artifacts.negatives_path,
            item_features_path=item_features_path,
            user_features_path=user_features_path
        )

    fit_first_level_models = [
        task(task_id=f"first_level_{model_class_name.split('.')[-1]}")(first_level_fitting)(
            train_path=artifacts.train_path,
            test_path=artifacts.test_path,
            model_class_name=model_class_name,
            model_kwargs=model_kwargs,
            model_path=artifacts.model_path(model_class_name),
            second_level_partial_train_path=artifacts.partial_train_path(model_class_name),
            first_level_model_predictions_path=artifacts.predictions_path(model_class_name),
            k=k,
            intermediate_datasets_mode="use",
            predefined_train_and_positives_path=(
                artifacts.first_level_train_path,
                artifacts.second_level_positives_path
            ),
            predefined_negatives_path=artifacts.negatives_path,
            item_features_path=item_features_path,
            user_features_path=user_features_path
        )
        for model_class_name, model_kwargs in models.items() if model_class_name != first_model_class_name
    ]

    combine_first_level_partial_trains = task(task_id="combine_partial_trains")(combine_datasets_for_second_level)(
        partial_datasets_paths=[artifacts.partial_train_path(model_class_name) for model_class_name in models],
        full_dataset_path=artifacts.full_first_level_train_path
    )

    combine_first_level_partial_tests = task(task_id="combine_partial_tests")(combine_datasets_for_second_level)(
        partial_datasets_paths=[artifacts.predictions_path(model_class_name) for model_class_name in models],
        full_dataset_path=artifacts.full_first_level_predictions_path
    )

    combine_first_level_partials = [combine_first_level_partial_trains, combine_first_level_partial_tests]

    fit_second_level_models = [
        task(task_id=f"second_level_{model_name}")(second_level_fitting)(
            model_name=model_name,
            train_path=artifacts.train_path,
            test_path=artifacts.test_path,
            final_second_level_train_path=artifacts.full_first_level_train_path,
            test_candidate_features_path=artifacts.full_first_level_predictions_path,
            second_level_model_path=artifacts.second_level_model_path,
            second_level_predictions_path=artifacts.second_level_predictions_path,
            k=k,
            second_model_type="lama",
            **model_kwargs
        )
        for model_name, model_kwargs in second_level_models.items()
    ]

    chain(splitting, fit_initial_first_level_model, fit_first_level_models)
    cross_downstream([fit_initial_first_level_model, *fit_first_level_models], combine_first_level_partials)
    cross_downstream(combine_first_level_partials, fit_second_level_models)


dag = build_full_dag()


if __name__ == "__main__":
    main()
