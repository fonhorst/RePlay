import importlib
import logging
import os
from typing import Dict, cast, Optional

from pyspark.sql import functions as sf, SparkSession
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator
from replay.model_handler import save, RandomRec
from replay.models.base_rec import BaseRecommender
from replay.scenarios import TwoStagesScenario
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info, join_with_col_renaming, get_top_k_recs

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


def get_spark_session() -> SparkSession:
    return SparkSession.builder.master("local[6]").getOrCreate()


def dataset_splitting(log_path: str, split_base_path: str):
    spark = get_spark_session()

    log = spark.read.parquet(log_path)

    # TODO: add splitting on train and test

    train_splitter = UserSplitter(item_test_size=0.5, shuffle=True, seed=42)
    first_level_train, second_level_positives = train_splitter.split(log)
    logger.debug("Log info: %s", get_log_info(log))
    logger.debug(
        "first_level_train info: %s", get_log_info(first_level_train)
    )
    logger.debug(
        "second_level_train info: %s", get_log_info(second_level_positives)
    )

    os.makedirs(split_base_path, exist_ok=True)
    first_level_train_path = os.path.join(split_base_path, "first_level_train.parquet")
    second_level_positives_path = os.path.join(split_base_path, "second_level_train.parquet")

    first_level_train.write.parquet(first_level_train_path)
    second_level_positives.write.parquet(second_level_positives_path)


def first_level_fitting(first_level_train_path: str, model_class_name: str, model_kwargs: Dict, model_path: str):
    # get session and read data
    spark = get_spark_session()
    first_level_train = spark.read.parquet(first_level_train_path)

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


def negative_sampling(
        num_negatives: int,
        first_level_train_path: str,
        second_level_positives_path: str,
        seconld_level_train_path: str,
        first_level_model_path: Optional[str] = None
):
    logger.info("Generate negative examples")

    spark = get_spark_session()

    log = spark.read.parquet(first_level_train_path)
    second_level_positive = spark.read.parquet(second_level_positives_path)
    k = num_negatives

    model = (
        spark.read.parquet(first_level_model_path)
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

    second_level_train.write.parquet(seconld_level_train_path)


def predict_for_second_level_train():
    logger.info("Adding features to second-level train dataset")
    # TODO: only partially
    # second_level_train_to_convert = self._add_features_for_second_level(
    #     log_to_add_features=second_level_train,
    #     log_for_first_level_models=first_level_train,
    #     user_features=user_features,
    #     item_features=item_features,
    # ).cache()

    full_second_level_train = log_to_add_features
    first_level_item_features_cached = cache_if_exists(
        self.first_level_item_features_transformer.transform(item_features)
    )
    first_level_user_features_cached = cache_if_exists(
        self.first_level_user_features_transformer.transform(user_features)
    )

    pairs = log_to_add_features.select("user_idx", "item_idx")
    for idx, model in enumerate(self.first_level_models):
        current_pred = self._predict_pairs_with_first_level_model(
            model=model,
            log=log_for_first_level_models,
            pairs=pairs,
            user_features=first_level_user_features_cached,
            item_features=first_level_item_features_cached,
        ).withColumnRenamed("relevance", f"rel_{idx}_{model}")
        full_second_level_train = full_second_level_train.join(
            sf.broadcast(current_pred),
            on=["user_idx", "item_idx"],
            how="left",
        )

        if self.use_first_level_models_feat[idx]:
            features = get_first_level_model_features(
                model=model,
                pairs=full_second_level_train.select(
                    "user_idx", "item_idx"
                ),
                user_features=first_level_user_features_cached,
                item_features=first_level_item_features_cached,
                prefix=f"m_{idx}",
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

    self.logger.info("Adding features from the dataset")
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

    if self.use_generated_features:
        if not self.features_processor.fitted:
            self.features_processor.fit(
                log=log_for_first_level_models,
                user_features=user_features,
                item_features=item_features,
            )
        self.logger.info("Adding generated features")
        full_second_level_train = self.features_processor.transform(
            log=full_second_level_train
        )

    self.logger.info(
        "Columns at second level: %s",
        " ".join(full_second_level_train.columns),
    )
    full_second_level_train_cached.unpersist()
    return full_second_level_train
    pass


def combine_datasets_for_second_level():
    # TODO: join individual dataframes together into one second_level_trains
    pass


def generate_features_for_second_level():

    self.features_processor.fit(
        log=first_level_train,
        user_features=user_features,
        item_features=item_features,
    )
    pass


def second_level_fitting():
    self.second_stage_model.fit(second_level_train_to_convert)
    pass


def combine_second_level_results():
    # TODO: combine all results (test prediction quality) into a single table
    pass


if __name__ == "__main__":
    main()
