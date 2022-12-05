from pyspark.sql import functions as sf, SparkSession
from rs_datasets import MovieLens

from replay.data_preparator import DataPreparator
from replay.scenarios import TwoStagesScenario
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info


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

    data = MovieLens("25m")
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


if __name__ == "__main__":
    main()
