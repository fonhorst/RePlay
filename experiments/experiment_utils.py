import json
import os
from typing import Tuple, Optional

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

from replay.models import (
    ALSWrap,
    SLIM,
    LightFMWrap,
    ItemKNN,
    Word2VecRec,
    PopRec,
    RandomRec,
    AssociationRulesItemRec,
    UserPopRec,
    Wilson,
    ClusterRec,
    UCB,
)
from replay.utils import log_exec_timer, getNumberOfAllocatedExecutors


def get_nmslib_hnsw_params(spark_app_id: str):
    index_params_str = os.environ.get("NMSLIB_HNSW_PARAMS")
    if not index_params_str:
        raise ValueError(
            f"To use nmslib hnsw index you need to set the 'NMSLIB_HNSW_PARAMS' env variable! "
            'For example, {"method":"hnsw","space":"negdotprod_sparse_fast","M":16,"efS":200,"efC":200,"post":0,'
            '"index_path":"/tmp/nmslib_hnsw_index_{spark_app_id}","build_index_on":"executor"}.'
        )
    nmslib_hnsw_params = json.loads(index_params_str)
    if (
        "index_path" in nmslib_hnsw_params
        and "{spark_app_id}" in nmslib_hnsw_params["index_path"]
    ):
        nmslib_hnsw_params["index_path"] = nmslib_hnsw_params[
            "index_path"
        ].replace("{spark_app_id}", spark_app_id)
    print(f"nmslib_hnsw_params: {nmslib_hnsw_params}")
    return nmslib_hnsw_params


def get_hnswlib_params(spark_app_id: str):
    index_params_str = os.environ.get("HNSWLIB_PARAMS")
    if not index_params_str:
        raise ValueError(
            f"To use hnswlib index you need to set the 'HNSWLIB_PARAMS' env variable! "
            'For example, {"space":"ip","M":100,"efS":2000,"efC":2000,"post":0,'
            '"index_path":"/tmp/hnswlib_index_{spark_app_id}","build_index_on":"executor"}.'
        )
    hnswlib_params = json.loads(index_params_str)
    if (
        "index_path" in hnswlib_params
        and "{spark_app_id}" in hnswlib_params["index_path"]
    ):
        hnswlib_params["index_path"] = hnswlib_params["index_path"].replace(
            "{spark_app_id}", spark_app_id
        )
    print(f"hnswlib_params: {hnswlib_params}")
    return hnswlib_params


def get_model(model_name: str, seed: int, spark_app_id: str):
    """Initializes model and returns an instance of it

    Args:
        model_name: model name indicating which model to use. For example, `ALS` and `ALS_HNSWLIB`, where second is ALS with the hnsw index.
        seed: seed
        spark_app_id: spark application id. used for model artifacts paths.
    """

    if model_name == "ALS":
        als_rank = int(os.environ.get("ALS_RANK", 100))
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))

        mlflow.log_params({"num_blocks": num_blocks, "ALS_rank": als_rank})

        model = ALSWrap(
            rank=als_rank,
            seed=seed,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
        )

    elif model_name == "Explicit_ALS":
        als_rank = int(os.environ.get("ALS_RANK", 100))
        mlflow.log_param("ALS_rank", als_rank)
        model = ALSWrap(rank=als_rank, seed=seed, implicit_prefs=False)
    elif model_name == "ALS_HNSWLIB":
        als_rank = int(os.environ.get("ALS_RANK", 100))
        num_blocks = int(os.environ.get("NUM_BLOCKS", 10))
        hnswlib_params = get_hnswlib_params(spark_app_id)
        mlflow.log_params(
            {
                "ALS_rank": als_rank,
                "num_blocks": num_blocks,
                "build_index_on": hnswlib_params["build_index_on"],
                "hnswlib_params": hnswlib_params,
            }
        )
        model = ALSWrap(
            rank=als_rank,
            seed=seed,
            num_item_blocks=num_blocks,
            num_user_blocks=num_blocks,
            hnswlib_params=hnswlib_params,
        )
    elif model_name == "SLIM":
        model = SLIM(seed=seed)
    elif model_name == "SLIM_NMSLIB_HNSW":
        nmslib_hnsw_params = get_nmslib_hnsw_params(spark_app_id)
        mlflow.log_params(
            {
                "build_index_on": nmslib_hnsw_params["build_index_on"],
                "nmslib_hnsw_params": nmslib_hnsw_params,
            }
        )
        model = SLIM(seed=seed, nmslib_hnsw_params=nmslib_hnsw_params)
    elif model_name == "ItemKNN":
        num_neighbours = int(os.environ.get("NUM_NEIGHBOURS", 10))
        mlflow.log_param("num_neighbours", num_neighbours)
        model = ItemKNN(num_neighbours=num_neighbours)
    elif model_name == "ItemKNN_NMSLIB_HNSW":
        nmslib_hnsw_params = get_nmslib_hnsw_params(spark_app_id)
        mlflow.log_params(
            {
                "build_index_on": nmslib_hnsw_params["build_index_on"],
                "nmslib_hnsw_params": nmslib_hnsw_params,
            }
        )
        model = ItemKNN(nmslib_hnsw_params=nmslib_hnsw_params)
    elif model_name == "LightFM":
        model = LightFMWrap(random_state=seed)
    elif model_name == "Word2VecRec":
        word2vec_rank = int(os.environ.get("WORD2VEC_RANK", 100))
        mlflow.log_param("word2vec_rank", word2vec_rank)
        model = Word2VecRec(rank=word2vec_rank, seed=seed)
    elif model_name == "Word2VecRec_NMSLIB_HNSW":
        nmslib_hnsw_params = get_nmslib_hnsw_params(spark_app_id)
        word2vec_rank = int(os.environ.get("WORD2VEC_RANK", 100))
        mlflow.log_params(
            {
                "build_index_on": nmslib_hnsw_params["build_index_on"],
                "nmslib_hnsw_params": nmslib_hnsw_params,
                "word2vec_rank": word2vec_rank,
            }
        )
        model = Word2VecRec(
            rank=word2vec_rank,
            seed=seed,
            nmslib_hnsw_params=nmslib_hnsw_params,
        )
    elif model_name == "Word2VecRec_HNSWLIB":
        hnswlib_params = get_hnswlib_params(spark_app_id)
        word2vec_rank = int(os.environ.get("WORD2VEC_RANK", 100))
        mlflow.log_params(
            {
                "build_index_on": hnswlib_params["build_index_on"],
                "hnswlib_params": hnswlib_params,
                "word2vec_rank": word2vec_rank,
            }
        )

        model = Word2VecRec(
            rank=word2vec_rank,
            seed=seed,
            hnswlib_params=hnswlib_params,
        )
    elif model_name == "PopRec":
        use_relevance = os.environ.get("USE_RELEVANCE", "False") == "True"
        model = PopRec(use_relevance=use_relevance)
        mlflow.log_param("USE_RELEVANCE", use_relevance)
    elif model_name == "UserPopRec":
        model = UserPopRec()
    elif model_name == "RandomRec_uniform":
        model = RandomRec(seed=seed, distribution="uniform")
    elif model_name == "RandomRec_popular_based":
        model = RandomRec(seed=seed, distribution="popular_based")
    elif model_name == "RandomRec_relevance":
        model = RandomRec(seed=seed, distribution="relevance")
    elif model_name == "AssociationRulesItemRec":
        model = AssociationRulesItemRec()
    elif model_name == "Wilson":
        model = Wilson()
    elif model_name == "ClusterRec":
        num_clusters = int(os.environ.get("NUM_CLUSTERS", "10"))
        mlflow.log_param("num_clusters", num_clusters)
        model = ClusterRec(num_clusters=num_clusters)
    elif model_name == "ClusterRec_HNSWLIB":
        num_clusters = int(os.environ.get("NUM_CLUSTERS", "10"))
        hnswlib_params = get_hnswlib_params(spark_app_id)
        mlflow.log_params(
            {
                "num_clusters": num_clusters,
                "build_index_on": hnswlib_params["build_index_on"],
                "hnswlib_params": hnswlib_params,
            }
        )
        model = ClusterRec(
            num_clusters=num_clusters, hnswlib_params=hnswlib_params
        )
    elif model_name == "UCB":
        model = UCB(seed=seed)
    else:
        raise ValueError("Unknown model.")

    return model


def get_datasets(
    dataset_name, spark: SparkSession, partition_num: int
) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
    """
    Reads prepared datasets from hdfs or disk and returns them.

    Args:
        dataset_name: Dataset name with size postfix (optional). For example `MovieLens__10m` or `MovieLens__25m`.
        spark: spark session
        partition_num: Number of partitions in output dataframes.

    Returns:
        train: train dataset
        test: test dataset
        user_features: dataframe with user features (optional)

    """
    user_features = None
    if dataset_name.startswith("MovieLens"):
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            dataset_version = "1m"
        else:
            dataset_version = dataset_params[1]

        with log_exec_timer(
            "Train/test datasets reading to parquet"
        ) as parquets_read_timer:
            train = spark.read.parquet(  # hdfs://node21.bdcl:9000
                f"/opt/spark_data/replay_datasets/MovieLens/train_{dataset_version}.parquet"
            )
            test = spark.read.parquet(  # hdfs://node21.bdcl:9000
                f"/opt/spark_data/replay_datasets/MovieLens/test_{dataset_version}.parquet"
            )
        train = train.repartition(partition_num)
        test = test.repartition(partition_num)
    elif dataset_name.startswith("MillionSongDataset"):
        # MillionSongDataset__{fraction} pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            fraction = "1.0"
        else:
            fraction = dataset_params[1]

        if fraction == "train_100m_users_1k_items":
            with log_exec_timer(
                "Train/test datasets reading to parquet"
            ) as parquets_read_timer:
                train = spark.read.parquet(
                    f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_train.parquet"
                )
                test = spark.read.parquet(
                    f"/opt/spark_data/replay_datasets/MillionSongDataset/fraction_{fraction}_test.parquet"
                )
                train = train.repartition(partition_num)
                test = test.repartition(partition_num)
        else:
            if partition_num in {6, 12, 24, 48}:
                with log_exec_timer(
                    "Train/test datasets reading to parquet"
                ) as parquets_read_timer:
                    train = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                        f"fraction_{fraction}_train_{partition_num}_partition.parquet"
                    )
                    test = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                        f"fraction_{fraction}_test_{partition_num}_partition.parquet"
                    )
            else:
                with log_exec_timer(
                    "Train/test datasets reading to parquet"
                ) as parquets_read_timer:
                    train = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                        f"fraction_{fraction}_train_24_partition.parquet"
                    )
                    test = spark.read.parquet(
                        f"/opt/spark_data/replay_datasets/MillionSongDataset/"
                        f"fraction_{fraction}_test_24_partition.parquet"
                    )
                    train = train.repartition(partition_num)
                    test = test.repartition(partition_num)
    elif dataset_name == "ml1m":
        with log_exec_timer(
            "Train/test/user_features datasets reading to parquet"
        ) as parquets_read_timer:
            train = spark.read.parquet(
                "/opt/spark_data/replay_datasets/ml1m_train.parquet"
            )
            test = spark.read.parquet(
                "/opt/spark_data/replay_datasets/ml1m_test.parquet"
            )
            # user_features = spark.read.parquet(
            #     "/opt/spark_data/replay_datasets/ml1m_user_features.parquet"
            # )
            # .select("user_idx", "gender_idx", "age", "occupation", "zip_code_idx")
            train = train.repartition(partition_num, "user_idx")
            test = test.repartition(partition_num, "user_idx")
    elif dataset_name == "ml1m_first_level_default":
        with log_exec_timer(
            "Train/test/user_features datasets reading to parquet"
        ) as parquets_read_timer:
            train = spark.read.parquet(
                "file:///opt/spark_data/replay/experiments/ml1m_first_level_default/train.parquet"
            )
            test = spark.read.parquet(
                "file:///opt/spark_data/replay/experiments/ml1m_first_level_default/test.parquet"
            )
            train = train.repartition(partition_num, "user_idx")
            test = test.repartition(partition_num, "user_idx")
    elif dataset_name == "ml1m_1m_users_3_7k_items":
        with log_exec_timer(
            "Train/test/user_features datasets reading to parquet"
        ) as parquets_read_timer:
            train = spark.read.parquet(
                "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_train.parquet"
            )
            test = spark.read.parquet(
                "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/ml1m_1m_users_3_7k_items_test.parquet"
            )
            user_features = spark.read.parquet(
                "hdfs://node21.bdcl:9000/opt/spark_data/replay_datasets/"
                "ml1m_1m_users_3_7k_items_user_features.parquet"
            )
            print(user_features.printSchema())
            train = train.repartition(partition_num, "user_idx")
            test = test.repartition(partition_num, "user_idx")
    elif dataset_name == "ml1m_1m_users_37k_items":
        with log_exec_timer(
            "Train/test/user_features datasets reading to parquet"
        ) as parquets_read_timer:
            train = spark.read.parquet(
                "/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_train.parquet"
            )
            test = spark.read.parquet(
                "/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_test.parquet"
            )
            user_features = spark.read.parquet(
                "/opt/spark_data/replay_datasets/ml1m_1m_users_37k_items_user_features.parquet"
            )
            train = train.repartition(partition_num, "user_idx")
            test = test.repartition(partition_num, "user_idx")
    else:
        raise ValueError("Unknown dataset.")

    mlflow.log_metric("parquets_read_sec", parquets_read_timer.duration)

    return train, test, user_features


def get_spark_configs_as_dict(spark_conf: SparkConf):
    return {
        "spark.driver.cores": spark_conf.get("spark.driver.cores"),
        "spark.driver.memory": spark_conf.get("spark.driver.memory"),
        "spark.memory.fraction": spark_conf.get("spark.memory.fraction"),
        "spark.executor.cores": spark_conf.get("spark.executor.cores"),
        "spark.executor.memory": spark_conf.get("spark.executor.memory"),
        "spark.executor.instances": spark_conf.get("spark.executor.instances"),
        "spark.sql.shuffle.partitions": spark_conf.get(
            "spark.sql.shuffle.partitions"
        ),
        "spark.default.parallelism": spark_conf.get(
            "spark.default.parallelism"
        ),
    }


def check_number_of_allocated_executors(spark: SparkSession):
    """
    Checks whether enough executors are allocated or not. If not, then throws an exception.

    Args:
        spark: spark session
    """

    spark_conf: SparkConf = spark.sparkContext.getConf()

    # if enough executors is not allocated in the cluster mode, then we stop the experiment
    if spark_conf.get("spark.executor.instances"):
        if getNumberOfAllocatedExecutors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")


def get_partition_num(spark_conf: SparkConf):
    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))

    return partition_num


def get_log_info(
    log: DataFrame, user_col="user_idx", item_col="item_idx"
) -> Tuple[int, int, int]:
    """
    Basic log statistics

    >>> from replay.session_handler import State
    >>> spark = State().session
    >>> log = spark.createDataFrame([(1, 2), (3, 4), (5, 2)]).toDF("user_idx", "item_idx")
    >>> log.show()
    +--------+--------+
    |user_idx|item_idx|
    +--------+--------+
    |       1|       2|
    |       3|       4|
    |       5|       2|
    +--------+--------+
    <BLANKLINE>
    >>> rows_count, users_count, items_count = get_log_info(log)
    >>> print((rows_count, users_count, items_count))
    (3, 3, 2)

    :param log: interaction log containing ``user_idx`` and ``item_idx``
    :param user_col: name of a columns containing users' identificators
    :param item_col: name of a columns containing items' identificators

    :returns: statistics string
    """
    cnt = log.count()
    user_cnt = log.select(user_col).distinct().count()
    item_cnt = log.select(item_col).distinct().count()
    return cnt, user_cnt, item_cnt
