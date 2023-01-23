"""
This script is a Spark application that executes replay recommendation models.
Parameters sets via environment variables.

Available models to execution:
    # LightFM
    # PopRec UserPopRec
    # Word2VecRec Word2VecRec_NMSLIB_HNSW Word2VecRec_HNSWLIB
    # ALS ALS_NMSLIB_HNSW ALS_HNSWLIB
    # SLIM SLIM_NMSLIB_HNSW
    # ItemKNN ItemKNN_NMSLIB_HNSW
    # ClusterRec ClusterRec_HNSWLIB

Available datasets:
    MovieLens
    MillionSongDataset

launch example:
    $ export DATASET=MovieLens
    $ export MODEL=ALS
    $ export ALS_RANK=100
    $ export SEED=22
    $ export K=10
    $ python experiments/run_experiment.py

or run in one line:
    $ DATASET=MovieLens MODEL=ALS ALS_RANK=100 SEED=22 K=10 python experiments/run_experiment.py

All params:
    DATASET: dataset name
    Available values:
        MovieLens__100k
        MovieLens==MovieLens__1m
        MovieLens__10m
        MovieLens__20m
        MovieLens__25m
        MillionSongDataset

    MODEL: model name
    Available values:
        LightFM
        PopRec
        UserPopRec
        ALS
        ALS_HNSWLIB
        ALS_NMSLIB_HNSW (!)
        Word2VecRec
        Word2VecRec_NMSLIB_HNSW
        Word2VecRec_HNSWLIB
        SLIM
        SLIM_NMSLIB_HNSW
        ItemKNN
        ItemKNN_NMSLIB_HNSW
        ClusterRec
        ClusterRec_HNSWLIB


    SEED: seed

    K: number of desired recommendations per user

    ALS_RANK: rank for ALS model, i.e. length of ALS factor vectors

    WORD2VEC_RANK: rank of Word2Vec model

    NUM_NEIGHBOURS: ItemKNN param

    NUM_CLUSTERS: number of clusters in Cluster model


    DRIVER_CORES:

    DRIVER_MEMORY:

    DRIVER_MAX_RESULT_SIZE:

    EXECUTOR_CORES:

    EXECUTOR_MEMORY:


"""

import os

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from experiment_utils import (
    get_model,
    get_datasets,
    get_spark_configs_as_dict,
    check_number_of_allocated_executors,
    get_partition_num,
)
from replay.dataframe_bucketizer import DataframeBucketizer
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.model_handler import save, load
from replay.models import (
    AssociationRulesItemRec,
    ClusterRec,
)
from replay.session_handler import get_spark_session
from replay.utils import (
    JobGroup,
    getNumberOfAllocatedExecutors,
    log_exec_timer,
)
from replay.utils import get_log_info2
from replay.utils import logger


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    check_number_of_allocated_executors(spark)

    k = int(os.environ.get("K", 10))
    k_list_metrics = list(map(int, os.environ["K_LIST_METRICS"].split(",")))
    seed = int(os.environ.get("SEED", 1234))
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8822"
    )
    model_name = os.environ.get("MODEL", "SLIM_NMSLIB_HNSW")
    # LightFM
    # PopRec
    # UserPopRec
    # Word2VecRec Word2VecRec_NMSLIB_HNSW Word2VecRec_HNSWLIB
    # ALS ALS_NMSLIB_HNSW ALS_HNSWLIB
    # SLIM SLIM_NMSLIB_HNSW
    # ItemKNN ItemKNN_NMSLIB_HNSW
    # ClusterRec ClusterRec_HNSWLIB

    partition_num = get_partition_num(spark_conf)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(os.environ.get("EXPERIMENT", "delete"))

    with mlflow.start_run():

        params = get_spark_configs_as_dict(spark_conf)
        params.update(
            {
                "spark.applicationId": spark.sparkContext.applicationId,
                "dataset": dataset_name,
                "seed": seed,
                "K": k,
            }
        )
        mlflow.log_params(params)

        train, test, user_features = get_datasets(
            dataset_name, spark, partition_num
        )

        use_bucketing = os.environ.get("USE_BUCKETING", "False") == "True"
        mlflow.log_param("USE_BUCKETING", use_bucketing)
        if use_bucketing:
            bucketizer = DataframeBucketizer(
                bucketing_key="user_idx",
                partition_num=partition_num,
                spark_warehouse_dir=spark_conf.get("spark.sql.warehouse.dir"),
            )

            with log_exec_timer("dataframe bucketing") as bucketing_timer:
                bucketizer.set_table_name(
                    f"bucketed_train_{spark.sparkContext.applicationId.replace('-', '_')}"
                )
                train = bucketizer.transform(train)

                bucketizer.set_table_name(
                    f"bucketed_test_{spark.sparkContext.applicationId.replace('-', '_')}"
                )
                test = bucketizer.transform(test)
            mlflow.log_metric("bucketing_sec", bucketing_timer.duration)

        with log_exec_timer("Train/test caching") as train_test_cache_timer:
            train = train.cache()
            test = test.cache()
            train.write.mode("overwrite").format("noop").save()
            test.write.mode("overwrite").format("noop").save()
        mlflow.log_metric(
            "train_test_cache_sec", train_test_cache_timer.duration
        )

        mlflow.log_metric("train_num_partitions", train.rdd.getNumPartitions())
        mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

        with log_exec_timer(
            "get_log_info2() execution"
        ) as get_log_info2_timer:
            train_info = get_log_info2(train)
            test_info = get_log_info2(test)
            logger.info(
                "train info: total lines: {}, total users: {}, total items: {}".format(
                    *train_info
                )
            )
            logger.info(
                "test info: total lines: {}, total users: {}, total items: {}".format(
                    *test_info
                )
            )
        mlflow.log_params(
            {
                "get_log_info_sec": get_log_info2_timer.duration,
                "train.total_users": train_info[1],
                "train.total_items": train_info[2],
                "train_size": train_info[0],
                "test_size": test_info[0],
                "test.total_users": test_info[1],
                "test.total_items": test_info[2],
            }
        )

        mlflow.log_param("model", model_name)
        model = get_model(model_name, seed, spark.sparkContext.applicationId)

        kwargs = {}
        if isinstance(model, ClusterRec):
            kwargs = {"user_features": user_features}

        with log_exec_timer(f"{model_name} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train, **kwargs)
        mlflow.log_metric("train_sec", train_timer.duration)

        with log_exec_timer(
            f"{model_name} prediction"
        ) as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            if isinstance(model, AssociationRulesItemRec):
                recs = model.get_nearest_items(
                    items=test,
                    k=k,
                )
            else:
                recs = model.predict(
                    k=k,
                    users=test.select("user_idx").distinct(),
                    log=train,
                    filter_seen_items=True,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_sec", infer_timer.duration)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(
                f"Metrics calculation"
            ) as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=True): k_list_metrics,
                        NDCG(use_scala_udf=True): k_list_metrics,
                        HitRate(use_scala_udf=True): k_list_metrics,
                    },
                )
                e.add_result(model_name, recs)
            mlflow.log_metric("metrics_sec", metrics_timer.duration)
            metrics = dict()
            for k in k_list_metrics:
                metrics["NDCG.{}".format(k)] = e.results.at[
                    model_name, "NDCG@{}".format(k)
                ]
                metrics["MAP.{}".format(k)] = e.results.at[
                    model_name, "MAP@{}".format(k)
                ]
                metrics["HitRate.{}".format(k)] = e.results.at[
                    model_name, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)

        with log_exec_timer(f"Model saving") as model_save_timer:
            save(
                model,
                path=f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}",  # file://
                overwrite=True,
            )
        mlflow.log_param(
            "model_save_dir",
            f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}",
        )
        mlflow.log_metric("model_save_sec", model_save_timer.duration)

        with log_exec_timer(f"Model loading") as model_load_timer:
            model_loaded = load(
                path=f"/tmp/replay/{model_name}_{dataset_name}_{spark.sparkContext.applicationId}"
            )
        mlflow.log_metric("_loaded_model_sec", model_load_timer.duration)

        with log_exec_timer(
            f"{model_name} prediction from loaded model"
        ) as infer_loaded_timer:
            if isinstance(model_loaded, AssociationRulesItemRec):
                recs = model_loaded.get_nearest_items(
                    items=test,
                    k=k,
                )
            else:
                recs = model_loaded.predict(
                    k=k,
                    users=test.select("user_idx").distinct(),
                    log=train,
                    filter_seen_items=True,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("_loaded_infer_sec", infer_loaded_timer.duration)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(
                f"Metrics calculation for loaded model"
            ) as metrics_loaded_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(use_scala_udf=True): k_list_metrics,
                        NDCG(use_scala_udf=True): k_list_metrics,
                        HitRate(use_scala_udf=True): k_list_metrics,
                    },
                )
                e.add_result(model_name, recs)
            mlflow.log_metric(
                "_loaded_metrics_sec", metrics_loaded_timer.duration
            )
            metrics = dict()
            for k in k_list_metrics:
                metrics["_loaded_NDCG.{}".format(k)] = e.results.at[
                    model_name, "NDCG@{}".format(k)
                ]
                metrics["_loaded_MAP.{}".format(k)] = e.results.at[
                    model_name, "MAP@{}".format(k)
                ]
                metrics["_loaded_HitRate.{}".format(k)] = e.results.at[
                    model_name, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.environ.get("DATASET", "ml1m")
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
