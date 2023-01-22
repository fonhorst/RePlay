import logging.config
import os

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType

from experiment_utils import get_model
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.models import (
    AssociationRulesItemRec,
    UserPopRec
)
from replay.session_handler import get_spark_session
from replay.utils import (
    JobGroup,
    getNumberOfAllocatedExecutors,
    log_exec_timer,
)
from replay.utils import get_log_info2
from replay.utils import logger


logger.setLevel(logging.DEBUG)


def main(spark: SparkSession, dataset_name: str):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    if spark_conf.get("spark.executor.instances"):
        if getNumberOfAllocatedExecutors(spark) < int(
            spark_conf.get("spark.executor.instances")
        ):
            raise Exception("Not enough executors to run experiment!")

    K = int(os.environ.get("K", 10))
    K_list_metrics = [10]
    SEED = int(os.environ.get("SEED", 1234))
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8811"
    )
    MODEL = os.environ.get("MODEL", "ALS_NMSLIB_HNSW")
    # PopRec
    # Word2VecRec Word2VecRec_NMSLIB_HNSW
    # ALS ALS_NMSLIB_HNSW
    # SLIM SLIM_NMSLIB_HNSW
    # ItemKNN ItemKNN_NMSLIB_HNSW
    # ClusterRec
    # UCB

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))  # 28

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(
        os.environ.get("EXPERIMENT", "delete")
    )  # os.environ["EXPERIMENT"]

    with mlflow.start_run():
        params = {
            "spark.driver.cores": spark_conf.get("spark.driver.cores"),
            "spark.driver.memory": spark_conf.get("spark.driver.memory"),
            "spark.memory.fraction": spark_conf.get("spark.memory.fraction"),
            "spark.executor.cores": spark_conf.get("spark.executor.cores"),
            "spark.executor.memory": spark_conf.get("spark.executor.memory"),
            "spark.executor.instances": spark_conf.get(
                "spark.executor.instances"
            ),
            "spark.sql.shuffle.partitions": spark_conf.get(
                "spark.sql.shuffle.partitions"
            ),
            "spark.default.parallelism": spark_conf.get(
                "spark.default.parallelism"
            ),
            "spark.applicationId": spark.sparkContext.applicationId,
            "dataset": dataset_name,
            "seed": SEED,
            "K": K,
        }
        mlflow.log_params(params)

        if dataset_name == "ml1m":        
            schema = (
                StructType()
                .add("relevance", IntegerType(), True)
                .add("timestamp", IntegerType(), True)
                .add("user_idx", IntegerType(), True)
                .add("item_idx", IntegerType(), True)
            )

            train70 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train70_ml1m.csv"
                )
            )
            train80 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train80_ml1m.csv"
                )
            )
            train_diff80 = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/train_dif80_ml1m.csv"
                )
            )
            test = (
                spark.read.option("header", True)
                .format("csv")
                .schema(schema)
                .load(
                    "file:///opt/spark_data/replay_datasets/MovieLens/posttraining/test_ml1m.csv"
                )
            )
        elif dataset_name == "MillionSongDataset":
            train70 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train70.parquet"
                )
            )
            train80 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train80.parquet"
                )
            )
            train_diff80 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train_diff80.parquet"
                )
            )
            test = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/test.parquet"
                )
            )
        elif dataset_name == "MillionSongDataset10x":
            train70 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train70_10x.parquet"
                )
            )
            train80 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train80_10x.parquet"
                )
            )
            train_diff80 = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/train_diff80_10x.parquet"
                )
            )
            test = (
                spark.read.parquet(
                    "/opt/spark_data/replay_datasets/MillionSongDataset/test_10x.parquet"
                )
            )
        else:
            ValueError("Unknown dataset.")

        kwargs = {}
        if MODEL == "ClusterRec":
            user_features = spark.read.parquet(
                "file:///opt/spark_data/replay_datasets/MovieLens/train80_ml1m_user_features.parquet"
            )
            kwargs = {"user_features": user_features}

        mlflow.log_param(
            "USE_BUCKETING", os.environ.get("USE_BUCKETING", "False")
        )
        if os.environ.get("USE_BUCKETING", "False") == "True":
            BUCKETING_KEY = "user_idx"

            with log_exec_timer("Train/test caching") as bucketing_timer:
                (
                    train70.repartition(partition_num, BUCKETING_KEY)
                    .write.mode("overwrite")
                    .bucketBy(partition_num, BUCKETING_KEY)
                    .sortBy(BUCKETING_KEY)
                    .saveAsTable(
                        f"bucketed_train_{spark.sparkContext.applicationId}",
                        format="parquet",
                        path=f"/spark-warehouse/bucketed_train_{spark.sparkContext.applicationId}",
                    )
                )

                train70 = spark.table(
                    f"bucketed_train_{spark.sparkContext.applicationId}"
                )

                (
                    test.repartition(partition_num, BUCKETING_KEY)
                    .write.mode("overwrite")
                    .bucketBy(partition_num, BUCKETING_KEY)
                    .sortBy(BUCKETING_KEY)
                    .saveAsTable(
                        f"bucketed_test_{spark.sparkContext.applicationId}",
                        format="parquet",
                        path=f"/spark-warehouse/bucketed_test_{spark.sparkContext.applicationId}",
                    )
                )
                test = spark.table(
                    f"bucketed_test_{spark.sparkContext.applicationId}"
                )

            mlflow.log_metric("bucketing_sec", bucketing_timer.duration)

        with log_exec_timer("Train/test caching") as train_test_cache_timer:
            train70 = train70.cache()
            train80 = train80.cache()
            train_diff80 = train_diff80.cache()
            test = test.cache()
            train70.write.mode("overwrite").format("noop").save()
            train80.write.mode("overwrite").format("noop").save()
            train_diff80.write.mode("overwrite").format("noop").save()
            test.write.mode("overwrite").format("noop").save()
        mlflow.log_metric(
            "train_test_cache_sec", train_test_cache_timer.duration
        )

        mlflow.log_metric(
            "train_num_partitions", train70.rdd.getNumPartitions()
        )
        mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

        with log_exec_timer(
            "get_log_info2() execution"
        ) as get_log_info2_timer:
            train_info = get_log_info2(train80)
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
        mlflow.log_params({
            "get_log_info_sec": get_log_info2_timer.duration,
            "train.total_users": train_info[1],
            "train.total_items": train_info[2],
            "train_size": train_info[0],
            "test_size": test_info[0],
            "test.total_users": test_info[1],
            "test.total_items": test_info[2],
        })

        mlflow.log_param("model", MODEL)
        model = get_model(MODEL, SEED, spark.sparkContext.applicationId)

        filter_seen_items = True
        if isinstance(model, UserPopRec):
            filter_seen_items = False

        print("\ndataset: train70")
        with log_exec_timer(f"{MODEL} training") as train_timer, JobGroup(
            "Model training", f"{model.__class__.__name__}.fit()"
        ):
            model.fit(log=train70, **kwargs)
        mlflow.log_metric("train70_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            if isinstance(model, AssociationRulesItemRec):
                recs = model.get_nearest_items(
                    items=test,
                    k=K,
                )
            else:
                recs = model.predict(
                    k=K,
                    users=test.select("user_idx").distinct(),
                    log=train70,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer70_sec", infer_timer.duration)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(): K_list_metrics,
                        NDCG(): K_list_metrics,
                        HitRate(): K_list_metrics,
                    },
                )
                e.add_result(MODEL, recs)
            mlflow.log_metric("metrics70_sec", metrics_timer.duration)
            for k in K_list_metrics:
                mlflow.log_metric(
                    "NDCG.{}".format(k), e.results.at[MODEL, "NDCG@{}".format(k)]
                )
                mlflow.log_metric(
                    "MAP.{}".format(k), e.results.at[MODEL, "MAP@{}".format(k)]
                )
                mlflow.log_metric(
                    "HitRate.{}".format(k),
                    e.results.at[MODEL, "HitRate@{}".format(k)],
                )

        print("\ndataset: train_diff")

        with log_exec_timer(
            f"{MODEL} training (additional)"
        ) as train_timer, JobGroup(
            "Model training (additional)", f"{model.__class__.__name__}.fit()"
        ):
            model.refit(log=train_diff80, previous_log=train70)
        mlflow.log_metric("train_diff_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model.__class__.__name__}.predict()"
        ):
            if isinstance(model, AssociationRulesItemRec):
                recs = model.get_nearest_items(
                    items=test,
                    k=K,
                )
            else:
                recs = model.predict(
                    k=K,
                    users=test.select("user_idx").distinct(),
                    log=train80,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_diff_sec", infer_timer.duration)

        if not isinstance(model, AssociationRulesItemRec):
            with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(): K_list_metrics,
                        NDCG(): K_list_metrics,
                        HitRate(): K_list_metrics,
                    },
                )
                e.add_result(MODEL, recs)
            mlflow.log_metric("metrics_diff_sec", metrics_timer.duration)
            for k in K_list_metrics:
                mlflow.log_metric(
                    "NDCG.{}_diff".format(k),
                    e.results.at[MODEL, "NDCG@{}".format(k)],
                )
                mlflow.log_metric(
                    "MAP.{}_diff".format(k),
                    e.results.at[MODEL, "MAP@{}".format(k)],
                )
                mlflow.log_metric(
                    "HitRate.{}_diff".format(k),
                    e.results.at[MODEL, "HitRate@{}".format(k)],
                )

        # unpersist all caches exclude train80
        for (_id, rdd) in spark.sparkContext._jsc.getPersistentRDDs().items():
            rdd.unpersist()
            print("Unpersisted {} rdd".format(_id))
        test = test.cache()
        train80 = train80.cache()
        train80.write.mode("overwrite").format("noop").save()
        test.write.mode("overwrite").format("noop").save()
        
        print("\ndataset: train80")
        del model
        model2 = get_model(MODEL, SEED, spark.sparkContext.applicationId)
        with log_exec_timer(f"{MODEL} training") as train_timer, JobGroup(
            "Model training", f"{model2.__class__.__name__}.fit()"
        ):
            model2.fit(log=train80, **kwargs)
        mlflow.log_metric("train80_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer, JobGroup(
            "Model inference", f"{model2.__class__.__name__}.predict()"
        ):
            if isinstance(model2, AssociationRulesItemRec):
                recs = model2.get_nearest_items(
                    items=test,
                    k=K,
                )
            else:
                recs = model2.predict(
                    k=K,
                    users=test.select("user_idx").distinct(),
                    log=train80,
                    filter_seen_items=filter_seen_items,
                    **kwargs,
                )
            recs = recs.cache()
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer80_sec", infer_timer.duration)

        if not isinstance(model2, AssociationRulesItemRec):
            with log_exec_timer(f"Metrics calculation") as metrics_timer, JobGroup(
                "Metrics calculation", "e.add_result()"
            ):
                e = Experiment(
                    test,
                    {
                        MAP(): K_list_metrics,
                        NDCG(): K_list_metrics,
                        HitRate(): K_list_metrics,
                    },
                )
                e.add_result(MODEL, recs)
            mlflow.log_metric("metrics80_sec", metrics_timer.duration)
            metrics = dict()
            for k in K_list_metrics:
                metrics["NDCG.{}_80".format(k)] = e.results.at[
                    MODEL, "NDCG@{}".format(k)
                ]
                metrics["MAP.{}_80".format(k)] = e.results.at[
                    MODEL, "MAP@{}".format(k)
                ]
                metrics["HitRate.{}_80".format(k)] = e.results.at[
                    MODEL, "HitRate@{}".format(k)
                ]
            mlflow.log_metrics(metrics)


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.environ.get("DATASET", "ml1m")  # ml1m
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
