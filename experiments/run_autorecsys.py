import logging
import sys
import os
from importlib.metadata import version

import mlflow
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession


from experiment_utils import (

    get_spark_configs_as_dict,
    check_number_of_allocated_executors,
    get_partition_num,
    get_models,
)
from replay.experiment import Experiment
from replay.metrics import HitRate, MAP, NDCG
from replay.model_handler import save, load
from replay.models import (
    AssociationRulesItemRec,
    ClusterRec,
    ALSWrap,
    SLIM,
    Word2VecRec,
    ItemKNN,
)
from replay.scenarios import OneStageScenario, AutoRecSysScenario
from replay.session_handler import get_spark_session
from replay.splitters import UserSplitter
from replay.utils import (
    JobGroup,
    log_exec_timer,
)
from replay.utils import logger

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# for k in logging.Logger.manager.loggerDict:
#     print(k)

StreamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
StreamHandler.setFormatter(formatter)

logging.basicConfig(
    level=logging.ERROR,
    handlers=[StreamHandler])

logging.getLogger("replay").setLevel(logging.DEBUG)

logger = logging.getLogger("replay")

print(logger)
print(logger.handlers)








# spark_logger = logging.getLogger("py4j")
# spark_logger.setLevel(logging.WARN)

# formatter = logging.Formatter(
#     "%(asctime)s %(levelname)s %(name)s: %(message)s",
#     datefmt="%d/%m/%y %H:%M:%S",
# )
# streamHandler = logging.StreamHandler()
# streamHandler.setFormatter(formatter)



# logger.addHandler(streamHandler)

# fileHandler = logging.FileHandler("/tmp/replay.log")
# fileHandler.setFormatter(formatter)
# logger.addHandler(fileHandler)

# logger.setLevel(logging.DEBUG)

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)s %(name)s: %(message)s",
#     handlers=[
#         # fileHandler,
#         # streamHandler
#     ],
# )
#
# # logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
# # logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
#
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("replay").setLevel(logging.INFO)


def main(spark: SparkSession, dataset_name: str):

    spark_conf: SparkConf = spark.sparkContext.getConf()

    check_number_of_allocated_executors(spark)

    k = int(os.environ.get("K", 100))
    k_list_metrics = list(
        map(int, os.environ.get("K_LIST_METRICS", "5,10,25,100").split(","))
    )
    seed = int(os.environ.get("SEED", 1234))
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://node2.bdcl:8822"
    )
    model_name = os.environ.get("MODEL", "some_mode")

    partition_num = get_partition_num(spark_conf)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("autorecsys_scenario")

    with mlflow.start_run():
        params = get_spark_configs_as_dict(spark_conf)
        params.update(
            {
                "spark.applicationId": spark.sparkContext.applicationId,
                "pyspark": version("pyspark"),
                "dataset": dataset_name,
                "seed": seed,
                "K": k,
            }
        )
        mlflow.log_params(params)

        train = spark.read.parquet(
            # "hdfs://node21.bdcl:9000"
            # "/opt/spark_data/replay/experiments/netflix_first_level_80_20/train.parquet"
            # "/opt/spark_data/replay/experiments/msd_first_level_80_20/train.parquet"
            # "/opt/spark_data/replay/experiments/ml25m_first_level_80_20/train.parquet"
            "/opt/spark_data/replay_datasets/ml1m_train.parquet"
        )
        test = spark.read.parquet(
            # "hdfs://node21.bdcl:9000"
            # "/opt/spark_data/replay/experiments/netflix_first_level_80_20/test.parquet"
            # "/opt/spark_data/replay/experiments/msd_first_level_80_20/test.parquet"
            # "/opt/spark_data/replay/experiments/ml25m_first_level_80_20/test.parquet"
            "/opt/spark_data/replay_datasets/ml1m_test.parquet"
        )

        logger.debug(f"partition num: {partition_num}")
        train = train.limit(25_000)
        test = test.limit(25_000)

        train = train.repartition(partition_num, "user_idx")
        test = test.repartition(partition_num, "user_idx")
        first_levels_models_params = {
            "replay.models.knn.ItemKNN": {"num_neighbours": int(os.environ.get("NUM_NEIGHBOURS", 100))},
            "replay.models.als.ALSWrap": {
                "rank": int(os.environ.get("ALS_RANK", 100)),
                "seed": seed,
                "num_item_blocks": int(os.environ.get("NUM_BLOCKS", 10)),
                "num_user_blocks": int(os.environ.get("NUM_BLOCKS", 10)),
                "hnswlib_params": {
                    "space": "ip",
                    "M": 100,
                    "efS": 2000,
                    "efC": 2000,
                    "post": 0,
                    "index_path": f"file:///tmp/als_hnswlib_index_{spark.sparkContext.applicationId}",
                    "build_index_on": "executor",
                },
            },
            "replay.models.word2vec.Word2VecRec": {
                "rank": int(os.environ.get("WORD2VEC_RANK", 100)),
                "seed": seed,
                "hnswlib_params": {
                    "space": "ip",
                    "M": 100,
                    "efS": 2000,
                    "efC": 2000,
                    "post": 0,
                    "index_path": f"file:///tmp/word2vec_hnswlib_index_{spark.sparkContext.applicationId}",
                    "build_index_on": "executor",
                },
            },
        }
        mlflow.log_params(first_levels_models_params)

        first_level_models = get_models(first_levels_models_params)

        mlflow.log_param(
            "first_level_models",
            [type(m).__name__ for m in first_level_models],
        )
        # mlflow.log_param(
        #     "use_first_level_models_feat", use_first_level_models_feat
        # )

        scenario = AutoRecSysScenario(
            task="user2item",
            subtask="user_recs",
            timeout=900,
        )

        mlflow.log_param("timer", scenario.timer.timeout)
        # Model fitting
        with log_exec_timer(
            f"{type(scenario).__name__} fitting"
        ) as timer, JobGroup(
            f"{type(scenario).__name__} fitting",
            f"{type(scenario).__name__}.fit()",
        ):
            scenario.fit(log=train,
                         user_features=None,
                         item_features=None)
        mlflow.log_metric(f"{type(scenario).__name__}.fit_sec", timer.duration)

        mlflow.log_param(f"fitted scenario", type(scenario.scenario).__name__)

        # Model inference
        with log_exec_timer(
            f"{type(scenario).__name__} inference"
        ) as timer, JobGroup(
            f"{type(scenario).__name__} inference",
            f"{type(scenario).__name__}.predict()",
        ):
            recs = scenario.predict(
                log=train,
                k=k,
                items=train.select("item_idx").distinct(),
                users=test.select("user_idx").distinct(),
                filter_seen_items=True,
            )
        mlflow.log_metric(
            f"{type(scenario).__name__}.predict_sec", timer.duration
        )

        with log_exec_timer("Metrics calculation") as metrics_timer, JobGroup(
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


if __name__ == "__main__":
    spark_sess = get_spark_session()
    spark_sess.sparkContext.setLogLevel('ERROR')
    dataset = os.environ.get("DATASET", "MovieLens_1m")
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
