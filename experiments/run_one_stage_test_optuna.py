import logging
import os, sys
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
from replay.scenarios import OneStageUser2ItemScenario, OneStageItem2ItemScenario
from replay.session_handler import get_spark_session
from replay.splitters import UserSplitter
from replay.utils import (
    JobGroup,
    log_exec_timer,
)
# from replay.utils import logger

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

StreamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
StreamHandler.setFormatter(formatter)

logging.basicConfig(
    level=logging.ERROR,
    handlers=[StreamHandler])

logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)


FIRST_LEVELS_MODELS_PARAMS_BORDERS = {
    "replay.models.als.ALSWrap": {
        "rank": [10, 300]
    },
    "replay.models.knn.ItemKNN": {
        "num_neighbours": [50, 1000],
                                  },
    "replay.models.slim.SLIM": {
        "beta": [1e-6, 1],  # 1e-6, 5 #0.01, 0.1
        "lambda_": [1e-8, 1e-5]  # [1e-6, 2] #0.01, 0.1
    },
    "replay.models.word2vec.Word2VecRec": {
        "rank": [10, 300],
        "window_size": [1, 50],
        # "max_iter": [1, 5],
        # "use_idf": [True, False]
    },
    "replay.models.association_rules.AssociationRulesItemRec": {
        "min_item_count": [1, 10],
        "min_pair_count": [1, 10],
        "num_neighbours": [50, 200]
    },
}

def main(spark: SparkSession, dataset_name: str):

    item2item = False
    do_optimization = True

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
    mlflow.set_experiment("one-stage")

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

        if item2item:
            rr_path = "/opt/spark_data/replay_datasets/"
            train = spark.read.parquet(os.path.join(rr_path, "retail_rocket_events_indexed_train30k.parquet"))
            train_opt = spark.read.parquet(os.path.join(rr_path, "retail_rocket_events_indexed_train_opt_30k.parquet"))
            val_opt = spark.read.parquet(os.path.join(rr_path, "retail_rocket_events_indexed_val_opt_30k.parquet"))
            # test = spark.read.parquet(os.path.join(rr_path, "retail_rocket_events_indexed_test30k.parquet"))
            train = train_opt


            test_first_item = spark.read.parquet(
                "/opt/spark_data/replay_datasets/retail_rocket_events_indexed_test_first_item_30k.parquet")
            test_other_items = spark.read.parquet(
                "/opt/spark_data/replay_datasets/retail_rocket_events_indexed_test_other_items_30k.parquet")

            test_first_item = test_first_item.repartition(partition_num, "user_idx")
            test_other_items = test_other_items.repartition(partition_num, "user_idx")
        else:
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

            train = train.limit(25_000)
            test = test.limit(25_000)
            test = test.repartition(partition_num, "user_idx")

        train = train.repartition(partition_num, "user_idx")

        # first_level_models_names = ["replay.models.als.ALSWrap",
        #                             "replay.models.slim.SLIM",
        #                             "replay.models.knn.ItemKNN",
        #                             "replay.models.word2vec.Word2VecRec",
        #                             "replay.models.association_rules.AssociationRulesItemRec"
        #                             ]


        first_levels_models_params = {
            "replay.models.knn.ItemKNN": {"num_neighbours": 100},
            # "replay.models.association_rules.AssociationRulesItemRec": {"num_neighbours":100,
            #                                                             "min_item_count":1,
            #                                                             "min_pair_count":1},}
            "replay.models.als.ALSWrap": {
                "rank": int(os.environ.get("ALS_RANK", 100)),
                "seed": seed,
                "num_item_blocks": int(os.environ.get("NUM_BLOCKS", 10)),
                "num_user_blocks": int(os.environ.get("NUM_BLOCKS", 10)),},
                # "hnswlib_params": {
                #     "space": "ip",
                #     "M": 100,
                #     "efS": 2000,
                #     "efC": 2000,
                #     "post": 0,
                #     "index_path": f"file:///tmp/als_hnswlib_index_{spark.sparkContext.applicationId}",
                #     "build_index_on": "executor",
                # },
            # },
            "replay.models.word2vec.Word2VecRec": {
                "rank": int(os.environ.get("WORD2VEC_RANK", 100)),
                "seed": seed,}}
            #     "hnswlib_params": {
            #         "space": "ip",
            #         "M": 100,
            #         "efS": 2000,
            #         "efC": 2000,
            #         "post": 0,
            #         "index_path": f"file:///tmp/word2vec_hnswlib_index_{spark.sparkContext.applicationId}",
            #         "build_index_on": "executor",
            #     },
            # },
        # }
        mlflow.log_params(first_levels_models_params)

        first_level_models = get_models(first_levels_models_params)

        mlflow.log_param(
            "first_level_models",
            [type(m).__name__ for m in first_level_models],
        )
        # mlflow.log_param(
        #     "use_first_level_models_feat", use_first_level_models_feat
        # )

        if item2item:
            scenario = OneStageItem2ItemScenario(
                first_level_models=first_level_models,  # first level model with default hyperparameters
                user_cat_features_list=None,
                item_cat_features_list=None,
                set_best_model=True
            )
        else:
            scenario = OneStageUser2ItemScenario(
                first_level_models=first_level_models,  # first level model with default hyperparameters
                user_cat_features_list=None,
                item_cat_features_list=None,
                set_best_model=True
            )

        if do_optimization:

            param_borders = [
                FIRST_LEVELS_MODELS_PARAMS_BORDERS[model_name] for model_name in first_levels_models_params.keys()
            ]
            param_borders.append(None)

            if item2item:
                scenario.optimize(train=train_opt, test=val_opt, param_borders=param_borders, budget=3)
            else:
                # scenario.optimize(train=train, test=test, param_borders=param_borders, budget=5)
                scenario.optimize(train=train, test=test, param_borders=param_borders, budget=2)
                scenario.optimize(train=train, test=test, param_borders=param_borders, budget=3, new_study=False)
        assert False
        # Model fitting
        with log_exec_timer(
            f"{type(scenario).__name__} fitting"
        ) as timer, JobGroup(
            f"{type(scenario).__name__} fitting",
            f"{type(scenario).__name__}.fit()",
        ):
            scenario.fit(log=train, user_features=None, item_features=None)
        mlflow.log_metric(f"{type(scenario).__name__}.fit_sec", timer.duration)

        # logger.debug("saving scenario")
        # save(scenario, "/tmp/one_stage", overwrite=True)
        # logger.debug("loading scenario")
        # scenario = load("/tmp/one_stage")


        # Model inference
        with log_exec_timer(
            f"{type(scenario).__name__} inference"
        ) as timer, JobGroup(
            f"{type(scenario).__name__} inference",
            f"{type(scenario).__name__}.predict()",
        ):

            if item2item:
                nearest_items = scenario._predict(
                    log=train,
                    k=k,
                    users=train.select("user_idx").distinct(),
                    items=test_first_item.select("item_idx").distinct(),
                  )

                logger.debug(f"nearest_items: {nearest_items}")
                logger.debug(f"nearest_items count: {nearest_items.count()}")
                nearest_items = nearest_items.filter(nearest_items.item_idx != nearest_items.neighbour_item_idx)
                # nearest_items.write.mode("overwrite").parquet("/opt/spark_data/replay_datasets/nearest_items_30k_cosine_filtered.parquet")

                nearest_items = nearest_items \
                    .withColumnRenamed("cosine_similarity", "similarity") \
                    .withColumnRenamed("confidence", "similarity")

                pred = test_first_item.select("user_idx", "item_idx", "timestamp").join(nearest_items, "item_idx",
                                                                                        how="left") \
                    .select("user_idx", "neighbour_item_idx", "similarity", "timestamp")

                pred = pred \
                    .withColumnRenamed("neighbour_item_idx", "item_idx") \
                    .withColumnRenamed("similarity", "relevance")

                logger.debug(f"pred: {pred}")
                logger.debug(f"pred count: {pred.count()}")
                pred = pred.filter(pred.item_idx.isNotNull())  # TODO: add smth to items which not in index

                # pred.write.mode("overwrite").parquet("/opt/spark_data/replay_datasets/preds_original_1m.parquet")
                # pred.write.mode("overwrite").parquet("/opt/spark_data/replay_datasets/preds_1m_w2w_v2.parquet")
                # pred.write.mode("overwrite").parquet("/opt/spark_data/replay_datasets/preds_hnsw_w2w.parquet")

                # pred.write.mode("overwrite").parquet("/opt/spark_data/replay_datasets/preds_hnsw_30k_cosine.parquet")

                test_other_items = test_other_items.select("user_idx", "item_idx", "relevance", "timestamp")
                logger.debug(f"users_other_items: {test_other_items}")

            else:

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
            if item2item:
                e = Experiment(
                    test_other_items,
                    {
                        MAP(use_scala_udf=True): k_list_metrics,
                        NDCG(use_scala_udf=True): k_list_metrics,
                        HitRate(use_scala_udf=True): k_list_metrics,
                    },
                )
                e.add_result(model_name, pred)
            else:

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