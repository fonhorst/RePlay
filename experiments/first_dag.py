import logging
import os
import pickle
import time

import mlflow
import pendulum
from airflow.decorators import task

from airflow.models.dag import dag
from kubernetes.client import models as k8s
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split

executor_config = {
            "pod_override": k8s.V1Pod(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",
                            resources=k8s.V1ResourceRequirements(requests={"cpu": 16}, limits={"cpu": 16})
                        )
                    ],
                )
            ),
        }


@task
def check_spark_data():
    files = os.listdir("/opt/spark_data")

    os.makedirs("/opt/spark_data/tmp", exist_ok=True)

    with open("/opt/spark_data/tmp/files_listing.txt", "w") as f:
        f.write("\n".join(files))

    time.sleep(300)


@task(executor_config=executor_config)
def test_executor_config():
    logger = logging.getLogger("airflow.task")
    logger.info("Hello World!")
    time.sleep(300)

@task
def check_lama():
    logger = logging.getLogger("airflow.task")

    use_algos = [["lgb"]]

    path = "/opt/spark_data/sampled_app_train.csv"
    task_type = "binary"
    roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}
    dtype = dict()

    data = pd.read_csv(path, dtype=dtype)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    task = Task(task_type)

    with mlflow.start_run(experiment_id="106") as run:
        mlflow.log_param("path", path)
        mlflow.log_param("task_type", task_type)

        num_threads = 6
        automl = TabularAutoML(
            task=task,
            cpu_limit=num_threads,
            timeout=3600 * 3,
            general_params={"use_algos": use_algos},
            reader_params={"cv": 5, "advanced_roles": False},
            lgb_params={"default_params": {"num_threads": num_threads}},
            # linear_l2_params={"default_params": {"cs": [1e-5]}},
            tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

        logger.info("Predicting on out of fold")

        score = task.get_dataset_metric()
        metric_value = score(oof_predictions)

        logger.info(f"Score for out-of-fold predictions: {metric_value}")
        mlflow.log_metric("OOF", metric_value)

        te_pred = automl.predict(test_data)
        te_pred.target = test_data[roles['target']]

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

        logger.info(f"Score for test predictions: {test_metric_value}")
        mlflow.log_metric("TEST", test_metric_value)

        logger.info("Predicting is finished")

        with open("model.pickle", "wb") as f:
            pickle.dump(automl, f)
        mlflow.log_artifact("model.pickle", "model.pickle")

        logger.info("Model saving is finished")


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=['first-dag-example'],
)
def build_full_dag():
    os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
    # check_spark_data()
    # test_executor_config()
    check_lama()


dag = build_full_dag()
