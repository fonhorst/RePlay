import logging
import os
import time

import pendulum
from airflow.decorators import task

from airflow.models.dag import dag
from kubernetes.client import models as k8s


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


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=['first-dag-example'],
)
def build_full_dag():
    check_spark_data()
    test_executor_config()


dag = build_full_dag()
