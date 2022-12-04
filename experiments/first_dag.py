import os

import pendulum
from airflow.decorators import task

from airflow.models.dag import dag


@task
def check_spark_data():
    files = os.listdir("/opt/spark_data")

    os.makedirs("/opt/spark_data/tmp")

    with open("/opt/spark_data/tmp/files_listing.txt", "w") as f:
        f.write("\n".join(files))


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=['first-dag-example'],
)
def build_full_dag():
    check_spark_data()


dag = build_full_dag()

k = 0