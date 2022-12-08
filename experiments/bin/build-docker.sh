#!/usr/bin/env bash

set -e

cp ../LightAutoML/dist/SparkLightAutoML_DEV-0.3.2-py3-none-any.whl .

cp ../LightAutoML/jars/spark-lightautoml_2.12-0.1.1.jar .

#poetry build

#poetry export --without-hashes -f requirements.txt > requirements.txt

docker build -t node2.bdcl:5000/airflow-worker:latest -f experiments/docker/airflow-worker.dockerfile .

rm SparkLightAutoML_DEV-0.3.2-py3-none-any.whl
rm spark-lightautoml_2.12-0.1.1.jar
