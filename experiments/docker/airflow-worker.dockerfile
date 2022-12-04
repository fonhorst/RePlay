FROM apache/airflow:2.4.1-python3.9

USER root

RUN mkdir -p /src && chmod 777 /src

RUN apt-get update && apt-get install -y gcc

USER airflow

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt

RUN pip install mlflow-skinny

COPY dist/replay_rec-0.10.0-py3-none-any.whl /src

RUN pip install /src/replay_rec-0.10.0-py3-none-any.whl

COPY experiments /src/experiments
