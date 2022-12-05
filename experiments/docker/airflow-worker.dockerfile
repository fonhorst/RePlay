FROM apache/airflow:2.4.1-python3.9

USER root

RUN apt-get update && \
	apt-get install -y openjdk-11-jre net-tools wget nano iputils-ping curl gcc && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN mkdir -p /src && chmod 777 /src

USER airflow

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt

RUN pip install mlflow-skinny

COPY dist/replay_rec-0.10.0-py3-none-any.whl /src

RUN pip install /src/replay_rec-0.10.0-py3-none-any.whl

COPY experiments /src/experiments

COPY scala/target/scala-2.12/replay_2.12-0.1.jar /src/

ENV REPLAY_JAR_PATH=/src/replay_2.12-0.1.jar
