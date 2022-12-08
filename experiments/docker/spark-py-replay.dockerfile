ARG base_image
FROM ${base_image}

ARG spark_jars_cache=jars_cache

USER root

RUN pip install pyspark==3.2.0

#USER ${spark_id}

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

RUN mkdir -p /src

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt

RUN pip install mlflow-skinny

COPY dist/replay_rec-0.10.0-py3-none-any.whl /src

RUN pip install /src/replay_rec-0.10.0-py3-none-any.whl

COPY scala/target/scala-2.12/replay_2.12-0.1.jar /src/

ENV REPLAY_JAR_PATH=/src/replay_2.12-0.1.jar

COPY spark-lightautoml_2.12-0.1.1.jar /src

ENV SLAMA_JAR_PATH=/src/spark-lightautoml_2.12-0.1.1.jar

COPY SparkLightAutoML_DEV-0.3.2-py3-none-any.whl /src/

#RUN pip install --force-reinstall --no-deps /src/SparkLightAutoML_DEV-0.3.2-py3-none-any.whl
RUN pip install /src/SparkLightAutoML_DEV-0.3.2-py3-none-any.whl

RUN pip install pyspark==3.2.0

ENV PYSPARK_PYTHON=python3

WORKDIR /root
