FROM apache/airflow:2.4.1-python3.9

USER root

RUN apt-get update && \
	apt-get install -y openjdk-11-jre net-tools wget nano iputils-ping curl gcc g++ && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN mkdir -p /src && chmod 777 /src

USER airflow

RUN pip install --upgrade pip

RUN pip install "apache-airflow-providers-apache-spark" \
--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.4.1/constraints-3.7.txt"

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

COPY SparkLightAutoML_DEV-0.3.2-py3-none-any.whl /src/

#RUN pip install --force-reinstall --no-deps /src/SparkLightAutoML_DEV-0.3.2-py3-none-any.whl
RUN pip install /src/SparkLightAutoML_DEV-0.3.2-py3-none-any.whl

RUN pip install pyspark==3.1.3

COPY core-site.xml /etc/hadoop/core-site.xml

COPY yarn-site.xml /etc/hadoop/yarn-site.xml

ENV HADOOP_CONF_DIR=/etc/hadoop/

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt

RUN pip install mlflow-skinny

RUN pip install sqlalchemy==1.4.41

RUN pip install pyspark==3.1.3
