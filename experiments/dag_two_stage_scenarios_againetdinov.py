import os
import pickle
import uuid
from datetime import timedelta
from typing import Dict, Optional, Union, List

import pendulum
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from dag_entities_againetdinov import ArtifactPaths, DEFAULT_CPU, DEFAULT_MEMORY, DatasetInfo, DATASETS, \
    SECOND_LEVELS_MODELS_CONFIGS, TASK_CONFIG_FILENAME_ENV_VAR
from dag_entities_againetdinov import EXTRA_BIG_CPU, EXTRA_BIG_MEMORY, SECOND_LEVELS_MODELS_PARAMS
from dag_entities_againetdinov import YARN_SUBMIT_CONF


def combine_1lvl_datasets_spark_submit(
        task_name: str,
        artifacts: ArtifactPaths,
        combined_train_path: str,
        combined_predicts_path: str,
        desired_models: Optional[List[str]] = None,
        mode: str = 'union'
):
    config_filename = f"task_config_{task_name}_{uuid.uuid4()}.pickle"
    with open(config_filename, "wb") as f:
        pickle.dump({
            "artifacts": artifacts,
            "combined_train_path": combined_train_path,
            "combined_predicts_path": combined_predicts_path,
            "desired_models": desired_models,
            "mode": mode,
        }, f)

    submit_job = SparkSubmitOperator(
        application="/opt/spark_data/spark_submit_files/dag_utils_againetdinov.py",
        name=f"submit_{task_name}",
        task_id=f"submit_{task_name}",
        files=f"{config_filename},/opt/spark_data/spark_submit_files/tabular_config.yml",
        conf=YARN_SUBMIT_CONF,
        env_vars={
            TASK_CONFIG_FILENAME_ENV_VAR: config_filename
        },
        py_files='/opt/spark_data/spark_submit_files/replay_rec-0.10.0-py3-none-any_againetdinov.whl,'
                 '/opt/spark_data/spark_submit_files/dag_entities_againetdinov.py,'
                 '/opt/spark_data/spark_submit_files/sparklightautoml_dev-0.3.2-py3-none-any_dalexandrov.whl',
        jars='/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar,'
             '/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar'
    )

    return submit_job


def fit_predict_second_level_model_spark_submit(
        task_name: str,
        artifacts: ArtifactPaths,
        model_name: str,
        k: int,
        train_path: str,
        first_level_predicts_path: str,
        second_model_type: str = "lama",
        second_model_params: Optional[Union[Dict, str]] = None,
        second_model_config_path: Optional[str] = None,
        cpu: int = DEFAULT_CPU,
        memory: int = DEFAULT_MEMORY):
    config_filename = f"task_config_{task_name}_{uuid.uuid4()}.pickle"
    with open(config_filename, "wb") as f:
        pickle.dump({
            "artifacts": artifacts,
            "model_name": model_name,
            "k": k,
            "train_path": train_path,
            "first_level_predicts_path": first_level_predicts_path,
            "second_model_type": second_model_type,
            "second_model_params": second_model_params,
            "second_model_config_path": second_model_config_path,
            "cpu": cpu,
            "memory": memory
        }, f)

    submit_job = SparkSubmitOperator(
        application="/opt/spark_data/spark_submit_files/dag_utils_againetdinov.py",
        name=task_name,
        task_id=f"submit_{task_name}",
        files=f"{config_filename},/opt/spark_data/spark_submit_files/tabular_config.yml",
        conf=YARN_SUBMIT_CONF,
        env_vars={
            TASK_CONFIG_FILENAME_ENV_VAR: config_filename
        },
        py_files='/opt/spark_data/spark_submit_files/replay_rec-0.10.0-py3-none-any_againetdinov.whl,'
                 '/opt/spark_data/spark_submit_files/dag_entities_againetdinov.py,'
                 '/opt/spark_data/spark_submit_files/sparklightautoml_dev-0.3.2-py3-none-any_dalexandrov.whl',
        jars='/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar,'
             '/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar,'
             '/opt/spark_data/spark_submit_files/synapseml_2.12-0.9.5.jar'
    )

    return submit_job


def _make_combined_2lvl(artifacts: ArtifactPaths,
                        model_name: str,
                        combiner_suffix: str,
                        k: int,
                        mode: str = 'union',
                        desired_1lvl_models: Optional[List[str]] = None):
    combined_train_path = artifacts.make_path(f"combined_train_{combiner_suffix}.parquet")
    combined_predicts_path = artifacts.make_path(f"combined_predicts_{combiner_suffix}.parquet")

    combiner = combine_1lvl_datasets_spark_submit(
        task_name=f"combiner_{combiner_suffix}",
        artifacts=artifacts,
        combined_train_path=combined_train_path,
        combined_predicts_path=combined_predicts_path,
        desired_models=desired_1lvl_models,
        mode=mode
    )

    second_level_model = fit_predict_second_level_model_spark_submit(
        task_name=f"2lvl_{model_name.split('.')[-1]}_{combiner_suffix}",
        artifacts=artifacts,
        model_name=f"{model_name}_{combiner_suffix}",
        k=k,
        train_path=combined_train_path,
        first_level_predicts_path=combined_predicts_path,
        second_model_type=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_type"],
        second_model_params=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_params"],
        second_model_config_path=SECOND_LEVELS_MODELS_CONFIGS.get(model_name, None),
        cpu=EXTRA_BIG_CPU,
        memory=EXTRA_BIG_MEMORY,
    )

    return combiner >> second_level_model


def build_combiner_second_level(dag_id: str, mlflow_exp_id: str, dataset: DatasetInfo, path_suffix: str = "default",
                                model_name: str = 'slama_fast'):
    os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8822"
    os.environ["MLFLOW_EXPERIMENT_ID"] = mlflow_exp_id

    # path_suffix = 'default'
    artifacts = ArtifactPaths(
        base_path=f"/opt/spark_data/replay/experiments/{dataset.name}_first_level_{path_suffix}",
        dataset=dataset
    )

    k = 100

    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
            tags=['two_stage', 'first level', 'second level', 'replay', 'slama']
    ) as dag:
        _make_combined_2lvl(
            artifacts=artifacts,
            model_name=model_name,
            combiner_suffix="4models_outer_join",  # an important part
            k=k,
            mode='union',
            desired_1lvl_models=["alswrap", "itemknn", "slim", "word2vecrec"]  # ['alswrap', 'slim']
        )

    return dag

# ml1m_combined_second_level_dag_80 = build_combiner_second_level(
#     dag_id="ml1m_combined_second_level_dag_4models_80_20_a",
#     mlflow_exp_id="two_stage_cluster",
#     dataset=DATASETS["ml1m"],
#     path_suffix="80_20",
#     model_name="slama_fast"
# )


ml1m_combined_second_level_dag_80 = build_combiner_second_level(
    dag_id="ml1m_combined_outer_join_second_level_4models_80_20",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["ml1m"],
    path_suffix="80_20",
    model_name="slama_fast"
)

ml25m_combined_second_level_dag_80 = build_combiner_second_level(
    dag_id="ml25m_combined_outer_join_second_level_4models_80_20",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["ml25m"],
    path_suffix="80_20",
    model_name="slama_fast"
)

msd_combined_second_level_dag_80 = build_combiner_second_level(
    dag_id="msd_combined_outer_join_second_level_4models_80_20",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["msd"],
    path_suffix="80_20",
    model_name="slama_fast"
)

ml25m_combined_second_level_dag_80_wo_tuning = build_combiner_second_level(
    dag_id="ml25m_combined_outer_join_second_level_dag_4models_80_20_wo_tuning",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["ml25m"],
    path_suffix="80_20",
    model_name="longer_slama_wo_tuning"
)
