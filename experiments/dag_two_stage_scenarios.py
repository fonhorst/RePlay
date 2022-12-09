import os
from datetime import timedelta
from typing import Dict, Any

import pendulum
from airflow import DAG
from airflow.decorators import task

from dag_entities import ArtifactPaths, DEFAULT_CPU, DEFAULT_MEMORY, DatasetInfo, BIG_CPU, BIG_MEMORY, \
    big_executor_config, _get_models_params, DATASETS


@task
def dataset_splitting(artifacts: ArtifactPaths, partitions_num: int):
    from dag_utils import do_dataset_splitting
    do_dataset_splitting(artifacts, partitions_num)


@task
def init_refitable_two_stage_scenario(artifacts: ArtifactPaths):
    from dag_utils import do_init_refitable_two_stage_scenario
    do_init_refitable_two_stage_scenario(artifacts)


@task
def fit_feature_transformers(artifacts: 'ArtifactPaths', cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    from dag_utils import do_fit_feature_transformers
    do_fit_feature_transformers(artifacts, cpu, memory)


@task
def presplit_data(artifacts: ArtifactPaths, cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    from dag_utils import do_presplit_data
    do_presplit_data(artifacts, cpu, memory)


def fit_predict_first_level_model(artifacts: ArtifactPaths,
                                  model_class_name: str,
                                  model_kwargs: Dict,
                                  k: int,
                                  cpu: int = DEFAULT_CPU,
                                  memory: int = DEFAULT_MEMORY):
    from dag_utils import do_fit_predict_first_level_model
    do_fit_predict_first_level_model(artifacts, model_class_name, model_kwargs, k, cpu, memory)


def build_fit_predict_first_level_models_dag(
        dag_id: str,
        mlflow_exp_id: str,
        model_params_map: Dict[str, Dict[str, Any]],
        dataset: DatasetInfo
):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
            tags=['two_stage', 'replay', 'first level']
    ) as dag:
        os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
        os.environ["MLFLOW_EXPERIMENT_ID"] = os.environ.get("MLFLOW_EXPERIMENT_ID", mlflow_exp_id)

        # TODO: fix it later
        # path_suffix = Variable.get(f'{dataset.name}_artefacts_dir_suffix', 'default')
        path_suffix = 'default'
        artifacts = ArtifactPaths(
            base_path=f"/opt/spark_data/replay/experiments/{dataset.name}_first_level_{path_suffix}",
            dataset=dataset
        )
        k = 100

        first_level_models = [
            task(
                task_id=f"fit_predict_first_level_model_{model_class_name.split('.')[-1]}",
                executor_config=big_executor_config
            )(fit_predict_first_level_model)(
                artifacts=artifacts,
                model_class_name=model_class_name,
                model_kwargs=model_kwargs,
                k=k,
                cpu=BIG_CPU,
                memory=BIG_MEMORY
            )
            for model_class_name, model_kwargs in model_params_map.items()
        ]

        dataset_splitting(artifacts, partitions_num=100) \
            >> presplit_data(artifacts) \
            >> fit_feature_transformers(artifacts) \
            >> first_level_models

    return dag


ml1m_first_level_dag = build_fit_predict_first_level_models_dag(
    dag_id="ml1m_first_level_dag",
    mlflow_exp_id="111",
    model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "cluster"),
    dataset=DATASETS["ml1m"]
)


# ml25m_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="ml25m_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "cluster"),
#     dataset=DATASETS["ml25m"]
# )
#
#
# msd_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="msd_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "cluster"),
#     dataset=DATASETS["msd"]
# )
