import os
from datetime import timedelta
from typing import Dict, Any, Optional, Union

import pendulum
from airflow import DAG
from airflow.decorators import task

from dag_entities import ArtifactPaths, DEFAULT_CPU, DEFAULT_MEMORY, DatasetInfo, BIG_CPU, BIG_MEMORY, \
    big_executor_config, _get_models_params, DATASETS, SECOND_LEVELS_MODELS_CONFIGS
from dag_entities import EXTRA_BIG_CPU, EXTRA_BIG_MEMORY, SECOND_LEVELS_MODELS_PARAMS


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


def fit_predict_second_level_model(
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
    from dag_utils import do_fit_predict_second_level
    do_fit_predict_second_level(
        artifacts,
        model_name,
        k,
        train_path,
        first_level_predicts_path,
        second_model_type,
        second_model_params,
        second_model_config_path,
        cpu,
        memory
    )


def build_fit_predict_first_level_models_dag(
        dag_id: str,
        mlflow_exp_id: str,
        model_params_map: Dict[str, Dict[str, Any]],
        dataset: DatasetInfo,
        cpu: int = BIG_CPU,
        memory: int = BIG_MEMORY
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
                cpu=cpu,
                memory=memory
            )
            for model_class_name, model_kwargs in model_params_map.items()
        ]

        dataset_splitting(artifacts, partitions_num=100) \
            >> presplit_data(artifacts) \
            >> fit_feature_transformers(artifacts) \
            >> first_level_models

    return dag


def build_fit_predict_second_level(
        dag_id: str,
        mlflow_exp_id: str,
        model_name: str,
        dataset: DatasetInfo
) -> DAG:
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

        flvl_model_names = [
            "ALSWrap",
            "ClusterRec",
            "ItemKNN",
            "SLIM",
            "UCB"
        ]

        train_paths = [
            "partial_train_replay__models__als__ALSWrap_b07691ef15114b9687126ee64a3bc8cf.parquet",
            "partial_train_replay__models__cluster__ClusterRec_717251fdc4ff4d63b52badff93331bd2.parquet",
            "partial_train_replay__models__knn__ItemKNN_372668465d6a4537b43173379109a66c.parquet",
            "partial_train_replay__models__slim__SLIM_b77f553e2ff94556ad24d640f4b1dee3.parquet",
            "partial_train_replay__models__ucb__UCB_43a7c02276d84b0f828f89b96ded5241.parquet"
        ]

        predicts_paths = [
            "partial_predict_replay__models__als__ALSWrap_b07691ef15114b9687126ee64a3bc8cf.parquet",
            "partial_predict_replay__models__cluster__ClusterRec_717251fdc4ff4d63b52badff93331bd2.parquet",
            "partial_predict_replay__models__knn__ItemKNN_372668465d6a4537b43173379109a66c.parquet",
            "partial_predict_replay__models__slim__SLIM_b77f553e2ff94556ad24d640f4b1dee3.parquet",
            "partial_predict_replay__models__ucb__UCB_43a7c02276d84b0f828f89b96ded5241.parquet"
        ]

        train_paths = [os.path.join(artifacts.base_path, train_path) for train_path in train_paths]
        predicts_paths = [os.path.join(artifacts.base_path, predicts_path) for predicts_path in predicts_paths]

        second_level_models = [
            task(
                task_id=f"2lvl_{model_name.split('.')[-1]}_{flvl_model_name}",
                executor_config=big_executor_config
            )(fit_predict_second_level_model)(
                artifacts=artifacts,
                model_name=f"{model_name}_{flvl_model_name}",
                k=k,
                train_path=train_path,
                first_level_predicts_path=first_level_predicts_path,
                second_model_type=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_type"],
                second_model_params=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_params"],
                second_model_config_path=SECOND_LEVELS_MODELS_CONFIGS.get(model_name, None),
                cpu=EXTRA_BIG_CPU,
                memory=EXTRA_BIG_MEMORY
            )
            for flvl_model_name, train_path, first_level_predicts_path in
            zip(flvl_model_names, train_paths, predicts_paths)
        ]

    return dag


ml1m_first_level_dag = build_fit_predict_first_level_models_dag(
    dag_id="ml1m_first_level_dag",
    mlflow_exp_id="111",
    model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "cluster"),
    dataset=DATASETS["ml1m"]
)


ml25m_first_level_dag = build_fit_predict_first_level_models_dag(
    dag_id="ml25m_first_level_dag",
    mlflow_exp_id="111",
    model_params_map=_get_models_params("als", "itemknn", "ucb", "slim"),
    dataset=DATASETS["ml25m"],
    cpu=EXTRA_BIG_CPU,
    memory=EXTRA_BIG_MEMORY
)


ml1m_second_level_dag = build_fit_predict_second_level(
    dag_id="ml1m_second_level_dag",
    mlflow_exp_id="111",
    model_name="lama_default",
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
