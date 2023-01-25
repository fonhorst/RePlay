import os
import pickle
import uuid
from datetime import timedelta
from typing import Dict, Any, Optional, Union, List

import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from dag_entities import ArtifactPaths, DEFAULT_CPU, DEFAULT_MEMORY, DatasetInfo, big_executor_config, \
    _get_models_params, DATASETS, SECOND_LEVELS_MODELS_CONFIGS, TASK_CONFIG_FILENAME_ENV_VAR
from dag_entities import EXTRA_BIG_CPU, EXTRA_BIG_MEMORY, SECOND_LEVELS_MODELS_PARAMS
from dag_entities import extra_big_executor_config
from dag_entities import YARN_SUBMIT_CONF


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


def fit_predict_first_level_model_spark_submit(
        task_name: str,
        artifacts: ArtifactPaths,
        model_class_name: str,
        model_kwargs: Dict,
        k: int):
    config_filename = f"task_config_{task_name}_{uuid.uuid4()}.pickle"
    with open(config_filename, "wb") as f:
        pickle.dump({
            "artifacts": artifacts,
            "model_class_name": model_class_name,
            "model_kwargs": model_kwargs,
            "k": k
        }, f)

    submit_job = SparkSubmitOperator(
        application="/src/experiments/dag_utils.py",
        task_id=f"submit_{task_name}",
        files=config_filename,
        conf=YARN_SUBMIT_CONF,
        env_vars={
            TASK_CONFIG_FILENAME_ENV_VAR: config_filename
        },
        py_files='/src/replay_rec-0.10.0-py3-none-any.whl,/src/experiments/*',
        jars='/src/replay_2.12-0.1.jar'
    )

    return submit_job


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


def combine_1lvl_datasets(
        artifacts: ArtifactPaths,
        combined_train_path: str,
        combined_predicts_path: str,
        desired_models: Optional[List[str]] = None,
        mode: str = 'union'
):
    from dag_utils import DatasetCombiner
    DatasetCombiner.do_combine_datasets(
        artifacts=artifacts,
        combined_train_path=combined_train_path,
        combined_predicts_path=combined_predicts_path,
        desired_models=desired_models,
        mode=mode
    )


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
            # task(
            #     task_id=f"fit_predict_first_level_model_{model_class_name.split('.')[-1]}",
            #     executor_config=extra_big_executor_config
            # )(fit_predict_first_level_model)(
            #     artifacts=artifacts,
            #     model_class_name=model_class_name,
            #     model_kwargs=model_kwargs,
            #     k=k,
            #     cpu=EXTRA_BIG_CPU,
            #     memory=EXTRA_BIG_MEMORY
            # )

            fit_predict_first_level_model_spark_submit(
                task_name=f"fit_predict_first_level_model_{model_class_name.split('.')[-1]}",
                artifacts=artifacts,
                model_class_name=model_class_name,
                model_kwargs=model_kwargs,
                k=k,
            )
            for model_class_name, model_kwargs in model_params_map.items()
        ]

        dataset_splitting(artifacts, partitions_num=100) \
            >> presplit_data(artifacts) \
            >> fit_feature_transformers(artifacts) \
            >> first_level_models

    return dag


def _make_combined_2lvl(artifacts: ArtifactPaths,
                        model_name: str,
                        combiner_suffix: str,
                        k: int,
                        mode: str = 'union',
                        desired_1lvl_models: Optional[List[str]] = None):
    combined_train_path = artifacts.make_path(f"combined_train_{combiner_suffix}.parquet")
    combined_predicts_path = artifacts.make_path(f"combined_predicts_{combiner_suffix}.parquet")

    combiner = task(
        task_id=f"combiner_{combiner_suffix}"
    )(combine_1lvl_datasets)(
        artifacts=artifacts,
        combined_train_path=combined_train_path,
        combined_predicts_path=combined_predicts_path,
        desired_models=desired_1lvl_models,
        mode=mode
    )

    second_level_model = task(
        task_id=f"2lvl_{model_name.split('.')[-1]}_{combiner_suffix}",
        executor_config=extra_big_executor_config
    )(fit_predict_second_level_model)(
        artifacts=artifacts,
        model_name=f"{model_name}_{combiner_suffix}",
        k=k,
        train_path=combined_train_path,
        first_level_predicts_path=combined_predicts_path,
        second_model_type=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_type"],
        second_model_params=SECOND_LEVELS_MODELS_PARAMS[model_name]["second_model_params"],
        second_model_config_path=SECOND_LEVELS_MODELS_CONFIGS.get(model_name, None),
        cpu=EXTRA_BIG_CPU,
        memory=EXTRA_BIG_MEMORY
    )

    return combiner >> second_level_model


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

        [
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


def build_combiner_second_level(dag_id: str, mlflow_exp_id: str, dataset: DatasetInfo):
    os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
    os.environ["MLFLOW_EXPERIMENT_ID"] = os.environ.get("MLFLOW_EXPERIMENT_ID", mlflow_exp_id)

    path_suffix = 'default'
    artifacts = ArtifactPaths(
        base_path=f"/opt/spark_data/replay/experiments/{dataset.name}_first_level_{path_suffix}",
        dataset=dataset
    )
    longer_model_name = 'longer_lama_default'
    slama_longer_model_name = 'longer_slama_default'
    k = 100

    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
            tags=['two_stage', 'replay', 'first level']
    ) as dag:
        _make_combined_2lvl(
            artifacts=artifacts,
            model_name=slama_longer_model_name,
            combiner_suffix="all_models_union",
            k=k,
            desired_1lvl_models=['itemknn', 'alswrap', 'slim', 'ucb'],
            mode='union'
        )

        _make_combined_2lvl(
            artifacts=artifacts,
            model_name=longer_model_name,
            combiner_suffix="all_models_leading_itemknn",
            desired_1lvl_models=['itemknn', 'alswrap', 'slim', 'ucb'],
            k=k,
            mode='leading_itemknn'
        )

        _make_combined_2lvl(
            artifacts=artifacts,
            model_name=longer_model_name,
            combiner_suffix="itemknn_slim",
            k=k,
            mode='union',
            desired_1lvl_models=['itemknn', 'slim']
        )

        _make_combined_2lvl(
            artifacts=artifacts,
            model_name=longer_model_name,
            combiner_suffix="itemknn_alswrap",
            k=k,
            mode='union',
            desired_1lvl_models=['itemknn', 'alswrap']
        )

        _make_combined_2lvl(
            artifacts=artifacts,
            model_name=longer_model_name,
            combiner_suffix="alswrap_slim",
            k=k,
            mode='union',
            desired_1lvl_models=['alswrap', 'slim']
        )

    return dag


# ml1m_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="ml1m_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "cluster"),
#     dataset=DATASETS["ml1m"]
# )
#
# ml10m_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="ml10m_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("ucb", "slim", "als", "itemknn"),
#     dataset=DATASETS["ml10m"]
# )
#
# ml20m_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="ml20m_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("ucb", "slim"),
#     dataset=DATASETS["ml20m"]
# )

# DAG SUBMIT series

ml1m_first_level_dag_submit = build_fit_predict_first_level_models_dag(
    dag_id="ml1m_first_level_dag_submit",
    mlflow_exp_id="111",
    model_params_map=_get_models_params("als"),
    dataset=DATASETS["ml1m"]
)

ml25m_first_level_dag_submit = build_fit_predict_first_level_models_dag(
    dag_id="ml25m_first_level_dag_submit",
    mlflow_exp_id="111",
    model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "word2vec"), #, "itemknn", "ucb", "slim"),
    dataset=DATASETS["ml25m"]
)

msd_first_level_dag_submit = build_fit_predict_first_level_models_dag(
    dag_id="msd_first_level_dag_submit",
    mlflow_exp_id="111",
    model_params_map=_get_models_params("als"), #, "itemknn", "ucb", "slim"),
    dataset=DATASETS["msd"]
)


# netflix_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="netflix_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("ucb", "slim"),
#     dataset=DATASETS["netflix"]
# )
#
# netflix_small_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="netflix_small_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("ucb", "slim", "als", "itemknn"),
#     dataset=DATASETS["netflix_small"]
# )
#
#
# msd_small_first_level_dag = build_fit_predict_first_level_models_dag(
#     dag_id="msd_small_first_level_dag",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("ucb", "slim", "als", "itemknn"),
#     dataset=DATASETS["msd_small"]
# )
#
#
# msd_small_combined_second_level_dag = build_combiner_second_level(
#     dag_id="msd_small_combined_second_level_dag",
#     mlflow_exp_id="111",
#     dataset=DATASETS["msd_small"]
# )
#
#
# msd_small_second_level_dag = build_fit_predict_second_level(
#     dag_id="msd_small_second_level_dag",
#     mlflow_exp_id="111",
#     model_name="lama_default",
#     dataset=DATASETS["msd_small"]
# )
#
#
# ml1m_second_level_dag = build_fit_predict_second_level(
#     dag_id="ml1m_second_level_dag",
#     mlflow_exp_id="111",
#     model_name="lama_default",
#     dataset=DATASETS["ml1m"]
# )
#
#
# ml1m_combined_second_level_dag = build_combiner_second_level(
#     dag_id="ml1m_combined_second_level_dag",
#     mlflow_exp_id="111",
#     dataset=DATASETS["ml1m"]
# )
