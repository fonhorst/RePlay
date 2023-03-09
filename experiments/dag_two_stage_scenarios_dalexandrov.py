import sys

sys.path.insert(0, "/opt/airflow/dags/dalexandrov_packages")

import os
import pickle
import uuid
from datetime import timedelta
from typing import Dict, Any, Optional, Union, List

import pendulum
from airflow import DAG
from airflow.decorators import task
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from dag_entities_dalexandrov import ArtifactPaths, DEFAULT_CPU, DEFAULT_MEMORY, DatasetInfo, big_executor_config, \
    _get_models_params, DATASETS, SECOND_LEVELS_MODELS_CONFIGS, TASK_CONFIG_FILENAME_ENV_VAR
from dag_entities_dalexandrov import EXTRA_BIG_CPU, EXTRA_BIG_MEMORY, SECOND_LEVELS_MODELS_PARAMS, FIRST_LEVELS_MODELS_PARAMS, MODELNAME2FULLNAME
from dag_entities_dalexandrov import extra_big_executor_config
from dag_entities_dalexandrov import YARN_SUBMIT_CONF


@task
def dataset_splitting(artifacts: ArtifactPaths, partitions_num: int):
    from dag_utils_dalexandrov import do_dataset_splitting
    do_dataset_splitting(artifacts, partitions_num)

@task
def collect_params(artifacts: ArtifactPaths, *model_class_names: str):
    from dag_utils_dalexandrov import do_collect_best_params
    do_collect_best_params(artifacts, *model_class_names)

@task
def fit_feature_transformers(artifacts: 'ArtifactPaths', cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    from dag_utils_dalexandrov import do_fit_feature_transformers
    do_fit_feature_transformers(artifacts, cpu, memory)


@task
def presplit_data(artifacts: ArtifactPaths, item_test_size: float,
                  cpu: int = DEFAULT_CPU, memory: int = DEFAULT_MEMORY):
    from dag_utils_dalexandrov import do_presplit_data
    do_presplit_data(artifacts, item_test_size, cpu, memory)


def fit_predict_first_level_model_spark_submit(
        task_name: str,
        artifacts: ArtifactPaths,
        model_class_name: str,
        # model_kwargs: Dict,
        k: int,
        get_optimized_params: bool = False,
        do_optimization: bool = False,):
    config_filename = f"task_config_{task_name}_{uuid.uuid4()}.pickle"
    with open(config_filename, "wb") as f:
        pickle.dump({
            "artifacts": artifacts,
            "model_class_name": model_class_name,
            # "model_kwargs": model_kwargs,
            "k": k,
            "get_optimized_params": get_optimized_params,
            "do_optimization": do_optimization,
        }, f)

    submit_job = SparkSubmitOperator(
        application="/opt/spark_data/spark_submit_files/dag_utils_dalexandrov.py",
        task_id=f"submit_{task_name}",
        files=config_filename,
        conf=YARN_SUBMIT_CONF,
        env_vars={
            TASK_CONFIG_FILENAME_ENV_VAR: config_filename
        },
        py_files='/opt/spark_data/spark_submit_files/replay_rec-0.10.0-py3-none-any_dalexandrov.whl,'
                 '/opt/spark_data/spark_submit_files/dag_entities_dalexandrov.py,'
                 '/opt/spark_data/spark_submit_files/sparklightautoml_dev-0.3.2-py3-none-any_dalexandrov.whl',
        jars='/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar,'
             '/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar'
    )

    return submit_job


def build_fit_predict_first_level_models_dag(
        dag_id: str,
        mlflow_exp_id: str,
        # model_params_map: Dict[str, Dict[str, Any]],
        models: list[str],
        dataset: DatasetInfo,
        path_suffix: str = 'default',
        item_test_size: float = 0.5,
        get_optimized_params: bool = False,
        do_optimization: bool = False,
        k: int = 100
):
    with DAG(
            dag_id=dag_id,
            schedule=timedelta(days=10086),
            start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
            catchup=False,
            tags=['two_stage', 'replay', 'first level']
    ) as dag:
        os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8822"
        os.environ["MLFLOW_EXPERIMENT_ID"] = os.environ.get("MLFLOW_EXPERIMENT_ID", mlflow_exp_id)

        # TODO: fix it later
        # path_suffix = Variable.get(f'{dataset.name}_artefacts_dir_suffix', 'default')
        # path_suffix = 'default'
        artifacts = ArtifactPaths(
            base_path=f"/opt/spark_data/replay/experiments/{dataset.name}_first_level_{path_suffix}",
            dataset=dataset
        )


        # if get_optimized_params:
        #     PARAMS =None
        #
        # else:
        #     model_params_map = _get_models_params(*models, FIRST_LEVELS_MODELS_PARAMS)

        # first_level_models = [
        #
        #     fit_predict_first_level_model_spark_submit(
        #         task_name=f"fit_predict_first_level_model_{model_class_name.split('.')[-1]}",
        #         artifacts=artifacts,
        #         model_class_name=model_class_name,
        #         model_kwargs=model_kwargs,
        #         k=k,
        #         do_optimization=do_optimization
        #     )
        #     for model_class_name, model_kwargs in model_params_map.items()
        # ]

        first_level_models = [

            fit_predict_first_level_model_spark_submit(
                task_name=f"fit_predict_first_level_model_{MODELNAME2FULLNAME[model_name].split('.')[-1]}",
                artifacts=artifacts,
                model_class_name=MODELNAME2FULLNAME[model_name],
                k=k,
                get_optimized_params=get_optimized_params,
                do_optimization=do_optimization
            )
            for model_name in models
        ]


        dataset_splitting(artifacts, partitions_num=100) \
        >> presplit_data(artifacts, item_test_size=item_test_size) \
        >> first_level_models
    #  >> fit_feature_transformers(artifacts) \

    return dag


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
        application="/opt/spark_data/spark_submit_files/dag_utils_dalexandrov.py",
        task_id=f"submit_{task_name}",
        files=f"{config_filename},/opt/spark_data/spark_submit_files/tabular_config.yml",
        conf=YARN_SUBMIT_CONF,
        env_vars={
            TASK_CONFIG_FILENAME_ENV_VAR: config_filename
        },
        py_files='/opt/spark_data/spark_submit_files/replay_rec-0.10.0-py3-none-any_dalexandrov.whl,'
                 '/opt/spark_data/spark_submit_files/dag_entities_dalexandrov.py,'
                 '/opt/spark_data/spark_submit_files/sparklightautoml_dev-0.3.2-py3-none-any_dalexandrov.whl',
        jars='/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar,'
             '/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar,'
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
        application="/opt/spark_data/spark_submit_files/dag_utils_dalexandrov.py",
        task_id=f"submit_{task_name}",
        files=f"{config_filename},/opt/spark_data/spark_submit_files/tabular_config.yml",
        conf=YARN_SUBMIT_CONF,
        env_vars={
            TASK_CONFIG_FILENAME_ENV_VAR: config_filename
        },
        py_files='/opt/spark_data/spark_submit_files/replay_rec-0.10.0-py3-none-any_dalexandrov.whl,'
                 '/opt/spark_data/spark_submit_files/dag_entities_dalexandrov.py,'
                 '/opt/spark_data/spark_submit_files/sparklightautoml_dev-0.3.2-py3-none-any_dalexandrov.whl',
        jars='/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar,'
             '/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar,'
             '/opt/spark_data/spark_submit_files/synapseml_2.12-0.9.5.jar'
    )

    return submit_job


def fit_predict_second_level_model_pure_lgbm_spark_submit(
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
        application="/opt/spark_data/spark_submit_files/dag_utils_dalexandrov.py",
        task_id=f"submit_{task_name}",
        files=f"{config_filename},/opt/spark_data/spark_submit_files/tabular_config.yml",
        conf=YARN_SUBMIT_CONF,
        env_vars={
            TASK_CONFIG_FILENAME_ENV_VAR: config_filename
        },
        py_files='/opt/spark_data/spark_submit_files/replay_rec-0.10.0-py3-none-any_dalexandrov.whl,'
                 '/opt/spark_data/spark_submit_files/dag_entities_dalexandrov.py,'
                 '/opt/spark_data/spark_submit_files/sparklightautoml_dev-0.3.2-py3-none-any_dalexandrov.whl',
        jars='/opt/spark_data/spark_submit_files/replay_2.12-0.1_againetdinov.jar,'
             '/opt/spark_data/spark_submit_files/spark-lightautoml_2.12-0.1.1.jar,'
             '/opt/spark_data/spark_submit_files/synapseml_2.12-0.9.5.jar'
    )

    return submit_job

def _make_combined_2lvl_pure_lgbm(artifacts: ArtifactPaths,
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

    second_level_model = fit_predict_second_level_model_pure_lgbm_spark_submit(
        task_name=f"pure_2lvl_{model_name.split('.')[-1]}_{combiner_suffix}",
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


def build_combiner_second_level_pure(dag_id: str, mlflow_exp_id: str, dataset: DatasetInfo, path_suffix: str = "default",
                                model_name: str = 'slama_fast'):
    os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
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
            tags=['two_stage', 'replay', 'first level']
    ) as dag:
        _make_combined_2lvl_pure_lgbm(
            artifacts=artifacts,
            model_name=model_name,
            combiner_suffix="4models",  # an important part
            k=k,
            mode='union',
            desired_1lvl_models=["alswrap", "itemknn", "slim", "word2vecrec"]  # ['alswrap', 'slim']
        )

    return dag

def build_combiner_second_level(dag_id: str, mlflow_exp_id: str, dataset: DatasetInfo, path_suffix: str = "default",
                                model_name: str = 'slama_fast'):
    os.environ["MLFLOW_TRACKING_URI"] = "http://node2.bdcl:8811"
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
            tags=['two_stage', 'replay', 'first level']
    ) as dag:
        _make_combined_2lvl(
            artifacts=artifacts,
            model_name=model_name,
            combiner_suffix="4models",  # an important part
            k=k,
            mode='union',
            desired_1lvl_models=["alswrap", "itemknn", "slim", "word2vecrec"]  # ['alswrap', 'slim']
        )

    return dag


# DAG SUBMIT series

# fair first lvl DAGS

# netflix_first_level_dag_submit_fair = build_fit_predict_first_level_models_dag(
#     dag_id="netflix_first_level_dag_submit_fair",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "slim", "word2vec"),
#     dataset=DATASETS["netflix"],
#     item_test_size=0.0,  # for second model
#     path_suffix="fair"
# )
#
# # first lvl DAGS

# ml1m_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
#     dag_id="ml1m_first_level_dag_submit_80_20",
#     mlflow_exp_id="222",
#     models=["als", "itemknn", "slim", "word2vec"],
#     dataset=DATASETS["ml1m"],
#     item_test_size=0.2,
#     path_suffix="80_20",
#     get_optimized_params=False,
#     do_optimization=False
# )

#
# ml25m_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
#     dag_id="ml25m_first_level_dag_submit_80_20",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "slim", "word2vec"),
#     dataset=DATASETS["ml25m"],
#     item_test_size=0.2,
#     path_suffix="80_20_2"
# )
#
msd_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
    dag_id="msd_first_level_dag_submit_80_20",
    mlflow_exp_id="222",
    models=["als", "slim", "word2vec", "itemknn"],
    dataset=DATASETS["msd"],
    item_test_size=0.2,
    path_suffix="80_20",
    get_optimized_params=False,
    do_optimization=False
)

msd_first_level_dag_submit_80_fair = build_fit_predict_first_level_models_dag(
    dag_id="msd_first_level_dag_submit_fair",
    mlflow_exp_id="222",
    models=["als", "slim", "word2vec", "itemknn"],
    dataset=DATASETS["msd"],
    item_test_size=0.0,
    path_suffix="fair",
    get_optimized_params=False,
    do_optimization=False
)

# ml25m_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
#     dag_id="ml25m_first_level_dag_submit_80",
#     mlflow_exp_id="222",
#     models=["als", "slim", "word2vec", "itemknn"],
#     dataset=DATASETS["ml25m"],
#     item_test_size=0.2,
#     path_suffix="80_20",
#     get_optimized_params=False,
#     do_optimization=False
# )
#
# netflix_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
#     dag_id="netflix_first_level_dag_submit_80_20",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "slim", "word2vec"),
#     dataset=DATASETS["netflix"],
#     item_test_size=0.2,  # for second model
#     path_suffix="80_20"
# )
#
# # first lvl DAGS with optimization
#
# ml1m_first_level_dag_submit_80_opt = build_fit_predict_first_level_models_dag(
#     dag_id="ml1m_first_level_dag_submit_80_20_opt",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "slim", "word2vec", optimized=False),
#     dataset=DATASETS["ml1m"],
#     item_test_size=0.2,
#     path_suffix="80_20",
#     optimize=True
# )

ml25m_first_level_dag_submit_80_opt = build_fit_predict_first_level_models_dag(
    dag_id="ml25m_first_level_dag_submit_80_opt",
    mlflow_exp_id="222",
    models=["als", "slim", "word2vec"],
    dataset=DATASETS["ml25m"],
    item_test_size=0.2,
    path_suffix="80_20",
    get_optimized_params=False,
    do_optimization=True
)

ml25m_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
    dag_id="ml25m_first_level_dag_submit_80",
    mlflow_exp_id="222",
    models=["als", "slim", "word2vec", "itemknn"],
    dataset=DATASETS["ml25m"],
    item_test_size=0.2,
    path_suffix="80_20",
    get_optimized_params=False,
    do_optimization=False
)



# ml1m_first_level_dag_submit_80_train_opt = build_fit_predict_first_level_models_dag(
#     dag_id="ml1m_first_level_dag_submit_80_20_train_opt",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "slim", "word2vec",
#                                         base_path=f"/opt/spark_data/replay/experiments/ml1m_first_level_80_20",
#                                         optimized=True,
#                                         ),
#     dataset=DATASETS["ml1m"],
#     item_test_size=0.2,
#     path_suffix="80_20",
#     optimize=False
# )

ml1m_first_level_dag_submit_80_train_opt = build_fit_predict_first_level_models_dag(
    dag_id="ml1m_first_level_dag_submit_80_20_train_opt",
    mlflow_exp_id="222",
    models=["als", "slim", "word2vec"],
    dataset=DATASETS["ml1m"],
    item_test_size=0.2,
    path_suffix="80_20",
    get_optimized_params=True,  # TODO: get default parameter if optimized not found
    do_optimization=False
)

ml25m_first_level_dag_submit_80_train_opt = build_fit_predict_first_level_models_dag(
    dag_id="ml25m_first_level_dag_submit_80_20_train_opt",
    mlflow_exp_id="222",
    models=["als", "slim", "word2vec"],
    dataset=DATASETS["ml25m"],
    item_test_size=0.2,
    path_suffix="80_20",
    get_optimized_params=True,  # TODO: get default parameter if optimized not found
    do_optimization=False
)

# second lvl DAGS
ml1m_combined_second_level_dag_80 = build_combiner_second_level(
    dag_id="ml1m_combined_second_level_dag_4models_80_20",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["ml1m"],
    path_suffix="80_20",
    model_name="slama_fast"
)


ml25m_combined_second_level_dag_80 = build_combiner_second_level(
    dag_id="ml25m_combined_second_level_dag_4models_80_20",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["ml25m"],
    path_suffix="80_20",
    model_name="longer_slama_default"

)

msd_combined_second_level_dag_80 = build_combiner_second_level(
    dag_id="msd_combined_second_level_dag_4models_80_20",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["msd"],
    path_suffix="80_20",
    model_name="longer_slama_default"
)

msd_combined_second_level_dag_80_wo_tuning = build_combiner_second_level(
    dag_id="msd_combined_second_level_dag_4models_80_20_wo_tuning",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["msd"],
    path_suffix="80_20",
    model_name="longer_slama_wo_tuning"
)


netflix_combined_second_level_dag_80 = build_combiner_second_level(
    dag_id="netflix_combined_second_level_dag_4models_80_20_fi",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["netflix"],
    path_suffix="80_20",
    model_name="longer_slama_default"
)

msd_combined_second_level_dag_80_pure = build_combiner_second_level_pure(
    dag_id="msd_combined_second_level_dag_4models_80_20_pure",
    mlflow_exp_id="two_stage_cluster",
    dataset=DATASETS["msd"],
    path_suffix="80_20",
    model_name="slama_fast"
)
#
# netflix_combined_second_level_dag_80 = build_combiner_second_level(
#     dag_id="netflix_combined_second_level_dag_4models_80_20",
#     mlflow_exp_id="two_stage_cluster",
#     dataset=DATASETS["netflix"],
#     path_suffix="80_20"
# )





# ===================================================================================


# ml1m_first_level_dag_submit = build_fit_predict_first_level_models_dag(
#     dag_id="ml1m_first_level_dag_submit_fair",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "word2vec", "cluster"),
#     dataset=DATASETS["ml1m"]
# )
#
# ml25m_first_level_dag_submit = build_fit_predict_first_level_models_dag(
#     dag_id="ml25m_first_level_dag_submit_fair",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "slim", "word2vec"),
#     dataset=DATASETS["ml25m"],
#     path_suffix="100_0"
# )
#
# msd_first_level_dag_submit = build_fit_predict_first_level_models_dag(
#     dag_id="msd_first_level_dag_submit_fair",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "word2vec", "cluster"),
#     dataset=DATASETS["msd"]
# )
#
# ml1m_first_level_dag_submit_50 = build_fit_predict_first_level_models_dag(
#     dag_id="ml1m_first_level_dag_submit_50_50",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "slim", ),
#     dataset=DATASETS["ml1m"],
#     path_suffix="50_50"
# )
#
# ml1m_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
#     dag_id="ml1m_first_level_dag_submit_80_20",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "slim", "word2vec"),
#     dataset=DATASETS["ml1m"],
#     path_suffix="80_20"
# )
#

#
# ml1m_combined_second_level_dag = build_combiner_second_level(
#     dag_id="ml1m_combined_second_level_dag_als_slim",
#     mlflow_exp_id="222",
#     dataset=DATASETS["ml1m"]
# )
#
# ml1m_combined_second_level_dag_50 = build_combiner_second_level(
#     dag_id="ml1m_combined_second_level_dag_als_slim_50_50",
#     mlflow_exp_id="two_stage_cluster",
#     dataset=DATASETS["ml1m"],
#     path_suffix="50_50"
# )
#
# ml1m_combined_second_level_dag_80 = build_combiner_second_level(
#     dag_id="ml1m_combined_second_level_dag_4models_80_20",
#     mlflow_exp_id="two_stage_cluster",
#     dataset=DATASETS["ml1m"],
#     path_suffix="80_20"
# )
#
# # 10m
# ml10m_first_level_dag_submit_80 = build_fit_predict_first_level_models_dag(
#     dag_id="ml10m_first_level_dag_submit_80_20",
#     mlflow_exp_id="222",
#     model_params_map=_get_models_params("als", "itemknn", "slim", "word2vec"),
#     dataset=DATASETS["ml10m"],
#     path_suffix="80_20"
# )
#
# ml10m_combined_second_level_dag_80 = build_combiner_second_level(
#     dag_id="ml10m_combined_second_level_dag_4models_80_20",
#     mlflow_exp_id="two_stage_cluster",
#     dataset=DATASETS["ml10m"],
#     path_suffix="80_20"
# )
# ml25m_first_level_dag_submit = build_fit_predict_first_level_models_dag(
#     dag_id="ml25m_first_level_dag_submit",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("als", "itemknn", "ucb", "slim", "word2vec"), #, "itemknn", "ucb", "slim"),
#     dataset=DATASETS["ml25m"]
# )
#
# msd_first_level_dag_submit = build_fit_predict_first_level_models_dag(
#     dag_id="msd_first_level_dag_submit",
#     mlflow_exp_id="111",
#     model_params_map=_get_models_params("als"), #, "itemknn", "ucb", "slim"),
#     dataset=DATASETS["msd"]
# )


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
