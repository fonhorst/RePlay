import dataclasses
import os
import shutil
import uuid

import pytest

from conftest import phase_report_key
from experiments.two_stage_scenarios import dataset_splitting, first_level_fitting, ArtifactPaths, \
    second_level_fitting, init_refitable_two_stage_scenario, \
    combine_train_predicts_for_second_level


@pytest.fixture
def resources_path() -> str:
    return "/opt/data/"


@pytest.fixture(scope="function")
def artifacts(request) -> ArtifactPaths:
    path = "/opt/experiments/test_exp"
    resources_path = "/opt/data/"

    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

    yield ArtifactPaths(
        base_path=path,
        log_path=os.path.join(resources_path, "ml100k_ratings.csv"),
        user_features_path=os.path.join(resources_path, "ml100k_users.csv"),
        item_features_path=os.path.join(resources_path, "ml100k_items.csv")
    )

    report = request.node.stash[phase_report_key]
    if report["setup"].failed:
        print("setting up a test failed or skipped", request.node.nodeid)
    elif ("call" not in report) or report["call"].failed:
        print("executing test failed or skipped", request.node.nodeid)
    else:
        print("it is succeded")

    # shutil.rmtree(path, ignore_errors=True)


def test_data_splitting(artifacts: ArtifactPaths):
    dataset_splitting.function(
        artifacts,
        partitions_num=4
    )

    assert os.path.exists(artifacts.base_path)
    assert os.path.exists(artifacts.train_path)
    assert os.path.exists(artifacts.test_path)

    # TODO: they are not empty, no crossing by ids


def test_init_refitable_two_stage_scenario(artifacts: ArtifactPaths):
    init_refitable_two_stage_scenario.function(
        artifacts
    )

    assert os.path.exists(artifacts.two_stage_scenario_path)
    assert os.path.exists(artifacts.base_path, "first_level_train.parquet")
    assert os.path.exists(artifacts.base_path, "second_level_positive.parquet")
    assert os.path.exists(artifacts.base_path, "first_level_candidates.parquet")


def test_first_level_fitting(resources_path: str, user_features_path: str, item_features_path: str, artifacts: ArtifactPaths):
    # first model (dump)
    assert len(os.listdir(artifacts.base_path)) == 0

    shutil.copytree(os.path.join(resources_path, "train.parquet"), artifacts.train_path)
    shutil.copytree(os.path.join(resources_path, "test.parquet"), artifacts.test_path)

    # alternative
    # model_class_name = "replay.models.knn.ItemKNN"
    # model_kwargs = {"num_neighbours": 10}

    # alternative
    model_class_name = "replay.models.als.ALSWrap"
    model_kwargs={"rank": 10}

    first_level_fitting(artifacts, model_class_name, model_kwargs)

    assert os.path.exists(artifacts.model_path(model_class_name))
    assert os.path.exists(artifacts.partial_train_path(model_class_name))
    assert os.path.exists(artifacts.partial_predicts_path(model_class_name))

    # second model (use)
    next_artifacts = dataclasses.replace(artifacts, uid=str(uuid.uuid4()).replace('-', ''))
    next_model_class_name = "replay.models.als.ALSWrap"
    next_model_kwargs = {"rank": 10}

    first_level_fitting(artifacts, next_model_class_name, next_model_kwargs)

    assert os.path.exists(next_artifacts.model_path(next_model_class_name))
    assert os.path.exists(next_artifacts.partial_train_path(next_model_class_name))
    assert os.path.exists(next_artifacts.partial_predicts_path(next_model_class_name))

    # TODO: restore this checking later
    # spark = _get_spark_session()
    #
    # ptrain_1_df = spark.read.parquet(artifacts.partial_train_path(model_class_name))
    # ppreds_1_df = spark.read.parquet(artifacts.predictions_path(model_class_name))
    # ptrain_2_df = spark.read.parquet(new_artifacts.partial_train_path(next_model_class_name))
    # ppreds_2_df = spark.read.parquet(new_artifacts.predictions_path(next_model_class_name))
    #
    # assert ptrain_1_df.count() == ptrain_2_df.count()
    # assert ppreds_1_df.count() == ppreds_2_df.count()
    #
    # pt_uniques_1_df = ptrain_1_df.select("user_idx", "item_idx").distinct()
    # pt_uniques_2_df = ptrain_2_df.select("user_idx", "item_idx").distinct()
    #
    # assert ptrain_1_df.count() == pt_uniques_1_df.join(pt_uniques_2_df, on=["user_idx", "item_idx"]).count()
    #
    # spark.stop()
    # TODO: both datasets can be combined


def test_combine_datasets(artifacts: ArtifactPaths):
    # the test's preparation
    shutil.rmtree(artifacts.base_path, ignore_errors=True)
    shutil.copytree("/opt/data/test_exp_folder_combine", artifacts.base_path)

    combine_train_predicts_for_second_level.function(artifacts)

    assert os.path.exists(artifacts.full_second_level_train_path)
    assert os.path.exists(artifacts.full_second_level_predicts_path)


def test_second_level_fitting(user_features_path: str, artifacts: ArtifactPaths):
    shutil.rmtree(artifacts.base_path, ignore_errors=True)
    shutil.copytree("/opt/data/test_exp_folder_second_model", artifacts.base_path)

    model_name = "test_lama_model"

    second_level_fitting(
        artifacts=artifacts,
        model_name=model_name,
        k=10,
        second_model_type="lama",
        second_model_params={
            "general_params": {"use_algos": [["lgb"]]},
            # "lgb_params": {
            #     'default_params': {'numIteration': 10}
            # },
            "reader_params": {"cv": 3, "advanced_roles": False}
        },
        second_model_config_path=None
    )

    # TODO: restore this test later
    # assert os.path.exists(artifacts.second_level_model_path(model_name))
    assert os.path.exists(artifacts.second_level_predicts_path(model_name))
