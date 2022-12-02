import dataclasses
import os
import shutil
import uuid
from typing import cast

import pytest

from conftest import phase_report_key
from experiments.two_stage_scenarios import dataset_splitting, first_level_fitting, ArtifactPaths, \
    second_level_fitting, init_refitable_two_stage_scenario, \
    combine_train_predicts_for_second_level, RefitableTwoStageScenario, _init_spark_session
from replay.history_based_fp import LogStatFeaturesProcessor
from replay.model_handler import load


@pytest.fixture
def resource_path() -> str:
    return "/opt/data/resources"


@pytest.fixture(scope='function')
def ctx(request, resource_path: str, artifacts: ArtifactPaths):
    # copy the context of dependencies
    context_name = request.param
    required_resource_path = os.path.join(resource_path, context_name)
    assert os.path.exists(required_resource_path)
    shutil.rmtree(artifacts.base_path, ignore_errors=True)
    shutil.copytree(required_resource_path, artifacts.base_path)


@pytest.fixture(scope="function")
def artifacts(request, resource_path: str) -> ArtifactPaths:
    path = "/opt/experiments/test_exp"
    data_base_path = "/opt/data/"

    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

    yield ArtifactPaths(
        base_path=path,
        log_path=os.path.join(data_base_path, "ml100k_ratings.csv"),
        user_features_path=os.path.join(data_base_path, "ml100k_users.csv"),
        item_features_path=os.path.join(data_base_path, "ml100k_items.csv")
    )

    report = request.node.stash[phase_report_key]
    if report["setup"].failed:
        print("setting up a test failed or skipped", request.node.nodeid)
    elif ("call" not in report) or report["call"].failed:
        print("executing test failed or skipped", request.node.nodeid)
    else:
        test_name = request.node.name
        # process cases like this "test_init_refitable_two_stage_scenario[test_data_splitting__out]"
        test_name = test_name[:test_name.find('[')] if test_name.endswith(']') else test_name
        resource_test_path = os.path.join(resource_path, f"{test_name}__out")
        if os.path.exists(resource_test_path):
            shutil.rmtree(resource_test_path)

        os.makedirs(resource_path, exist_ok=True)
        shutil.copytree(path, resource_test_path)
        print(f"Copied the exp dir state ({path}) to the resource folder "
              f"of this test ({resource_test_path}) for next tests")

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


@pytest.mark.parametrize('ctx', ['test_data_splitting__out'], indirect=True)
def test_init_refitable_two_stage_scenario(artifacts: ArtifactPaths, resource_path: str, ctx):
    init_refitable_two_stage_scenario.function(
        artifacts
    )

    assert os.path.exists(artifacts.two_stage_scenario_path)
    assert os.path.exists(os.path.join(artifacts.base_path, "first_level_train.parquet"))
    assert os.path.exists(os.path.join(artifacts.base_path, "second_level_positive.parquet"))
    assert os.path.exists(os.path.join(artifacts.base_path, "first_level_candidates.parquet"))

    scenario = cast(RefitableTwoStageScenario, load(artifacts.two_stage_scenario_path))
    assert scenario._are_candidates_dumped
    assert scenario._are_split_data_dumped
    assert scenario.features_processor.fitted


@pytest.mark.parametrize('ctx', ['test_init_refitable_two_stage_scenario__out'], indirect=True)
def test_first_level_fitting(artifacts: ArtifactPaths, ctx):
    # alternative
    model_class_name = "replay.models.knn.ItemKNN"
    model_kwargs = {"num_neighbours": 10}

    # alternative
    # model_class_name = "replay.models.als.ALSWrap"
    # model_kwargs={"rank": 10}

    first_level_fitting(artifacts, model_class_name, model_kwargs, k=10)

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


@pytest.mark.parametrize('ctx', ['test_first_level_fitting__out'], indirect=True)
def test_combine_datasets(artifacts: ArtifactPaths, ctx):
    # the test's preparation
    shutil.rmtree(artifacts.base_path, ignore_errors=True)
    shutil.copytree("/opt/data/test_exp_folder_combine", artifacts.base_path)

    combine_train_predicts_for_second_level.function(artifacts)

    assert os.path.exists(artifacts.full_second_level_train_path)
    assert os.path.exists(artifacts.full_second_level_predicts_path)


@pytest.mark.parametrize('ctx', ['test_combine_datasets__out'], indirect=True)
def test_second_level_fitting(artifacts: ArtifactPaths, ctx):
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


@pytest.mark.parametrize('ctx', ['test_data_splitting__out'], indirect=True)
def test_log_stat_features_processor_save_load(artifacts: ArtifactPaths, ctx):
    with _init_spark_session():
        processor = LogStatFeaturesProcessor()
        processor.fit(artifacts.train)

        result = processor.transform(artifacts.test)
        assert result.count() == artifacts.test.count()

        path = os.path.join(artifacts.base_path, "log_stat_processor")

        processor.save(path)
        loaded_processor = LogStatFeaturesProcessor.load(path)

        new_result = loaded_processor.transform(artifacts.test)
        assert sorted(result.columns) == sorted(new_result.columns)