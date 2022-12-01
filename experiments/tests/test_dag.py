import os
import shutil

import dataclasses
import uuid

import pytest

from experiments.two_stage_scenarios import dataset_splitting, first_level_fitting, ArtifactPaths


@pytest.fixture
def resources_path() -> str:
    return "/opt/data/"


@pytest.fixture
def log_path(resources_path: str) -> str:
    return os.path.join(resources_path, "ml100k_ratings.csv")


@pytest.fixture
def user_features_path(resources_path: str) -> str:
    return os.path.join(resources_path, "ml100k_users.csv")


@pytest.fixture
def item_features_path(resources_path: str) -> str:
    return os.path.join(resources_path, "ml100k_items.csv")


@pytest.fixture(scope="function")
def artifacts() -> ArtifactPaths:
    path = "/opt/experiments/test_exp"

    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

    yield ArtifactPaths(base_path=path)

    shutil.rmtree(path, ignore_errors=True)


def test_data_splitting(log_path: str, artifacts: ArtifactPaths):

    # assert not os.path.exists(artifacts.base_path)

    dataset_splitting.function(
        log_path=log_path,
        base_path=artifacts.base_path,
        train_path=artifacts.train_path,
        test_path=artifacts.test_path,
        cores=1
    )

    assert os.path.exists(artifacts.base_path)
    assert os.path.exists(artifacts.train_path)
    assert os.path.exists(artifacts.test_path)

    # TODO: they are not empty, no crossing by ids


def test_first_level_fitting(resources_path: str, user_features_path: str, item_features_path: str, artifacts: ArtifactPaths):
    # first model (dump)
    assert len(os.listdir(artifacts.base_path)) == 0

    shutil.copytree(os.path.join(resources_path, "train.parquet"), artifacts.train_path)
    shutil.copytree(os.path.join(resources_path, "test.parquet"), artifacts.test_path)

    model_class_name = "replay.models.knn.ItemKNN"

    # alternative
    # model_class_name = "replay.models.als.ALSWrap"
    # model_kwargs={"rank": 10}

    first_level_fitting(
        train_path=artifacts.train_path,
        test_path=artifacts.test_path,
        model_class_name=model_class_name,
        model_kwargs={"num_neighbours": 10},
        model_path=artifacts.model_path(model_class_name),
        second_level_partial_train_path=artifacts.partial_train_path(model_class_name),
        first_level_model_predictions_path=artifacts.predictions_path(model_class_name),
        k=10,
        intermediate_datasets_mode="dump",
        predefined_train_and_positives_path=(
            artifacts.first_level_train_path,
            artifacts.second_level_positives_path
        ),
        predefined_negatives_path=artifacts.negatives_path,
        item_features_path=item_features_path,
        user_features_path=user_features_path
    )

    assert os.path.exists(artifacts.model_path(model_class_name))
    assert os.path.exists(artifacts.partial_train_path(model_class_name))
    assert os.path.exists(artifacts.predictions_path(model_class_name))
    assert os.path.exists(artifacts.first_level_train_path)
    assert os.path.exists(artifacts.second_level_positives_path)
    assert os.path.exists(artifacts.negatives_path)

    # second model (use)
    # TODO: model exists
    # TODO: dataset exists

    artifacts = dataclasses.replace(artifacts, uid=str(uuid.uuid4()).replace('-', ''))
    next_model_class_name = "replay.models.knn.ItemKNN"

    first_level_fitting(
        train_path=artifacts.train_path,
        test_path=artifacts.test_path,
        model_class_name=next_model_class_name,
        model_kwargs={"num_neighbours": 10},
        model_path=artifacts.model_path(next_model_class_name),
        second_level_partial_train_path=artifacts.partial_train_path(next_model_class_name),
        first_level_model_predictions_path=artifacts.predictions_path(next_model_class_name),
        k=10,
        intermediate_datasets_mode="use",
        predefined_train_and_positives_path=(
            artifacts.first_level_train_path,
            artifacts.second_level_positives_path
        ),
        predefined_negatives_path=artifacts.negatives_path,
        item_features_path=item_features_path,
        user_features_path=user_features_path
    )

    assert os.path.exists(artifacts.model_path(next_model_class_name))
    assert os.path.exists(artifacts.partial_train_path(next_model_class_name))
    assert os.path.exists(artifacts.predictions_path(next_model_class_name))

    # TODO: both datasets can be combined

#
# def test_combine_datasets():
#     # TODO: dataset exists
#     # TODO: no Nones
#     pass


# def second_level_fitting():
#     # first run
#     # TODO: model exists
#     # TODO: dataset exists
#
#     # second run
#     # TODO: model exists
#     # TODO: dataset exists
#
#     # TODO: both datasets can be combined
#     pass