import os
import shutil

import pytest

from experiments.two_stage_scenarios import dataset_splitting, first_level_fitting, ArtifactPaths


@pytest.fixture(scope="function")
def artifacts() -> ArtifactPaths:
    path = "/opt/experiments/test_exp"

    shutil.rmtree(path, ignore_errors=True)

    yield ArtifactPaths(base_path=path)

    shutil.rmtree(path, ignore_errors=True)


def test_data_splitting(artifacts: ArtifactPaths):

    assert not os.path.exists(artifacts.base_path)

    dataset_splitting.function(
        log_path="/opt/data/ml100k_ratings.csv",
        base_path=artifacts.base_path,
        train_path=artifacts.train_path,
        test_path=artifacts.test_path,
        cores=1
    )

    assert os.path.exists(artifacts.base_path)
    assert os.path.exists(artifacts.train_path)
    assert os.path.exists(artifacts.test_path)

    # TODO: they are not empty, no crossing by ids


# def test_first_level_fitting(artifacts, train_path: str, test_path: str):
#     # first model (dump)
#     first_level_fitting(
#         train_path=train_path,
#         test_path=test_path,
#         model_class_name="replay.models.als.ALSWrap",
#         model_kwargs={"rank": 128},
#         model_path=os.path.join(artifacts, "initial_model"),
#         second_level_partial_train_path=os.path.join(artifacts, "first_lvl_partial_model_1_train.parquet"),
#         first_level_model_predictions_path=os.path.join(artifacts, "first_lvl_partial_model_1_preds.parquet"),
#         k=10,
#         intermediate_datasets_mode="dump",
#         predefined_train_and_positives_path=(first_level_train_path, second_level_positives_path),
#         predefined_negatives_path=negatives_path,
#         item_features_path=item_features_path,
#         user_features_path=user_features_path
#     )
#     # TODO: test_exp exists
#     # TODO: model exists
#     # TODO: dataset exists
#
#     # second model (use)
#     # TODO: model exists
#     # TODO: dataset exists
#
#     # TODO: both datasets can be combined
#     pass
#
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