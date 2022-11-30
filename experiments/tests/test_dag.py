import os
import shutil

import pytest

from experiments.two_stage_scenarios import dataset_splitting


@pytest.fixture(scope="function")
def base_path() -> str:
    path = "/opt/experiments/test_exp"

    shutil.rmtree(path, ignore_errors=True)

    yield path

    shutil.rmtree(path, ignore_errors=True)


def test_data_splitting(base_path: str):
    train_path = os.path.join(base_path, "train.parquet")
    test_path = os.path.join(base_path, "test.parquet")

    # TODO: test_exp doesn't exists

    dataset_splitting.function(
        log_path="/opt/data/ml100k_ratings.csv",
        base_path=base_path,
        train_path=train_path,
        test_path=test_path,
        cores=1
    )

    assert os.path.exists(base_path)
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # TODO: they are not empty, no crossing by ids


# def test_first_level_fitting():
#     # first model (dump)
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


def second_level_fitting():
    # first run
    # TODO: model exists
    # TODO: dataset exists

    # second run
    # TODO: model exists
    # TODO: dataset exists

    # TODO: both datasets can be combined
    pass