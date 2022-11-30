from experiments.two_stage_scenarios import dataset_splitting


def test_data_splitting():

    base_path = "/opt/experiments/test_exp"

    # TODO: test_exp doesn't exists


    dataset_splitting(
        log_path="/opt/data/ratings.parquet",
        base_path="/opt/experiments/test_exp",
        train_pat="/opt/experiments/test_exp/train.parquet",
        test_path="/opt/experiments/test_exp/test.parquet",
        cores=1
    ).execute_callable()

    # TODO: test_exp exists
    # TODO: train and test exists
    # TODO: they are not empty, no crossing by ids

    pass


def test_first_level_fitting():
    # first model (dump)
    # TODO: test_exp exists
    # TODO: model exists
    # TODO: dataset exists

    # second model (use)
    # TODO: model exists
    # TODO: dataset exists

    # TODO: both datasets can be combined
    pass


def test_combine_datasets():
    # TODO: dataset exists
    # TODO: no Nones
    pass


def second_level_fitting():
    # first run
    # TODO: model exists
    # TODO: dataset exists

    # second run
    # TODO: model exists
    # TODO: dataset exists

    # TODO: both datasets can be combined
    pass