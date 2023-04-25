# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
from pyspark.sql import functions as sf
from replay.models.base_rec import BaseRecommender
from replay.models import ALSWrap, PopRec, SLIM
from replay.scenarios import OneStageScenario
from replay.history_based_fp import HistoryBasedFeaturesProcessor
from replay.data_preparator import ToNumericFeatureTransformer
from replay.splitters import DateSplitter

from tests.utils import (
    spark,
    sparkDataFrameEqual,
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
)


@pytest.fixture
def one_stages_kwargs():
    return {
        "first_level_models": [
            ALSWrap(rank=4),
            SLIM(seed=22),
        ],
        "train_val_splitter": DateSplitter(test_start=0.1),
        "user_cat_features_list": ["gender"],
        "item_cat_features_list": ["class"],
        "custom_features_processor": None,
        "set_best_model": True
    }


def test_init(one_stages_kwargs):

    one_stage_scenario = OneStageScenario(**one_stages_kwargs)
    assert isinstance(one_stage_scenario.fallback_model, PopRec)
    assert isinstance(
        one_stage_scenario.features_processor, HistoryBasedFeaturesProcessor
    )
    assert isinstance(
        one_stage_scenario.first_level_item_features_transformer,
        ToNumericFeatureTransformer,
    )


def test_fit(
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
    one_stages_kwargs,
):
    one_stage_scenario = OneStageScenario(**one_stages_kwargs)
    one_stage_scenario.fit(
        long_log_with_features,
        user_features,
        item_features.filter(sf.col("iq") > 4),
    )
    assert one_stage_scenario.first_level_item_len == 8
    assert one_stage_scenario.first_level_user_len == 3

    assert isinstance(one_stage_scenario.best_model, BaseRecommender)

    one_stage_scenario.first_level_item_features_transformer.transform(item_features)


def test_predict(
    long_log_with_features, user_features, item_features, one_stages_kwargs,
):
    one_stage_scenario = OneStageScenario(**one_stages_kwargs)

    one_stage_scenario.fit(
        long_log_with_features,
        user_features,
        item_features.filter(sf.col("iq") > 4),
    )
    pred = one_stage_scenario.predict(
        log=long_log_with_features,
        k=2,
        user_features=user_features,
        item_features=item_features,
        filter_seen_items=False
    )
    assert pred.count() == 6
    assert sorted(pred.select(sf.collect_set("user_idx")).collect()[0][0]) == [
        0,
        1,
        2,
    ]


def test_optimize(
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
    one_stages_kwargs,
):
    one_stage_scenario = OneStageScenario(**one_stages_kwargs)
    param_borders = [{"rank": [1, 10]}, {}, None]
    # with fallback
    first_level_params, fallback_params, metrics_values = one_stage_scenario.optimize(
        train=long_log_with_features,
        test=short_log_with_features,
        user_features=user_features,
        item_features=item_features,
        param_borders=param_borders,
        k=1,
        budget=1,
    )

    assert len(first_level_params) == 2
    assert first_level_params[1] is None
    assert list(first_level_params[0].keys()) == ["rank"]
    assert fallback_params is None

    # no fallback works
    one_stage_scenario.fallback_model = None
    one_stage_scenario.optimize(
        train=long_log_with_features,
        test=short_log_with_features,
        user_features=user_features,
        item_features=item_features,
        param_borders=param_borders[:2],
        k=1,
        budget=1,
    )
