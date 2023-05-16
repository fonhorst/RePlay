# pylint: disable-all
import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.models import ALSWrap
from replay.scenarios.two_stages.two_stages_scenario import (
    get_first_level_model_features,
)
from tests.utils import log, spark


@pytest.fixture
def model():
    model = ALSWrap(2, implicit_prefs=False)
    model._seed = 42
    return model


@pytest.fixture
def model_with_ann(tmp_path):
    index_path = str((tmp_path / "nmslib_index"))
    model = ALSWrap(
        rank=2,
        implicit_prefs=False,
        seed=42,
        hnswlib_params=HnswlibParam(
            space="ip",
            M=100,
            efC=2000,
            post=0,
            efS=2000,
            # build_index_on="driver"
            build_index_on="executor",
            index_path=index_path
        )
    )
    return model


def test_works(log, model):
    try:
        pred = model.fit_predict(log, k=1)
        assert pred.count() == 4
    except:  # noqa
        pytest.fail()


def test_diff_feedback_type(log, model):
    pred_exp = model.fit_predict(log, k=1)
    model.implicit_prefs = True
    pred_imp = model.fit_predict(log, k=1)
    assert not np.allclose(
        pred_exp.toPandas().sort_values("user_idx")["relevance"].values,
        pred_imp.toPandas().sort_values("user_idx")["relevance"].values,
    )


def test_enrich_with_features(log, model):
    model.fit(log.filter(sf.col("user_idx").isin([0, 2])))
    res = get_first_level_model_features(
        model, log.filter(sf.col("user_idx").isin([0, 1]))
    )

    cold_user_and_item = res.filter(
        (sf.col("user_idx") == 1) & (sf.col("item_idx") == 3)
    )
    row_dict = cold_user_and_item.collect()[0].asDict()
    assert row_dict["_if_0"] == row_dict["_uf_0"] == row_dict["_fm_1"] == 0.0

    warm_user_and_item = res.filter(
        (sf.col("user_idx") == 0) & (sf.col("item_idx") == 0)
    )
    row_dict = warm_user_and_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_fm_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [4.093189725967505, row_dict["_fm_1"]],
    )

    cold_user_warm_item = res.filter(
        (sf.col("user_idx") == 1) & (sf.col("item_idx") == 0)
    )
    row_dict = cold_user_warm_item.collect()[0].asDict()
    np.allclose(
        [row_dict["_if_1"], row_dict["_if_1"] * row_dict["_uf_1"]],
        [-2.938199281692505, 0],
    )


def test_ann_predict(log, model, model_with_ann):
    model.fit(log)
    recs1 = model.predict(log, k=1)

    model_with_ann.fit(log)
    recs2 = model_with_ann.predict(log, k=1)

    recs1 = recs1.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    recs2 = recs2.toPandas().sort_values(
        ["user_idx", "item_idx"], ascending=False
    )
    assert recs1.user_idx.equals(recs2.user_idx)
    assert recs1.item_idx.equals(recs2.item_idx)
