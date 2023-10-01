# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
import numpy as np

from pyspark.sql import functions as sf

from replay.models.extensions.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.models.extensions.ann.index_builders.executor_nmslib_index_builder import (
    ExecutorNmslibIndexBuilder,
)
from replay.models.extensions.ann.index_builders.nmslib_index_builder_mixin import NmslibIndexBuilderMixin
from replay.models.extensions.ann.index_stores.shared_disk_index_store import (
    SharedDiskIndexStore,
)
from replay.models import SLIM
from tests.utils import log, spark


@pytest.fixture
def model():
    return SLIM(0.0, 0.01, seed=42)


@pytest.fixture
def model_with_ann(tmp_path):
    nmslib_hnsw_params = NmslibHnswParam(
        space="negdotprod_sparse",
        m=10,
        ef_s=200,
        ef_c=200,
        post=0,
    )
    return SLIM(
        0.0,
        0.01,
        seed=42,
        index_builder=ExecutorNmslibIndexBuilder(
            index_params=nmslib_hnsw_params,
            index_store=SharedDiskIndexStore(
                warehouse_dir=str(tmp_path), index_dir="nmslib_hnsw_index"
            ),
        ),
    )


def test_fit(log, model):
    model.fit(log)
    assert np.allclose(
        model.similarity.toPandas()
        .sort_values(["item_idx_one", "item_idx_two"])
        .to_numpy(),
        [
            (0, 1, 0.60048005),
            (0, 2, 0.12882786),
            (0, 3, 0.12860215),
            (1, 0, 1.06810235),
            (1, 2, 0.23784898),
            (2, 0, 0.25165837),
            (2, 1, 0.26372437),
            (3, 0, 1.32888889),
        ],
    )


def test_predict(log, model):
    model.fit(log)
    recs = model.predict(log, k=1)
    assert np.allclose(
        recs.toPandas()
        .sort_values(["user_idx", "item_idx"], ascending=False)
        .relevance,
        [0.4955047, 0.12860215, 0.60048005, 0.12860215],
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


@pytest.mark.parametrize(
    "beta,lambda_", [(0.0, 0.0), (-0.1, 0.1), (0.1, -0.1)]
)
def test_exceptions(beta, lambda_):
    with pytest.raises(ValueError):
        SLIM(beta, lambda_)


def test_build_index_udf(log, model, tmp_path):
    """This test used for test ANN functionality using similarity dataframe from SLIM model."""
    nmslib_hnsw_params = NmslibHnswParam(
        space="negdotprod_sparse",
        m=10,
        ef_s=200,
        ef_c=200,
        post=0,
    )
    index_store = SharedDiskIndexStore(
        warehouse_dir=str(tmp_path),
        index_dir="nmslib_hnsw_index",
        cleanup=False,
    )
    model.fit(log)
    similarity_pdf = model.similarity.select(
        "similarity", "item_idx_one", "item_idx_two"
    ).toPandas()
    nmslib_hnsw_params.items_count = (
        log.select(sf.max("item_idx")).first()[0] + 1
    )
    NmslibIndexBuilderMixin.build_and_save_index(
        similarity_pdf, nmslib_hnsw_params, index_store
    )
