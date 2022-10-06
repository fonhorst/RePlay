# pylint: disable-all

import pandas as pd
import pytest
import pyspark.sql.functions as sf

from replay.models import ClusterRec
from replay.utils import convert2spark


@pytest.fixture
def train():
    return convert2spark(pd.DataFrame({"user_idx": [1, 2, 3], "item_idx": [1, 2, 3]}))


@pytest.fixture
def test():
    return convert2spark(pd.DataFrame({"user_idx": [4, 5], "item_idx": [1, 2]}))


@pytest.fixture
def user_features():
    return convert2spark(pd.DataFrame(
        {
            "user_idx": [1, 2, 3, 4, 5],
            "age": [18, 20, 80, 16, 69],
            "sex": [1, 0, 0, 1, 0],
        }
    ))


@pytest.fixture
def model():
    return ClusterRec()


def test_works(model, train, test, user_features):
    model.fit(train, user_features)
    model.predict(user_features, k=1)
    res = model.optimize(train, test, user_features, k=1, budget=1)
    assert type(res["num_clusters"]) == int


def test_predict_pairs(model, train, test, user_features):
    model.fit(train, user_features=user_features)
    res = model.predict_pairs(
        train.filter(sf.col("user_idx") == 1),
        log=train,
        user_features=user_features,
    )
    assert res.count() == 1
    assert res.select("item_idx").collect()[0][0] == 1


def test_raises(model, train, test, user_features):
    with pytest.raises(ValueError, match="User features are missing for predict"):
        model.fit(train, user_features=user_features)
        model.predict_pairs(
            train.filter(sf.col("user_idx") == 1)
        )
