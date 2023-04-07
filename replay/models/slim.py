from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from scipy.sparse import csc_matrix
from sklearn.linear_model import ElasticNet

from replay.models.base_rec import NeighbourRec
from replay.session_handler import State


class SLIM(NeighbourRec):
    """`SLIM: Sparse Linear Methods for Top-N Recommender Systems
    <http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf>`_"""

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": "",
            "params": self._nmslib_hnsw_params,
            "index_type": "sparse",
        }

    @property
    def _use_ann(self) -> bool:
        return self._nmslib_hnsw_params is not None

    _search_space = {
        "beta": {"type": "loguniform", "args": [1e-6, 5]},
        "lambda_": {"type": "loguniform", "args": [1e-6, 2]},
    }

    def __init__(
        self,
        beta: float = 0.01,
        lambda_: float = 0.01,
        seed: Optional[int] = None,
        nmslib_hnsw_params: Optional[dict] = None,
    ):
        """
        :param beta: l2 regularization
        :param lambda_: l1 regularization
        :param seed: random seed
        :param nmslib_hnsw_params: parameters for nmslib-hnsw methods:
        {"method":"hnsw",
        "space":"negdotprod_sparse_fast",
        "M":16,"efS":200,"efC":200,
        ...}
            The reasonable range of values for M parameter is 5-100,
            for efC and eFS is 100-2000.
            Increasing these values improves the prediction quality but increases index_time and inference_time too.
            We recommend using these settings:
              - M=16, efC=200 and efS=200 for simple datasets like MovieLens
              - M=50, efC=1000 and efS=1000 for average quality with an average prediction time
              - M=75, efC=2000 and efS=2000 for the highest quality with a long prediction time

            note: choosing these parameters depends on the dataset and quality/time tradeoff
            note: while reducing parameter values the highest range metrics like Metric@1000 suffer first
            note: even in a case with a long training time,
                profit from ann could be obtained while inference will be used multiple times

        for more details see https://github.com/nmslib/nmslib/blob/master/manual/methods.md
        """
        if beta < 0 or lambda_ <= 0:
            raise ValueError("Invalid regularization parameters")
        self.beta = beta
        self.lambda_ = lambda_
        self.seed = seed
        self._nmslib_hnsw_params = nmslib_hnsw_params

    @property
    def _init_args(self):
        return {
            "beta": self.beta,
            "lambda_": self.lambda_,
            "seed": self.seed,
            "nmslib_hnsw_params": self._nmslib_hnsw_params,
        }

    def _save_model(self, path: str):
        if self._nmslib_hnsw_params:
            self._save_nmslib_hnsw_index(path, sparse=True)

    def _load_model(self, path: str):
        if self._nmslib_hnsw_params:
            self._load_nmslib_hnsw_index(path, sparse=True)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        pandas_log = log.select("user_idx", "item_idx", "relevance").toPandas()

        interactions_matrix = csc_matrix(
            (pandas_log.relevance, (pandas_log.user_idx, pandas_log.item_idx)),
            shape=(self._user_dim, self._item_dim),
        )
        similarity = (
            State()
            .session.createDataFrame(pandas_log.item_idx, st.IntegerType())
            .withColumnRenamed("value", "item_idx_one")
        )
        # DEBUG
        # similarity = similarity.limit(10)
        #
        alpha = self.beta + self.lambda_
        l1_ratio = self.lambda_ / alpha

        regression = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=5000,
            random_state=self.seed,
            selection="random",
            positive=True,
        )

        def slim_column(pandas_df: pd.DataFrame) -> pd.DataFrame:
            """
            fit similarity matrix with ElasticNet
            :param pandas_df: pd.Dataframe
            :return: pd.Dataframe
            """
            k = 100
            idx = int(pandas_df["item_idx_one"][0])
            column = interactions_matrix[:, idx]
            column_arr = column.toarray().ravel()
            interactions_matrix[
                interactions_matrix[:, idx].nonzero()[0], idx
            ] = 0

            # print("data to fit X: (interaction matrix)")
            # print(interactions_matrix.count_nonzero())
            # print("data to fix Y: (column_arr)")
            # print("total size")
            # print(column_arr.size)
            # print("total positives")
            # print(np.count_nonzero(column_arr))

            # zero_idx = np.where(column_arr == 0)[0]
            # sample_size = int(k * column_arr.nonzero()[0].size)
            # if sample_size > zero_idx.size:
            #     sample_size = zero_idx.size
            # zero_idx_new = np.random.choice(zero_idx, size=sample_size, replace=False)
            # new_idx = np.concatenate((column_arr.nonzero()[0], zero_idx_new), axis=0)
            # column_arr = column_arr[new_idx]
            # interactions_matrix = interactions_matrix[new_idx, :]
            # print("total size")
            # print(column_arr.size)
            # print("total positives")
            # print(np.count_nonzero(column_arr))
            # End of sampling
            regression.fit(interactions_matrix, column_arr)
            # regression.fit(interactions_matrix[new_idx, :], column_arr)  # перемешались индексы
            interactions_matrix[:, idx] = column
            # print("regression coef_")
            # print(np.count_nonzero(regression.coef_))
            # print(regression.coef_)
            good_idx = np.argwhere(regression.coef_ > 0).reshape(-1)
            # print("good_idx")
            # print(good_idx)
            # good_idx = new_idx[good_idx]  # for index mapping
            # print("good_idx")
            # print(good_idx)
            good_values = regression.coef_[good_idx]
            similarity_row = {
                "item_idx_one": good_idx,
                "item_idx_two": idx,
                "similarity": good_values,
            }
            # print("similarity row")
            # print(similarity_row)
            return pd.DataFrame(data=similarity_row)

        self.similarity = similarity.groupby("item_idx_one").applyInPandas(
            slim_column,
            "item_idx_one int, item_idx_two int, similarity double",
        )
        self.similarity.cache().count()
        # assert False
    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        return self._predict_pairs_inner(
            log=log,
            filter_df=items.withColumnRenamed("item_idx", "item_idx_filter"),
            condition=sf.col("item_idx_two") == sf.col("item_idx_filter"),
            users=users,
        )
