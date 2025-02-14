from typing import Optional, Dict, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window

from replay.models.base_rec import NeighbourRec, PartialFitMixin
from replay.optuna_objective import ItemKNNObjective
from replay.utils import unionify

import warnings


class ItemKNN(NeighbourRec, PartialFitMixin):
    """Item-based ItemKNN with modified cosine similarity measure."""

    def _get_ann_infer_params(self) -> Dict[str, Any]:
        return {
            "features_col": "",
            "params": self._nmslib_hnsw_params,
            "index_type": "sparse",
        }

    @property
    def _use_ann(self) -> bool:
        return self._nmslib_hnsw_params is not None

    all_items: Optional[DataFrame]
    dot_products: Optional[DataFrame]
    item_norms: Optional[DataFrame]
    bm25_k1 = 1.2
    bm25_b = 0.75
    _objective = ItemKNNObjective
    _search_space = {
        "num_neighbours": {"type": "int", "args": [1, 100]},
        "shrink": {"type": "int", "args": [0, 100]},
        "weighting": {"type": "categorical", "args": [None, "tf_idf", "bm25"]}
    }

    def __init__(
        self,
        num_neighbours: int = 10,
        use_relevance: bool = False,
        shrink: float = 0.0,
        weighting: str = None,
        nmslib_hnsw_params: Optional[dict] = None,
    ):
        """
        :param num_neighbours: number of neighbours
        :param use_relevance: flag to use relevance values as is or to treat them as 1
        :param shrink: term added to the denominator when calculating similarity
        :param weighting: item reweighting type, one of [None, 'tf_idf', 'bm25']
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
        self.shrink = shrink
        self.use_relevance = use_relevance
        self.num_neighbours = num_neighbours

        valid_weightings = self._search_space["weighting"]["args"]
        if weighting not in valid_weightings:
            raise ValueError(f"weighting must be one of {valid_weightings}")
        self.weighting = weighting
        self._nmslib_hnsw_params = nmslib_hnsw_params
        self.similarity = None

    @property
    def _init_args(self):
        return {
            "shrink": self.shrink,
            "use_relevance": self.use_relevance,
            "num_neighbours": self.num_neighbours,
            "weighting": self.weighting,
            "nmslib_hnsw_params": self._nmslib_hnsw_params,
        }

    def _save_model(self, path: str):
        if self._nmslib_hnsw_params:
            self._save_nmslib_hnsw_index(path, sparse=True)

    def _load_model(self, path: str):
        if self._nmslib_hnsw_params:
            self._load_nmslib_hnsw_index(path, sparse=True)

    @staticmethod
    def _shrink(dot_products: DataFrame, shrink: float) -> DataFrame:
        return dot_products.withColumn(
            "similarity",
            sf.col("dot_product")
            / (sf.col("norm1") * sf.col("norm2") + shrink),
        ).select("item_idx_one", "item_idx_two", "similarity")

    def _get_similarity(self, log: DataFrame, previous_log: Optional[DataFrame] = None) -> DataFrame:
        """
        Calculate item similarities

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: similarity matrix `[item_idx_one, item_idx_two, similarity]`
        """

        if previous_log is not None:
            log_part = log
            log = previous_log
        else:
            log_part = None

        dot_products = self._get_products(log, log_part)
        similarity = self._shrink(dot_products, self.shrink)
        return similarity

    def _reweight_log(self, log: DataFrame):
        """
        Reweight relevance according to TD-IDF or BM25 weighting.

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: log `[user_idx, item_idx, relevance]`
        """
        if self.weighting == "bm25":
            log = self._get_tf_bm25(log)

        idf = self._get_idf(log)

        log = log.join(idf, how="inner", on="user_idx").withColumn(
            "relevance",
            sf.col("relevance") * sf.col("idf"),
        )

        return log

    def _get_tf_bm25(self, log: DataFrame):
        """
        Adjust relevance by BM25 term frequency.

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: log `[user_idx, item_idx, relevance]`
        """
        item_stats = log.groupBy("item_idx").agg(
            sf.count("user_idx").alias("n_users_per_item")
        )
        avgdl = item_stats.select(sf.mean("n_users_per_item")).take(1)[0][0]
        log = log.join(item_stats, how="inner", on="item_idx")

        log = (
            log.withColumn(
                "relevance",
                sf.col("relevance") * (self.bm25_k1 + 1) / (
                    sf.col("relevance") + self.bm25_k1 * (
                        1 - self.bm25_b + self.bm25_b * (
                            sf.col("n_users_per_item") / avgdl
                        )
                    )
                )
            )
            .drop("n_users_per_item")
        )

        return log

    def _get_idf(self, log: DataFrame):
        """
        Return inverse document score for log reweighting.

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: idf `[idf]`
        :raises: ValueError if self.weighting not in ["tf_idf", "bm25"]
        """
        df = log.groupBy("user_idx").agg(sf.count("item_idx").alias("DF"))
        n_items = log.select("item_idx").distinct().count()

        if self.weighting == "tf_idf":
            idf = (
                df.withColumn("idf", sf.log1p(sf.lit(n_items) / sf.col("DF")))
                .drop("DF")
            )
        elif self.weighting == "bm25":
            idf = (
                df.withColumn(
                    "idf",
                    sf.log1p(
                        (sf.lit(n_items) - sf.col("DF") + 0.5)
                        / (sf.col("DF") + 0.5)
                    ),
                )
                .drop("DF")
            )
        else:
            raise ValueError("weighting must be one of ['tf_idf', 'bm25']")

        return idf

    def _get_products(self, log: DataFrame, log_part: Optional[DataFrame] = None) -> DataFrame:
        """
        Calculate item dot products

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: similarity matrix `[item_idx_one, item_idx_two, norm1, norm2, dot_product]`
        """
        if self.weighting:
            log = self._reweight_log(log)
            log_part = self._reweight_log(log_part) if log_part is not None else None

        left = log.withColumnRenamed(
            "item_idx", "item_idx_one"
        ).withColumnRenamed("relevance", "rel_one")
        right = (log_part if log_part is not None else log).withColumnRenamed(
            "item_idx", "item_idx_two"
        ).withColumnRenamed("relevance", "rel_two")

        dot_products = (
            left.join(right, how="inner", on="user_idx")
            .filter(sf.col("item_idx_one") != sf.col("item_idx_two"))
            .withColumn("relevance", sf.col("rel_one") * sf.col("rel_two"))
            .groupBy("item_idx_one", "item_idx_two")
            .agg(sf.sum("relevance").alias("dot_product"))
        )

        item_norms = (
            log.withColumn("relevance", sf.col("relevance") ** 2)
            .groupBy("item_idx")
            .agg(sf.sum("relevance").alias("square_norm"))
            .select(sf.col("item_idx"), sf.sqrt("square_norm").alias("norm"))
        )
        norm1 = item_norms.withColumnRenamed(
            "item_idx", "item_id1"
        ).withColumnRenamed("norm", "norm1")

        if log_part is not None:
            item_norms_part = (
                log_part.withColumn("relevance", sf.col("relevance") ** 2)
                .groupBy("item_idx")
                .agg(sf.sum("relevance").alias("square_norm"))
                .select(sf.col("item_idx"), sf.sqrt("square_norm").alias("norm"))
            )

            norm2 = item_norms_part.withColumnRenamed(
                "item_idx", "item_id2"
            ).withColumnRenamed("norm", "norm2")
        else:
            norm2 = item_norms.withColumnRenamed(
                "item_idx", "item_id2"
            ).withColumnRenamed("norm", "norm2")

        dot_products = dot_products.join(
            norm1, how="inner", on=sf.col("item_id1") == sf.col("item_idx_one")
        )
        dot_products = dot_products.join(
            norm2, how="inner", on=sf.col("item_id2") == sf.col("item_idx_two")
        )

        return dot_products

    def _get_k_most_similar(self, similarity_matrix: DataFrame) -> DataFrame:
        """
        Leaves only top-k neighbours for each item

        :param similarity_matrix: dataframe `[item_idx_one, item_idx_two, similarity]`
        :return: cropped similarity matrix
        """
        return (
            similarity_matrix.withColumn(
                "similarity_order",
                sf.row_number().over(
                    Window.partitionBy("item_idx_one").orderBy(
                        sf.col("similarity").desc(),
                        sf.col("item_idx_two").desc(),
                    )
                ),
            )
            .filter(sf.col("similarity_order") <= self.num_neighbours)
            .drop("similarity_order")
        )

    def fit_partial(self, log: DataFrame, previous_log: Optional[DataFrame] = None) -> None:
        super().fit_partial(log, previous_log)

        if self._use_ann:
            warnings.warn("ItemKNN fit_partial is used wth 'use_ann' flag. "
                          "It means full ann index rebuilding for this particular model.", RuntimeWarning)
            vectors = self._get_vectors_to_build_ann(log)
            ann_params = self._get_ann_build_params(log)
            self._build_ann_index(vectors, **ann_params)

    def _fit_partial(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            previous_log: Optional[DataFrame] = None) -> None:
        log = log.select("user_idx", "item_idx", "relevance" if self.use_relevance else sf.lit(1).alias("relevance"))

        # TODO: fit_partial integration with ANN index
        # TODO: no need for special integration, because you need to rebuild the whole
        #  index if set of items have been chnaged
        #  and no need for rebuilding if only user sets have changes
        similarity_matrix = self._get_similarity(log, previous_log)
        similarity_matrix = unionify(similarity_matrix, self.similarity)
        self.similarity = self._get_k_most_similar(similarity_matrix)
        self.similarity.cache().count()

    def _project_fields(self, log: Optional[DataFrame]):
        if log is None:
            return None
        return log.select("user_idx", "item_idx", "relevance" if self.use_relevance else sf.lit(1))

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
