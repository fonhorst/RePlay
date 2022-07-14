from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.window import Window

from replay.models.base_rec import NeighbourRec
from replay.optuna_objective import ItemKNNObjective


class ItemKNN(NeighbourRec):
    """Item-based ItemKNN with modified cosine similarity measure."""

    all_items: Optional[DataFrame]
    dot_products: Optional[DataFrame]
    item_norms: Optional[DataFrame]
    k1 = 1.2
    b = 0.75
    _objective = ItemKNNObjective
    _search_space = {
        "num_neighbours": {"type": "int", "args": [1, 100]},
        "shrink": {"type": "int", "args": [0, 100]},
    }

    def __init__(
        self,
        num_neighbours: int = 10,
        use_relevance: bool = False,
        shrink: float = 0.0,
        weighting: str = None,
    ):
        """
        :param num_neighbours: number of neighbours
        :param use_relevance: flag to use relevance values as is or to treat them as 1
        :param shrink: term added to the denominator when calculating similarity
        :param weighting: item reweighting type, one of [None, 'tf_idf', 'bm25']
        """
        self.shrink = shrink
        self.use_relevance = use_relevance
        self.num_neighbours = num_neighbours
        self.weighting = weighting

    @property
    def _init_args(self):
        return {
            "shrink": self.shrink,
            "use_relevance": self.use_relevance,
            "num_neighbours": self.num_neighbours,
        }

    @staticmethod
    def _shrink(dot_products: DataFrame, shrink: float) -> DataFrame:
        return dot_products.withColumn(
            "similarity",
            sf.col("dot_product")
            / (sf.col("norm1") * sf.col("norm2") + shrink),
        ).select("item_idx_one", "item_idx_two", "similarity")

    def _get_similarity(self, log: DataFrame) -> DataFrame:
        """
        Calculate item similarities

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: similarity matrix `[item_idx_one, item_idx_two, similarity]`
        """
        dot_products = self._get_products(log)
        similarity = self._shrink(dot_products, self.shrink)
        return similarity

    def _get_item_vectors(self, log: DataFrame):
        if self.weighting == "bm25":
            left, right = self._get_tf_bm25(log)
        else:
            left = log.withColumnRenamed(
                "item_idx", "item_idx_one"
            ).withColumnRenamed("relevance", "rel_one")
            right = log.withColumnRenamed(
                "item_idx", "item_idx_two"
            ).withColumnRenamed("relevance", "rel_two")

        if self.weighting in ["tf_idf", "bm25"]:
            if self.weighting == "tf_idf":
                idf1, idf2 = ItemKNN._get_idf(log)
            elif self.weighting == "bm25":
                idf1, idf2 = ItemKNN._get_idf_bm25(log)

            left = left.join(idf1, how="inner", on="item_idx_one")
            left = left.withColumn(
                "rel_one",
                sf.col("rel_one") * sf.col("idf1"),
            )

            right = right.join(idf2, how="inner", on="item_idx_two")
            right = right.withColumn(
                "rel_two",
                sf.col("rel_two") * sf.col("idf2"),
            )

        return left, right

    def _get_tf_bm25(self, log: DataFrame):
        user_stats = log.groupBy("user_idx").agg(
            sf.count("item_idx").alias("n_items_per_user")
        )
        avgdl = user_stats.select(sf.mean("n_items_per_user")).take(1)[0][0]
        log = log.join(user_stats, how="inner", on="user_idx")

        left = (
            log.withColumn(
                "relevance",
                sf.col("relevance") * (self.k1 + 1) / (
                    sf.col("relevance") + self.k1 * (
                        1 - self.b + self.b * (
                            sf.col("n_items_per_user") / avgdl
                        )
                    )
                )
            )
            .drop("n_items_per_user")
            .withColumnRenamed("item_idx", "item_idx_one")
            .withColumnRenamed("relevance", "rel_one")
        )

        right = (
            log.withColumn(
                "relevance",
                sf.col("relevance") * (self.k1 + 1) / (
                    sf.col("relevance") + self.k1 * (
                        1 - self.b + self.b * (
                            sf.col("n_items_per_user") / avgdl
                        )
                    )
                )
            )
            .drop("n_items_per_user")
            .withColumnRenamed("item_idx", "item_idx_two")
            .withColumnRenamed("relevance", "rel_two")
        )
        return left, right

    @staticmethod
    def _get_idf(log: DataFrame):
        df = log.groupBy("item_idx").agg(sf.count("user_idx").alias("DF"))
        n_users = log.select("user_idx").distinct().count()

        idf1 = (
            df.withColumn("idf1", sf.log1p(sf.lit(n_users) / sf.col("DF")))
            .drop("DF")
            .withColumnRenamed("item_idx", "item_idx_one")
        )

        idf2 = (
            df.withColumn("idf2", sf.log1p(sf.lit(n_users) / sf.col("DF")))
            .drop("DF")
            .withColumnRenamed("item_idx", "item_idx_two")
        )

        return idf1, idf2

    @staticmethod
    def _get_idf_bm25(log: DataFrame):
        df = log.groupBy("item_idx").agg(sf.count("user_idx").alias("DF"))
        n_users = log.select("user_idx").distinct().count()

        idf1 = (
            df.withColumn(
                "idf1",
                sf.log1p(
                    (sf.lit(n_users) - sf.col("DF") + 0.5)
                    / (sf.col("DF") + 0.5)
                ),
            )
            .drop("DF")
            .withColumnRenamed("item_idx", "item_idx_one")
        )

        idf2 = (
            df.withColumn(
                "idf2",
                sf.log1p(
                    (sf.lit(n_users) - sf.col("DF") + 0.5)
                    / (sf.col("DF") + 0.5)
                ),
            )
            .drop("DF")
            .withColumnRenamed("item_idx", "item_idx_two")
        )

        return idf1, idf2

    def _get_products(self, log: DataFrame) -> DataFrame:
        """
        Calculate item dot products

        :param log: DataFrame with interactions, `[user_idx, item_idx, relevance]`
        :return: similarity matrix `[item_idx_one, item_idx_two, norm1, norm2]`
        """
        left, right = self._get_item_vectors(log)

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

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        df = log.select("user_idx", "item_idx", "relevance")
        if not self.use_relevance:
            df = df.withColumn("relevance", sf.lit(1))

        similarity_matrix = self._get_similarity(df)
        self.similarity = self._get_k_most_similar(similarity_matrix).cache()
