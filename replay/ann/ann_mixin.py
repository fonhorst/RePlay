import logging
import os
import shutil
import tempfile
import uuid
import weakref
from abc import abstractmethod
from typing import Optional, Dict, Union, Any, Tuple

from cached_property import cached_property
from pyarrow import fs
from pyspark import SparkFiles
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

from replay.ann.entities.base_hnsw_param import BaseHnswParam
from replay.models.base_rec import BaseRecommender
from replay.utils import get_filesystem, FileSystem

logger = logging.getLogger("replay")


class ANNMixin(BaseRecommender):
    """
    This class overrides the `_fit_wrap` and `_inner_predict_wrap` methods of the base class,
    adding an index construction in the `_fit_wrap` step
    and an index inference in the `_inner_predict_wrap` step.
    """

    INDEX_FILENAME = "index"

    @cached_property
    def _spark_index_file_uid(self) -> str:  # pylint: disable=no-self-use
        """
        Cached property that returns the uuid for the index file name.
        The unique name is needed to store the index file in `SparkFiles`
        without conflicts with other index files.
        """
        return uuid.uuid4().hex[-12:]

    @property
    @abstractmethod
    def _use_ann(self) -> bool:
        """
        Property that determines whether the ANN (index) is used.
        If `True`, then the index will be built (at the `fit` stage)
        and index will be inferred (at the `predict` stage).
        """

    @abstractmethod
    def _get_vectors_to_build_ann(self, log: DataFrame) -> DataFrame:
        """Implementations of this method must return a dataframe with item vectors.
        Item vectors from this method are used to build the index.

        Args:
            log: DataFrame with interactions

        Returns: DataFrame[item_idx int, vector array<double>] or DataFrame[vector array<double>].
        Column names in dataframe can be anything.
        """

    @abstractmethod
    def _get_ann_build_params(self, log: DataFrame) -> Dict[str, Any]:
        """Implementation of this method must return dictionary
        with arguments for `_build_ann_index` method.

        Args:
            log: DataFrame with interactions

        Returns: Dictionary with arguments to build index. For example: {
            "id_col": "item_idx",
            "features_col": "item_factors",
            "params": self._hnswlib_params,
        }

        """

    def _fit_wrap(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        """Wrapper extends `_fit_wrap`, adds construction of ANN index by flag.

        Args:
            log: historical log of interactions
                ``[user_idx, item_idx, timestamp, relevance]``
            user_features: user features
                ``[user_idx, timestamp]`` + feature columns
            item_features: item features
                ``[item_idx, timestamp]`` + feature columns

        """
        super()._fit_wrap(log, user_features, item_features)

        if self._use_ann:
            vectors = self._get_vectors_to_build_ann(log)
            ann_params = self._get_ann_build_params(log)
            self._build_ann_index(vectors, **ann_params)

    @abstractmethod
    def _get_vectors_to_infer_ann_inner(
        self, log: DataFrame, users: DataFrame
    ) -> DataFrame:
        """Implementations of this method must return a dataframe with user vectors.
        User vectors from this method are used to infer the index.

        Args:
            log: DataFrame with interactions
            users: DataFrame with users

        Returns: DataFrame[user_idx int, vector array<double>] or DataFrame[vector array<double>].
        Vector column name in dataframe can be anything.
        """

    def _get_vectors_to_infer_ann(
        self, log: DataFrame, users: DataFrame, filter_seen_items: bool
    ) -> DataFrame:
        """This method wraps `_get_vectors_to_infer_ann_inner`
        and adds seen items to dataframe with user vectors by flag.

        Args:
            log: DataFrame with interactions
            users: DataFrame with users
            filter_seen_items: flag to remove seen items from recommendations based on ``log``.

        Returns:

        """
        users = self._get_vectors_to_infer_ann_inner(log, users)

        # here we add `seen_item_idxs` to filter the viewed items in UDFs (see infer_index)
        if filter_seen_items:
            user_to_max_items = log.groupBy("user_idx").agg(
                sf.count("item_idx").alias("num_items"),
                sf.collect_set("item_idx").alias("seen_item_idxs"),
            )
            users = users.join(user_to_max_items, on="user_idx")

        return users

    @abstractmethod
    def _get_ann_infer_params(self) -> Dict[str, Any]:
        """Implementation of this method must return dictionary
        with arguments for `_infer_ann_index` method.

        Returns: Dictionary with arguments to infer index. For example: {
            "features_col": "user_vector",
            "params": self._hnsw_params,
        }

        """

    @abstractmethod
    def _build_ann_index(
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Union[int, str]],
        id_col: Optional[str] = None,
    ) -> None:
        """The method implements the construction of the ANN index.

        Args:
            vectors: DataFrame with vectors to build index
            features_col: Name of column from `vectors` dataframe
                that contains vectors to build index
            params: Index params
            id_col: Name of column that contains identifiers of vectors.
                None if `vectors` dataframe have no id column.

        """

    @abstractmethod
    def _infer_ann_index(  # pylint: disable=too-many-arguments
        self,
        vectors: DataFrame,
        features_col: str,
        params: Dict[str, Union[int, str]],
        k: int,
        filter_seen_items: bool,
    ) -> DataFrame:
        """The method implements the inference of the ANN index.

        Args:
            vectors: Dataframe that contains vectors to inference
            features_col: Column name from `vectors` dataframe that contains vectors to inference
            params: Index params
            k: Desired number of neighbour vectors for vectors from `vectors` dataframe
            filter_seen_items: flag to filter seen items before output

        Returns: DataFrame[user_idx int, item_idx int, relevance double]

        """

    def _inner_predict_wrap(  # pylint: disable=too-many-arguments
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        """Override base `_inner_predict_wrap` and adds ANN inference by condition"""
        if self._use_ann:
            vectors = self._get_vectors_to_infer_ann(
                log, users, filter_seen_items
            )
            ann_params = self._get_ann_infer_params()
            return self._infer_ann_index(
                vectors,
                k=k,
                filter_seen_items=filter_seen_items,
                **ann_params,
            )
        else:
            return self._predict(
                log,
                k,
                users,
                items,
                user_features,
                item_features,
                filter_seen_items,
            )

    @staticmethod
    def _unpack_infer_struct(inference_result: DataFrame) -> DataFrame:
        """Transforms input dataframe.
        Unpacks and explodes arrays from `neighbours` struct.

        >>>
        >> inference_result.printSchema()
        root
         |-- user_idx: integer (nullable = true)
         |-- neighbours: struct (nullable = true)
         |    |-- item_idx: array (nullable = true)
         |    |    |-- element: integer (containsNull = true)
         |    |-- distance: array (nullable = true)
         |    |    |-- element: double (containsNull = true)
        >> ANNMixin._unpack_infer_struct(inference_result).printSchema()
        root
         |-- user_idx: integer (nullable = true)
         |-- item_idx: integer (nullable = true)
         |-- relevance: double (nullable = true)

        Args:
            inference_result: output of infer_index UDF
        """
        res = inference_result.select(
            "user_idx",
            sf.explode(
                sf.arrays_zip("neighbours.item_idx", "neighbours.distance")
            ).alias("zip_exp"),
        )

        # Fix arrays_zip random behavior.
        # It can return zip_exp.0 or zip_exp.item_idx in different machines.
        fields = res.schema["zip_exp"].jsonValue()["type"]["fields"]
        item_idx_field_name: str = fields[0]["name"]
        distance_field_name: str = fields[1]["name"]

        res = res.select(
            "user_idx",
            sf.col(f"zip_exp.{item_idx_field_name}").alias("item_idx"),
            (sf.lit(-1.0) * sf.col(f"zip_exp.{distance_field_name}")).alias(
                "relevance"
            ),
        )
        return res

    def _filter_seen(
        self, recs: DataFrame, log: DataFrame, k: int, users: DataFrame
    ):
        """
        Overridden _filter_seen method from base class.
        Filtering is not needed for ann methods, because the data is already filtered in udf.
        """
        if self._use_ann:
            return recs

        return super()._filter_seen(recs, log, k, users)

    def _save_index_files(
        self,
        target_dir: str,
        index_params: BaseHnswParam,
        additional_file_extensions: Optional[Tuple[str]] = (),
    ):
        if index_params.build_index_on == "executor":
            index_path = index_params.index_path
        elif index_params.build_index_on == "driver":
            index_path = SparkFiles.get(
                f"{self.INDEX_FILENAME}_{self._spark_index_file_uid}"
            )
        else:
            raise ValueError("Unknown 'build_index_on' param.")

        source = get_filesystem(index_path)
        target = get_filesystem(target_dir)

        source_paths = [source.path] + [
            source.path + ext for ext in additional_file_extensions
        ]
        index_file_target_path = os.path.join(target.path, self.INDEX_FILENAME)
        target_paths = [index_file_target_path] + [
            index_file_target_path + ext for ext in additional_file_extensions
        ]

        if source.filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(source.hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()
        if target.filesystem == FileSystem.HDFS:
            destination_filesystem = fs.HadoopFileSystem.from_uri(
                target.hdfs_uri
            )
        else:
            destination_filesystem = fs.LocalFileSystem()

        for source_path, target_path in zip(source_paths, target_paths):
            print(source_path, target_path)
            logger.debug(
                "Index file coping from '%s' to '%s'", source_path, target_path
            )
            fs.copy_files(
                source_path,
                target_path,
                source_filesystem=source_filesystem,
                destination_filesystem=destination_filesystem,
            )
            # param use_threads=True (?)

    def _load_index(
        self,
        path: str,
        index_params: BaseHnswParam,
        additional_file_extensions: Optional[Tuple[str]] = (),
    ):
        """Loads hnsw index from `path` directory to local dir.
        Index file name is 'hnswlib_index'.
        And adds index file to the `SparkFiles`.
        `path` can be a hdfs path or a local path.


        Args:
            path: directory path, where index file is stored
        """
        source = get_filesystem(path + f"/{self.INDEX_FILENAME}")

        temp_dir = tempfile.mkdtemp()
        weakref.finalize(self, shutil.rmtree, temp_dir)
        target_path = os.path.join(
            temp_dir, f"{self.INDEX_FILENAME}_{self._spark_index_file_uid}"
        )

        source_paths = [source.path] + [
            source.path + ext for ext in additional_file_extensions
        ]
        target_paths = [target_path] + [
            target_path + ext for ext in additional_file_extensions
        ]

        if source.filesystem == FileSystem.HDFS:
            source_filesystem = fs.HadoopFileSystem.from_uri(source.hdfs_uri)
        else:
            source_filesystem = fs.LocalFileSystem()

        destination_filesystem = fs.LocalFileSystem()

        for source_path, target_path in zip(source_paths, target_paths):
            print(source_path, target_path)
            logger.debug(
                "Index file coping from '%s' to '%s'", source_path, target_path
            )
            fs.copy_files(
                source_path,
                target_path,
                source_filesystem=source_filesystem,
                destination_filesystem=destination_filesystem,
            )

        spark = SparkSession.getActiveSession()
        for target_path in target_paths:
            spark.sparkContext.addFile("file://" + target_path)

        index_params.build_index_on = "driver"
