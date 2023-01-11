import logging
import os
from typing import Any, Dict, Iterator, Optional
import uuid
import math

import numpy as np
import pandas as pd
import scann
import tempfile

from pyarrow import fs
from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame, functions as sf
from pyspark.sql.functions import pandas_udf
from replay.session_handler import State

from replay.utils import FileSystem, JobGroup, get_filesystem

logger = logging.getLogger("replay")


class ScannIndexFileManager:
    """Loads index from hdfs, local disk or SparkFiles dir and keep it in a memory.
    Instance of `ScannIndexFileManager` broadcasts to executors and is used in pandas_udf.
    """

    def __init__(
        self,
        index_params,
        index_path: Optional[str] = None,
        filesystem: Optional[FileSystem] = None,
        hdfs_uri: Optional[str] = None,
        index_filename: Optional[str] = None,
    ) -> None:

        self._index_path = index_path
        self._filesystem = filesystem
        self._hdfs_uri = hdfs_uri
        self._index_filename = index_filename
        self._index = None

    @property
    def index(self):
        if self._index:
            return self._index

        if self._index_path:
            if self._filesystem == FileSystem.HDFS:
                with tempfile.TemporaryDirectory() as temp_path:
                    source_filesystem = fs.HadoopFileSystem.from_uri(
                        self._hdfs_uri
                    )
                    fs.copy_files(
                        self._index_path,
                        "file://" + temp_path,
                        source_filesystem=source_filesystem,
                    )
                    print("self._index_path:")
                    print(self._index_path)
                    print("os.listdir(temp_path)")
                    print(os.listdir(temp_path))

                    self._index = scann.scann_ops_pybind.load_searcher(
                        temp_path
                    )
            else:
                # self._index.load_index(self._index_path)
                self._index = scann.scann_ops_pybind.load_searcher(
                    self._index_path
                )
        else:
            # self._index.load_index(SparkFiles.get(self._index_filename))
            self._index = scann.scann_ops_pybind.load_searcher(
                SparkFiles.get(self._index_filename)
            )

        return self._index


class ScannMixin:
    """Mixin that provides methods to build scann index and infer it.
    Also provides methods to saving and loading index to/from disk.
    """

    def __init__(self):
        # A unique id for the object.
        self.uid = uuid.uuid4().hex[-12:]

    def _build_scann_index(
        self,
        item_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        id_col: Optional[str],
    ):
        """Builds scann index and dump it to hdfs or disk.

        Args:
            item_vectors (DataFrame): DataFrame with item vectors
            params (Dict[str, Any]): hnsw params
        """

        with JobGroup(
            f"{self.__class__.__name__}._build_scann_index()",
            "all _build_scann_index()",
        ):
            if params["build_index_on"] == "executor":
                # to execution in one executor
                item_vectors = item_vectors.repartition(1)

                filesystem, hdfs_uri, index_path = get_filesystem(
                    params["index_path"]
                )

                def build_index(iterator: Iterator[pd.DataFrame]):

                    ids_dataframes = []
                    vectors = []
                    for pdf in iterator:
                        ids_dataframes.append(pdf[[id_col]])
                        vectors.append(
                            np.array(pdf[features_col].values.tolist())
                        )
                    vectors = np.vstack(vectors)
                    # print(vectors[:5])
                    # print(f"vectors.shape: {str(vectors.shape)}")

                    # vectors[np.linalg.norm(vectors, axis=1) == 0] = 1.0 / np.sqrt(vectors.shape[1])
                    # vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]

                    # normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

                    # print("int(math.sqrt(len(vectors)))")
                    # print(int(math.sqrt(len(vectors))))

                    # ).score_brute_force().build()
                    searcher = (
                        scann.scann_ops_pybind.builder(
                            vectors,
                            params[
                                "num_neighbors"
                            ],  # 10 + int(max_items_to_retrieve/20),
                            params["distance_measure"],
                        )
                        .tree(
                            num_leaves=params["num_leaves"],
                            num_leaves_to_search=200,  # 1
                            training_sample_size=len(vectors),
                            spherical=True,
                            quantize_centroids=True,
                        )
                        .score_ah(2, anisotropic_quantization_threshold=0.2)
                        .reorder(200)
                        .build()
                    )  # 1
                    #     .tree(
                    #     num_leaves=2000, #  int(math.sqrt(len(vectors))),  # int(math.sqrt(len(vectors))),  # 2000
                    #     num_leaves_to_search=200,
                    #     training_sample_size=len(vectors),
                    #     spherical=True,
                    #     quantize_centroids=True
                    # ).score_ah(
                    #     2, anisotropic_quantization_threshold=0.2
                    # ).reorder(200).set_n_training_threads(8).build()

                    # .score_brute_force().build()

                    if filesystem == FileSystem.HDFS:
                        # temp_path = tempfile.mkdtemp()
                        with tempfile.TemporaryDirectory() as temp_path:
                            # tmp_file_path = os.path.join(
                            #     temp_path, "scann_index"
                            # )
                            # index.save_index(tmp_file_path)
                            searcher.serialize(temp_path)

                            destination_filesystem = (
                                fs.HadoopFileSystem.from_uri(hdfs_uri)
                            )
                            fs.copy_files(
                                "file://" + temp_path,
                                index_path,
                                destination_filesystem=destination_filesystem,
                            )
                            # param use_threads=True (?)
                    else:
                        print(index_path)
                        os.makedirs(index_path, exist_ok=True)
                        searcher.serialize(index_path)

                    index_2_vector_id_pdf = pd.concat(ids_dataframes)
                    index_2_vector_id_pdf.reset_index(drop=True, inplace=True)
                    index_2_vector_id_pdf[
                        "index"
                    ] = index_2_vector_id_pdf.index
                    index_2_vector_id_pdf.to_csv("/tmp/1.csv", index=False)
                    yield index_2_vector_id_pdf[["index", id_col]]

                # builds index on executor and writes it to shared disk or hdfs
                index_2_item_idx_df = (
                    item_vectors.select(id_col, features_col)
                    .mapInPandas(build_index, f"index int, {id_col} int")
                    .cache()
                )
                index_2_item_idx_df.write.mode("overwrite").format(
                    "noop"
                ).save()

                return index_2_item_idx_df
            else:
                item_vectors = item_vectors.toPandas()
                vectors = np.array(item_vectors[features_col].values.tolist())

                # ).score_brute_force().build()
                searcher = (
                    scann.scann_ops_pybind.builder(
                        vectors,
                        params[
                            "num_neighbors"
                        ],  # 10 + int(max_items_to_retrieve/20),
                        params["distance_measure"],
                    )
                    .tree(
                        num_leaves=params["num_leaves"],
                        num_leaves_to_search=200,  # 1
                        training_sample_size=len(vectors),
                        spherical=True,
                        quantize_centroids=True,
                    )
                    .score_ah(2, anisotropic_quantization_threshold=0.2)
                    .reorder(200)  # 1
                    .build()
                )

                # saving index to local `temp_path` directory and sending it to executors
                temp_path = tempfile.mkdtemp()
                searcher.serialize(temp_path)
                spark = SparkSession.getActiveSession()
                # spark.sparkContext.addFile("file://" + tmp_file_path)
                spark.sparkContext.addFile(
                    path="file://" + temp_path, recursive=True
                )

    def _infer_scann_index(
        self,
        user_vectors: DataFrame,
        features_col: str,
        params: Dict[str, Any],
        k: int,
    ):

        if params["build_index_on"] == "executor":
            filesystem, hdfs_uri, index_path = get_filesystem(
                params["index_path"]
            )
            _index_file_manager = ScannIndexFileManager(
                params, index_path, filesystem, hdfs_uri
            )
        else:
            _index_file_manager = ScannIndexFileManager(
                params,
                index_filename="scann_index_" + self.uid,
            )

        index_file_manager_broadcast = State().session.sparkContext.broadcast(
            _index_file_manager
        )

        return_type = "item_idx array<int>, distance array<double>"

        @pandas_udf(return_type)
        def infer_index(
            vectors: pd.Series, num_items: pd.Series
        ) -> pd.DataFrame:
            index_file_manager = index_file_manager_broadcast.value
            searcher = index_file_manager.index

            # max number of items to retrieve per batch
            max_items_to_retrieve = num_items.max()

            neighbors, distances = searcher.search_batched(
                np.array(vectors.values.tolist()),
                final_num_neighbors=k + max_items_to_retrieve,
                pre_reorder_num_neighbors=params[
                    "pre_reorder_num_neighbors"
                ],  # 35, # (k + max_items_to_retrieve)*2,
                leaves_to_search=params[
                    "leaves_to_search"
                ],  # 13 #(k + max_items_to_retrieve)*2
            )
            # neighbors, distances = searcher.search_batched(np.array(vectors.values.tolist()))

            pd_res = pd.DataFrame(
                {"item_idx": list(neighbors), "distance": list(distances)}
            )

            return pd_res

        with JobGroup(
            "infer_index()",
            "infer_hnsw_index (inside 1)",
        ):
            res = user_vectors.select(
                "user_idx", infer_index(features_col, "num_items").alias("r")
            )
            # res = res.cache()
            # res.write.mode("overwrite").format("noop").save()

        with JobGroup(
            "res.withColumn('zip_exp', ...",
            "infer_hnsw_index (inside 2)",
        ):
            res = res.select(
                "user_idx",
                sf.explode(sf.arrays_zip("r.item_idx", "r.distance")).alias(
                    "zip_exp"
                ),
            )

            # Fix arrays_zip random behavior. It can return zip_exp.0 or zip_exp.item_idx in different machines
            fields = res.schema["zip_exp"].jsonValue()["type"]["fields"]
            item_idx_field_name: str = fields[0]["name"]
            distance_field_name: str = fields[1]["name"]

            res = res.select(
                "user_idx",
                sf.col(f"zip_exp.{item_idx_field_name}").alias("vector_idx"),
                (sf.col(f"zip_exp.{distance_field_name}")).alias("relevance"),
            )
            # res = res.cache()
            # res.write.mode("overwrite").format("noop").save()

        return res
