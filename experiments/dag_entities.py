import os
import uuid
from dataclasses import dataclass
from typing import Optional, List, Sequence, Dict, Any

from kubernetes.client import models as k8s
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as sf

DEFAULT_CPU = 6
BIG_CPU = 12
EXTRA_BIG_CPU = 20

DEFAULT_MEMORY = 30
BIG_MEMORY = 40
EXTRA_BIG_MEMORY = 80


big_executor_config = {
    "pod_override": k8s.V1Pod(
        spec=k8s.V1PodSpec(
            containers=[
                k8s.V1Container(
                    name="base",
                    resources=k8s.V1ResourceRequirements(
                        requests={"cpu": BIG_CPU, "memory": f"{BIG_MEMORY}Gi"},
                        limits={"cpu": BIG_CPU, "memory": f"{BIG_MEMORY}Gi"})
                )
            ],
        )
    ),
}


extra_big_executor_config = {
    "pod_override": k8s.V1Pod(
        spec=k8s.V1PodSpec(
            containers=[
                k8s.V1Container(
                    name="base",
                    resources=k8s.V1ResourceRequirements(
                        requests={"cpu": EXTRA_BIG_CPU, "memory": f"{EXTRA_BIG_MEMORY}Gi"},
                        limits={"cpu": EXTRA_BIG_CPU, "memory": f"{EXTRA_BIG_MEMORY}Gi"})
                )
            ],
        )
    ),
}

dense_hnsw_params = {
    "method": "hnsw",
    "space": "negdotprod",
    "M": 100,
    "efS": 2000,
    "efC": 2000,
    "post": 0,
    # hdfs://node21.bdcl:9000
    # "index_path": f"/opt/spark_data/replay_datasets/nmslib_hnsw_index_{spark.sparkContext.applicationId}",
    "build_index_on": "executor"
}

FIRST_LEVELS_MODELS_PARAMS = {
    "replay.models.als.ALSWrap": {"rank": 100, "seed": 42, "nmslib_hnsw_params": dense_hnsw_params},
    "replay.models.knn.ItemKNN": {"num_neighbours": 1000},
    "replay.models.cluster.ClusterRec": {"num_clusters": 100},
    "replay.models.slim.SLIM": {"seed": 42},
    "replay.models.word2vec.Word2VecRec": {"rank": 100, "seed": 42, "nmslib_hnsw_params": dense_hnsw_params},
    "replay.models.ucb.UCB": {"seed": 42}
}

MODELNAME2FULLNAME = {
    "als": "replay.models.als.ALSWrap",
    "itemknn": "replay.models.knn.ItemKNN",
    "cluster": "replay.models.cluster.ClusterRec",
    "slim": "replay.models.slim.SLIM",
    "word2vec": "replay.models.word2vec.Word2VecRec",
    "ucb": "replay.models.ucb.UCB"
}

SECOND_LEVELS_MODELS_PARAMS = {
    "test": {
            "general_params": {"use_algos": [["lgb"]]},
            # "lgb_params": {
            #     'default_params': {'numIteration': 10}
            # },
            "reader_params": {"cv": 2, "advanced_roles": False}
    },

    "lama_default": {
        "second_model_type": "lama",
        "second_model_params": {
            "cpu_limit": EXTRA_BIG_CPU,
            "memory_limit": int(EXTRA_BIG_MEMORY * 0.95),
            "timeout": 10800,
            "general_params": {"use_algos": [["lgb_tuned"]]},
            "reader_params": {"cv": 5, "advanced_roles": False},
            "tuning_params": {'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
        }
    },

    "longer_lama_default": {
        "second_model_type": "lama",
        "second_model_params": {
            "cpu_limit": EXTRA_BIG_CPU,
            "memory_limit": int(EXTRA_BIG_MEMORY * 0.95),
            "timeout": 10800,
            "general_params": {"use_algos": [["lgb_tuned"]]},
            "reader_params": {"cv": 5, "advanced_roles": False},
            "tuning_params": {'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 7200}
        }
    }
}

SECOND_LEVELS_MODELS_CONFIGS = dict()


def _get_models_params(*model_names: str) -> Dict[str, Any]:
    return {
        MODELNAME2FULLNAME[model_name]: FIRST_LEVELS_MODELS_PARAMS[MODELNAME2FULLNAME[model_name]]
        for model_name in model_names
    }


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    log_path: str
    user_features_path: Optional[str] = None
    item_features_path: Optional[str] = None
    user_cat_features: Optional[List[str]] = None
    item_cat_features: Optional[List[str]] = None


DATASETS = {
    dataset.name: dataset  for dataset in [
        DatasetInfo(
            name="ml100k",
            log_path="/opt/spark_data/replay/ml100k_ratings.csv",
            user_features_path="/opt/spark_data/replay/ml100k_users.csv",
            item_features_path="/opt/spark_data/replay/ml100k_items.csv",
            user_cat_features=["gender", "age", "occupation", "zip_code"],
            item_cat_features=['title', 'release_date', 'imdb_url', 'unknown',
                               'Action', 'Adventure', 'Animation', "Children's",
                               'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                               'Sci-Fi', 'Thriller', 'War', 'Western']
        ),

        DatasetInfo(
            name="ml1m",
            log_path="/opt/spark_data/replay/ml1m_ratings.csv",
            user_features_path="/opt/spark_data/replay/ml1m_users.csv",
            item_features_path="/opt/spark_data/replay/ml1m_items.csv",
            user_cat_features=["gender", "age", "occupation", "zip_code"],
            item_cat_features=['Crime', 'Romance', 'Thriller', 'Adventure',
                               "Children's", 'Drama', 'War', 'Documentary',
                               'Fantasy', 'Mystery', 'Musical', 'Animation',
                               'Film-Noir', 'Horror', 'Western', 'Comedy',
                               'Action', 'Sci-Fi']
        ),

        DatasetInfo(
            name="ml25m",
            log_path="/opt/spark_data/replay/ml25m_ratings.csv"
        ),

        DatasetInfo(
            name="msd",
            log_path="/opt/spark_data/replay_datasets/MillionSongDataset/original.parquet"
        )
    ]
}


@dataclass
class ArtifactPaths:
    base_path: str
    dataset: DatasetInfo
    uid: str = f"{uuid.uuid4()}".replace('-', '')
    partial_train_prefix: str = "partial_train"
    partial_predict_prefix: str = "partial_predict"
    second_level_model_prefix: str = "second_level_model"
    second_level_predicts_prefix: str = "second_level_predicts"
    first_level_train_predefined_path: Optional[str] = None
    second_level_positives_predefined_path: Optional[str] = None

    template_fields: Sequence[str] = ("base_path",)

    def _fs_prefix(self, path: str) -> str:
        return path

    @property
    def train_path(self) -> str:
        return self._fs_prefix(os.path.join(self.base_path, "train.parquet"))

    @property
    def test_path(self) -> str:
        return self._fs_prefix(os.path.join(self.base_path, "test.parquet"))

    @property
    def two_stage_scenario_path(self) -> str:
        return self._fs_prefix(os.path.join(self.base_path, "two_stage_scenario"))

    @property
    def partial_train_paths(self) -> List[str]:
        if not os.path.exists(self.base_path):
            return []

        return sorted([
            self._fs_prefix(os.path.join(self.base_path, path))
            for path in os.listdir(self.base_path)
            if path.startswith(self.partial_train_prefix)
        ])

    @property
    def partial_predicts_paths(self) -> List[str]:
        if not os.path.exists(self.base_path):
            return []

        return sorted([
            self._fs_prefix(os.path.join(self.base_path, path))
            for path in os.listdir(self.base_path)
            if path.startswith(self.partial_predict_prefix)
        ])

    @property
    def full_second_level_train_path(self) -> str:
        return self._fs_prefix(os.path.join(self.base_path, "full_second_level_train.parquet"))

    @property
    def full_second_level_predicts_path(self) -> str:
        return self._fs_prefix(os.path.join(self.base_path, "full_second_level_predicts.parquet"))

    @property
    def log(self) -> DataFrame:
        if self.dataset.log_path.endswith('.csv'):
            return self._get_session().read.csv(self.dataset.log_path, header=True)

        if self.dataset.log_path.endswith('.parquet'):
            return self._get_session().read.parquet(self.dataset.log_path)

        raise Exception("Unsupported format of the file, only csv and parquet are supported")

    @property
    def user_features(self) -> Optional[DataFrame]:
        if self.dataset.user_features_path is None:
            return None

        return (
            self._get_session().read.csv(self.dataset.user_features_path, header=True)
            .withColumnRenamed('user_id', 'user_idx')
            .withColumn('user_idx', sf.col('user_idx').cast('int'))
            .drop('_c0')
        )

    @property
    def item_features(self) -> Optional[DataFrame]:
        if self.dataset.item_features_path is None:
            return None

        return (
            self._get_session().read.csv(self.dataset.item_features_path, header=True)
            .withColumnRenamed('item_id', 'item_idx')
            .withColumn('item_idx', sf.col('item_idx').cast('int'))
            .drop('_c0')
        )

    @property
    def train(self) -> DataFrame:
        return self._get_session().read.parquet(self.train_path)

    @property
    def test(self) -> DataFrame:
        return self._get_session().read.parquet(self.test_path)

    @property
    def full_second_level_train(self) -> DataFrame:
        return self._get_session().read.parquet(self.full_second_level_train_path)

    @property
    def full_second_level_predicts(self) -> DataFrame:
        return self._get_session().read.parquet(self.full_second_level_predicts_path)

    @property
    def first_level_train_path(self) -> str:
        path = self.first_level_train_predefined_path if self.first_level_train_predefined_path is not None \
            else os.path.join(self.base_path, "first_level_train.parquet")

        return self._fs_prefix(path)

    @property
    def second_level_positives_path(self) -> str:
        path = self.second_level_positives_predefined_path if self.second_level_positives_predefined_path is not None \
            else os.path.join(self.base_path, "second_level_positives.parquet")

        return self._fs_prefix(path)

    @property
    def user_features_transformer_path(self) -> str:
        return self._fs_prefix(os.path.join(self.base_path, "user_features_transformer"))

    @property
    def item_features_transformer_path(self) -> str:
        return self._fs_prefix(os.path.join(self.base_path, "item_features_transformer"))

    @property
    def history_based_transformer_path(self):
        return self._fs_prefix(os.path.join(self.base_path, "history_based_transformer"))

    def partial_two_stage_scenario_path(self, model_cls_name: str) -> str:
        return self._fs_prefix(os.path.join(self.base_path, f"two_stage_scenario_{model_cls_name.split('.')[-1]}_{self.uid}"))

    def model_path(self, model_cls_name: str) -> str:
        return self._fs_prefix(os.path.join(self.base_path, f"model_{model_cls_name.replace('.', '__')}_{self.uid}"))

    def hnsw_index_path(self, model_cls_name: str) -> str:
        return self._fs_prefix(os.path.join(self.base_path, f"hnsw_model_index_{model_cls_name.replace('.', '__')}_{self.uid}"))

    def partial_train_path(self, model_cls_name: str) -> str:
        return self._fs_prefix(os.path.join(self.base_path,
                            f"{self.partial_train_prefix}_{model_cls_name.replace('.', '__')}_{self.uid}.parquet"))

    def partial_predicts_path(self, model_cls_name: str):
        return self._fs_prefix(os.path.join(self.base_path,
                            f"{self.partial_predict_prefix}_{model_cls_name.replace('.', '__')}_{self.uid}.parquet"))

    def second_level_model_path(self, model_name: str) -> str:
        return self._fs_prefix(os.path.join(self.base_path, f"{self.second_level_model_prefix}_{model_name}"))

    def second_level_predicts_path(self, model_name: str) -> str:
        return self._fs_prefix(os.path.join(self.base_path, f"{self.second_level_predicts_prefix}_{model_name}.parquet"))

    def make_path(self, relative_path: str) -> str:
        return self._fs_prefix(os.path.join(self.base_path, relative_path))

    def _get_session(self) -> SparkSession:
        return SparkSession.getActiveSession()
