# pylint: disable=wildcard-import,invalid-name,unused-wildcard-import,unspecified-encoding
import json
import os
import shutil
from inspect import getfullargspec
from os.path import exists, join

import pyspark.sql.types as st
from pyspark.ml.feature import StringIndexerModel, IndexToString

from replay.data_preparator import Indexer
from replay.models import *
from replay.models.base_rec import BaseRecommender
from replay.session_handler import State
from replay.splitters import *
from replay.utils import save_picklable_to_parquet, load_pickled_from_parquet


def prepare_dir(path):
    """
    Create empty `path` dir
    """
    if exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def save(model: BaseRecommender, path: str, overwrite: bool = False):
    """
    Save fitted model to disk as a folder

    :param model: Trained recommender
    :param path: destination where model files will be stored
    :return:
    """
    spark = State().session

    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    if not overwrite:
        is_exists = fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))
        if is_exists:
            raise FileExistsError(f"Path '{path}' already exists. Mode is 'overwrite = False'.")

    fs.mkdirs(spark._jvm.org.apache.hadoop.fs.Path(join(path, "model")))
    model._save_model(join(path, "model"))

    init_args = model._init_args
    init_args["_model_name"] = str(model)
    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    df.coalesce(1).write.mode("overwrite").json(join(path, "init_args.json"))

    dataframes = model._dataframes
    df_path = join(path, "dataframes")
    for name, df in dataframes.items():
        if df is not None:
            df.write.parquet(join(df_path, name))
    model.fit_users.write.mode("overwrite").parquet(join(df_path, "fit_users"))
    model.fit_items.write.mode("overwrite").parquet(join(df_path, "fit_items"))

    save_picklable_to_parquet(model.study, join(path, "study"))


def load(path: str) -> BaseRecommender:
    """
    Load saved model from disk

    :param path: path to model folder
    :return: Restored trained model
    """
    spark = State().session
    args = spark.read.json(join(path, "init_args.json")).first().asDict(recursive=True)
    name = args["_model_name"]
    del args["_model_name"]

    model_class = globals()[name]
    init_args = getfullargspec(model_class.__init__).args
    init_args.remove("self")
    extra_args = set(args) - set(init_args)
    if len(extra_args) > 0:
        extra_args = {key: args[key] for key in args}
        init_args = {key: args[key] for key in init_args}
    else:
        init_args = args
        extra_args = {}

    model = model_class(**init_args)
    for arg in extra_args:
        model.arg = extra_args[arg]

    df_path = join(path, "dataframes")
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    statuses = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(df_path))
    dataframes_paths = [str(f.getPath()) for f in statuses]
    for dataframe_path in dataframes_paths:
        df = spark.read.parquet(dataframe_path)
        attr_name = dataframe_path.split("/")[-1]
        setattr(model, attr_name, df)

    model._load_model(join(path, "model"))
    model.study = load_pickled_from_parquet(join(path, "study"))

    return model


def save_indexer(indexer: Indexer, path: str, overwrite: bool = False):
    """
    Save fitted indexer to disk as a folder

    :param indexer: Trained indexer
    :param path: destination where indexer files will be stored
    """
    spark = State().session
    
    if not overwrite:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        is_exists = fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))
        if is_exists:
            raise FileExistsError(f"Path '{path}' already exists. Mode is 'overwrite = False'.")

    init_args = indexer._init_args
    init_args["user_type"] = str(indexer.user_type)
    init_args["item_type"] = str(indexer.item_type)
    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    df.coalesce(1).write.mode("overwrite").json(join(path, "init_args.json"))

    indexer.user_indexer.write().overwrite().save(join(path, "user_indexer"))
    indexer.item_indexer.write().overwrite().save(join(path, "item_indexer"))
    indexer.inv_user_indexer.write().overwrite().save(join(path, "inv_user_indexer"))
    indexer.inv_item_indexer.write().overwrite().save(join(path, "inv_item_indexer"))


def load_indexer(path: str) -> Indexer:
    """
    Load saved indexer from disk

    :param path: path to folder
    :return: restored Indexer
    """
    spark = State().session
    args = spark.read.json(join(path, "init_args.json")).first().asDict()

    user_type = args["user_type"]
    del args["user_type"]
    item_type = args["item_type"]
    del args["item_type"]

    indexer = Indexer(**args)

    indexer.user_type = getattr(st, user_type)()
    indexer.item_type = getattr(st, item_type)()

    indexer.user_indexer = StringIndexerModel.load(join(path, "user_indexer"))
    indexer.item_indexer = StringIndexerModel.load(join(path, "item_indexer"))
    indexer.inv_user_indexer = IndexToString.load(
        join(path, "inv_user_indexer")
    )
    indexer.inv_item_indexer = IndexToString.load(
        join(path, "inv_item_indexer")
    )

    return indexer


def save_splitter(splitter: Splitter, path: str, overwrite: bool = False):
    """
    Save initialized splitter

    :param splitter: Initialized splitter
    :param path: destination where splitter files will be stored
    """
    init_args = splitter._init_args
    init_args["_splitter_name"] = str(splitter)
    spark = State().session
    sc = spark.sparkContext
    df = spark.read.json(sc.parallelize([json.dumps(init_args)]))
    if overwrite:
        df.coalesce(1).write.mode("overwrite").json(join(path, "init_args.json"))
    else:
        df.coalesce(1).write.json(join(path, "init_args.json"))


def load_splitter(path: str) -> Splitter:
    """
    Load splitter

    :param path: path to folder
    :return: restored Splitter
    """
    spark = State().session
    args = spark.read.json(join(path, "init_args.json")).first().asDict()
    name = args["_splitter_name"]
    del args["_splitter_name"]
    splitter = globals()[name]
    return splitter(**args)
