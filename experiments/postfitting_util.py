import os

import numpy as np
import pandas as pd

from pyspark.sql import functions as sf, types as st, SparkSession, Window, DataFrame
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType, FloatType
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import IndexToString, StringIndexer
from replay.session_handler import State
from replay.data_preparator import DataPreparator
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info, get_top_k_recs
from replay.models import ALSWrap, Word2VecRec, SLIM
from replay.model_handler import save, load
from replay.metrics import Coverage, HitRate, NDCG, MAP, Precision
from replay.experiment import Experiment
# from rs_datasets import MillionSongDataset, MovieLens
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge

from tqdm.notebook import tqdm
# from utils import filter_seen, calc_precision


spark_sess = (
    SparkSession
    .builder
    .master("local[6]")
    .config('spark.jars.packages', 'com.github.jelmerk:hnswlib-spark_3.3_2.12:1.0.1')
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryoserializer.buffer.max", "512m")
    .config("spark.driver.memory", "64g")
    .config("spark.executor.memory", "64g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.local.dir", "/tmp")
    .getOrCreate()

)

spark = State(spark_sess).session
spark.sparkContext.setLogLevel("ERROR")

# DS_PATH = "/opt/spark_data/replay_datasets/MovieLens/posttraining/"
DS_PATH = "/mnt/ess_storage/DN_1/storage/SLAMA/kaggle_used_cars_dataset/replay_datasets/MovieLens/posttraining/"

check_types_dict = {0: "old_users_old_items", # new interactions with old users and items
                    1: "new_users_old_items",
                    2: "old_users_new_items",
                    4: "new_users_new_items", # all interactions with new users and items
                    5: "all"} # all new log

schema = StructType() \
      .add("relevance", IntegerType(),True) \
      .add("timestamp", IntegerType(),True) \
      .add("user_idx", IntegerType(),True) \
      .add("item_idx", IntegerType(),True)


def postfit_check(Model, ds='ml1m', check_type=0, k=1000):


    if ds == 'ml1m':
        postfix = '_ml1m' #for ml 1m
    elif ds == 'ml10m':
        postfix = ''  # for ml10m
    else:
        print("incorrect ds name, try one of ['ml1m', 'ml10m']")
        return 0


    train70 = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"train70{postfix}.csv"))

    train80 = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"train80{postfix}.csv"))
    train_diff80 = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"train_dif80{postfix}.csv"))

    train90 = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"train90{postfix}.csv"))
    train_diff90 = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"train_dif90{postfix}.csv"))

    train100 = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"train{postfix}.csv"))
    train_diff100 = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"train_dif100{postfix}.csv"))

    test = spark.read.option("header", True).format("csv").schema(schema).load(
        os.path.join(DS_PATH, f"test{postfix}.csv"))


    print(get_log_info(train70))
    print(get_log_info(train_diff80))
    print(get_log_info(test))


    if check_type == 0:
        old_users = train70.select('user_idx').distinct().toPandas()['user_idx'].values
        old_items = train70.select('item_idx').distinct().toPandas()['item_idx'].values

        train80 = train80.filter(
            (train80.user_idx.isin(old_users.tolist())) & (train80.item_idx.isin(old_items.tolist())))

        train_diff80 = train_diff80.filter(
            (train_diff80.user_idx.isin(old_users.tolist())) & (train_diff80.item_idx.isin(old_items.tolist())))

        train90 = train90.filter(
            (train90.user_idx.isin(old_users.tolist())) & (train90.item_idx.isin(old_items.tolist())))

        train_diff90 = train_diff90.filter(
            (train_diff90.user_idx.isin(old_users.tolist())) & (train_diff90.item_idx.isin(old_items.tolist())))

        train100 = train100.filter(
            (train100.user_idx.isin(old_users.tolist())) & (train100.item_idx.isin(old_items.tolist())))

        train_diff100 = train_diff100.filter(
            (train_diff100.user_idx.isin(old_users.tolist())) & (train_diff100.item_idx.isin(old_items.tolist())))

        test1 = test.union(train_diff90).union(train_diff100)
        test2 = test.union(train_diff100)
        test3 = test

    elif check_type == 1:
        # TODO
        pass

    elif check_type == 2:
        # TODO
        pass

    elif check_type == 3:
        # TODO
        pass
    elif check_type == 4:
        # TODO
        pass
    elif check_type == 5:
        # TODO
        pass
    else:
        print('unknown check type')


    # Stage 1 (70+10 VS 80)

    # refit model
    model = Model()
    model.fit(log=train70)
    model.refit(log=train_diff80)

    # make a predictions only for train & test intersection users
    train_test_users80 = test1.select('user_idx').distinct().join(train80.select('user_idx').distinct(), how='inner',
                                                                 on='user_idx').select('user_idx')

    predict_refit = model.predict(log=train80, k=k, users=train_test_users80, filter_seen_items=True)
    predict_refit = predict_refit.cache()
    predict_refit.count()

    # fit model
    model1 = Model()
    model1.fit(log=train80)
    predict_fit = model1.predict(log=train80, k=k, users=train_test_users80, filter_seen_items=True)
    predict_fit = predict_fit.cache()
    predict_fit.count()

    prediction_quality1 = Experiment(test1, {NDCG(): [5, 10, 25, 100, 500, 1000],
                                           MAP(): [5, 10, 25, 100, 500, 1000],
                                           HitRate(): [5, 10, 25, 100, 500, 1000],
                                           })
    prediction_quality1.add_result('fit_stage1', predict_fit)
    prediction_quality1.add_result('refit_stage1', predict_refit)

    # TODO: make a comparison assert

    # Stage 2 (70+10+10 VS 90):
    model.refit(log=train_diff90)
    train_test_users90 = test2.select('user_idx').distinct().join(train90.select('user_idx').distinct(), how='inner',
                                                                  on='user_idx').select('user_idx')
    predict_refit = model.predict(log=train90, k=k, users=train_test_users90, filter_seen_items=True)
    predict_refit = predict_refit.cache()
    predict_refit.count()

    # fit model
    model1 = Model()
    model1.fit(log=train90)
    predict_fit = model1.predict(log=train90, k=k, users=train_test_users90, filter_seen_items=True)
    predict_fit = predict_fit.cache()
    predict_refit.count()

    prediction_quality2 = Experiment(test2, {NDCG(): [5, 10, 25, 100, 500, 1000],
                                           MAP(): [5, 10, 25, 100, 500, 1000],
                                           HitRate(): [5, 10, 25, 100, 500, 1000],
                                           })
    prediction_quality2.add_result('fit_stage2', predict_fit)
    prediction_quality2.add_result('refit_stage2', predict_refit)

    # TODO: make a comparison assert

    # Stage 3 (70+10+10+10 VS 100):

    model.refit(log=train_diff100)
    train_test_users100 = test3.select('user_idx').distinct().join(train100.select('user_idx').distinct(), how='inner',
                                                                  on='user_idx').select('user_idx')
    predict_refit = model.predict(log=train100, k=k, users=train_test_users100, filter_seen_items=True)
    predict_refit = predict_refit.cache()
    predict_refit.count()

    model1 = Model()
    model1.fit(log=train100)
    predict_fit = model1.predict(log=train100, k=k, users=train_test_users100, filter_seen_items=True)
    predict_fit = predict_fit.cache()
    predict_refit.count()

    prediction_quality3 = Experiment(test3, {NDCG(): [5, 10, 25, 100, 500, 1000],
                                           MAP(): [5, 10, 25, 100, 500, 1000],
                                           HitRate(): [5, 10, 25, 100, 500, 1000],
                                           })
    prediction_quality3.add_result('fit_stage3', predict_fit)
    prediction_quality3.add_result('refit_stage3', predict_refit)

    # TODO: make a comparison assert



for i in range(5):

    postfit_check(Model=model, check_type=i)

