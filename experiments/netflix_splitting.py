import logging
import uuid

from experiments.dag_entities import ArtifactPaths, DatasetInfo
from experiments.dag_utils import _init_spark_session
from pyspark.sql import functions as sf

from replay.data_preparator import DataPreparator
from replay.splitters import DateSplitter
from replay.utils import get_log_info

logger = logging.getLogger(__name__)

partitions_num = 6

dataset = DatasetInfo(
    name="netflix",
    log_path="/opt/data/netflix.csv"
)

artifacts = ArtifactPaths(
    base_path=f"/tmp/{uuid.uuid4()}",
    dataset=dataset
)

with _init_spark_session():
    data = (
        artifacts.log
            .withColumn('user_id', sf.col('user_id').cast('int'))
            .withColumn('item_id', sf.col('item_id').cast('int'))
    )

    if 'timestamp' in data.columns:
        data = data.withColumn('timestamp', sf.col('timestamp').cast('long'))

    # splitting on train and test
    preparator = DataPreparator()

    if artifacts.dataset.name.startswith('ml') or artifacts.dataset.name.startswith('netflix'):
        log = preparator.transform(
            columns_mapping={"user_id": "user_id", "item_id": "item_id",
                             "relevance": "rating", "timestamp": "timestamp"},
            data=data
        ).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")
    else:
        raise Exception(f"Unsupported dataset name: {artifacts.dataset.name}")

    print(get_log_info(log))

    log = log.repartition(partitions_num).cache()
    log.write.mode('overwrite').format('noop').save()

    only_positives_log = log.filter(sf.col('relevance') >= 3).withColumn('relevance', sf.lit(1))
    logger.info(get_log_info(only_positives_log))

    # train/test split ml
    train_spl = DateSplitter(
        test_start=0.2,
        drop_cold_items=True,
        drop_cold_users=True,
    )

    train, test = train_spl.split(only_positives_log)
    logger.info(f'train info: \n{get_log_info(train)}')
    logger.info(f'test info:\n{get_log_info(test)}')