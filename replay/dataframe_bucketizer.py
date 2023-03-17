from pyspark.ml import Transformer
from pyspark.ml.param import TypeConverters, Params, Param
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import DataFrame, SparkSession


class DataframeBucketizer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    """
    Buckets the input dataframe, dumps it to spark warehouse directory, and returns a bucketed dataframe.
    """

    bucketingKey = Param(
        Params._dummy(),
        "bucketingKey",
        "bucketing key (also used as sort key)",
        typeConverter=TypeConverters.toString,
    )

    partitionNum = Param(
        Params._dummy(),
        "partitionNum",
        "number of buckets",
        typeConverter=TypeConverters.toInt,
    )

    tableName = Param(
        Params._dummy(),
        "tableName",
        "parquet file name (for storage  in 'spark-warehouse') and spark table name",
        typeConverter=TypeConverters.toString,
    )

    sparkWarehouseDir = Param(
        Params._dummy(),
        "sparkWarehouseDir",
        "sparkWarehouseDir",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self, bucketing_key: str, partition_num: int, spark_warehouse_dir: str, table_name: str = ""):
        """Makes bucketed dataframe from input dataframe.

        Args:
            bucketing_key: bucketing key (also used as sort key)
            partition_num: number of buckets
            table_name: parquet file name (for storage  in 'spark-warehouse') and spark table name
            spark_warehouse_dir: spark warehouse dir, i.e. value of 'spark.sql.warehouse.dir' property
        """
        super().__init__()
        self.set(self.bucketingKey, bucketing_key)
        self.set(self.partitionNum, partition_num)
        self.set(self.tableName, table_name)
        self.set(self.sparkWarehouseDir, spark_warehouse_dir)

    def set_table_name(self, table_name: str):
        self.set(self.tableName, table_name)

    def _transform(self, df: DataFrame):
        bucketing_key = self.getOrDefault(self.bucketingKey)
        partition_num = self.getOrDefault(self.partitionNum)
        table_name = self.getOrDefault(self.tableName)
        spark_warehouse_dir = self.getOrDefault(self.sparkWarehouseDir)

        if not table_name:
            raise ValueError("Parameter 'table_name' is not set! Please set it via method 'set_table_name'.")

        (
            df.repartition(partition_num, bucketing_key)
            .write.mode("overwrite")
            .bucketBy(partition_num, bucketing_key)
            .sortBy(bucketing_key)
            .saveAsTable(
                table_name,
                format="parquet",
                path=f"{spark_warehouse_dir}/{table_name}",
            )
        )

        spark = SparkSession.getActiveSession()

        return spark.table(table_name)
