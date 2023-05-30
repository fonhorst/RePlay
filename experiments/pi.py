from pyspark.sql import SparkSession
import sparklightautoml
import replay

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    result = spark.sparkContext.parallelize(list(range(5))).sum()

    print(f"Successful run: {result}")
