
# pip install pyspark findspark boto3 numpy pandas scikit-learn datetime
#AWS S3
#ML Libs
#Normal Libs
#Package pyspark
import findspark
findspark.find()
findspark.init()
from datetime import datetime
from io import StringIO
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics, RegressionMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import RandomForest, RandomForestModel, DecisionTreeModel
from pyspark.mllib.tree import RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession	
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import boto3
import numpy as np
import os
import pandas as pd
import pyspark
import shutil
import sys

def main():

    sc, spark = sparkInit()
    model_dt, model_rf = loadModels(sc)
    validation = loadCleanData(spark)
    predictionTesting(validation, model_dt, model_rf)

def sparkInit(): 
    # Initialize Spark
    conf = SparkConf().setAppName("Model Testing")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    return sc, spark

def loadModels(sc):
    # Define S3 bucket
    s3 = boto3.client('s3')
    bucket_name = 'neel-cs643'
    model_dt_path = "s3a://"+bucket_name+"/models/model_dt.model"
    model_rf_path = "s3a://"+bucket_name+"/models/model_rf.model"

    # Load the models
    model_dt = DecisionTreeModel.load(sc, model_dt_path)
    print(">>>>>DT Model Loaded")

    # Load the models
    model_rf = RandomForestModel.load(sc, model_rf_path)
    print(">>>>>RF Model Loaded")
    return model_dt, model_rf

def loadCleanData(spark):
    bucket_name = 'neel-cs643'
    file_key = "ValidationDataset.csv"
    dataset_path = "s3a://"+bucket_name+"/"+file_key
    # Load the validation dataset
    validation = spark.read.csv(dataset_path, inferSchema=True, header=True, sep=';')
    validation = validation.withColumnRenamed('""""quality"""""', "myLabel")
    print(">>>>>Quality Column renamed")

    # Removing the quotes from column names
    for column in validation.columns:
        validation = validation.withColumnRenamed(column, column.replace('"', ''))

    # Converting to double/integer
    for idx, col_name in enumerate(validation.columns):
        if idx not in [6 - 1, 7 - 1, len(validation.columns) - 1]:
            validation = validation.withColumn(col_name, col(col_name).cast("double"))
        
        if idx in [6 - 1, 7 - 1, len(validation.columns) - 1]:
            validation = validation.withColumn(col_name, col(col_name).cast("integer"))
    print(">>>>>Data Cleaned, printing TestData...")
    validation.printSchema()
    validation.show()
    return validation

def predictionTesting(validation, model_dt, model_rf):
    # Split features and labels
    validation_DT = validation.rdd.map(lambda row: (float(row[-1]), [float(feature) for feature in row[:-1]]))
    validation_RF = validation.rdd.map(lambda row: (float(row[-1]), [float(feature) for feature in row[:-1]]))
    print(">>>>>Splitted features and labels")

    predictions_dt = model_dt.predict(validation_DT.map(lambda x: x[1]))
    labelsAndPredictions_dt = validation_DT.map(lambda lp: lp[0]).zip(predictions_dt)
    print(">>>>>DT Predictions complete")
    predictions_rf = model_rf.predict(validation_RF.map(lambda x: x[1]))
    labelsAndPredictions_rf = validation_RF.map(lambda lp: lp[0]).zip(predictions_rf)
    print(">>>>>RF Predictions complete")

    metrics_dt = MulticlassMetrics(labelsAndPredictions_dt)
    metrics_rf = MulticlassMetrics(labelsAndPredictions_rf)
    print(f'Decision Tree Model - Accuracy: {metrics_dt.accuracy}, F1 Score: {metrics_dt.weightedFMeasure()}')
    print(f'Random Forest Model - Accuracy: {metrics_rf.accuracy}, F1 Score: {metrics_rf.weightedFMeasure()}')

    print(">>>>>PROGRAM END")


if __name__ == "__main__": main()

# # Split features and labels
# validation_DT = validation.rdd.map(lambda row: (float(row[-1]), [float(feature) for feature in row[:-1]]))
# print(">>>>>Splitted features and labels")
# predictions_dt = model_dt.predict(validation_DT.map(lambda x: x[1]))
# labelsAndPredictions_dt = validation_DT.map(lambda lp: lp[0]).zip(predictions_dt)
# print(">>>>>DT Predictions complete")

# metrics_dt = MulticlassMetrics(labelsAndPredictions_dt)
# print(f'Decision Tree Model - Accuracy: {metrics_dt.accuracy}, F1 Score: {metrics_dt.weightedFMeasure()}')


# # Split features and labels
# validation_RF = validation.rdd.map(lambda row: (float(row[-1]), [float(feature) for feature in row[:-1]]))
# print(">>>>>Splitted features and labels")
# predictions_rf = model_rf.predict(validation_RF.map(lambda x: x[1]))
# labelsAndPredictions_rf = validation_RF.map(lambda lp: lp[0]).zip(predictions_rf)
# print(">>>>>RF Predictions complete")

# metrics_rf = MulticlassMetrics(labelsAndPredictions_rf)
# print(f'Random Forest Model - Accuracy: {metrics_rf.accuracy}, F1 Score: {metrics_rf.weightedFMeasure()}')