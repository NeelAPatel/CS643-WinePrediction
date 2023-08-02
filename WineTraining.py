# pip install pyspark findspark boto3 numpy pandas scikit-learn datetime

# Name: Neel Patel 
# ID: nap48
# Class: CS643 - Cloud Computing - Project 2 Wine Quality Prediction

import findspark
findspark.find()
findspark.init()

from datetime import datetime
from io import StringIO
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import RandomForest, RandomForestModel, DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.session import SparkSession	
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , f1_score
import boto3
import numpy as np
import os
import pandas as pd
import pyspark
import shutil
import sys


def main():
    print(">>>>>> PROGRAM STARTS")

    # s3_path_train = sys.argv[1]
    # s3_path_val = sys.argv[2]
    conf = SparkConf().setAppName('NP-CS643-WineQuality-Training')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    s3Client = boto3.client('s3')
    #s3Client = ""
    bucket_name = 'neel-cs643'
    train_key = 'TrainingDataset.csv'

    df_train = spark.read.csv("s3a://"+bucket_name+"/"+train_key, header=True, sep=";")
    df_train.printSchema() # Column info
    df_train.show()


    df_train = df_train.withColumnRenamed('""""quality"""""', "myLabel")

    # Removing the quotes from column names
    for column in df_train.columns:
        df_train = df_train.withColumnRenamed(column, column.replace('"', ''))

    # Converting to double/integer
    for idx, col_name in enumerate(df_train.columns):
        if idx not in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("double"))
        elif idx in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("integer"))


    # Convert DataFrame to RDD
    df_train = df_train.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))


    print("Training RandomForest model...")
    model_rf = RandomForest.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                    numTrees=10, featureSubsetStrategy="auto",
                                    impurity='gini', maxDepth=10, maxBins=32)
    print("Model - Randomforest Created")
    
    # Save RandomForest model
    model_path = "s3a://"+bucket_name+"/model_rf.model"
    model_rf.save(sc, model_path)

    print(">>>>>Random Forest model saved")


    print("Training DecisionTree model...")
    model_dt = DecisionTree.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                            impurity='gini', maxDepth=10, maxBins=32)
    print("Model - DecisionTree Created")

    # Save DecisionTree model
    model_path = "s3a://"+bucket_name+"/model_dt.model"
    model_dt.save(sc, model_path)

    print(">>>>> DecisionTree model saved")


if __name__ == "__main__": main()