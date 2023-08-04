#!/bin/bash
spark-submit --master yarn CS643-WinePrediction/WineTraining.py
spark-submit --master yarn CS643-WinePrediction/WineTesting.py
