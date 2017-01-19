#!/usr/bin/env bash

spark-submit --master yarn-client \
 --conf spark.app.name=userlabel_train \
 --num-executors 80 \
 --driver-memory 4g \
 --executor-memory 8g \
 --executor-cores 4 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 train


 spark-submit --master yarn-client \
 --conf spark.app.name=userlabel \
 --conf spark.buffer.pageSize=16m \
 --conf spark.shuffle.memoryFraction=0.3 \
 --num-executors 80 \
 --driver-memory 4g \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 sql_predict