#!/usr/bin/env bash

source /data/bossapp/software/bashrc

spark-submit --master yarn-client \
 --conf spark.app.name=userlabel_train \
 --driver-memory 4g \
 --num-executors 100 \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 train


 spark-submit --master yarn-client \
 --conf spark.app.name=userlabel \
 --conf spark.buffer.pageSize=16m \
 --conf spark.shuffle.memoryFraction=0.3 \
 --driver-memory 4g \
 --num-executors 100 \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 sql_predict

 spark-shell --num-executors 10 --driver-memory 2g  --executor-memory 8g  --executor-cores 4