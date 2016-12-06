#!/usr/bin/env bash

spark-submit --master yarn-client \
 --conf spark.app.name=user_label \
 --num-executors 10 \
 --driver-memory 2g \
 --executor-memory 8g \
 --executor-cores 4 \
 --class com.jd.szad.user.user_label midpage.jar \
 train


 spark-submit --master yarn-client \
 --conf spark.app.name=user_label \
 --conf spark.buffer.pageSize=16m \
 --conf spark.shuffle.memoryFraction=0.3 \
 --num-executors 80 \
 --driver-memory 4g \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.user.user_label midpage.jar \
 sql_predict