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
 --num-executors 50 \
 --driver-memory 2g \
 --executor-memory 8g \
 --executor-cores 4 \
 --class com.jd.szad.user.user_label midpage.jar \
 predict