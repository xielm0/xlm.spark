#!/usr/bin/env bash

#spark2.0
source /data/bossapp/software/bashrc

spark-submit --master yarn-client \
 --conf spark.app.name=userlabel_train \
 --driver-memory 4g \
 --num-executors 100 \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 train


