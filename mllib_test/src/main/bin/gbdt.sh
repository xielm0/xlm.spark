#!/usr/bin/env bash

spark-submit --master yarn-client  \
   --num-executors 50 \
   --executor-memory 8g \
   --executor-cores 4 \
   --class com.jd.szad.itemcf.compute_itemcor midpage.jar \