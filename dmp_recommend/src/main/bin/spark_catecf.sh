#!/usr/bin/env bash


spark-submit --master yarn-client \
   --num-executors 20 --executor-memory 10g --executor-cores 4 \
   --class com.jd.szad.catecf.CateCF midpage.jar