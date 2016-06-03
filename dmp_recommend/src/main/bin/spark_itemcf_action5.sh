#!/usr/bin/env bash


spark-submit --master yarn-client \
   --num-executors 80 --executor-memory 10g --executor-cores 4 \
   --class com.jd.szad.catecf.itemCF \
   jar/dmp_recommend.jar  \
   app.db/app_szad_m_dmp_itemcf_train_day/action_type=5 800  app.db/app_szad_m_dmp_itemcf_res/action_type=5