#!/usr/bin/env bash


spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
   --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
    --executor-memory 8g --executor-cores 4 \
   --class com.jd.szad.itemcf.app \
   dmp_recommend.jar  \
   train app.db/app_szad_m_dmp_itemcf_train_day/action_type=1 1000  app.db/app_szad_m_dmp_itemcf_res/action_type=1


   # --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
   # --conf "spark.executor.extraJavaOptions=-XX:+UseParallelGC -XX:+UseParallelOldGC"