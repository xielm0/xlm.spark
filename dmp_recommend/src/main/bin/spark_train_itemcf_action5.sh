#!/usr/bin/env bash

dt=`date -d last-day +%Y-%m-%d`
queue=bdp_jmart_adv.bdp_jmart_sz_ad

spark-submit --master yarn-client \
   --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
   --executor-memory 10g --executor-cores 4 \
   --class com.jd.szad.itemcf.app  dmp_recommend.jar \
   train app.db/app_szad_m_dmp_itemcf_train_day/action_type=5 800  app.db/app_szad_m_dmp_itemcf_res/action_type=5
