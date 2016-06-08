#!/usr/bin/env bash

 #spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
 spark-submit --master yarn-client  \
 --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.maxExecutors=100 \
  --executor-memory 8g --executor-cores 4 \
 --class com.jd.szad.intrest.lda_jd  dmp_intrest.jar  \
  train  22 36 1.3 1.05 app.db/app_szad_m_dmp_label_intrest_jdpin_train  300 app.db/app_szad_m_dmp_label_intrest_res_jdpin