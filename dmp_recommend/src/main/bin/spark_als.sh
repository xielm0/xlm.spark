#!/usr/bin/env bash

#spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
spark-submit --master yarn-client \
 --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
 --executor-memory 8g --executor-cores 4 \
 --conf spark.shuffle.memoryFraction=0.3 \
 --class com.jd.szad.als.als dmp_recommend.jar \
 app.db/app_szad_m_dmp_als_train/user_type=1 20 5 800 app.db/app_szad_m_dmp_als_res/user_type=1 app.db/app_szad_m_dmp_als_model/user_type=1

 # app.db/app_szad_m_dmp_als_train/user_type=1/000000_0.lzo 20 5 20 app.db/app_szad_m_dmp_als_res/user_type=1 app.db/app_szad_m_dmp_als_model/user_type=1

