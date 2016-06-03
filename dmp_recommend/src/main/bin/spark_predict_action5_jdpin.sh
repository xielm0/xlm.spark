#!/usr/bin/env bash
# 获取昨天
# dt=`date -d last-day +%Y-%m-%d`
#获取上月
# dt=`date -d "last month" +%Y-%m`


spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
   --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
    --executor-memory 8g --executor-cores 4 \
   --class com.jd.szad.itemcf.predict_itemcf \
   jar/dmp_recommend.jar  \
   app.db/app_szad_m_dmp_itemcf_train_day/action_type=5 1000  app.db/app_szad_m_dmp_recommend_itemcf_res_jdpi/action_type=5