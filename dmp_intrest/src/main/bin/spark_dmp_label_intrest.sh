#!/usr/bin/env bash
# 获取昨天
# dt=`date -d last-day +%Y-%m-%d`
#获取上月
dt=`date -d "last month" +%Y-%m`
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo 'start spark-job dmp_intrest task' $dt


# spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
 spark-submit --master yarn-client \
   --num-executors 40 --executor-memory 8g --executor-cores 4 \
   --class org.apache.spark.mllib.clusteringB.APP  dmp_intrest.jar \
   app.db/app_szad_m_dmp_label_intrest_t2/entrance_type=1  22 38 1.3 1.05  train 600


