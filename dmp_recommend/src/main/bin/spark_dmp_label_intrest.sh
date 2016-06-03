#!/usr/bin/env bash
# 获取昨天
# dt=`date -d last-day +%Y-%m-%d`
#获取上月
dt=`date -d "last month" +%Y-%m`
echo 'start spark-job gender_model_apply_jdpin task' $dt

queue=bdp_jmart_adv.bdp_jmart_sz_ad


 spark-submit --master yarn-client
   --class org.apache.spark.mllib.clusteringB.APP
   --num-executors 40 --executor-memory 8g --executor-cores 4
   /home/jd_ad/xlm/dmp1.4.jar
   app.db/app_szad_m_dmp_label_intrest_t2/entrance_type=1  22 38 1.3 1.05  train 400
