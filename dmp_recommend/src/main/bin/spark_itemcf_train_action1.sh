#!/usr/bin/env bash

dt=`date -d last-day +%Y-%m-%d`
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo $dt

source /home/jd_ad/.bashrc
jars=$(echo $SPARK_HOME/lib/datanucleus*.jar | tr ' ' ',')

dir='/home/jd_ad/spark_task/dmp_recommend'
cd ${dir}
#spark-submit --master yarn-client \
spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
 --conf spark.app.name=itemcf_train1 \
 --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
 --driver-memory 4g \
 --executor-memory 8g \
 --executor-cores 4 \
 --class com.jd.szad.itemcf.app dmp_recommend.jar \
 train  app.db/app_szad_m_dmp_itemcf_train_day/action_type=1 1000  app.db/app_szad_m_dmp_itemcf_res/action_type=1



