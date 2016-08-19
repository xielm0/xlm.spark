#!/usr/bin/env bash

dt=`date -d last-day +%Y-%m-%d`
echo $dt

source /home/jd_ad/.bashrc
jars=$(echo $SPARK_HOME/lib/datanucleus*.jar | tr ' ' ',')
dir='/home/jd_ad/spark_task/dmp_recommend'
cd ${dir}

#spark-submit --master yarn-client \
spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
 --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
 --executor-memory 8g \
 --executor-cores 4 \
 --conf spark.shuffle.memoryFraction=0.3 \
 --class com.jd.szad.als.als dmp_recommend.jar \
 app.db/app_szad_m_dmp_als_train/user_type=1 20 30 800 app.db/app_szad_m_dmp_als_model/user_type=1


