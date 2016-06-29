#!/usr/bin/env bash
# 获取昨天
# dt=`date -d last-day +%Y-%m-%d`
#获取上月
# dt=`date -d "last month" +%Y-%m`
dt=`date -d last-day +%Y-%m-%d`
echo $dt

source /home/jd_ad/.bashrc
jars=$(echo $SPARK_HOME/lib/datanucleus*.jar | tr ' ' ',')
dir='/data0/task/edw/etl/spark/dmp_recommend'
cd ${dir}

#spark-submit --master yarn-client \
spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
   --jars ${jars} \
   --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
   --executor-memory 8g \
   --executor-cores 4 \
   --class com.jd.szad.itemcf.app dmp_recommend.jar \
   predict app.db/app_szad_m_dmp_itemcf_apply_day_jdpin/action_type=1  400  \
   app.db/app_szad_m_dmp_itemcf_res/action_type=1  app.db/app_szad_m_dmp_recommend_itemcf_res_jdpin/action_type=1