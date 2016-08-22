#!/usr/bin/env bash
# 获取昨天
# dt=`date -d last-day +%Y-%m-%d`
#获取上月
# dt=`date -d "last month" +%Y-%m`
dt=`date -d last-day +%Y-%m-%d`
echo $dt

source /home/jd_ad/.bashrc
jars=$(echo $SPARK_HOME/lib/datanucleus*.jar | tr ' ' ',')
dir='/home/jd_ad/spark_task/dmp_recommend'
cd ${dir}

#spark-submit --master yarn-client \
spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
   --conf spark.app.name=itemcf_predict1 \
   --num-executors 50 \
   --executor-memory 8g \
   --executor-cores 4 \
   --class com.jd.szad.itemcf.app dmp_recommend.jar \
   predict app.db/app_szad_m_dmp_itemcf_apply_day_jdpin/action_type=1  500  \
   app.db/app_szad_m_dmp_itemcf_res/action_type=1  app.db/app_szad_m_dmp_recommend_itemcf_res_jdpin/action_type=1