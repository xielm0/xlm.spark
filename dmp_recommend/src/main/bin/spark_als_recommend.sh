#!/usr/bin/env bash

dt=`date -d last-day +%Y-%m-%d`
echo $dt

sku_id=$1
user_num=$2
user_type=$3
#sku_id=2600137
#user_num=1000000
#user_type=1

source /home/jd_ad/.bashrc
jars=$(echo $SPARK_HOME/lib/datanucleus*.jar | tr ' ' ',')
dir='/data0/task/edw/etl/spark/dmp_recommend'
cd ${dir}

#spark-submit --master yarn-client \
spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
 --jars $jars \
 --conf "spark.driver.extraJavaOptions=-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/data1/yarn-logs/am_dump -XX:PermSize=1g -XX:MaxPermSize=1g" \
 --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=10 \
 --driver-memory 12g \
 --executor-memory 8g \
 --executor-cores 4 \
 --class com.jd.szad.als.als_recommend dmp_recommend.jar \
 $sku_id $user_num $user_type

 # 2600137 1000000 1
