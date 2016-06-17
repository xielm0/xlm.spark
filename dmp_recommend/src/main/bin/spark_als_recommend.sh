#!/usr/bin/env bash

sku_id = $1
user_num =$2
user_type = $3
#sku_id =


spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
 --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true  --conf spark.dynamicAllocation.maxExecutors=100 \
 --executor-memory 8g --executor-cores 4 \
 --class com.jd.szad.als.als dmp_recommend.jar \
 $sku_id $user_num $user_type
