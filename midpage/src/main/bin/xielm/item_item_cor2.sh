#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo ${dt}


#--conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true \
#--conf spark.dynamicAllocation.maxExecutors=100 --conf spark.dynamicAllocation.minExecutors=50 \
#spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
spark-submit --master yarn-client \
 --conf spark.app.name=item_condition_prob \
 --num-executors 5 \
 --driver-memory 4g \
 --executor-memory 8g \
 --executor-cores 4 \
 --class com.jd.szad.midpage.item_condition_prob midpage.jar \
${dt} 20
