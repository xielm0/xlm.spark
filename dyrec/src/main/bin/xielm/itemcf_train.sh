#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
echo ${dt}
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad

#spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
spark-submit --master yarn-client \
 --conf spark.app.name=CF.app \
 --num-executors 50 \
 --driver-memory 8g \
 --executor-memory 12g \
 --executor-cores 4 \
 --class com.jd.szad.CF.app dyrec.jar \
 train  1000  app.db/app_szad_m_dyrec_itemcf_model
