#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
echo ${dt}
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad

#spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
spark-submit --master yarn-client \
 --conf spark.app.name=compute_item_cor \
 --num-executors 100 \
 --driver-memory 8g \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.itemcf.compute_itemcor midpage.jar \
 train  app.db/app_szad_m_midpage_jd_itemcf_train_day/action_type=1 1000  app.db/app_szad_m_midpage_item_item_cor1_res ${dt}

