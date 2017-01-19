#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo ${dt}


#spark-submit --master yarn-client \
spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
   --conf spark.app.name=itemcf_predict1 \
   --num-executors 100 \
   --executor-memory 8g \
   --executor-cores 4 \
   --class com.jd.szad.itemcf.compute_itemcor midpage.jar \
   predict app.db/app_szad_m_midpage_uid_item_train_day_jdpin_t/type=30  500  \
   app.db/app_szad_m_midpage_item_item_cor1/dt=${dt} ${dt}
