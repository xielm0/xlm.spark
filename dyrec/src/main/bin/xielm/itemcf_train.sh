#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
echo ${dt}
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad

#spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
spark-submit --master yarn-client \
 --conf spark.app.name=CF.app \
 --num-executors 100 \
 --driver-memory 8g \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.CF.app dyrec.jar \
 train  app.db/app_szad_m_midpage_jd_itemcf_train_day/action_type=1 1000  app.db/app_szad_m_dyrec_itemcf_model \
 app.db/app_szad_m_dyrec_itemcf_predict_res/user_type=1

