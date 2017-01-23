#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo ${dt}


spark-submit --master yarn-client \
   --conf spark.app.name=itemcf_predict \
   --driver-memory 6g \
   --num-executors 100 \
   --executor-memory 8g \
   --executor-cores 4 \
   --class com.jd.szad.CF.app dyrec.jar \
 predict  app.db/app_szad_m_dyrec_itemcf_apply_day/user_type=1 1000  app.db/app_szad_m_dyrec_itemcf_model \
 app.db/app_szad_m_dyrec_itemcf_predict_res/user_type=1



 nohup spark-submit --master yarn-client \
   --conf spark.app.name=itemcf_predict \
   --driver-memory 6g \
   --num-executors 10 \
   --executor-memory 8g \
   --executor-cores 4 \
   --class com.jd.szad.CF.app dyrec.jar \
 predict  app.db/app_szad_m_dyrec_itemcf_apply_day/user_type=1/000033_0.lzo 40  app.db/app_szad_m_dyrec_itemcf_model \
 app.db/app_szad_m_dyrec_itemcf_predict_res/user_type=1 &