#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
dt_15=`date -d "-16 days" +%Y-%m-%d`
echo ${dt}
echo ${ftime}
echo ${dt_15}
queue=bdp_jmart_adv.bdp_jmart_sz_ad


spark-submit --master yarn-client \
   --conf spark.app.name=itemcf_predict \
   --driver-memory 6g \
   --num-executors 50 \
   --executor-memory 12g \
   --executor-cores 6 \
   --class com.jd.szad.CF.predict dyrec.jar \
 predict 1 ${dt} ${dt_15}  app.db/app_szad_m_dyrec_model_predict_res/user_type=1/type=CF




