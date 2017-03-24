#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
dt_07=`date -d "-8 days" +%Y-%m-%d`
echo ${dt}
echo ${ftime}
echo ${dt_07}
queue=bdp_jmart_adv.bdp_jmart_sz_ad

spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 50 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict2 sku_feature sku_feature app.db/app_szad_m_dyrec_model_predict_res/user_type=1/type=sku_feature



