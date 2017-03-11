#!/usr/bin/env bash

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo ${dt}

spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 50 \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict2 short_sku short_sku2click app.db/app_szad_m_dyrec_userlabel_predict_res2/user_type=1/type=short_sku2click



