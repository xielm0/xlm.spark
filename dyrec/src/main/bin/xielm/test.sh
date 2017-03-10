#!/bin/sh

#spark2.0
#source /data/bossapp/software/bashrc

spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 50 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict2 short_sku2click short_sku2click app.db/app_szad_m_dyrec_userlabel_predict_res2/user_type=1/type=short_sku2click

spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 50 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict2 short_sku2click short_sku2browse app.db/app_szad_m_dyrec_userlabel_predict_res2/user_type=1/type=short_sku2browse

 spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 50 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict2 short_cate2click short_cate2click app.db/app_szad_m_dyrec_userlabel_predict_res2/user_type=1/type=short_cate2click

 spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 50 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict2 short_cate2click short_cate2browse app.db/app_szad_m_dyrec_userlabel_predict_res2/user_type=1/type=short_cate2browse



