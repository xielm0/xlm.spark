#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo $dt

source /home/jd_ad/.bashrc
dir='/home/jd_ad/spark_task/dmp_intrest'
cd ${dir}

 #spark-submit --master yarn-cluster --queue bdp_jmart_adv.bdp_jmart_sz_ad \
 # --conf spark.dynamicAllocation.enabled=true  --conf spark.shuffle.service.enabled=true --conf spark.dynamicAllocation.maxExecutors=100 \
 spark-submit --master yarn-client  \
 --driver-memory 4g \
 --num-executors 50 \
 --executor-memory 8g \
 --executor-cores 4 \
 --class com.jd.szad.intrest.lda_jd  dmp_intrest.jar  \
  train  22 36 1.3 1.05 app.db/app_szad_m_dmp_label_intrest_jdpin_train  800 app.db/app_szad_m_dmp_label_intrest_res_jdpin