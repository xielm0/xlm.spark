#!/usr/bin/env bash

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo ${dt}

spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 100 \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict type='browse_top20brand'

  #predict 1=1

spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 100 \
 --executor-memory 10g \
 --executor-cores 5 \
 --class edu.nju.pasalab.marlin.examples.SparseMultiply dyrec.jar \
 100000000 100000 1000000 1 3


 spark-submit --master yarn-client \
 --driver-memory 8g \
 --num-executors 100 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.multiply dyrec.jar \
block



#spark2.0
source /data/bossapp/software/bashrc
