#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
dt_07=`date -d "-7 days" +%Y-%m-%d`
echo ${dt}
echo ${dt_07}
queue=bdp_jmart_adv.bdp_jmart_sz_ad

 spark-submit --master yarn-client \
 --driver-memory 4g \
 --num-executors 50 \
 --executor-memory 10g \
 --executor-cores 5 \
 --class com.jd.szad.userlabel.catehot dyrec.jar \
 user_type=1/type=short_cate  click 20 user_type=1/type=short_cate 20 ${dt_07}



