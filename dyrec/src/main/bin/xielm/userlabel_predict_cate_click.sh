#!/bin/sh

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo ${dt}


spark-submit --master yarn-client \
 --driver-memory 6g \
 --num-executors 80 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict2 browse_top20cate 7_click_top20cate app.db/app_szad_m_dyrec_userlabel_predict_res2/user_type=1/type=7_click_top20cate



#spark2.0
#source /data/bossapp/software/bashrc
