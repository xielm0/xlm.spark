#!/usr/bin/env bash
# 获取昨天
# dt=`date -d last-day +%Y-%m-%d`
#获取上月
dt=`date -d "last month" +%Y-%m`
echo 'start spark-job gender_model_apply_jdpin task' $dt

#exesql="INSERT overwrite TABLE app_szad_m_dmp_label_gender_jdpin_result partition(dt='${dt}')"
#echo $exesql

queue=bdp_jmart_adv.bdp_jmart_sz_ad

/data0/spark/bin/spark-submit \
--master yarn-client \
--class com.jd.szad.gender.gender_model_apply /home/jd_ad/wangjing/dmp/original-dmp-1.0.0.jar hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_gender_apply_jdpin/*.lzo hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_gender_jdpin_model_save/ app_szad_m_dmp_label_gender_result_jdpin ${dt}
