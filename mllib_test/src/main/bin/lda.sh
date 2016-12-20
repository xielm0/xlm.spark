#!/usr/bin/env bash

queue=bdp_jmart_adv.bdp_jmart_sz_ad
tday=`date +%Y%m%d`
yesday=`date -d last-day +%Y-%m-%d`
echo 'start spark-job kwd_cate_coo task' $tday

/data0/spark/bin/spark-submit \
--master yarn-client \
--class com.jd.szad.kwd_cate.kwd_cate_coo /home/jd_ad/wangjing/dmp/original-dmp-1.0.0.jar hdfs://ns3/user/jd_ad/jiangxue/search_cps_ads/${tday}/query_category/ app_szad_m_dmp_kwd_cate_coo ${yesday}

#/data0/spark/bin/spark-submit --master yarn-client --class com.jd.szad.kwd_cate.kwd_cate_coo /home/jd_ad/wangjing/dmp/original-dmp-1.0.0.jar hdfs://ns3/user/jd_ad/jiangxue/search_cps_ads/20160525/query_category/ app_szad_m_dmp_kwd_cate_coo 2016-05-24
#export SPARK_CLASSPATH=$SPARK_CLASSPATH:/home/jd_ad/huangzhong/jtfm/lib/elephant-bird-core-4.13.jar:/home/jd_ad/huangzhong/jtfm/lib/elephant-bird-hadoop-compat-4.13.jar:/home/jd_ad/huangzhong/jtfm/lib/guava-14.0.1.jar:/home/jd_ad/huangzhong/jtfm/lib/protobuf-java-2.5.0.jar:/home/jd_ad/huangzhong/jtfm/original-jtfm-1.0.0.jar
     