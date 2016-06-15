#!/usr/bin/env bash

queue=bdp_jmart_adv.bdp_jmart_sz_ad

/data0/spark/bin/spark-submit \
--master yarn-client \
--class com.jd.szad.gender.gender_model_train /home/jd_ad/wangjing/dmp/original-dmp-1.0.0.jar hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_gender_train_jdpin/*.lzo hdfs://ns3/user/jd_ad/app.db/app_szad_m_dmp_label_gender_jdpin_model_save/
