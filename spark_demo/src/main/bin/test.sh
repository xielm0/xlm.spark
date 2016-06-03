#!/usr/bin/env bash

queue=bdp_jmart_adv.bdp_jmart_sz_ad

/software/servers/spark-1.5.2-HADOOP-2.2.0/bin/spark-submit \
--master yarn-client \
--jars $(echo /home/jd_ad/data_dir/huangzhong/lib/*.jar | tr ' ' ',') \
--num-executors 2 \
--driver-memory 2g \
--executor-memory 2g \
--executor-cores 2 \
--queue $queue \
--conf spark.rdd.compress=true \
--conf spark.akka.frameSize=100 \
--conf spark.storage.memoryFraction=0.3 \
--class com.jd.ads.tools.JTMFTask /home/jd_ad/data_dir/huangzhong/original-jtfm-1.0.0.jar yarn-client gdt /user/jd_ad/huangzhong/jtfm.txt


#export SPARK_CLASSPATH=$SPARK_CLASSPATH:/home/jd_ad/huangzhong/jtfm/lib/elephant-bird-core-4.13.jar:/home/jd_ad/huangzhong/jtfm/lib/elephant-bird-hadoop-compat-4.13.jar:/home/jd_ad/huangzhong/jtfm/lib/guava-14.0.1.jar:/home/jd_ad/huangzhong/jtfm/lib/protobuf-java-2.5.0.jar:/home/jd_ad/huangzhong/jtfm/original-jtfm-1.0.0.jar
     