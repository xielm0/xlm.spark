#!/usr/bin/env bash

dt=`date -d last-day +%Y-%m-%d`
ftime=`date -d last-day +%Y%m%d`
echo ${ftime}
queue=bdp_jmart_adv.bdp_jmart_sz_ad
echo ${dt}

spark-submit --master yarn-client \
 --driver-memory 8g \
 --num-executors 100 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.app dyrec.jar \
 predict


 spark-submit --master yarn-client \
 --driver-memory 8g \
 --num-executors 100 \
 --executor-memory 16g \
 --executor-cores 8 \
 --class com.jd.szad.userlabel.multiply dyrec.jar \
 predict vec



#spark2.0
source /data/bossapp/software/bashrc

spark-sql --master yarn-client \
 --conf spark.app.name=userlabel \
 --driver-memory 6g \
 --num-executors 100 \
 --executor-memory 10g \
 --executor-cores 5 \
 -e "
 set mapreduce.output.fileoutputformat.compress=true;
 set hive.exec.compress.output=true;
 set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec;

insert overwrite table app.app_szad_m_dyrec_userlabel_apply partition (user_type=1)
select uid,'browse_top20sku' as type,sku as label,1 as rate ,rn
  from app.app_szad_m_dyrec_user_top100_sku
 where user_type=1 and length(uid)>20
   and action_type=1
   and dt='2017-02-07'
   and sku is not null
   and rn<=20
union all
select uid,'browse_top20cate' as type,3rd_cate_id as label,count(1) as rate ,min(rn) as rn
  from app.app_szad_m_dyrec_user_top100_sku
 where user_type=1 and length(uid)>20
   and action_type=1
   and dt='2017-02-07'
   and 3rd_cate_id is not null
   and rn<=20
group by uid,3rd_cate_id
union all
select uid,'browse_top20cate' as type,brand_id as label,count(1) as rate ,min(rn) as rn
  from app.app_szad_m_dyrec_user_top100_sku
 where user_type=1 and length(uid)>20
   and action_type=1
   and dt='2017-02-07'
   and brand_id is not null
   and rn<=20
group by uid,brand_id
"


 spark-sql --master yarn-client \
 --conf spark.app.name=userlabel \
 --driver-memory 6g \
 --num-executors 100 \
 --executor-memory 10g \
 --executor-cores 5 \
 -e "
 set spark.sql.shuffle.partitions = 1000;
 set mapreduce.output.fileoutputformat.compress=true;
 set hive.exec.compress.output=true;
 set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec;

insert overwrite table app.app_szad_m_dyrec_userlabel_predict_res partition (user_type=1)
select uid ,sku ,score,rn
  from( select uid ,sku ,score ,row_number() over(partition by uid order by score desc) rn
          from(select uid,sku,sum(round(rate*score,1)) as score
                from (select * from app_szad_m_dyrec_userlabel_apply where user_type=1 and rn <=20) a
                join (select type,label,sku,score
                       from (select type,label,sku,score,row_number() over(partition by type,label order by score desc ) rn
                               from app.app_szad_m_dyRec_userlabel_model )t
                       where rn <=50 )b
                 on (a.type=b.type and a.label=b.label)
              group by uid,sku
                )t1
       )t2
 where rn <=100"