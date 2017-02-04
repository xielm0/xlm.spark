#!/usr/bin/env bash

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
select uid,type,label,rate,rn
  from(select uid,type,label,rate,row_number() over(partition by uid order by type,label) rn
        from(select gdt_openid as uid,label_code as type,label_value as label,1 as rate,
                    row_number() over(partition by gdt_openid order by label_code,label_value) rn
              from app.app_szad_m_dmp_label_gdt_openid
             where label_type in('209','210','211','309','310','311','331','332','333','363','364','365')
               and label_code in('209001','210001','211001','309001','310001','311001','331001','332001','333001','363001','364001','365001')
              )a1
       )a2
  where rn <100"


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