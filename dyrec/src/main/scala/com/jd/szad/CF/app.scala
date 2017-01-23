package com.jd.szad.CF


import com.jd.szad.tools.Writer
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by xieliming on 2016/6/7.
 */
object app {
  def main(args:Array[String]) {
    val conf = new SparkConf()
      .setAppName("OCPA.itemCF")
      .set("spark.akka.timeout", "1000")
      .set("spark.rpc.askTimeout", "500")

    val sc = new SparkContext(conf)

    val model_type :String = args(0)  // val model_type="train"
    val input_path :String = args(1)
    val part_num :Int = args(2).toInt
    val model_path :String =args(3)
    val res_path :String = args(4)

    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)

    if (model_type =="train") {

      val data=sc.textFile(input_path).repartition(part_num)

      //sku
      val s1 =
        s"""select bigint(outerid) from app.app_szad_m_dyrec_sku_list_day
       """.stripMargin
      val ad_sku = hiveContext.sql(s1).map(t=> t(0).asInstanceOf[Long]).collect().zipWithIndex.toMap

      val user_action = data.map( _.split("\t") match{ case Array(user,item,rate) =>UserItem(user,item.toLong)})

      //计算相似度
      val similary = itemCF.compute_sim(sc,user_action,ad_sku,30)
        .repartition(100)
        .map(t=>t.itemid1 +"\t" + t.itemid2 +"\t" + t.similar  )

      //保存到hdfs
      Writer.write_table(similary,model_path,"lzo")

    } else if (model_type =="predict") {

      val user =sc.textFile(input_path).repartition(part_num)
        .map(_.split("\t"))
        .map{x =>
          if (x(1) != "\\N") UserPref(x(0) ,x(1).toLong,x(2).toInt)
          else UserPref(x(0) ,x(1).toLong,x(2).toInt)
        }

      // 取sim的top k
      val k = 6
      val sim = sc.textFile(model_path).map(_.split("\t")).map(t=>(t(0).toLong ,(t(1).toLong,t(2).toDouble)))
      .groupByKey()
      .flatMap{
        case( a, b)=>  //b=Interable[(item2,score)]
          val bb= b.toBuffer.sortWith{ (a,b) => a._2>b._2 } //desc
          if (bb.length >k ) bb.remove( k,bb.length - k)
          bb.map{ t => ItemSimi(a,t._1,t._2) }
      }

      //预测用户的推荐
      val res = itemCF.recommendItem(sim,user,50)
        .map(t=> t._1 +"\t" + t._2 +"\t" + t._3 +"\t" + t._4)

      //保存到hdfs
      Writer.write_table(res,res_path,"lzo")

    }else if (model_type =="sql_predict") {

      val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
      sqlContext.sql("set spark.sql.shuffle.partitions = 1000")
      val sql=
        """
          |insert overwrite table app.app_szad_m_dyRec_userlabel_predict_res partition (user_type=1)
          |select t3.*
          |  from (select uid ,sku ,score,rn
          |          from( select t1.uid,t1.sku2  ,t1.score ,row_number() over(partition by t1.uid order by t1.score desc) rn
          |                  from(select uid,sku2,sum(round(rate*score,4)) as score
          |                        from (select uid,sku, count(1) as rate
          |                                from app.app_szad_m_dyrec_itemcf_apply_day
          |                               where user_type=1
          |                               group by  uid,sku )a
          |                        join (select sku,type,label,score
          |                               from (select sku1,sku2,row_number() over(partition by sku1 order by cor desc ) rn
          |                                       from app.app_szad_m_dyrec_itemcf_model)t
          |                               where rn <=6 )b
          |                         on (a.sku=b.sku1)
          |                      group by uid,sku2
          |                        )t1
          |               )t2
          |          where rn <=50
          |          )t3
          |left join (select uid,sku,count(1) as rate
          |            from app.app_szad_m_dyrec_itemcf_apply_day
          |           where user_type=1
          |           group by  uid,sku )c
          |     on  (t3.uid =c.uid and t3.sku2 = c.sku)
          |  where c.rate is null
        """.stripMargin
      sqlContext.sql(sql)

    }


  }
}
