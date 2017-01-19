package com.jd.szad.userlabel

/**
 * Created by xieliming on 2017/1/12.
 */
case class UserLabelSku (
                          val User_id:String ,
                          val Label_id :String ,
                          val Sku_id :Long,
                          val rate :Int)extends  Serializable

