package com.jd.szad.CF

/**
 * Created by xieliming on 2016/6/3.
 */
case class ItemSimi(
                     val itemid1:Long ,
                     val itemid2 :Long ,
                     val similar :Double  )extends  Serializable


case class UserPref(
                     val userid:String ,
                     val itemid :Long ,
                     val score :Int  )extends  Serializable

case class UserItem(
                     val userid:String ,
                     val itemid :Long   )extends  Serializable
