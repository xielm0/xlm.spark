package com.jd.szad.itemcf

/**
 * Created by xieliming on 2016/6/3.
 */
case class ItemSimi(
                     val itemid1:Long ,
                     val itemid2 :Long ,
                     val similar :Double  )


case class UserPref(
                     val userid:String ,
                     val itemid :Long ,
                     val score :Int  )
