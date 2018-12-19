package utils

/**
 * Created by sara on 11/3/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
class MetaFile (name: String, utype: String, uripath:String) {
  var uri : String = uripath
  var uriType: String = utype
  var objName : String = name
  def this() = this ("", "", "")
}
