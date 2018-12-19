package utils

/**
 * Created by sara on 11/4/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object GalaxyException extends Exception{
  def getString(e: Exception): String ={
    val sb = new StringBuilder
    sb.append(e.toString + "\t")
    e.getStackTrace.foreach(s=>sb.append(s.toString + "\t"))
    val error = "ERROR. tool unsuccessful:" + sb.mkString
    return error
  }

}
