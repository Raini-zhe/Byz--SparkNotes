package utils

import java.io.{File, PrintWriter}

/**
 * Created by sara on 11/5/15.
  * * Copyright (C) 2015 by Sara Riazi
  * University of Oregon
  * All rights reserved.
 */
object FileLogger {
  var FILE : PrintWriter= null
  def open(logfile: String) : Unit = {
    FILE = new PrintWriter(new File(logfile));
  }
  def println(x: Any) : Unit ={
    FILE.println(x);
  }
  def error(x: Any) : Unit = {
    FILE.print("Error: ");
    FILE.println(x);
  }
  def close(): Unit = {
    FILE.close();
  }

}
