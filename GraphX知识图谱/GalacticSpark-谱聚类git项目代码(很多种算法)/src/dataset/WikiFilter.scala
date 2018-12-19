package dataset

import org.apache.spark.{SparkContext, SparkConf}
import utils.{MetaFile, GalaxyException, FileHandler, FileLogger}

/**
 * Created by sara on 12/24/15.
 */
object WikiFilter {
  def main(args: Array[String]) {
    try {
      val filename = args(0)
      val output = args(1)
      val log = FileLogger.open(args(2))
      val reg = args(3)
      val col = args(4).toInt

      val conf = new SparkConf().setAppName("filter")
      val sc = new SparkContext(conf)

      val metaFiles = FileHandler.loadInput(filename)
      FileLogger.println("Search for " + reg);
      val uritype = metaFiles(0).uriType
      val data = metaFiles(0).uri

      val documents = sc.objectFile[(String, String)](data);
      val filtered = documents.filter{case (uri, short)=>
          if (col == 1) {uri.contains(reg)} else {short.contains(reg)} }
      val filteredFile = FileHandler.getFullName(sc, uritype, "filtered")
      filtered.saveAsObjectFile(filteredFile);


      FileLogger.println("Num of matched items: " + filtered.count() )

      println("Num of matched items: " + filtered.count() )
      filtered.takeSample(false,10).foreach{case (uri, short)=> FileLogger.println(uri + " " + short.mkString(""))}

      val objects = new Array[MetaFile](1)
      val obj = new MetaFile
      obj.objName = "FilteredAbstract"
      obj.uri = filteredFile
      obj.uriType = uritype;

      objects.update(0, obj);
      FileHandler.dumpOutput(output, objects)
      FileLogger.close();



    } catch {
      case e: Exception => println(GalaxyException.getString(e))
    }
  }
}
