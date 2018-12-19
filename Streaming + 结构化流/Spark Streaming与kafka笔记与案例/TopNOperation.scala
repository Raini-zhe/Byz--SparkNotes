package com.xtwy.streaming

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.SparkConf
import org.apache.spark.streaming.Seconds
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.hive.HiveContext


object TopNOperation {
  def main(args: Array[String]): Unit = {
    val conf=new SparkConf().setMaster("local[2]").setAppName("TopNOperation")
    val  ssc=new StreamingContext(conf,Seconds(5));    
    val fileDS=ssc.socketTextStream("hadoop1", 9999);
   val pairsDS= fileDS.map { log => (log.split(",")(1)+"_"+log.split(",")(2) ,1)}
   val pairsCountsDS = pairsDS.reduceByKeyAndWindow((v1:Int,v2:Int)=>(v1+v2),Seconds(60),Seconds(10))
    pairsCountsDS.foreachRDD( categoryproductRDD =>{
    val rowRDD=  categoryproductRDD.map( tuple =>{
     val category=   tuple._1.split("_")(0);
     val product=   tuple._1.split("_")(1);
     val count=tuple._2;
     Row(category,product,count)     
      })
      
      val structType=StructType(
      Array(
      StructField("category",StringType,true), 
      StructField("product",StringType,true), 
      StructField("count",IntegerType,true)
      )    
      )
      
      val hiveContext=new HiveContext(categoryproductRDD.context);
      val categoryProductCountsDF= hiveContext.createDataFrame(rowRDD, structType)
      categoryProductCountsDF.registerTempTable("product_count")
      
      val sql="""
        select category,product,count from 
        (select category,product,count,row_number() over(partition by category order by count desc) rank
        from product_count) tmp
        where tmp.rank <= 3
        
        """
     val top3DF= hiveContext.sql(sql);
      
      top3DF.show();
    })
    ssc.start();
   ssc.awaitTermination();
  }
}