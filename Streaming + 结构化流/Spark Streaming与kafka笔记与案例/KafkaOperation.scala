package com.xtwy.streaming

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.SparkConf
import org.apache.spark.streaming.kafka.KafkaUtils
import kafka.serializer.StringDecoder
import org.apache.spark.SparkContext

object KafkaOperation {
  def main(args: Array[String]): Unit = {
    
    
    val conf=new SparkConf().setMaster("local[2]").setAppName("KafkaOperation")
    val sc=new SparkContext(conf);
    val  ssc=new StreamingContext(sc,Seconds(2));
     ssc.checkpoint(".")
     val kafkaParams=Map("metadata.broker.list" -> "hadoop1:9092");
     val topics=Set("xtwy");
     /**
      * k:其实就是偏移量 offset
      * V:就是我们消费的数据
      * InputDStream[(K, V)]
      * 
      * k:数据类型
      * v:数据类型
      * 
      * k的解码器
      * v的解码器
      * [K, V, KD <: Decoder[K], VD <: Decoder[V]]
      */
    val kafkaDS= KafkaUtils.createDirectStream[String,String,StringDecoder,StringDecoder](ssc,kafkaParams,topics)
     .map(_._2)
     
    val wordcountDS= kafkaDS.flatMap { line => line.split("\t") }
    .map { word => (word,1) }
    .reduceByKey(_+_)//window  mapwithstate updatewithstateByKey topK
    wordcountDS.print();
    ssc.start();
    ssc.awaitTermination();
     
  }
}