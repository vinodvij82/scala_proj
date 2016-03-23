import org.apache.spark.streaming._
import org.apache.spark.SparkConf

object ScalaStreamingApp extends App{
  //initialize spark context
  val conf         = new SparkConf().setAppName("Streaming Application").setMaster("local")
  val ssc          = new StreamingContext(conf, Seconds(5))
  val textRDD      = ssc.socketTextStream("localhost",9999)
  // Split each line into words
  val words = textRDD.flatMap(_.split(" "))

  // Count each word in each batch
  val pairs = words.map(word => (word, 1))
  val wordCounts = pairs.reduceByKey{ case (x,y) => x+y}

  // Print the first ten elements of each RDD generated in this DStream to the console
  wordCounts.print()
  
  ssc.start()
  ssc.awaitTermination()
}