package com
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.streaming.{Seconds, StreamingContext}

/**
  * Created by hduser on 27/3/16.
  */
object ScalaMachLearnApp{
  def main(args:Array[String]): Unit = {

    // Create the context with a 1 second batch size
    val sparkConf = new SparkConf().setAppName("ScalaStreamingApp").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val spam = sc.textFile("/home/hduser/IdeaProjects/files/spam.txt")
    val ham = sc.textFile("/home/hduser/IdeaProjects/files/ham.txt")

    val tf = new HashingTF(numFeatures = 10000)

    val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
    val hamFeatures = ham.map(email => tf.transform(email.split(" ")))

    val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))
    val negativeExamples = hamFeatures.map(features => LabeledPoint(0, features))

    val trainingData = positiveExamples.union(negativeExamples)

    //trainingData.cache()

    val model = new LogisticRegressionWithSGD().run(trainingData)

    val posTest = tf.transform(
      "O M G GET cheap stuff by sending money to ...".split(" "))

    println(s"Prediction for positive test example: ${model.predict(posTest)}")

    sc.stop()
  }
}

