/**
  *  Using spark ML linear regression to predict spam
  *  Author: Vinod Vijayan
  *  Date: 5 Apr 2016
  *
spam.txt:
-------------- START COPY FROM BELOW LINE------------------------
Dear sir, I am a Prince in a far kingdom you have not heard of.  I want to send you money via wire transfer so please ...
Get Viagra real cheap!  Send money right away to ...
Oh my gosh you can be really strong too with these drugs found in the rainforest. Get them cheap right now ...
YOUR COMPUTER HAS BEEN INFECTED!  YOU MUST RESET YOUR PASSWORD.  Reply to this email with your password and SSN ...
THIS IS NOT A SCAM!  Send money and get access to awesome stuff really cheap and never have to ...
-------------- STOP COPY WITH THE ABOVE LINE----------------------

ham.txt
-------------- START COPY FROM BELOW LINE------------------------
Dear Spark Learner, Thanks so much for attending the Spark Summit 2014!  Check out videos of talks from the summit at ...
Hi Mom, Apologies for being late about emailing and forgetting to send you the package.  I hope you and bro have been ...
Wow, hey Fred, just heard about the Spark petabyte sort.  I think we need to take time to try it out immediately ...
Hi Spark user list, This is my first question to this list, so thanks in advance for your help!  I tried running ...
Thanks Tom for your email.  I need to refer you to Alice for this one.  I haven't yet figured out that part either ...
Good job yesterday!  I was attending your talk, and really enjoyed it.  I want to try out GraphX ...
Summit demo got whoops from audience!  Had to let you know. --Joe
-------------- STOP COPY WITH THE ABOVE LINE----------------------

sampleinput.txt
-------------- START COPY FROM BELOW LINE-------------------------
O M G GET cheap stuff by sending money to ...
Hi Dad, I have started learning Spark using Scala since the other ...
-------------- STOP COPY WITH THE ABOVE LINE----------------------

Output when run:
(O M G GET cheap stuff by sending money to ...) --> prob=[0.10202581946958277,0.8979741805304173], prediction=1.0
(Hi Dad, I have started learning Spark using Scala since the other ...) --> prob=[0.6280674046224966,0.37193259537750334], prediction=0.0

Please Note: There is always a better way to do stuff.
  */
package com

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.functions._

object ScalaMachLearnMLApp{
  def main(args:Array[String]): Unit = {

    // Create the context with a 1 second batch size
    val sparkConf        = new SparkConf().setAppName("ScalaStreamingApp").setMaster("local[*]")
    val sc               = new SparkContext(sparkConf)
    val sqlContext       = new SQLContext(sc)

    // UDF functions that return positive(1) and negative(0) lables
    val getPosLabel: (String => Double) = (arg: String) => {1.0}
    val getNegLabel: (String => Double) = (arg: String) => {0.0}

    val sqlPosLblFunc = udf(getPosLabel)
    val sqlNegLblFunc = udf(getNegLabel)

    // UDF function to filter out empty rows
    val filterEmptyString: (String => Boolean) = (arg: String) => {if(arg.length > 0) true else false}
    val sqlStrFiltFunc = udf(filterEmptyString)

    // Read Spam and Normal text from HDFS to train the Linear Regression model
    val dfSpam = sqlContext.read.text("/home/hduser/IdeaProjects/files/spam.txt").withColumnRenamed("value","text")
    val dfNorm = sqlContext.read.text("/home/hduser/IdeaProjects/files/ham.txt").withColumnRenamed("value","text")

    // Read Sample data from HDFS to predict each line is spam or not
    val dfSamp = sqlContext.read.text("/home/hduser/IdeaProjects/files/sampleinput.txt").withColumnRenamed("value","text")

    //Filter empty rows with no text data
    val dfSpamFullLines = dfSpam.filter(sqlStrFiltFunc(dfSpam("text")))
    val dfNormFullLines = dfNorm.filter(sqlStrFiltFunc(dfNorm("text")))

    //Create a column "label" in training data with 1.0 labeled Spam and 0.0 for normal text
    val dfLabelSpam = dfSpamFullLines.withColumn("label", sqlPosLblFunc(col("text")))
    val dfLabelNorm = dfNormFullLines.withColumn("label", sqlNegLblFunc(col("text")))

    // Create a union of the Spam and Normal DataFrame which will be the training data for the regression model
    val dfTrainingData = dfLabelSpam.unionAll(dfLabelNorm)

    // Just to ensure the label is double in training data
    //val toDouble = udf[Double, String]( _.toDouble)
    //val dfTrainingDataDoub = dfTrainingData.withColumn("label", toDouble(dfTrainingData("label")))


    val tok = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(10000)
      .setInputCol(tok.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)

    //Create pipeline stages of Tokenizer, hashing and linear regression
    val pipeline = new Pipeline()
      .setStages(Array(tok, hashingTF, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(dfTrainingData)

    // Make predictions on test documents.
    model.transform(dfSamp)
      .select("text", "probability", "prediction")
      .collect()
      .foreach { case Row(text: String, prob: Vector, prediction: Double) =>
        println(s"($text) --> prob=$prob, prediction=$prediction")
      }

    sc.stop()
  }
}

