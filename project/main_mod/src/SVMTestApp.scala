/**
 * Created by rusty_lai on 2015/5/29.
 */
import _root_.org.apache.spark.SparkConf
import _root_.org.apache.spark.SparkContext
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint

object SVMTestApp {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SVM Test")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile("/root/kddcup.data")

    val parsedData = rawData.map{ line =>
      val parts = line.split(",")
      var label = "0"
      val buffer = line.split(",").toBuffer
      buffer.remove(1,3)
      val strLabel = buffer.remove(buffer.length-1)
      if(strLabel == "normal.")
        label = "1"
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      LabeledPoint(label.toDouble, Vectors.dense(buffer.map(_.toDouble).toArray))
    }


    val numIterations = 20
    val model = SVMWithSGD.train(parsedData, numIterations)
    val labelAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count
    val n1 = labelAndPreds.filter(r => (r._1==1)&&(r._2==1)).count.toDouble
    val n2 = labelAndPreds.filter(r => (r._1==0)&&(r._2==0)).count.toDouble
    val d1 = labelAndPreds.filter(r => (r._1==1)).count.toDouble
    val d2 = labelAndPreds.filter(r => (r._1==0)).count.toDouble
    val sensitivity = n1/d1
    val specificity = n2/d2

    println("\nTraining Error = " + trainErr)
    println("Sensitivity = " + sensitivity)
    println("Specificity = " + specificity)
    println();
  }
}
