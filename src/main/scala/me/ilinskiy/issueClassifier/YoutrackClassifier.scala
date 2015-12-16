package me.ilinskiy.issueClassifier

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map

/**
  * Author: Svyatoslav Ilinskiy
  * Date: 16.12.15.
  */
object YoutrackClassifier {
  val project = "SCL"

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Youtrack Issue Classifier")
    val sc = new SparkContext(conf)
    val issues = YoutrackStats.getIssues(sc, project)
    val randomSplit = issues.randomSplit(Array(0.8, 0.2))
    val (trainingSet, testingSet) = (randomSplit.head, randomSplit(1))
    val model = train(trainingSet)
    YoutrackUtil.measure(model, testingSet.collect())
  }

  //todo: give more weight to summary than to description
  //todo: include comments during training
  //todo: include attachments
  //todo: stem words
  def train(_issues: RDD[Issue]): TrainingResult = {
    val issues = _issues.filter(_.subsystem.id != "No Subsystem")
    val issuesAndTheirWords: RDD[(Issue, Seq[String])] = issues.map { i =>
      (i, YoutrackUtil.tokenize(i.description + " " + i.summary))
    }

    val docsWithWords: Map[String, Int] = {
      issuesAndTheirWords.flatMap { case (issue, words) =>
        words
      }.groupBy(identity).map {
        case (s, usages) => (s, usages.size)
      }.collect().groupBy(identity).toSeq.map {
        case ((s, _), array) => (s, array.map(_._2).sum)
      }.toMap
    }

    val wordIds: Map[String, Int] = docsWithWords.keys.zipWithIndex.toMap
    //have to do it that way because after mapValues Map is not serializable
    val subsystemIds: Map[String, Double] = issues.map(_.subsystem.name).collect().
      zipWithIndex.toMap.mapValues(_.toDouble).map(identity)
    val numDocs = issues.count().toDouble
    val input: RDD[LabeledPoint] = issuesAndTheirWords.map {
      case (i, words) =>
        val data = words.map { word =>
          val numDocsWithThisWord = docsWithWords(word).toDouble
          val numOccurrences = words.count(_ == word).toDouble

          val value = (numDocs / numDocsWithThisWord) * (numOccurrences / words.length.toDouble)
          (wordIds(word), value)
        }
        val s = data.groupBy(_._1).toSeq.map {
          case (id, b) => (id, b.map(_._2).sum)
        }
        val vector = Vectors.sparse(wordIds.size, s)
        LabeledPoint(subsystemIds(i.subsystem.name), vector)
    }

    val model = NaiveBayes.train(input)
    TrainingResult(model, docsWithWords, numDocs, wordIds, subsystemIds.map(_.swap))
  }

  def predict(issue: Issue, trainingResult: TrainingResult): String = {
    val words: Seq[String] = YoutrackUtil.tokenize(issue.description + " " + issue.summary)
    val ids: Map[String, Int] = trainingResult.wordIds
    val data: Seq[(Int, Double)] = words.filter(ids.contains).map { word =>
      val numDocsWithThisWord = trainingResult.docsWithWords(word).toDouble
      val numOccurrences = words.count(_ == word).toDouble
      val value = (trainingResult.numDocs / numDocsWithThisWord) * (numOccurrences / words.length.toDouble)
      (ids(word), value)
    }.groupBy(identity).map {
      case ((id, value), values) => (id, values.size.toDouble)
    }.toSeq
    val vector = Vectors.sparse(ids.size, data)
    val predicted = trainingResult.model.predict(vector)
    val resString = trainingResult.idToSubsystemName(predicted)
    println(s"${issue.fullIssueId}: $resString")
    resString
  }
}

case class TrainingResult(model: NaiveBayesModel, docsWithWords: Map[String, Int], numDocs: Double,
                          wordIds: Map[String, Int], idToSubsystemName: Map[Double, String])
