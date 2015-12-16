package me.ilinskiy.issueClassifier

import java.io.{PrintStream, File, FileWriter}

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.{BufferedSource, Source}
import scala.xml.{NodeSeq, Elem, Node, XML}

/**
  * Author: Svyatoslav Ilinskiy
  * Date: 12.12.15.
  */
object YoutrackStats {
  val issueOutputDirName = "issues"

  def getIssues(sc: SparkContext, project: String): RDD[Issue] = {
    val xmlNodes: RDD[NodeSeq] = getAllXmlIssuesInProject(project, sc)
    val issues = YoutrackUtil.nodesToIssues(xmlNodes)
    issues
  }

  private def getAllXmlIssuesInProject(project: String, sc: SparkContext): RDD[NodeSeq] = {
    def url(skip: Int) = s"https://youtrack.jetbrains.com/rest/issue/byproject/$project?max=1000&after=$skip"
    val dir = new File(issueOutputDirName)
    if (!dir.exists()) {
      def makeRequest(url: String): String = {
        val source = scala.io.Source.fromURL(url)
        try {
          source.mkString
        } finally {
          source.close()
        }
      }
      dir.mkdir()
      var skip = 0
      while (skip >= 0) {
        val text = makeRequest(url(skip))
        if (text.lines.size <= 1) {
          skip = -1
        } else {
          val fileWriter = new FileWriter(s"$issueOutputDirName/$skip.xml")
          try {
            fileWriter.write(text)
          } finally {
            fileWriter.close()
          }
          XML.save(s"$issueOutputDirName/$skip.xml", XML.loadString(text), "UTF-8")
          skip += 1000
        }
      }
    }
    sc.parallelize(dir.listFiles()).map { file =>
      val source: BufferedSource = Source.fromFile(file)
      try {
        XML.loadString(source.mkString) \ "issue"
      } finally {
        source.close()
      }
    }
  }
}

case class Issue(project: String, issueId: Int, summary: String, description: String, created: Long, updated: Long,
                 updaterName: Name, reporterName: Name, votesCount: Int, subsystem: NameIdPair, issueType: NameIdPair,
                 priority: NameIdPair, state: NameIdPair, assignees: Seq[Name], comments: Seq[Comment]) {
  def fullIssueId: String = s"$project-$issueId"
}

case class Name(shortName: String, fullName: String)

object Name {
  def apply(s: (String, String)): Name = Name(s._1, s._2)
}

case class NameIdPair(name: String, id: String)

case class Comment(author: Name, text: String, created: Long, deleted: Boolean)