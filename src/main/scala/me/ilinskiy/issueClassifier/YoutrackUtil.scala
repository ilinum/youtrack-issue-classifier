package me.ilinskiy.issueClassifier


import me.ilinskiy.issueClassifier.{TrainingResult, YoutrackClassifier}
import org.apache.spark.rdd.RDD

import scala.collection.immutable
import scala.io.{BufferedSource, Source}
import scala.xml.{NodeSeq, Elem, Node}


/**
  * Author: Svyatoslav Ilinskiy
  * Date: 15.12.15.
  */
object YoutrackUtil {
  def nodesToIssues(issueXmls: RDD[NodeSeq]): RDD[Issue] = {
    issueXmls.flatMap(_.map(nodeToIssue))
  }

  def nodeToIssue(node: Node): Issue = {
    val fields = node \ "field"
    val fieldsWithNames: immutable.Seq[(Node, String)] = fields.map(f => (f, (f \ "@name").text))
    def fieldWithName(fieldName: String, default: String): String = {
      fieldsWithNames.find(_._2 == fieldName).map {
        case (n, _) => (n \ "value").text
      }.getOrElse(default)
    }

    def fieldWithNameIdPair(fieldName: String, default: (String, String)): NameIdPair = {
      fieldsWithNames.find(_._2 == fieldName).map {
        case (n, _) => NameIdPair((n \ "value").text, (n \ "valueId").text)
      }.getOrElse(NameIdPair(default._1, default._2))
    }

    val project = fieldWithName("projectShortName", "NoProject")
    val issueId = fieldWithName("numberInProject", "-1").toInt
    val summary = fieldWithName("summary", "")
    val description = fieldWithName("description", "")
    val created = fieldWithName("created", "0").toLong
    val updated = fieldWithName("updated", "0").toLong
    val updaterName = (fieldWithName("updaterName", "NoName"), fieldWithName("updaterFullName", "NoFullName"))
    val reporterName = (fieldWithName("reporterName", "NoName"), fieldWithName("reporterFullName", "NoFullName"))
    val votesCount = fieldWithName("votes", "-1").toInt
    val subsystem = fieldWithNameIdPair("Subsystem", ("NoSubsystem", ""))
    val issueType = fieldWithNameIdPair("Type", ("NoType", ""))
    val priority = fieldWithNameIdPair("Priority", ("NoPriority", ""))
    val state = fieldWithNameIdPair("State", ("NoState", ""))
    val assignees: Seq[Name] = fieldsWithNames.filter(_._2 == "Assignee").map {
      case (n, _) =>
        val value = n \ "value"
        Name(value.text, (value \ "@fullName").text)
    }

    val comments: Seq[Comment] = (node \ "comment").map { commentNode =>
      val author: Name = Name((commentNode \ "@author").text, (commentNode \ "@authorFullName").text)
      val text: String = (commentNode \ "@text").text
      val created: Long = (commentNode \ "@created").text.toLong
      val deleted: Boolean = (commentNode \ "@deleted").text.toBoolean
      Comment(author, text, created, deleted)
    }

    Issue(project, issueId, summary, description, created, updated, Name(updaterName), Name(reporterName), votesCount,
      subsystem, issueType, priority, state, assignees, comments)
  }

  def fetchIssue(id: Int, project: String): Issue = {
    val address: String = s"http://youtrack.jetbrains.com/rest/issue/$project-$id"
    val source = Source.fromURL(address)
    val text = try {
      source.mkString
    } finally {
      source.close()
    }
    val elem: Elem = scala.xml.XML.loadString(text)
    nodeToIssue(elem)
  }

  def measure(model: TrainingResult, issues: Seq[Issue]): Unit = {
    val results = issues.map { i =>
      (i.subsystem.id, YoutrackClassifier.predict(i, model))
    }.filterNot(_._1 == "No Subsystem")

    val correct = results.count(t => t._1 == t._2)
    val wrong = results.count(t => t._1 != t._2)
    val accuracy = correct.toDouble / (wrong.toDouble + correct)
    println(s"Correct: $correct\nWrong: $wrong\nTotal: ${results.size}\nAccuracy: $accuracy")
  }

  def tokenize(s: String): Seq[String] = {
    //remove all non-alphabetic and non-numeric characters
    s.split("[ \n]").map(_.replaceAll("[^\\p{L}\\p{Nd}]+", "")).map(_.toLowerCase)
  }
}