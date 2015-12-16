# Youtrack Issue Classifier
Youtrack Issue Classifier is an Machine Learning algorithm that tries to determine the subsystem of a ticket based on its context.
It uses [Apache Spark](http://spark.apache.org/) and its [MLlib](http://spark.apache.org/docs/latest/mllib-guide.html)

The algorithm uses [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) to study how old tickets were divided into subsystems on [YouTrack](https://www.jetbrains.com/youtrack/) and tries to guess what subsystem should new tickets be assigned to

#How to run
If you're running locally, you should use the following command:
```
$SPARK_HOME/bin/spark-submit --class "YoutrackClassifier" --master local[4] $OUTPUT_JAR
```
If you're running in IntelliJ IDEA, you should set a VM Option: `-Dspark.master=local[4]`
This will download the issues and put them in issues directory. Next time it will use this file instead of fetching all issues from Youtrack
