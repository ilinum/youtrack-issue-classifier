# Youtrack Issue Classifier
Youtrack Issue Classifier is an Machine Learning algorithm that tries to determine the subsystem of a ticket based on its context.
It uses [Apache Spark](http://spark.apache.org/) and its [MLlib](http://spark.apache.org/docs/latest/mllib-guide.html)

The algorithm uses [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) to study how old tickets were divided into subsystems on [YouTrack](https://www.jetbrains.com/youtrack/) and tries to guess what subsystem should new tickets be assigned to

#How to run
##Terminal
To run you should use the following command:
```
$SPARK_HOME/bin/spark-submit --class "me.ilinskiy.issueClassifier.YoutrackClassifier" --master local[4] $OUTPUT_JAR
```

##IntellliJ IDEA
All you need to do is set VM options to: `-Dspark.master=local[4]`
Then you can run the main method normally. 

#Misc
Issues are downloaded in issues directory if the directory is not present. Later, they are just reused, without redownloading
