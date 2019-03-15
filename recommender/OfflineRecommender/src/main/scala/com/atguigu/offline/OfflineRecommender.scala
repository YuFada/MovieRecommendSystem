package com.atguigu.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession

/**
  * @author :YuFada
  * @date ： 2019/3/15 0015 下午 15:18
  *       Description：
  */
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String,language: String, genres: String, actors: String, directors: String )

case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int)

case class MongoConfig(uri:String, db:String)

// 标准推荐对象，mid,score
case class Recommendation(mid: Int, score:Double)

// 用户推荐
case class UserRecs(uid: Int, recs: Seq[Recommendation])

// 电影相似度（电影推荐）
case class MovieRecs(mid: Int, recs: Seq[Recommendation])

object OfflineRecommmeder {

    // 定义常量
    val MONGODB_RATING_COLLECTION = "Rating"
    val MONGODB_MOVIE_COLLECTION = "Movie"

    // 推荐表的名称
    val USER_RECS = "UserRecs"
    val MOVIE_RECS = "MovieRecs"

    val USER_MAX_RECOMMENDATION = 20

    def main(args: Array[String]): Unit = {
        // 定义配置
        val config = Map(
            "spark.cores" -> "local[*]",
            "mongo.uri" -> "mongodb://hadoop102:27017/recommender",
            "mongo.db" -> "recommender"
        )

        // 创建spark session
        val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("StatisticsRecommender")
        val spark = SparkSession.builder().config(sparkConf).getOrCreate()

        implicit val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))

        import spark.implicits._
        //读取mongoDB中的业务数据
        val ratingRDD = spark
            .read
            .option("uri",mongoConfig.uri)
            .option("collection",MONGODB_RATING_COLLECTION)
            .format("com.mongodb.spark.sql")
            .load()
            .as[MovieRating]
            .rdd
            .map(rating=> (rating.uid, rating.mid, rating.score)).cache()
        //用户的数据集 RDD[Int]
        val userRDD = ratingRDD.map(_._1).distinct()

        //电影数据集 RDD[Int]
        val movieRDD = spark
            .read
            .option("uri",mongoConfig.uri)
            .option("collection",MONGODB_MOVIE_COLLECTION)
            .format("com.mongodb.spark.sql")
            .load()
            .as[Movie]
            .rdd
            .map(_.mid).cache()

        //创建训练数据集
        val trainData = ratingRDD.map(x => Rating(x._1,x._2,x._3))
        // rank 是模型中隐语义因子的个数, iterations 是迭代的次数, lambda 是ALS的正则化参
        val (rank,iterations,lambda) = (50, 5, 0.01)
        // 调用ALS算法训练隐语义模型
        val model = ALS.train(trainData,rank,iterations,lambda)

        //计算用户推荐矩阵
        val userMovies = userRDD.cartesian(movieRDD)
        // model已训练好，把id传进去就可以得到预测评分列表RDD[Rating] (uid,mid,rating)
        val preRatings = model.predict(userMovies)

        val userRecs = preRatings
            .filter(_.rating > 0)
            .map(rating => (rating.user,(rating.product, rating.rating)))
            .groupByKey()
            .map{
                case (uid,recs) => UserRecs(uid,recs.toList.sortWith(_._2 >
                    _._2).take(USER_MAX_RECOMMENDATION).map(x => Recommendation(x._1,x._2)))
            }.toDF()

        userRecs.write
            .option("uri",mongoConfig.uri)
            .option("collection",USER_RECS)
            .mode("overwrite")
            .format("com.mongodb.spark.sql")
            .save()

        //TODO：计算电影相似度矩阵

        // 关闭spark
        spark.stop()
    }
}