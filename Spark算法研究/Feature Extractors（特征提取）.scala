1.
TF-IDF（词频-逆向文档频率）
    ：因为是词频，去除噪音词效果肯定比较好。（优化，去除停用词）

    import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "其中 乾 清宫 非常 精美 ， 午门 紫禁城 正门 。"),
      (0.0, "故宫 著名景点 乾 清宫 、 太和殿 和 午门 。"),
      (1.0, "我 我 我 是 是 闭雨哲")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(30) //设置哈希表的大小

    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.select("words", "rawFeatures").show(3,false)
    +--------------------------------------+-------------------------------------------------------------------+
    |words                                 |rawFeatures                                                        |
    +--------------------------------------+-------------------------------------------------------------------+
    |[其中, 乾, 清宫, 非常, 精美, ，, 午门, 紫禁城, 正门, 。]|(30,[1,2,4,5,7,10,18,20,27],[1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0]) |
    |[故宫, 著名景点, 乾, 清宫, 、, 太和殿, 和, 午门, 。]   |(30,[1,2,4,7,14,19,20,21,23],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
    |[我, 我, 我, 是, 是, 闭雨哲]                  |(30,[17,18,25],[3.0,2.0,1.0])                                      |
    +--------------------------------------+-------------------------------------------------------------------+

    /** 另外，countvectorizer 也～可以获得词频向量 - alternatively, CountVectorizer can also be used to get term frequency vectors*/

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show(false)


1、HashingTF 是使用哈希表来存储分词，并计算分词频数（TF），生成HashMap表。在Map中，K为分词对应索引号，V为分词的频数。
   ：在声明HashingTF 时，需要设置numFeatures，该属性实为设置哈希表的大小；如果设置numFeatures过小，则在存储分词时会出现重叠现象，所以不要设置太小，一般情况下设置为30w~50w之间。
   ：参数-- setBinary(value: Boolean)，来实验是否要追踪词语出现次数，还是只简单的记录词语出现与否。
   ：	    If true, all non-zero counts are set to 1. 这对于模拟二进制事件而不是整数计数的离散概率模型很有用。 (default = false)

2、IDF是计算每个分词出现在文章中的次数，并计算log值。
   ：参数-- minDocFreq，即过滤掉出现文章数小于minDocFreq的分词。默认0，一般1/2.

3、IDFModel 主要是计算TF*IDF，另外IDFModel也可以将IDF数据保存下来（即模型的保存），在测试语料时，只需要计算测试语料中每个分词的在该篇文章中的词频TF，就可以计算TFIDF。





#----------------------------

2.Word2Vec
    Word2Vec是一个Estimator(评估器)，它采用表示文档的单词序列，并训练一个Word2VecModel。
    该模型将每个单词映射到一个唯一的固定大小向量。 Word2VecModel使用文档中所有单词的平均值将每个文档转换为向量;
    该向量然后可用作预测，文档相似性计算等功能。

    import org.apache.spark.ml.feature.Word2Vec
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.Row

    // Input data: Each row is a bag of words from a sentence or document.
    val documentDF1 = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")
    val documentDF = spark.createDataFrame(Seq(
      (0.0, "其中 乾 清宫 非常 精美 ， 午门 紫禁城 正门 。".split(" ")),
      (0.0, "故宫 著名景点 乾 清宫 、 太和殿 和 午门 。".split(" ")),
      (1.0, "我 我 我 是 是 闭雨哲".split(" "))
    )).toDF("label", "text")


    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec().
      setInputCol("text").
      setOutputCol("result").
      setMaxSentenceLength(3000).
      setVectorSize(10).
      setMinCount(0)
    val model = word2Vec.fit(documentDF)

    val result = model.transform(documentDF)
    result.select("text","result").collect().foreach { case Row(text: Seq[_], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }


      Value Members
      def
        setLearningRate(learningRate: Double): Word2Vec.this.type
        Sets initial learning rate (default: 0.025).
      def
        setMaxSentenceLength(maxSentenceLength: Int): Word2Vec.this.type
        Sets the maximum length (in words) of each sentence in the input data.
        设置每个句子最大值 长度（ in words ）在输入数据中。
        任何超过此阈值的句子将被分成最大大小（默认值：1000）
      def
        setMinCount(minCount: Int): Word2Vec.this.type
        Sets minCount, the minimum number of times a token must appear to be included in the word2vec model's vocabulary (default: 5).
      def
        setNumIterations(numIterations: Int): Word2Vec.this.type
        Sets number of iterations (default: 1), which should be smaller than or equal to number of partitions.
      def
        setNumPartitions(numPartitions: Int): Word2Vec.this.type
        Sets number of partitions (default: 1).
      def
        setSeed(seed: Long): Word2Vec.this.type
        Sets random seed (default: a random long integer).
      def
        setVectorSize(vectorSize: Int): Word2Vec.this.type
        Sets vector size (default: 100).词向量的维度，默认值是100。
        这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。
      def
        setWindowSize(window: Int): Word2Vec.this.type
        Sets the window of words (default: 5)


3.CountVectorizer
    CountVectorizer和CountVectorizerModel旨在帮助将文本文档集合转换为标记数的向量。
    当先验词典不可用时，CountVectorizer可以用作估计器来提取词汇表，并生成CountVectorizerModel。 该模型通过词汇生成文档的稀疏表示，然后可以将其传递给其他算法，如LDA。
    在拟合过程中，
      .setVocabSize(3): 将选择通过语料库按术语频率排序的top前几vocabSize词。
      .setMinDF(2): 通过指定术语必须出现以包含在词汇表中的文档的最小数量（或小于1.0）来影响拟合过程。
      .setBinary(true): 可选的二进制切换参数控制输出向量。 如果设置为true，则所有非零计数都设置为1.对于模拟二进制而不是整数的离散概率模型，这是非常有用的。


    import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c","g")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")

    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("features").
      setVocabSize(4). // topN个词
      setMinDF(1).     // 词出现的最小次数
      setBinary(true). // 所有有统计的计算量都替换成1.0，而不是原来的一个词出现几次了
      fit(df)

    cvModel.transform(df).show(false)
