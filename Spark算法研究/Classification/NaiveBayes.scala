朴素贝叶斯是用于分类数据的非常简单的生成模型。该实现学习了一个多项朴素贝叶斯分类器，当所有特征都是分类值的计数时，该分类器是适用的。
它假设给定类标签，功能可以彼此独立地生成。

这个API和原MLlib决策树的API之间的主要差异：
    ：支持 ML Pipelines
    ：分类与回归决策树的分离
    ：使用 DataFrame metadata 区分连续和分类特征
    ：决策树的 Pipelines API比原始API提供了更多的功能。特别是，
      分类，用户可以得到每个类的预测概率（名类条件概率）；
      回归，用户可以得到有偏样本方差的预测。
    ：多棵树（随机森林和梯度提升树）后续描述。

Input Columns
    labelCol	Double	"label"	- Label to predict
    featuresCol	Vector	"features"	- Feature vector

Output Columns
    （Param - name -	Type(s) -	Default -	Description -	Notes）
    predictionCol	Double	"prediction"	Predicted label
    （Classification only）rawPredictionCol	Vector	"rawPrediction"	Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction
    （Classification only）probabilityCol	Vector	"probability"	Vector of length # classes equal to rawPrediction normalized to a multinomial distribution
    （Regression only）    varianceCol	Double		The biased sample variance of prediction


val nbModel = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setPredictionCol("prediction")  // 设置预测列 prediction column name (default: prediction)
      .setProbabilityCol("")           // 先验概率
      .setThresholds(Array(0.0,0.1))   // 数量与类别相同,加起来是1
      .setModelType("multinomial")      // "multinomialNB"它假设特征就是出现次数。因此我们以词频当作特征，即分类器可以很好的处理TF-IDF向量。
					// 另一种是 "bernoulli"-和multinomial类似 但更适合判断词语是否出现了这种二值特征 而不是词频统计 ， 默认"multinomial"
					// 还有一种不知道spark有没有支持，GaussianNB-它假设特征是正太分布的，例：根据给定的人物的高度和宽度，判定这个人的性别。-文本提取出词语个数 很明显不是正太分布。
      .setWeightCol("列名")            // 权重列，如果不设置 则认为每列权重都为1



	rawPredictionCol: 可以设置一个名字，如"score"，代表输出的DataFrame列名。所得结果表示为预测为每一类的得分，有多少类就有多少个得分，取得分最小的 则判别为该类。

	smoothing: 设置平滑参数alpha。采用treeAggregate的RDD方法，进行聚合计算，计算每个样本的权重向量、误差值，然后对所有样本权重向量及误差值进行累加。(default = 1.0).ke其设为0.01/0.05/0.5/1/


	.setThresholds(Array(0.6,0.4)) // 在多（或二）分类中设置阈值以调整每个类的预测概率 。(如果预测得分小于0.6，则预测为1类)数组长度必须=类数目，值大于0 且最多一个值可置为0; 这个类预测的最大值( p/t ) p:该类的原始概率 t:该类的阈值 --（预测为该类的概率，所有加起来不用=1,设置可能需要不断的取调整）还是不太明白。网上找不到，貌似是可以跳过的。
	
	thresholds: // 模型预测的值并不是恰好为1或0,预测的输出通常是实数，必须转换为预测类别。这就是通过（分类器决策函数-打分函数）使用（阈值）来实现，例如：二分类中设置阈值为0.5, 于是，如果类别1的概率估计超过50%，这个模型会将样本标记为类别1,否则标记为类别0.

		//在多类分类中的阈值，以调整每个类的预测概率。数组必须具有与类的数目相等的长度，值 > 0，除了至多一个值可能是0。这个最大值 p/t 是类的预测，其中 p是该类原来的概率（original probability 先验概率）和 t是该类的阈值（未定义）

	
	优化：
	     使用uni-grams、big-grams、trig-grams作为TfidfVectorizer中ngram_range参数的参数值，进行交叉验证。-scikit-learn
	    （spark中也可以尝试类似数据处理）



Probability:
  1、prior probability：
　　先验概率（prior probability）是指根据以往经验和分析得到的概率，如全概率公式，它往往作为"由因求果"问题中的"因"出现· 先验概率的分类 利用过去历史资料计算得到的先验概率，称为客观先验概率； 当历史资料无从取得或资料不完全时，凭人们的主观经验来判断而得到的先验概率，称为主观先验概率。 先验概率的条件 先验概率是通过古典概率模型加以定义的，故又称为古典概率。
　　2、empirical probability
　　经验概率(Empirical probability)或称实验概率(experimental probability)，也称为相对频率(Relative frequency)，是指特定的事件发生的次数占总体实验样本的比率。



scala> nbModel.explainParams
res2: String =
featuresCol: features column name (default: features)
labelCol: label column name (default: label, current: label)
modelType: The model type which is a string (case-sensitive). Supported options: multinomial (default) and bernoulli. (default: multinomial)
predictionCol: prediction column name (default: prediction)
probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities (default: probability)
rawPredictionCol: raw prediction (a.k.a. confidence) column name (default: rawPrediction)
smoothing: The smoothing parameter. (default: 1.0)
thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0 excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold (undefined)
weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0 (undefined)





阈值的选取：
    val trainingSummary = lrModel.summary

    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    val roc = binarySummary.roc
    roc.show()
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    val fMeasure = binarySummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
        .select("threshold").head().getDouble(0)
    lrModel.setThreshold(bestThreshold)




缺点：
7. 朴素贝叶斯(Naive Bayes)，“Naive”在何处？
    加上条件独立假设的贝叶斯方法就是朴素贝叶斯方法（Naive Bayes）。
    Naive的发音是“乃一污”，意思是“朴素的”、“幼稚的”、“蠢蠢的”。咳咳，也就是说，大神们取名说该方法是一种比较萌蠢的方法，为啥？
    将句子（“我”,“司”,“可”,“办理”,“正规发票”) 中的 （“我”,“司”）与（“正规发票”）调换一下顺序，
    就变成了一个新的句子（“正规发票”,“可”,“办理”, “我”, “司”)。新句子与旧句子的意思完全不同。

    但由于乘法交换律，朴素贝叶斯方法中算出来二者的条件概率完全一样！计算过程如下：
      P(（“我”,“司”,“可”,“办理”,“正规发票”)|S)
        =P(“我”|S)P(“司”|S)P(“可”|S)P(“办理”|S)P(“正规发票”|S)
        =P(“正规发票”|S)P(“可”|S)P(“办理”|S)P(“我”|S)P(“司”|S）
        =P(（“正规发票”,“可”,“办理”,“我”,“司”)|S)

    也就是说，在朴素贝叶斯眼里，“我司可办理正规发票”与“正规发票可办理我司”完全相同。
    朴素贝叶斯失去了词语之间的顺序信息。这就相当于把所有的词汇扔进到一个袋子里随便搅和，
    贝叶斯都认为它们一样。因此这种情况也称作词袋子模型(bag of words)。

    词袋子模型与人们的日常经验完全不同。比如，在条件独立假设的情况下，“武松打死了老虎”与“老虎打死了武松”被它认作一个意思了。
    恩，朴素贝叶斯就是这么单纯和直接，对比于其他分类器，好像是显得有那么点萌蠢。
