（正则化的前提是-特征值要进行归一化。）- 这句话还不能好好理解，是不是训练模型前需要做归一化处理，再使用模型里的正则化选项。
正则化的前提是-特征值要进行归一化。
	：在实际应该过程中，为了增强模型的(泛化能力)，防止我们训练的模型过拟合，特别是对于大量的稀疏特征，模型复杂度比较高，需要进行降维，我们需要保证在-(训练误差最小化)的基础上，通过加上正则化项减小模型复杂度。
	：在逻辑回归中，有L1、L2进行正则化。

	：在损失函数里加入一个正则化项，正则化项就是权重的L1或者L2范数乘以一个λ，用来控制损失函数和正则化项的比重，直观的理解，首先防止过拟合的目的就是防止最后训练出来的模型过分的依赖某一个特征，当最小化损失函数的时候，(某一维度很大)，拟合出来的函数值与真实的值之间的差距很小，通过正则化可以使(整体的cost)变大，从而避免了过分依赖某一维度的结果。当然加正则化的前提是特征值要进行归一化。



与二分类回归的不同：
    ：需设置.setFamily("multinomial")
    ：不能使用 trainingSummary.asInstanceOf[BinaryLogisticRegressionTrainingSummary] 评估模型指标
    ：coefficients(系数) and intercept(截距) 在多元回归中是不支持的. 使用 coefficientMatrix and interceptVector 代替.



所有参数：
  scala> println(lr.explainParams())
      aggregationDepth: suggested depth for treeAggregate (>= 2) (default: 2)
      elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty (default: 0.0)
      family: The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial. (default: auto)
      featuresCol: features column name (default: features)
      fitIntercept: whether to fit an intercept term (default: true)
      labelCol: label column name (default: label)
      maxIter: maximum number of iterations (>= 0) (default: 100)
      predictionCol: prediction column name (default: prediction)
      probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities (default: probability)
      rawPredictionCol: raw prediction (a.k.a. confidence) column name (default: rawPrediction)
      regParam: regularization parameter (>= 0) (default: 0.0)
      standardization: whether to standardize the training features before fitting the model (default: true)
      threshold: threshold in binary classification prediction, in range [0, 1] (default: 0.5)
      thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0 excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold (undefined)
      tol: the convergence tolerance for iterative algorithms (>= 0) (default: 1.0E-6)
      weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0 (undefined)


lr Parameter setters：
    val lr = new LogisticRegression()
    lr
      .setElasticNetParam(0.8)   // lr.elasticNetParam.doc ->(the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty
        ratio:在0到1之间，代表在l1惩罚和l2惩罚之间，如果_ratio=1，则为lasso，是调节模型性能的一个重要指标。
        ElasticNet 是一种使用L1和L2先验作为正则化矩阵的线性回归模型.这种组合用于只有很少的权重非零的稀疏模型，比如:class:Lasso, 但是又能保持:class:Ridge 的正则化属性.
        我们可以使用 ratio 参数来调节L1和L2的凸组合(一类特殊的线性组合)。 
        当多个特征和另一个特征相关的时候弹性网络非常有用。Lasso 倾向于随机选择其中一个，而弹性网络更倾向于选择两个. 
        在实践中，Lasso 和 Ridge 之间权衡的一个优势是它允许在循环过程（Under rotate）中继承 Ridge 的稳定性. 

        L1正则化使得模型更加稀疏，L2使得模型参数更趋近于0，提高泛化能力
        而正则化是通过约束参数的范数使其不要太大，所以可以在一定程度上减少过拟合情况。 
        弹性网络（Elastic Net）：实际上是L1，L2的综合
        L0范数：就是指矩阵中非零元素的个数，很显然，在损失函数后面加上L0正则项就能够得到稀疏解，但是L0范数很难求解，是一个NP问题，因此转为求解相对容易的L1范数（l1能够实现稀疏性是因为l1是L0范数的最优凸近似） 
        L1范数：矩阵中所有元素的绝对值的和。损失函数后面加上L1正则项就成了著名的Lasso问题（Least Absolute Shrinkage and Selection Operator），L1范数可以约束方程的稀疏性，该稀疏性可应用于特征选择： 
            比如，有一个分类问题，其中一个类别Yi(i=0,1),特征向量为Xj（j=0,1~~~1000），那么构造一个方程 
            Yi = W0*X0+W1*X1···Wj*Xj···W1000*X1000+b; 
            其中W为权重系数，那么通过L1范数约束求解，得到的W系数是稀疏的，那么对应的X值可能就是比较重要的，这样就达到了特征选择的目的（该例子是自己思考后得出的，不知道正不正确，欢迎指正）。
        L2范数： 其实就是矩阵所有元素的平方和开根号，即欧式距离，在回归问题中，在损失函数（或代价函数）后面加上L2正则项就变成了岭回归（Ridge Regression），也有人叫他权重衰减，L2正则项的一个很大的用处就是用于防止机器学习中的过拟合问题，同L1范数一样，L2范数也可以对方程的解进行约束，但他的约束相对L1更平滑，在模型预测中，L2往往比L1好。L2会让W的每个元素都很小，接近于0，但是不会等于0.而越小的参数模型越简单，越不容易产生过拟合，以下引自另一篇文章： 到目前为止，我们只是解释了L2正则化项有让w“变小”的效果（公式中的lamda越大，最后求得的w越小），但是还没解释为什么w“变小”可以防止overfitting？一个所谓“显而易见”的解释就是：更小的权值w，从某种意义上说，表示网络的复杂度更低，对数据的拟合刚刚好（这个法则也叫做奥卡姆剃刀），而在实际应用中，也验证了这一点，L2正则化的效果往往好于未经正则化的效果。



      .setFamily("multinomial")  // Default is "auto".
      .setFeaturesCol("feature") //
      .setLabelCol("label")
      .setFitIntercept(true)     // 设置算法是否应该加 截距; （权重优化，进行梯度下降学习，返回最优权重。
      .setMaxIter(10)            // Default is 100.
      .setPredictionCol("")      // Default is "prediction" -(prediction column name)
      .setRawPredictionCol("")   // Default: "rawPrediction" (可以设置一个名字，如"score"，代表输出的DataFrame列名。所得结果表示为预测为每一类的得分，有多少类就有多少个得分，取得分最小的 则判别为该类。)
      .setProbabilityCol("")     // Default: "probability" (预测类条件概率的列名,得到样本属于N个类的概率.)。注：并非所有模型输出都有校准概率估计！这些结果应被视为机密，不精确的概率。
      .setRegParam(0.3)          // Default is 0.0. (设置正则化参数[0,1]。0=L1,1=L2) - Set the regularization parameter
      .setStandardization(true)  // Default is true.(在模型拟合前是否规范训练特征) - 注:有/没有标准化，模型都会收敛到相同的解决方案。
      .setThreshold(0.5)         // Default is 0.5. (设置在二分类问题中, in range [0, 1])。注：When setThreshold(), 任何用户设置的 thresholds()都将被清除; If both（threshold and thresholds）are set in a ParamMap, 他们必须等价.
      .setThresholds(Array(0.6,0.4))// 在多（或二）分类中设置阈值以调整每个类的预测概率 。(如果预测得分小于0.6，则预测为1类)数组长度必须=类数目，值大于0 且最多一个值可置为0; 这个类预测的最大值( p/t ) p:该类的原始概率 t:该类的阈值 --（预测为该类的概率，所有加起来不用=1,设置可能需要不断的取调整）还是不太明白。网上找不到，貌似是可以跳过的。
      .setTol(1e-6)              // Default is 1E-6.(设置迭代的收敛容差-步长)较小的值将导致更高的精度与更多的迭代成本。默认是1e-6。
      .setWeightCol("")          // 默认不用设置. （权重列-权重较高的列）-- 如果不设置,将所有特征值权重看作1.0
      .setAggregationDepth(2)    // Default is 2.（树深度）- 如果特征或分区多，建议调大该参数 - (greater than or equal to 2)






lrModel Members：
    val mlrModel = mlr.fit(training)
