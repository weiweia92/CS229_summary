## Adaboost

假设我们的训练集样本是

$$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_m,y_m))\}$$

训练集的在第k个弱学习器的输出权重为

$$D(k)=(w_{k1},w_{k2},\cdots,w_{km}); w_{1i}=\frac{1}{m};i=1,2,\cdots,m$$
 
首先我们看看Adaboost的分类问题。

分类问题的误差率很好理解和计算。由于多元分类是二元分类的推广，这里假设我们是二元分类问题，输出为{-1，1}，则第k个弱分类器𝐺𝑘(𝑥)在训练集上的加权误差率为
𝑒𝑘=𝑃(𝐺𝑘(𝑥𝑖)≠𝑦𝑖)=∑𝑖=1𝑚𝑤𝑘𝑖𝐼(𝐺𝑘(𝑥𝑖)≠𝑦𝑖)
　　　　接着我们看弱学习器权重系数,对于二元分类问题，第k个弱分类器𝐺𝑘(𝑥)的权重系数为
𝛼𝑘=12𝑙𝑜𝑔1−𝑒𝑘𝑒𝑘
　　　　为什么这样计算弱学习器权重系数？从上式可以看出，如果分类误差率𝑒𝑘越大，则对应的弱分类器权重系数𝛼𝑘越小。也就是说，误差率小的弱分类器权重系数越大。具体为什么采用这个权重系数公式，我们在讲Adaboost的损失函数优化时再讲。

　　　　第三个问题，更新更新样本权重D。假设第k个弱分类器的样本集权重系数为𝐷(𝑘)=(𝑤𝑘1,𝑤𝑘2,...𝑤𝑘𝑚)，则对应的第k+1个弱分类器的样本集权重系数为
𝑤𝑘+1,𝑖=𝑤𝑘𝑖𝑍𝐾𝑒𝑥𝑝(−𝛼𝑘𝑦𝑖𝐺𝑘(𝑥𝑖))
　　　　这里𝑍𝑘是规范化因子
𝑍𝑘=∑𝑖=1𝑚𝑤𝑘𝑖𝑒𝑥𝑝(−𝛼𝑘𝑦𝑖𝐺𝑘(𝑥𝑖))
　　　　从𝑤𝑘+1,𝑖计算公式可以看出，如果第i个样本分类错误，则𝑦𝑖𝐺𝑘(𝑥𝑖)<0，导致样本的权重在第k+1个弱分类器中增大，如果分类正确，则权重在第k+1个弱分类器中减少.具体为什么采用样本权重更新公式，我们在讲Adaboost的损失函数优化时再讲。

　　　　最后一个问题是集合策略。Adaboost分类采用的是加权表决法，最终的强分类器为
𝑓(𝑥)=𝑠𝑖𝑔𝑛(∑𝑘=1𝐾𝛼𝑘𝐺𝑘(𝑥))
　　　　

　　　　接着我们看看Adaboost的回归问题。由于Adaboost的回归问题有很多变种，这里我们以Adaboost R2算法为准。

　　　　我们先看看回归问题的误差率的问题，对于第k个弱学习器，计算他在训练集上的最大误差
𝐸𝑘=𝑚𝑎𝑥|𝑦𝑖−𝐺𝑘(𝑥𝑖)|𝑖=1,2...𝑚
　　　　然后计算每个样本的相对误差
𝑒𝑘𝑖=|𝑦𝑖−𝐺𝑘(𝑥𝑖)|𝐸𝑘
　　　　这里是误差损失为线性时的情况，如果我们用平方误差，则𝑒𝑘𝑖=(𝑦𝑖−𝐺𝑘(𝑥𝑖))2𝐸2𝑘,如果我们用的是指数误差，则𝑒𝑘𝑖=1−𝑒𝑥𝑝（−|𝑦𝑖−𝐺𝑘(𝑥𝑖)|)𝐸𝑘）
　　　　最终得到第k个弱学习器的 误差率
𝑒𝑘=∑𝑖=1𝑚𝑤𝑘𝑖𝑒𝑘𝑖
　　　　我们再来看看如何得到弱学习器权重系数𝛼。这里有：
𝛼𝑘=𝑒𝑘1−𝑒𝑘
　　　　对于更新更新样本权重D，第k+1个弱学习器的样本集权重系数为
𝑤𝑘+1,𝑖=𝑤𝑘𝑖𝑍𝑘𝛼1−𝑒𝑘𝑖𝑘
　　　　这里𝑍𝑘是规范化因子
𝑍𝑘=∑𝑖=1𝑚𝑤𝑘𝑖𝛼1−𝑒𝑘𝑖𝑘
　　　　最后是结合策略，和分类问题稍有不同，采用的是对加权的弱学习器取权重中位数对应的弱学习器作为强学习器的方法，最终的强回归器为
𝑓(𝑥)=𝐺𝑘∗(𝑥)
　　　　其中，𝐺𝑘∗(𝑥)是所有𝑙𝑛1𝛼𝑘,𝑘=1,2,....𝐾的中位数值对应序号𝑘∗对应的弱学习器。　

3. AdaBoost分类问题的损失函数优化
　　　　刚才上一节我们讲到了分类Adaboost的弱学习器权重系数公式和样本权重更新公式。但是没有解释选择这个公式的原因，让人觉得是魔法公式一样。其实它可以从Adaboost的损失函数推导出来。

　　　　从另一个角度讲，　Adaboost是模型为加法模型，学习算法为前向分步学习算法，损失函数为指数函数的分类问题。

　　　　模型为加法模型好理解，我们的最终的强分类器是若干个弱分类器加权平均而得到的。

　　　　前向分步学习算法也好理解，我们的算法是通过一轮轮的弱学习器学习，利用前一个强学习器的结果和当前弱学习器来更新当前的强学习器的模型。也就是说，第k-1轮的强学习器为
𝑓𝑘−1(𝑥)=∑𝑖=1𝑘−1𝛼𝑖𝐺𝑖(𝑥)
　　　　而第k轮的强学习器为
𝑓𝑘(𝑥)=∑𝑖=1𝑘𝛼𝑖𝐺𝑖(𝑥)
　　　　上两式一比较可以得到
𝑓𝑘(𝑥)=𝑓𝑘−1(𝑥)+𝛼𝑘𝐺𝑘(𝑥)
　　　　可见强学习器的确是通过前向分步学习算法一步步而得到的。

　　　　Adaboost损失函数为指数函数，即定义损失函数为
𝑎𝑟𝑔𝑚𝑖𝑛𝛼,𝐺∑𝑖=1𝑚𝑒𝑥𝑝(−𝑦𝑖𝑓𝑘(𝑥))
　　　　利用前向分步学习算法的关系可以得到损失函数为
(𝛼𝑘,𝐺𝑘(𝑥))=𝑎𝑟𝑔𝑚𝑖𝑛𝛼,𝐺∑𝑖=1𝑚𝑒𝑥𝑝[(−𝑦𝑖)(𝑓𝑘−1(𝑥)+𝛼𝐺(𝑥))]
　　　　令𝑤′𝑘𝑖=𝑒𝑥𝑝(−𝑦𝑖𝑓𝑘−1(𝑥)), 它的值不依赖于𝛼,𝐺,因此与最小化无关，仅仅依赖于𝑓𝑘−1(𝑥),随着每一轮迭代而改变。

　　　　将这个式子带入损失函数,损失函数转化为
(𝛼𝑘,𝐺𝑘(𝑥))=𝑎𝑟𝑔𝑚𝑖𝑛𝛼,𝐺∑𝑖=1𝑚𝑤′𝑘𝑖𝑒𝑥𝑝[−𝑦𝑖𝛼𝐺(𝑥)]
　　　　

　　　　首先，我们求𝐺𝑘(𝑥).，
∑𝑖=1𝑚𝑤′𝑘𝑖𝑒𝑥𝑝(−𝑦𝑖𝛼𝐺(𝑥𝑖))=∑𝑦𝑖=𝐺𝑘(𝑥𝑖)𝑤′𝑘𝑖𝑒−𝛼+∑𝑦𝑖≠𝐺𝑘(𝑥𝑖)𝑤′𝑘𝑖𝑒𝛼=(𝑒𝛼−𝑒−𝛼)∑𝑖=1𝑚𝑤′𝑘𝑖𝐼(𝑦𝑖≠𝐺𝑘(𝑥𝑖))+𝑒−𝛼∑𝑖=1𝑚𝑤′𝑘𝑖(1)(2)
　　　　基于上式， 可以得到
𝐺𝑘(𝑥)=𝑎𝑟𝑔𝑚𝑖𝑛𝐺∑𝑖=1𝑚𝑤′𝑘𝑖𝐼(𝑦𝑖≠𝐺(𝑥𝑖))
　　　　将𝐺𝑘(𝑥)带入损失函数，并对𝛼求导，使其等于0，则就得到了
𝛼𝑘=12𝑙𝑜𝑔1−𝑒𝑘𝑒𝑘
　　　　其中，𝑒𝑘即为我们前面的分类误差率。
𝑒𝑘=∑𝑖=1𝑚𝑤′𝑘𝑖𝐼(𝑦𝑖≠𝐺(𝑥𝑖))∑𝑖=1𝑚𝑤′𝑘𝑖=∑𝑖=1𝑚𝑤𝑘𝑖𝐼(𝑦𝑖≠𝐺(𝑥𝑖))
　　　　最后看样本权重的更新。利用𝑓𝑘(𝑥)=𝑓𝑘−1(𝑥)+𝛼𝑘𝐺𝑘(𝑥)和𝑤′𝑘𝑖=𝑒𝑥𝑝(−𝑦𝑖𝑓𝑘−1(𝑥))，即可得：
𝑤′𝑘+1,𝑖=𝑤′𝑘𝑖𝑒𝑥𝑝[−𝑦𝑖𝛼𝑘𝐺𝑘(𝑥)]
　　　　这样就得到了我们第二节的样本权重更新公式。

4. AdaBoost二元分类问题算法流程
　　　　这里我们对AdaBoost二元分类问题算法流程做一个总结。

　　　　输入为样本集𝑇={(𝑥,𝑦1),(𝑥2,𝑦2),...(𝑥𝑚,𝑦𝑚)}，输出为{-1, +1}，弱分类器算法, 弱分类器迭代次数K。

　　　　输出为最终的强分类器𝑓(𝑥)
　　　　1) 初始化样本集权重为
𝐷(1)=(𝑤11,𝑤12,...𝑤1𝑚);𝑤1𝑖=1𝑚;𝑖=1,2...𝑚
　　　　2) 对于k=1,2，...K:

　　　　　　a) 使用具有权重𝐷𝑘的样本集来训练数据，得到弱分类器𝐺𝑘(𝑥)
　　　　　　b)计算𝐺𝑘(𝑥)的分类误差率
𝑒𝑘=𝑃(𝐺𝑘(𝑥𝑖)≠𝑦𝑖)=∑𝑖=1𝑚𝑤𝑘𝑖𝐼(𝐺𝑘(𝑥𝑖)≠𝑦𝑖)
　　　　　　c) 计算弱分类器的系数
𝛼𝑘=12𝑙𝑜𝑔1−𝑒𝑘𝑒𝑘
　　　　　　d) 更新样本集的权重分布
𝑤𝑘+1,𝑖=𝑤𝑘𝑖𝑍𝐾𝑒𝑥𝑝(−𝛼𝑘𝑦𝑖𝐺𝑘(𝑥𝑖))𝑖=1,2,...𝑚
　　　　　　　　这里𝑍𝑘是规范化因子
𝑍𝑘=∑𝑖=1𝑚𝑤𝑘𝑖𝑒𝑥𝑝(−𝛼𝑘𝑦𝑖𝐺𝑘(𝑥𝑖))
　　　　3) 构建最终分类器为：
𝑓(𝑥)=𝑠𝑖𝑔𝑛(∑𝑘=1𝐾𝛼𝑘𝐺𝑘(𝑥))
　　　　

 

　　　　对于Adaboost多元分类算法，其实原理和二元分类类似，最主要区别在弱分类器的系数上。比如Adaboost SAMME算法，它的弱分类器的系数
𝛼𝑘=12𝑙𝑜𝑔1−𝑒𝑘𝑒𝑘+𝑙𝑜𝑔(𝑅−1)
　　　　其中R为类别数。从上式可以看出，如果是二元分类，R=2，则上式和我们的二元分类算法中的弱分类器的系数一致。

5. Adaboost回归问题的算法流程
　　　　这里我们对AdaBoost回归问题算法流程做一个总结。AdaBoost回归算法变种很多，下面的算法为Adaboost R2回归算法过程。

　　　　输入为样本集𝑇={(𝑥,𝑦1),(𝑥2,𝑦2),...(𝑥𝑚,𝑦𝑚)}，，弱学习器算法, 弱学习器迭代次数K。

　　　　输出为最终的强学习器𝑓(𝑥)
　　　　1) 初始化样本集权重为
𝐷(1)=(𝑤11,𝑤12,...𝑤1𝑚);𝑤1𝑖=1𝑚;𝑖=1,2...𝑚
　　　　2) 对于k=1,2，...K:

　　　　　　a) 使用具有权重𝐷𝑘的样本集来训练数据，得到弱学习器𝐺𝑘(𝑥)
　　　　　　b) 计算训练集上的最大误差
𝐸𝑘=𝑚𝑎𝑥|𝑦𝑖−𝐺𝑘(𝑥𝑖)|𝑖=1,2...𝑚
　　　　　　c) 计算每个样本的相对误差:

　　　　　　　　如果是线性误差，则𝑒𝑘𝑖=|𝑦𝑖−𝐺𝑘(𝑥𝑖)|𝐸𝑘；

　　　　　　　　如果是平方误差，则𝑒𝑘𝑖=(𝑦𝑖−𝐺𝑘(𝑥𝑖))2𝐸2𝑘
　　　　　　　　如果是指数误差，则𝑒𝑘𝑖=1−𝑒𝑥𝑝（−|𝑦𝑖−𝐺𝑘(𝑥𝑖)|𝐸𝑘）　　　　　　　　

　　　　　　d) 计算回归误差率
𝑒𝑘=∑𝑖=1𝑚𝑤𝑘𝑖𝑒𝑘𝑖
　　　　　　c) 计算弱学习器的系数
𝛼𝑘=𝑒𝑘1−𝑒𝑘
　　　　　　d) 更新样本集的权重分布为
𝑤𝑘+1,𝑖=𝑤𝑘𝑖𝑍𝑘𝛼1−𝑒𝑘𝑖𝑘
　　　　　　　　这里𝑍𝑘是规范化因子
𝑍𝑘=∑𝑖=1𝑚𝑤𝑘𝑖𝛼1−𝑒𝑘𝑖𝑘
　　　　3) 构建最终强学习器为：
𝑓(𝑥)=𝐺𝑘∗(𝑥)
　　　　其中，𝐺𝑘∗(𝑥)是所有𝑙𝑛1𝛼𝑘,𝑘=1,2,....𝐾的中位数值乘以对应序号𝑘∗对应的弱学习器。　　

6. Adaboost算法的正则化
　　　　为了防止Adaboost过拟合，我们通常也会加入正则化项，这个正则化项我们通常称为步长(learning rate)。定义为𝜈,对于前面的弱学习器的迭代
𝑓𝑘(𝑥)=𝑓𝑘−1(𝑥)+𝛼𝑘𝐺𝑘(𝑥)
　　　　如果我们加上了正则化项，则有
𝑓𝑘(𝑥)=𝑓𝑘−1(𝑥)+𝜈𝛼𝑘𝐺𝑘(𝑥)
　　　　𝜈的取值范围为0<𝜈≤1。对于同样的训练集学习效果，较小的𝜈意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。

7. Adaboost小结
　　　　到这里Adaboost就写完了，前面有一个没有提到，就是弱学习器的类型。理论上任何学习器都可以用于Adaboost.但一般来说，使用最广泛的Adaboost弱学习器是决策树和神经网络。对于决策树，Adaboost分类用了CART分类树，而Adaboost回归用了CART回归树。

　　　　这里对Adaboost算法的优缺点做一个总结。

　　　　Adaboost的主要优点有：

　　　　1）Adaboost作为分类器时，分类精度很高

　　　　2）在Adaboost的框架下，可以使用各种回归分类模型来构建弱学习器，非常灵活。

　　　　3）作为简单的二元分类器时，构造简单，结果可理解。

　　　　4）不容易发生过拟合

　　　　Adaboost的主要缺点有：

　　　　1）对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性。