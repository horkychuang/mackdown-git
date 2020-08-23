# Decision Tree
Decision tree builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast and Rainy). Leaf node (e.g., Play) represents a classification or decision. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.

![](https://i.imgur.com/JiyO5c1.png)

## Algorithm
The core algorithm for building decision trees called ID3 by J. R. Quinlan which employs a top-down, greedy search through the space of possible branches with no backtracking. ID3 uses *Entropy* & and *Information Gain* to construct a decision tree. 

Entropy
A decision tree is built top-down from a root node and involves partitioning the data into subsets that contain instances with similar values (homogenous). ID3 algorithm uses entropy to calculate the homogeneity of a sample. If the sample is completely homogeneous the entropy is zero and if the sample is an equally divided it has entropy of one. 熵可以衡量混亂程度 

![](https://i.imgur.com/e9nTp8U.png)

To build a decision tree, we need to calculate two types of entropy using frequency tables as follows:
a) Entropy using the frequency table of one attribute:
$$E\left(S\right)=\sum_{i=1}^{n}-p_{i}\log_{2}p_{i}$$
![](https://i.imgur.com/88eOKsG.png)

b) Entropy using the frequency table of two attributes:
$$E\left(T,X\right)=\sum_{c\in X}P\left(c\right)E\left(c\right)$$
![](https://i.imgur.com/9LkHRHF.png)

Information Gain		
The information gain is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about finding attribute that returns the highest information gain (i.e., the most homogeneous branches).

Step 1: Calculate entropy of the target. 

$\begin{array}{ll}
Entropy\left(PlayGolf\right) & =Entropy\left(5,9\right)\\
 & =Entropy\left(0.36,0.64\right)\\
 & =-\left(0.36\log_{2}0.36\right)-\left(0.64\log_{2}0.64\right)\\
 & =0.94
\end{array}$

Step 2: The dataset is then split on the different attributes. The entropy for each branch is calculated. Then it is added proportionally, to get total entropy for the split. The resulting entropy is subtracted from the entropy before the split. The result is the Information Gain, or decrease in entropy.

![](https://i.imgur.com/TMKYvan.png)
$Gain\left(T,X\right)=Entropy\left(T\right)-Entropy\left(T,X\right)$

$\begin{array}{ll}
G\left(PlayGolf,Outlook\right) & =E\left(PlayGolf\right)-E\left(PlayGolf,Outlook\right)\\
 & =0.940-0.693=0.247
\end{array}$

Step 3: Choose attribute with the largest information gain as the decision node, divide the dataset by its branches and repeat the same process on every branch.

![](https://i.imgur.com/keb85Fg.png)

Step 4a: A branch with entropy of 0 is a leaf node
![](https://i.imgur.com/aue5G1E.png)

Step 4b: A branch with entropy more than 0 needs further splitting.
![](https://i.imgur.com/3rqDWrH.png)

Step 5: The ID3 algorithm is run recursively on the non-leaf branches, until all data is classified.

Decision Tree to Decision Rules
A decision tree can easily be transformed to a set of rules by mapping from the root node to the leaf nodes one by one.

![](https://i.imgur.com/msTl4Wh.png)

Decision Tree - Overfitting
Overfitting is a significant practical difficulty for decision tree models and many other predictive models. Overfitting happens when the learning algorithm continues to develop hypotheses that reduce training set error at the cost of an increased test set error. There are several approaches to avoiding overfitting in building decision trees. 

Pre-pruning that stop growing the tree earlier, before it perfectly classifies the training set.
Post-pruning that allows the tree to perfectly classify the training set, and then post prune the tree. 

Practically, the second approach of post-pruning overfit trees is more successful because it is not easy to precisely estimate when to stop growing the tree. 		
The important step of tree pruning is to define a criterion be used to determine the correct final tree size using one of the following methods:		
1. Use a distinct dataset from the training set (called validation set), to evaluate the effect of post-pruning nodes from the tree.
2. Build the tree by using the training set, then apply a statistical test to estimate whether pruning or expanding a particular node is likely to produce an improvement beyond the training set.
     Error estimation
     Significance testing (e.g., Chi-square test)
3. Minimum Description Length principle : Use an explicit measure of the complexity for encoding the training set and the decision tree, stopping growth of the tree when this encoding size (size(tree) + size(misclassifications(tree)) is minimized.

The first method is the most common approach. In this approach, the available data are separated into two sets of examples: a training set, which is used to build the decision tree, and a validation set, which is used to evaluate the impact of pruning the tree. The second method is also a common approach. Here, we explain the error estimation and Chi2 test.

Post-pruning using Error estimation
error estimate for a sub-tree is weighted sum of error estimates for all its leaves. The error estimate $(e)$ for a node is:

$$e=\left(f+\frac{z^{2}}{2N}+z\sqrt{\frac{f}{N}-\frac{f^{2}}{N}+\frac{z^{2}}{4N^{2}}}\right)/\left(1+\frac{z^{2}}{N}\right)$$
where
$f$ is the error on the training data
$N$ is the number of instances cover by leaf
$z$ from normal distrbution

In the following example we set Z to 0.69 which is equal to a confidence level of 75%.

![](https://i.imgur.com/PejUshV.png)

The error rate at the parent node is 0.46 and since the error rate for its children (0.51) increases with the split, we do not want to keep the children.

Post-pruning using Chi2 test		
In Chi2 test we construct the corresponding frequency table and calculate the Chi2 value and its probability.		



|          | Bronze   | Silver   |Gold |
| -------- | -------- | -------- |-----|
|  Bad     |   4      |    1     |  4  |
|  Good    |   2      |    1     | 2   |
Chi2 = 0.21          Probability = 0.90         degree of freedom=2

if we require that the probability has to be less than a limit (e.g., 0.05), therefore we decide not to split the node.

```
#------------------
# Data Preparation
#------------------

#Read datasets
#Download the data from http://www.saedsayad.com/datasets/CreditData.zip
train <- read.csv("Credit_train.csv")
test <- read.csv("Credit_test.csv")

#Rows and Cols
dim(train)
dim(test)

#Columns name
colnames(train)
colnames(test)

#Show  
head(train)
head(test)


#---------------
# Decision tree
#---------------
library(caret)
library(rpart)
library(rpart.plot)	
library(AUC)

#train
model.Dtree <- rpart(DEFAULT~., data = train, method="class")
prp(model.Dtree)

#lift chart
pb <- NULL
pb <- predict(model.Dtree, test)
pb <- as.data.frame(pb)
pred.Dtree <- data.frame(test$DEFAULT, pb$Y)
colnames(pred.Dtree) <- c("target","score")
lift.Dtree <- lift(target ~ score, data = pred.Dtree, cuts=10, class="Y")
xyplot(lift.Dtree, main="Decision Tree - Lift Chart", type=c("l","g"), lwd=2
       , scales=list(x=list(alternating=FALSE,tick.number = 10)
                     ,y=list(alternating=FALSE,tick.number = 10)))

#confusion matrix
pc <- NULL
pc <- ifelse(pb$N > pb$Y, "N", "Y")
summary(as.data.frame(pc))
xtab <- table(pc, test$DEFAULT)
caret::confusionMatrix(xtab, positive = "Y")

#roc chart
labels <- as.factor(ifelse(pred.Dtree$target=="Y", 1, 0))
predictions <- pred.Dtree$score
auc(roc(predictions, labels), min = 0, max = 1)
plot(roc(predictions, labels), min=0, max=1, type="l", main="Decision Tree - ROC Chart")

```
By default, rpart uses gini impurity to select splits. 
mytree <- rpart(
  Fraud ~ RearEnd, 
  data = train, 
  method = "class",
  parms = list(split = 'information'), 
  minsplit = 2, 
  minbucket = 1
) 


```{r}
install.packages("rpart.plot")
install.packages("rpart")
library(rpart)
library(rpart.plot)

fit <- rpart(survived~., data = data_train, method = 'class')
rpart.plot(fit, extra = 106) # The extra features are set to 101 to display the probability of the 2nd class (useful for binary responses)

prp(cart.model,         # 模型
    faclen=0,           # 呈現的變數不要縮寫
    fallen.leaves=TRUE, # 讓樹枝以垂直方式呈現
    shadow.col="gray",  # 最下面的節點塗上陰影
    # number of correct classifications / number of observations in that node
    extra=2) 

```

Tune the hyper-parameters
Decision tree has various parameters that control aspects of the fit. In rpart library, you can control the parameters using the rpart.control() function. 

printcp(cart.model) # 先觀察未修剪的樹，CP欄位代表樹的成本複雜度參數
plotcp(cart.model) # 畫圖觀察未修剪的樹

自動選擇xerror最小時候對應的cp值來剪枝
prunetree_cart.model <- prune(cart.model, cp = cart.model$cptable[which.min(cart.model$cptable[,"xerror"]),"CP"]) # 利用能使決策樹具有最小誤差的CP來修剪樹

The complexity measure is a combination of the size of a tree and the ability of the tree to separate the classes of the target variable.
cp是引數複雜度（complexity parameter）作為控制樹規模的懲罰因子，簡而言之，就是cp越大，樹分裂規模（nsplit）越小。輸出引數（rel error）指示了當前分類模型樹與空樹之間的平均偏差比值。xerror為交叉驗證誤差，xstd為交叉驗證誤差的標準差。可以看到，當nsplit為3的時候，即有四個葉子結點的樹，要比nsplit為4，即五個葉子結點的樹的交叉誤差要小。而決策樹剪枝的目的就是為了得到更小交叉誤差（xerror）的樹
The‘CP’ stands for Complexity Parameter of the tree. We want the cp value of the smallest tree that has smallest cross validation error. we can we pick the tree having CP = XXXXX as it has least cross validation error (xerror). 


# prune the tree 
pfit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(pfit, uniform=TRUE, 
   main="Pruned Classification Tree for Kyphosis")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)
post(pfit, file = "c:/ptree.ps", 
   title = "Pruned Classification Tree for Kyphosis")

nice site
https://www.gormanalysis.com/blog/decision-trees-in-r-using-rpart/





