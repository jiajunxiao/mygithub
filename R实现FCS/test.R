library(ggplot2)
require(cowplot)
# 导入数据
svm_score=read.table("E:/matlab文件/评价指标/svm_score.txt",header=FALSE,sep ="")
tree_score=read.table("E:/matlab文件/评价指标/tree_score.txt",header=FALSE,sep ="")
print("svm数据")
print(svm_score)
print("随机森林数据")
print(tree_score)
# 绘制svm的FCS图
s <- ggplot(svm_score,aes(V1,fill=factor(V2)))+geom_histogram(position='identity',bins=30,alpha=0.5)+
  labs(x="Score",y="Frequency",title="svm的FCS图")
s
# 绘制随机森林的svm图
t <- ggplot(tree_score,aes(V1,fill=factor(V2)))+geom_histogram(position='identity',bins=30,alpha=0.5)+
  labs(x="Score",y="Frequency",title="随机森林的FCS图")
t
# 合并
p5 <- plot_grid(s, t, ncol = 2, labels = LETTERS[1:2])
p5
