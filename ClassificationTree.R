data <- read.csv("stats.csv")
data <- data[,-c(1,11,12)]
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

tree <- rpart(result ~ ., data = data, method = "class")

fancyRpartPlot(tree)

prunedTree <- prune(tree, cp = 0.02)

fancyRpartPlot(prunedTree)
