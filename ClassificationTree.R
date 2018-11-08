data <- read.csv("inputdata.csv")
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

tree <- rpart(FTR ~ ., data = data, method = "class")

fancyRpartPlot(tree)

prunedTree <- prune(tree, cp = 0.015)

fancyRpartPlot(prunedTree)
