#wczytanie danych stats.csv
data2<-read.csv(file="stats.csv", header = TRUE, sep = ",")
#standaryzacja
data_temp<-data2
#data_temp[,1]<-(data2[,1]-mean(data2[,1]))/sd(data2[,1])
data_temp[,2]<-(data2[,2]-mean(data2[,2]))/sd(data2[,2])
data_temp[,3]<-(data2[,3]-mean(data2[,3]))/sd(data2[,3])
data_temp[,4]<-(data2[,4]-mean(data2[,4]))/sd(data2[,4])
data_temp[,5]<-(data2[,5]-mean(data2[,5]))/sd(data2[,5])
data_temp[,6]<-(data2[,6]-mean(data2[,6]))/sd(data2[,6])
data_temp[,7]<-(data2[,7]-mean(data2[,7]))/sd(data2[,7])
data_temp[,8]<-(data2[,8]-mean(data2[,8]))/sd(data2[,8])
data_temp[,9]<-(data2[,9]-mean(data2[,9]))/sd(data2[,9])
data_temp[,10]<-(data2[,10]-mean(data2[,10]))/sd(data2[,10])
#data_temp[,11]<-(data2[,11]-mean(data2[,11]))/sd(data2[,11])
#data_temp[,12]<-(data2[,12]-mean(data2[,12]))/sd(data2[,12])
data_temp[,13]<-(data2[,13]-mean(data2[,13]))/sd(data2[,13])
data_temp[,14]<-(data2[,14]-mean(data2[,14]))/sd(data2[,14])
data_temp[,15]<-(data2[,15]-mean(data2[,15]))/sd(data2[,15])
data_temp[,16]<-(data2[,16]-mean(data2[,16]))/sd(data2[,16])
data_temp[,17]<-(data2[,17]-mean(data2[,17]))/sd(data2[,17])
data_temp[,18]<-(data2[,18]-mean(data2[,18]))/sd(data2[,18])
data_temp[,19]<-(data2[,19]-mean(data2[,19]))/sd(data2[,19])
data_temp[,20]<-(data2[,20]-mean(data2[,20]))/sd(data2[,20])
data_temp[,21]<-(data2[,21]-mean(data2[,21]))/sd(data2[,21])

data2<-data_temp
write.csv(data2,file = "data_log.csv")
#zmiana wartości zmiennej objaśnianej
data2[,22]<-replace(data2[,22],data2$result==0, 0.5 )
data2[,22]<-replace(data2[,22],data2$result==2, 0)
#usunięcie remisów z danych w celu stworzenia modelu logitowego
datawl<-data2[data2$result==0,]
datawl2<-data2[data2$result==1,]
datawins<-rbind(datawl, datawl2)
data2<-datawins
write.csv(data2,file = "data_logit.csv")
#model
#library("MASS")
#log_mod<-glm(data2[,22]~data2[,1]+data2[,2]+data2[,3]+data2[,4]+data2[,5]+data2[,6]+data2[,7]+data2[,8]+data2[,9]+data2[,10]+data2[,11]+data2[,12]+data2[,13]+data2[,14]+data2[,15]+data2[,16]+data2[,17]+data2[,18]+data2[,19]+data2[,20]+data2[,21], method="binomial")
#sumlog<-summary(log_mod)
