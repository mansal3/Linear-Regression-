#data acquistion
library(MASS)
?Boston
View(Boston)
str(Boston)

#divide data into traning and testing
set.seed(123)
cut<-sample(2,nrow(Boston),replace=TRUE,  prob = c(0.7,0.3))
traning<-Boston[cut==1,]
testing<-Boston[cut==2,]
pairs(Boston)

#colinearity
crr<-cor(Boston)
crr
install.packages("corrplot")
library(corrplot)
corrplot(crr,type="lower")

#building model and checking multicolinearity
install.packages("car")
library(car)
model<-lm(medv~.,data = traning)
vif(model)
summary(model)

#predicting
pred<-predict(model,testing)
pred
tab<-table(pred,testing$medv)
tab
#accuray of model
1-(sum(diag(tab)))/sum(tab)
