install.packages("dplyr")
library(dplyr)
install.packages("mlbench")
library(mlbench)
install.packages("ggplot2")
library(ggplot2)
install.packages("corrplot")
library(corrplot)
install.packages("caret")
library(caret)
install.packages("class")
library(class)
install.packages("e1071")
library(e1071)

data(PimaIndiansDiabetes2)
PimaIndiansDiabetes2
#prepare data - Since some of the data provided is incomplete, we have removed those incomplete cases to form a complete data set.
PI <-PimaIndiansDiabetes2[complete.cases(PimaIndiansDiabetes2),]
PI1 = PI[-9]

#Boxplot for dataset
PI%>%ggplot(aes(x=diabetes, y=glucose)) + geom_boxplot()

#matrix scatter plot
pairs.panels(PI[,-10],
             method = "pearson", # correlation method
             hist.col = "steelblue",
             pch = 21, bg = c("pink", "light green", "light blue"),
             density = TRUE, # show density plots
             ellipses = FALSE # show correlation ellipses
)

#knn regression
#Split the data into training and test sets
set.seed(100)
training.idx <- sample(1: nrow(PI1), nrow(PI1)*0.8)
train.data <- PI1[training.idx, ]
test.data <- PI1[-training.idx, ]

# Fit the model on the training set. what number to put instead of 10? result or diabetes
library(caret)
set.seed(101)
model.knn <- train(
  glucose~., data = train.data, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 10
)

# Best tuning parameter k that minimize the RMSE
model.knn$bestTune
#[k=11]

# Make predictions based on the test data
predictions<-predict(model.knn, test.data)
head(predictions)

# Compute the prediction error RMSE
RMSE(predictions, test.data$glucose)
#[output= 21.34912]


#Prediction- linear regression
set.seed(100)
training.idx <- sample(1: nrow(PI1), nrow(PI1)*0.8)
train.data <- PI1[training.idx, ]
test.data <- PI1[-training.idx, ]

#Prediction : Linear Regression
lmodel<- lm(glucose ~., data = train.data)
summary(lmodel)

RMSE(predictions, test.data$glucose)
#rmse = 21.0049

#create multiple plots on the same page
par(mfrow=c(2,2))
plot(lmodel)

#Visualize the correlation between the outcome medv and each predictor
corrplot(cor(train.data), type="upper", method="color",addCoef.col = "black",number.cex = 0.6)

#remove the outlier
PI2<-PI1[-c(585),]
set.seed(100)
training1.idx <- sample(1: nrow(PI2), size=nrow(PI2)*0.8)
train.data1 <- PI2[-training.idx,]
test.data1 <- PI2[-training.idx, ]
#second order term is added
lmodel1=lm(glucose
           ~I(insulin^2)+I(age^2)+I(insulin*age)+pregnant+pressure+triceps+insulin+mass+pedigree+age, data= train.data1)
summary(lmodel1)
predictions1 <- predict(lmodel1, test.data1)
RMSE(predictions1, test.data1$glucose)
par(mfrow=c(2,2))
plot(lmodel1)
#18.74496

#logistic classification
PI_numeric=PI
PI_numeric[,1:8]<- sapply(PI[,1:8], as.numeric)
PI_numeric
PI_numeric<-PI_numeric%>%mutate(result=factor(ifelse(diabetes=="pos", 1,0)))%>%select(-diabetes)

set.seed(100)
training.idx <- sample(1: nrow(PI_numeric), size=nrow(PI_numeric)*0.8)
train.data <-PI_numeric[training.idx, ]
test.data <- PI_numeric[-training.idx, ]
mlogit <- glm(result ~., data = train.data, family = "binomial")
summary(mlogit)

Pred.p <-predict(mlogit, newdata =test.data, type = "response")
result_pred_num <-ifelse(Pred.p > 0.5, 1, 0)
result_pred <-factor(result_pred_num, levels=c(0, 1))
mean(result_pred ==test.data$result )
tab <-table(result_pred,test.data$result)
tab

#Knn Classification
nor <-function(x) { (x -min(x))/(max(x)-min(x)) }
PI_numeric[,1:8] <- sapply(PI_numeric[,1:8], nor)
set.seed(100)
training.idx <- sample(1: nrow(PI_numeric), size=nrow(PI_numeric)*0.8)
train.data <-PI_numeric[training.idx, ]
test.data <- PI_numeric[-training.idx, ]

library(class)
set.seed(101)
knn1<-knn(train.data[,1:8], test.data[,1:8], cl=train.data$result, k=2)
mean(knn1 ==test.data$result)
table(knn1,test.data$result)

#try for a better k
ac<-rep(0, 30)
for(i in 1:30){
  set.seed(101)
  knn.i<-knn(train.data[,1:8], test.data[,1:8], cl=train.data$result, k=i)
  ac[i]<-mean(knn.i ==test.data$result)
  cat("k=", i, " accuracy=", ac[i], "\n")
}
plot(ac, type="b", xlab="K",ylab="Accuracy")

set.seed(101)
knn2<-knn(train.data[,1:8], test.data[,1:8], cl=train.data$result, k=24)
mean(knn2 ==test.data$result)
table(knn2,test.data$result)

#SVM Classification
#split the data
set.seed(100)
training.idx <- sample(1: nrow(PI_numeric), size=nrow(PI_numeric)*0.8)
train.data <-PI_numeric[training.idx, ]
test.data <- PI_numeric[-training.idx, ]

#svm classification
library(e1071)
set.seed(100)
m.svm<-svm(result~., data = train.data, kernel = "linear")
summary(m.svm)

#predict newdata in test set
pred.svm <- predict(m.svm, newdata=test.data[,1:8])

#evaluate classification performance and check accuracy
table(pred.svm, test.data$result)
mean(pred.svm ==test.data$result)

#improve the model
set.seed(100)
m.svm.tune<-tune.svm(result~., data=train.data, kernel="radial", cost=10^(-1:2), gamma=c(.1,.5,1,2))
summary(m.svm.tune)

#visualize results of parameter tuning
plot(m.svm.tune)

#confusion matrix and accuracy
best.svm = m.svm.tune$best.model
pred.svm.tune = predict(best.svm, newdata=test.data[,1:8])
table(pred.svm.tune, test.data$result)
mean(pred.svm.tune ==test.data$result)

#In this case, the linear kernel gives a better classification as compared to the radial kernel. This could be due to the way the data is scattered. Hence, we see that SVM with linear kernel classifies over 79% of the patients correctly into diabetic and non-diabetic.
