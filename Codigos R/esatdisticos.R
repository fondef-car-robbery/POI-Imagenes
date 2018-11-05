rm(list=ls())

library(randomForest)
library(rpart)
library(rpart.plot)

#### ZONAS
########################################################################
#                                   DATOS
########################################################################
########################################################################
#                           CARGA DE DATOs DE ZONAS
########################################################################
data <- read.csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/data_train_test_PI.csv', header = TRUE)
etiquetas_train <- read.csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_zona_train.csv', header = TRUE)
inner_train <- merge(data, etiquetas_train)

x_train <- inner_train[2:521]
y_train <- inner_train[524]
train <- inner_train[c(2:521,524)]

etiquetas_test <- read.csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/Base de entrenamiento/df_zona_test.csv', header = TRUE)
inner_test <- merge(data, etiquetas_test)

x_test <- inner_test[2:521]
y_test <- inner_test[524]
test <- inner_test[c(2:521,524)]

train01km <- train[c(1:104,521)]
summary(train01km)

train01km_scale <- scale(train01km)

########################################################################
#                           CARGA DE DATOs DE LUGARES
########################################################################

#### LUGARES
data_random_lug_train <- read.csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/caracterizacion_random_Santiago.csv', header = TRUE)
etiquetas_random_lug_train <- read.csv('file:///C:/Users/Bgm9/Dropbox/CapHuida/df_lugar_train.csv', header = TRUE)


data_random_lug_test <- read.csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/caracterizacion_random_Santiago.csv', header = TRUE)
etiquetas_random_lug_test <- read.csv('file:///C:/Users/Bgm9/Dropbox/CapHuida/df_lugar_test.csv', header = TRUE)


data_prose_lug_train <- read.csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/caracterizacion_PROSE_Santiago.csv', header = TRUE)
etiquetas_prose_lug_train <- read.csv('file:///C:/Users/Bgm9/Dropbox/CapHuida/df_zona_train_PROSE_v3.csv', header = TRUE)


data_prose_lug_test <- read.csv('file:///D:/Modelos Tensorflow/base de datos - Random de Santiago/caracterizacion_PROSE_Santiago.csv', header = TRUE)
etiquetas_prose_lug_test <- read.csv('file:///C:/Users/Bgm9/Dropbox/CapHuida/df_zona_test_PROSE_v3.csv', header = TRUE)

x_inner_lug_train <- merge(data_random_lug_train, etiquetas_random_lug_train, by='id')[1:521]
y_inner_lug_train <- merge(data_random_lug_train, etiquetas_random_lug_train, by='id')[c('id','label_lugar')]
x_inner_lug_train_prose <- merge(data_prose_lug_train, etiquetas_prose_lug_train, by='id')[1:521]
y_inner_lug_train_prose <- merge(data_prose_lug_train, etiquetas_prose_lug_train, by='id')[c('id','label_lugar')]

x_inner_lug_train_f <- rbind(x_inner_lug_train, x_inner_lug_train_prose)
y_inner_lug_train_f <- rbind(y_inner_lug_train, y_inner_lug_train_prose)


x_inner_lug_test <- merge(data_random_lug_test, etiquetas_random_lug_test, by='id')[1:521]
y_inner_lug_test <- merge(data_random_lug_test, etiquetas_random_lug_test, by='id')[c('id','label_lugar')]
x_inner_lug_test_prose <- merge(data_prose_lug_test, etiquetas_prose_lug_test, by='id')[1:521]
y_inner_lug_test_prose <- merge(data_prose_lug_test, etiquetas_prose_lug_test, by='id')[c('id','label_lugar')]


x_inner_lug_test_f <- rbind(x_inner_lug_test, x_inner_lug_test_prose)
y_inner_lug_test_f <- rbind(y_inner_lug_test, y_inner_lug_test_prose)

x_inner_lug_train_f_01 <- x_inner_lug_train_f[1:104]
x_inner_lug_train_f_01$label_lugar <- y_inner_lug_train_f$label_lugar

########################################################################
#                                   MODELOS
########################################################################
########################################################################
#                           REGRESIÓN LOGISTICA
########################################################################
library(MASS)
model <- glm(label_zona ~.,family=binomial(link='logit'),data=train01km)
predictedval <- predict(model, newdata=x_test)
predictedval <- ifelse(predictedval > 0.5,1,0)

misClasificError <- mean(predictedval != y_test)
print(paste('Accuracy',1-misClasificError))

stepAIC(model,direction="backward")
stepAIC(model,direction="forward")

########################################################################
#                          ARBOLES DE DECISIÓN
########################################################################
fit1 <- rpart(label_zona ~ ., data=train01km)
fit1.pred <- predict(fit1, x_test)
fit1.pred <- ifelse(fit1.pred > 0.5,1,0)
table(pred = fit1.pred, true = y_test)

rpart.plot(fit1)

print(fit1$variable.importance)
plot(fit1);text(fit1,pretty=0)

summary(fit1)
print(fit1) 

########################################################################
#                             RandomForest
########################################################################
train01km <- x_inner_lug_train_f_01[2:105]

nvar = integer(log2(ncol(train01km)+1))
model1 <- randomForest(label_lugar ~ ., data = train01km, importance = TRUE, ntrees=500,mtry=nvar)
importance_randomforest <-model1$importance


########################################################################
#                                 SVM
########################################################################
install.packages("rpart")
library(e1071)
library(rpart)
svm.model <- svm(label_zona ~ ., data = train01km, cost = 100, kernel = "linear")
svm.pred <- predict(svm.model, x_test)
svm.pred <- ifelse(svm.pred > 0.5,1,0)


confusion_matrix <- table(svm.pred, y_test$label_zona)
TP <- confusion_matrix[1][1]
FN <- confusion_matrix[3][1]
FP <- confusion_matrix[2][1]
TN <- confusion_matrix[4][1]

acc <- (TP+TN)/(TP+TN+FP+FN)
precision <- (TP)/(TP+FP)
recall <- (TP)/(TP+FN)
f1 <- (2 * precision * recall) / (precision + recall)
## Analisis de los pesos de SVM
w <- t(svm.model$coefs) %*% svm.model$SV # weight vectors
w <- apply(w, 2, function(v){sqrt(sum(v^2))})  # weight
w <- sort(w, decreasing = T)
print(w)

library (ROCR)

#retrieved <- sum(svm.pred)
#accuracy <- sum(svm.pred & y_test) / nrow(y_test)
#precision <- sum(svm.pred & y_test) / retrieved
#recall <- sum(svm.pred & y_test) / sum(y_test)
#Fmeasure <- 2 * precision * recall / (precision + recall)

### PERSON CORRELATION
library(Hmisc)
rcorr(train, type="pearson") # type can be pearson or spearman

correlaciones <- cor(train01km, method = c("pearson", "kendall", "spearman"))
summary(correlaciones)

fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError))


summary(model)


