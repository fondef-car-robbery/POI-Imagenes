rm(list=ls())

library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)


########################################################################
#                           CARGA DE DATOs DE ZONAS
########################################################################
data <- read.csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa1_PI.csv', header = TRUE)
etiquetas <- read.csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas.csv', header = TRUE)
inner <- merge(data, etiquetas)

########################################
#   METODO 1: Random en todos lados
########################################

smp_size <- floor(0.8 * nrow(inner))
set.seed(123)
train_ind <- sample(seq_len(nrow(inner)), size = smp_size)

train <- inner[train_ind, ][c(4:211,213)]
sum(train$label_zona)

train_robo <- train[ sample( which( train$label_zona == 1 ) , 3000 ) , ]
train_norobo <- train[ sample( which( train$label_zona == 0 ) , 3000 ) , ]
train <- merge(train_robo, train_norobo, all=TRUE) 

test <- inner[-train_ind, ][c(4:211,213)]
test_robo <- train[ sample( which( train$label_zona == 1 ) , 1000 ) , ]
test_norobo <- train[ sample( which( train$label_zona == 0 ) , 1000 ) , ]
test <- merge(test_robo, test_norobo, all=TRUE) 

########################################
#   METODO 2: Linea de separacion
########################################

x_plaza <- -33.437914
y_plaza <- -70.650339
n  <- sample(1:112003, 1)

#x_ale <- inner[n,'x']
#y_ale <- inner[n,'y']
x_ale = -33.383787 
y_ale = -70.526801
m <- (y_ale-y_plaza)/(x_ale-x_plaza)

inner['train'] <- 0
for(i in 1:nrow(inner)){
  y_recta <- m*(inner[i,'x']-x_plaza)+y_plaza
  y_real <- inner[i,'y']
  if(y_real > y_recta){
    inner[i,'train'] <- 1 #train
  }
}


train <- inner[ sample( which( inner$train == 1 ) ) , ]
train_robo <- train[ sample( which( train$label_zona == 1 ) , 3000 ) , ]
train_norobo <- train[ sample( which( train$label_zona == 0 ) , 3000 ) , ]
train <- merge(train_robo, train_norobo, all=TRUE) 
train <- subset( train, select = -train )
train <- train[c(4:211,213)]


test <- inner[ sample( which( inner$train == 0 ) ) , ]
test_robo <- test[ sample( which( test$label_zona == 1 ) , 1000 ) , ]
test_norobo <- test[ sample( which( test$label_zona == 0 ) , 1000 ) , ]
test <- merge(test_robo, test_norobo, all=TRUE) 
test <- subset( test, select = -train )
test <- test[c(4:211,213)]


########################################
#   METODO 3: Por cuadrantes
########################################

#hacer una lista de cuadrantes
#segmentar los cuadrantes aleatoriamente
#seleccionar

########################################
#   Procese 1: Normalizacion de datos
########################################

for(i in 1:(ncol(train01km)-1)){
  mean <- mean(train01km[[i]])
  print(mean)
  sdv <- sd(train01km[[i]])
  print(sdv)
  train01km[[i]] <- (((train01km[[i]]-mean)/sdv)+1)
}
summary(train01km)

########################################
#   Feature Selection 1: Boruta Method
########################################
#install.packages('Boruta')
library(Boruta)
Boruta.model <- Boruta(label_zona ~ ., data = train, doTrace = 2, ntree = 500, maxRuns = 12)
plot(Boruta.model)
attStats(Boruta.model)
plotZHistory(Boruta.Madelon)
########################################################################
#                           REGRESIÓN LOGISTICA
########################################################################
library(MASS)
summary(model)
model <- glm(formula = label_zona ~ ., family = binomial(link = "logit"), data = train)

finalmodel <- stepAIC(model,direction="backward")
finalmodel <- glm(formula = label_zona ~ agencias.de.turismo.01km + aseo.y.limpieza.01km + 
                    barberias.slash.barber.shop.01km + cafes.slash.cafeterias.01km + 
                    centros.bip.01km + chilena.01km + comidas.rapidas.01km + 
                    compras.personales.01km + construccion.01km + cumpleanos.slash.cotillon.01km + 
                    deco.hogar.01km + delivery.a.domicilios.01km + deporte.01km + 
                    deportivos.01km + educativos.01km + embajadas.01km + financieros.01km + 
                    fruterias.slash.verdulerias.01km + fundaciones.01km + hogar.01km + 
                    juegos.01km + kioscos.01km + mantenimiento.de.maquinaria.01km + 
                    mascotas.01km + medicos.y.especialistas.01km + municipalidades.01km + 
                    nan.01km + otras.compras.01km + otros.comercios.01km + otros.organismos.ciudad.01km + 
                    otros.restaurantes.01km + otros.servicios.01km + para.deportistas.01km + 
                    para.el.auto.slash.moto.01km + para.el.hogar.01km + para.ninos.y.bebes.01km + 
                    para.verse.bien.01km + para.vestirse.01km + parques.y.plazas.01km + 
                    pescados.y.mariscos.01km + picadas.01km + plasticos.01km + 
                    prenderias.slash.casas.de.empeno.01km + prevision.de.salud.01km + 
                    puntos.de.informacion.01km + reciclaje.slash.chatarra.01km + 
                    regalos.01km + servicios.contables.01km + servicios.especializados.01km + 
                    tecnologia.01km + terminal.de.transporte.01km + teterias.01km + 
                    tiendas.de.musica.slash.disqueria.01km + venta.de.entradas.01km + 
                    vida.social.01km + vivienda.01km + zapateria.01km + aeropuertos.12km + 
                    agencias.de.empleo.12km + agencias.de.turismo.12km + agricola.12km + 
                    aseo.y.limpieza.12km + barberias.slash.barber.shop.12km + 
                    centros.bip.12km + ciclovias.12km + cine.12km + climatizacion.slash.aire.acondicionado.12km + 
                    comisarias.slash.carabineros.12km + compras.personales.12km + 
                    comunicaciones.12km + concesionarios.y.automotoras.12km + 
                    cultura.12km + cumpleanos.slash.cotillon.12km + deco.hogar.12km + 
                    delivery.a.domicilios.12km + deportivos.12km + electrodomesticos.12km + 
                    empresas.slash.oficinas.12km + entidades.oficiales.slash.ministerios.12km + 
                    financieros.12km + fruterias.slash.verdulerias.12km + helados.slash.postres.12km + 
                    hogar.12km + hospitales.clinicas.centros.de.salud.12km + 
                    juegos.12km + mantenimiento.de.maquinaria.12km + metro.12km + 
                    municipalidades.12km + nan.12km + oficinas.y.direccion.publica.12km + 
                    ong.12km + otras.compras.12km + otros.comercios.12km + otros.organismos.ciudad.12km + 
                    otros.restaurantes.12km + para.deportistas.12km + para.el.hogar.12km + 
                    para.ninos.y.bebes.12km + para.verse.bien.12km + pescados.y.mariscos.12km + 
                    prenderias.slash.casas.de.empeno.12km + prevision.de.salud.12km + 
                    puntos.de.informacion.12km + reciclaje.slash.chatarra.12km + 
                    regalos.12km + seminarios.12km + servicios.especializados.12km + 
                    servicios.personales.12km + spa.12km + tecnologia.12km + 
                    teterias.12km + tiendas.de.musica.slash.disqueria.12km + 
                    tiendas.electronicas.12km + transantiago.12km + vegetariano.slash.naturista.12km + 
                    venta.de.entradas.12km + vivienda.12km, family = binomial(link = "logit"), 
                  data = train)
summary(finalmodel)


nullmod <- glm(train$label_zona~1, family="binomial")
McFadden <- 1-logLik(finalmodel)/logLik(nullmod)
summary(finalmodel)

#finalmodel = model
predictedval <- predict(finalmodel, newdata=test)
predictedval <- ifelse(predictedval > 0.5,1,0)

#sum(predictedval) con esto pude ver en que orden estaba la predicción y valor real en la matriz de confusión
#sum(y_test$label_zona) con esto pude ver en que orden estaba la predicción y valor real en la matriz de confusión
#misClasificError <- mean(predictedval != y_test)
#print(paste('Accuracy',1-misClasificError))


confusion_matrix <- table(predictedval, test$label_zona)
TN <- confusion_matrix[1][1]
FP <- confusion_matrix[2][1]
FN <- confusion_matrix[3][1]
TP <- confusion_matrix[4][1]

acc <- (TP+TN)/(TP+TN+FP+FN)
precision <- (TP)/(TP+FP)
recall <- (TP)/(TP+FN)
f1 <- (2 * precision * recall) / (precision + recall)
#stepAIC(model,direction="forward")

########################################################################
#                             RandomForest
########################################################################
#install.packages('cforest')

train$label_zona <- as.character(train$label_zona)
test$label_zona <- as.character(test$label_zona)
train$label_zona <- as.factor(train$label_zona)
test$label_zona <- as.factor(test$label_zona)

nvar = log2(ncol(train)+1)
nvar = 8
model1 <- randomForest(label_zona ~ ., data = train, importance = TRUE, ntrees=5000, mtry=nvar)
importance_randomforest <-model1$importance

predictedval <- predict(model1, newdata=test)
predictedval <- ifelse(predictedval > 0.5,1,0)
confusion_matrix <- table(predictedval, test$label_zona)
TN <- confusion_matrix[1][1]
FP <- confusion_matrix[2][1]
FN <- confusion_matrix[3][1]
TP <- confusion_matrix[4][1]

acc <- (TP+TN)/(TP+TN+FP+FN)
precision <- (TP)/(TP+FP)
recall <- (TP)/(TP+FN)
f1 <- (2 * precision * recall) / (precision + recall)

########################################################################
#                                 SVM
########################################################################
#install.packages("rpart")

svm.model <- svm(label_zona ~ ., data = train, cost = 1, kernel='linear')
svm.pred <- predict(svm.model, test)
svm.pred <- ifelse(svm.pred > 0.5,1,0)


confusion_matrix <- table(svm.pred, test$label_zona)
TN <- confusion_matrix[1][1]
FP <- confusion_matrix[2][1]
FN <- confusion_matrix[3][1]
TP <- confusion_matrix[4][1]

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

