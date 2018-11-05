rm(list=ls())

library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)
library(MASS)
library(geosphere)
########################################################################
#                           CARGA DE DATOs DE LUGARES
########################################################################

#### LUGARES
data <- read.csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa1_PI.csv', header = TRUE)
etiquetas <- read.csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas.csv', header = TRUE)
inner <- merge(data, etiquetas)
## Para calcular distancia al centro de la ciudad
plaza_de_armas = c(-70.650471, -33.437637)
baquedano = c(-70.634449, -33.4369436)
costanera = c(-70.606261, -33.417313)

smp_size <- floor(0.8 * nrow(inner))
set.seed(123)
train_ind <- sample(seq_len(nrow(inner)), size = smp_size)

train <- inner[train_ind, ][c(2:211,215)]
sum(train$label_lugar)

train_robo <- train[ sample( which( train$label_lugar == 1 ) , 5000 ) , ]
train_norobo <- train[ sample( which( train$label_lugar == 0 ) , 5000 ) , ]
train <- merge(train_robo, train_norobo, all=TRUE) 

test <- inner[-train_ind, ][c(2:211,215)]
test_robo <- test[ sample( which( test$label_lugar == 1 ) , 1000 ) , ]
test_norobo <- test[ sample( which( test$label_lugar == 0 ) , 1000 ) , ]
test <- merge(test_robo, test_norobo, all=TRUE) 


for(i in 1:nrow(train)){
  train[i,'dist_plazarmas'] <- distm(c(train[i,'y'], train[i,'x']), plaza_de_armas, fun = distHaversine)
  #train[i,'dist_baquedano'] <- distm(c(train[i,'y'], train[i,'x']), plaza_de_armas, fun = distHaversine)
  train[i,'dist_costanera'] <- distm(c(train[i,'y'], train[i,'x']), costanera, fun = distHaversine)
  #inner_train[i,'dist_baquedano'] <- distm(c(inner_train[i,'y'], inner_train[i,'x']), baquedano, fun = distHaversine)
}
for(i in 1:nrow(test)){
  test[i,'dist_plazarmas'] <- distm(c(test[i,'y'], test[i,'x']), plaza_de_armas, fun = distHaversine)
  #test[i,'dist_baquedano'] <- distm(c(test[i,'y'], test[i,'x']), plaza_de_armas, fun = distHaversine)
  test[i,'dist_costanera'] <- distm(c(test[i,'y'], test[i,'x']), costanera, fun = distHaversine)
  #inner_train[i,'dist_baquedano'] <- distm(c(inner_train[i,'y'], inner_train[i,'x']), baquedano, fun = distHaversine)
}

train <- train[c(3:213)]
test <- test[c(3:213)]

########################################
#   METODO 0: Estadistica descriptiva
########################################

matrix_cor <- cor(test, use="complete.obs", method="kendall") 
corrplot(matrix_cor, type="lower")
corrplot(matrix_cor, type="upper", order="hclust", tl.col="black", tl.srt=45)

data_robo <- data[ which(train$label_lugar==1), ] 
data_norobo <- data[ which(train$label_lugar==0), ] 

boxplot(data_robo[c(3:211)])
boxplot(data_norobo[c(3:211)])

########################################
#   METODO 2: Linea de separacion
########################################
data <- read.csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa1_PI.csv', header = TRUE)
etiquetas <- read.csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa2_raizetiquetas.csv', header = TRUE)
inner <- merge(data, etiquetas)

#x_ale <- inner[n,'x']
#y_ale <- inner[n,'y']
x_ale = -33.383787 
y_ale = -70.526801
n  <- sample(1:112003, 1)

x_ale <- inner[n,'x']
y_ale <- inner[n,'y']
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
train_robo <- train[ sample( which( train$label_lugar == 1 ) , 3000 ) , ]
train_norobo <- train[ sample( which( train$label_lugar == 0 ) , 3000 ) , ]
train <- merge(train_robo, train_norobo, all=TRUE) 
train <- subset( train, select = -train )

for(i in 1:nrow(train)){
  train[i,'dist_plazarmas'] <- distm(c(train[i,'y'], train[i,'x']), plaza_de_armas, fun = distHaversine)
  #train[i,'dist_baquedano'] <- distm(c(train[i,'y'], train[i,'x']), plaza_de_armas, fun = distHaversine)
  train[i,'dist_costanera'] <- distm(c(train[i,'y'], train[i,'x']), costanera, fun = distHaversine)
  #inner_train[i,'dist_baquedano'] <- distm(c(inner_train[i,'y'], inner_train[i,'x']), baquedano, fun = distHaversine)
}

train <- train[c(4:211,215,217,218)]



test <- inner[ sample( which( inner$train == 0 ) ) , ]
test_robo <- test[ sample( which( test$label_lugar == 1 ) , 500 ) , ]
test_norobo <- test[ sample( which( test$label_lugar == 0 ) , 500 ) , ]
test <- merge(test_robo, test_norobo, all=TRUE) 
test <- subset( test, select = -train )

for(i in 1:nrow(test)){
  test[i,'dist_plazarmas'] <- distm(c(test[i,'y'], test[i,'x']), plaza_de_armas, fun = distHaversine)
  #test[i,'dist_baquedano'] <- distm(c(test[i,'y'], test[i,'x']), plaza_de_armas, fun = distHaversine)
  test[i,'dist_costanera'] <- distm(c(test[i,'y'], test[i,'x']), costanera, fun = distHaversine)
  #inner_train[i,'dist_baquedano'] <- distm(c(inner_train[i,'y'], inner_train[i,'x']), baquedano, fun = distHaversine)
}
test <- test[c(4:211,215,217,218)]
########################################
#   Feature Selection 1: Boruta Method
########################################
#install.packages('Boruta')
library(Boruta)
Boruta.model <- Boruta(label_lugar ~ ., data = train, doTrace = 5, ntree = 500)
summary(Boruta.model)
plot(Boruta.model)
options(max.print=999999)
attStats(Boruta.model)

########################################################################
#                           REGRESIÓN LOGISTICA
########################################################################
library(MASS)

model <- glm(formula = label_lugar ~ ., family = binomial(link = "logit"), data = train)
summary(model)

modelo_sugerido <- glm(formula = label_lugar ~ academias.y.cursos.01km + aeropuertos.01km + 
      agencias.de.turismo.01km + barberias.slash.barber.shop.01km + 
      centros.de.pilates.slash.yoga.01km + chilena.01km + comercios.especializados.01km + 
      compras.personales.01km + comunicaciones.01km + cumpleanos.slash.cotillon.01km + 
      deco.hogar.01km + deportivos.01km + educativos.01km + electrodomesticos.01km + 
      entidades.oficiales.slash.ministerios.01km + financieros.01km + 
      fruterias.slash.verdulerias.01km + fundaciones.01km + helados.slash.postres.01km + 
      hogar.01km + hospedaje.01km + iglesias.01km + internacional.01km + 
      joyas.y.accesorios.01km + juegos.01km + metro.01km + para.el.hogar.01km + 
      para.ninos.y.bebes.01km + para.verse.bien.01km + para.vestirse.01km + 
      parques.y.plazas.01km + parrilladas.01km + picadas.01km + 
      reciclaje.slash.chatarra.01km + teterias.01km + tiendas.de.musica.slash.disqueria.01km + 
      transantiago.01km + uniformes.01km + urbano.01km + vegetariano.slash.naturista.01km + 
      venta.de.entradas.01km + vida.social.01km + aeropuertos.12km + 
      alimentos.12km + aseo.y.limpieza.12km + barberias.slash.barber.shop.12km + 
      bares.y.discotheque.12km + centros.comerciales.slash.mall.12km + 
      chilena.12km + comercios.especializados.12km + comidas.rapidas.12km + 
      comisarias.slash.carabineros.12km + companias.de.bomberos.12km + 
      construccion.12km + cultura.12km + educativos.12km + electrodomesticos.12km + 
      entidades.oficiales.slash.ministerios.12km + estacionamientos.12km + 
      financieros.12km + helados.slash.postres.12km + hospedaje.12km + 
      juegos.12km + kioscos.12km + municipalidades.12km + otras.compras.12km + 
      para.el.hogar.12km + para.verse.bien.12km + picadas.12km + 
      plasticos.12km + prenderias.slash.casas.de.empeno.12km + 
      puntos.de.interes.12km + reciclaje.slash.chatarra.12km + 
      servicios.personales.12km + spa.12km + teterias.12km + tiendas.de.musica.slash.disqueria.12km + 
      transantiago.12km + uniformes.12km + urbano.12km + venta.de.entradas.12km + 
      vida.social.12km + vivienda.12km + zapateria.12km
      +dist_plazarmas+dist_costanera, family = binomial(link = "logit"), 
    data = train)


dep_labels = 'lugar de robo'
stargazer(modelo_sugerido
          , title="Resultados", align=TRUE, type='html', no.space=TRUE, dep.var.labels = dep_labels
          , ci=TRUE, ci.level=0.95, single.row=TRUE)

model <- glm(label_lugar ~.,family=binomial(link='logit'),data=train)

finalmodel <- stepAIC(model,direction="backward")
summary(finalmodel)

#model_sugerido = model
predictedval <- predict(finalmodel, newdata=test)
predictedval <- ifelse(predictedval > 0.5,1,0)

#misClasificError <- mean(predictedval != y_test)
#print(paste('Accuracy',1-misClasificError))


confusion_matrix <- table(predictedval, test$label_lugar)
TN <- confusion_matrix[1][1]
FP <- confusion_matrix[2][1]
FN <- confusion_matrix[3][1]
TP <- confusion_matrix[4][1]

acc <- (TP+TN)/(TP+TN+FP+FN)
precision <- (TP)/(TP+FP)
recall <- (TP)/(TP+FN)
f1 <- (2 * precision * recall) / (precision + recall)

########################################################################
#                          ARBOLES DE DECISIÓN
########################################################################
fit1 <- rpart(label_lugar ~ ., data=x_inner_lug_train_f_01)
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

nvar = log2(126+1)
nvar = 7
model1_lugar <- randomForest(label_lugar ~ ., data = train, importance = TRUE, ntrees=5000, mtry=8)

model2_lugar <- randomForest(label_lugar ~ dist_costanera+servicios.personales.12km+para.verse.bien.12km+servicios.personales.01km+internacional.12km+dist_plazarmas+cafes.slash.cafeterias.12km+comunicaciones.12km+parques.y.plazas.12km+academias.y.cursos.12km+cafes.slash.cafeterias.01km+educativos.12km+educativos.01km+comunicaciones.01km+concesionarios.y.automotoras.12km+hospitales.clinicas.centros.de.salud.12km+deco.hogar.12km+otros.servicios.12km+construccion.01km+transantiago.12km+comidas.rapidas.12km+regalos.12km+hospitales.clinicas.centros.de.salud.01km+construccion.12km+para.el.auto.slash.moto.12km+agricola.12km+alimentos.12km+otros.servicios.01km+deco.hogar.01km+para.ninos.y.bebes.12km+comercios.especializados.12km+internacional.01km+para.el.auto.slash.moto.01km+para.verse.bien.01km+mascotas.12km+parques.y.plazas.01km+otros.comercios.12km+vivienda.12km+otros.restaurantes.12km+compras.personales.12km+servicios.especializados.12km+alimentos.01km+chilena.12km+comidas.rapidas.01km+compras.personales.01km+joyas.y.accesorios.12km+para.vestirse.12km+hospedaje.12km+regalos.01km+hogar.12km+bares.y.discotheque.12km+para.el.hogar.12km+concesionarios.y.automotoras.01km+tecnologia.12km+joyas.y.accesorios.01km+mascotas.01km+centros.de.pilates.slash.yoga.12km+hospedaje.01km+terapias.naturales.12km+cultura.12km+vida.social.12km+transantiago.01km+empresas.slash.oficinas.12km+chilena.01km+otros.restaurantes.01km+academias.y.cursos.01km+tecnologia.01km+spa.12km+servicio.tecnico.12km+spa.01km+helados.slash.postres.12km+zapateria.12km+otras.compras.12km+agricola.01km+para.vestirse.01km+financieros.12km+otros.organismos.ciudad.12km+uniformes.12km+medicos.y.especialistas.12km+plasticos.12km+financieros.01km+bares.y.discotheque.01km+otros.comercios.01km+para.ninos.y.bebes.01km+cine.12km+reciclaje.slash.chatarra.12km+aseo.y.limpieza.01km+tiendas.electronicas.12km+electrodomesticos.12km+urbano.12km+vegetariano.slash.naturista.12km+reciclaje.slash.chatarra.01km+vegetariano.slash.naturista.01km+para.el.hogar.01km+fundaciones.12km+prevision.de.salud.12km+aseo.y.limpieza.12km+estacionamientos.12km+para.deportistas.12km+centros.comerciales.slash.mall.12km+servicios.especializados.01km+agencias.de.turismo.12km+terapias.naturales.01km+hogar.01km+embajadas.12km+otros.organismos.ciudad.01km+vivienda.01km+servicios.contables.12km+agencias.de.turismo.01km+parrilladas.12km+delivery.a.domicilios.12km+picadas.12km+empresas.slash.oficinas.01km+loteria.12km+vida.social.01km+comercios.especializados.01km+urbano.01km+otras.compras.01km+helados.slash.postres.01km+plasticos.01km+cultura.01km+pescados.y.mariscos.12km+uniformes.01km+ong.12km+servicios.contables.01km+centros.comerciales.slash.mall.01km, data = train, importance = TRUE, ntrees=5000, mtry=7)


importance_randomforest <-model2_lugar$importance

predictedval <- predict(model2_lugar, newdata=test)
predictedval <- ifelse(predictedval > 0.5,1,0)
confusion_matrix <- table(predictedval, test$label_lugar)
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
install.packages("rpart")
library(e1071)
library(rpart)
svm.model <- svm(label_lugar ~ ., data = train, kernel = 'linear')
svm.pred <- predict(svm.model, test)
svm.pred <- ifelse(svm.pred > 0.5,1,0)


confusion_matrix <- table(svm.pred, test$label_lugar)
TN <- confusion_matrix[1][1]
FP <- confusion_matrix[2][1]
FN <- confusion_matrix[3][1]
TP <- confusion_matrix[4][1]

acc <- (TP+TN)/(TP+TN+FP+FN)
precision <- (TP)/(TP+FP)
recall <- (TP)/(TP+FN)
f1 <- (2 * precision * recall) / (precision + recall)
## Analisis de los pesos de SVM
w <- t(svm.model$coefs) %*% svm.model$SV # weight vectors # Esto es lo use.
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
