rm(list=ls())

library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)
library(MASS)
library(geosphere)
########################################################################
#                           GENERACION DE DATOS A BINARIOS
########################################################################
data <- read.csv('file:///D:/Modelos Tensorflow/Conjunto train y test/etapa1_PI.csv', header = TRUE)

for(j in 4:ncol(data)){
  data[,j][data[,j] > 0] <- 1
}
  
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

finalmodel_BI <- stepAIC(model,direction="backward")

modelo_sugerido_BWbinario <- glm(formula = label_lugar ~ agencias.de.empleo.01km + alimentos.01km + aseo.y.limpieza.01km + 
                                cafes.slash.cafeterias.01km + chilena.01km + comidas.rapidas.01km + 
                                concesionarios.y.automotoras.01km + deco.hogar.01km + embajadas.01km + 
                                fundaciones.01km + hogar.01km + hospedaje.01km + joyas.y.accesorios.01km + 
                                juegos.01km + kioscos.01km + mantenimiento.de.maquinaria.01km + 
                                mascotas.01km + medicos.y.especialistas.01km + ong.01km + 
                                otras.compras.01km + otros.organismos.ciudad.01km + para.deportistas.01km + 
                                para.el.hogar.01km + pescados.y.mariscos.01km + plasticos.01km + 
                                reciclaje.slash.chatarra.01km + seminarios.01km + servicio.tecnico.01km + 
                                servicios.personales.01km + spa.01km + terminal.de.transporte.01km + 
                                vegetariano.slash.naturista.01km + vida.social.01km + vivienda.01km + 
                                zapateria.01km + centros.comerciales.slash.mall.12km + centros.de.pilates.slash.yoga.12km + 
                                comercios.especializados.12km + compras.personales.12km + 
                                comunicaciones.12km + concesionarios.y.automotoras.12km + 
                                construccion.12km + educativos.12km + financieros.12km + 
                                fundaciones.12km + loteria.12km + maquilladores.12km + oficinas.y.direccion.publica.12km + 
                                ong.12km + otros.comercios.12km + para.el.hogar.12km + parques.y.plazas.12km + 
                                parrilladas.12km + plasticos.12km + prenderias.slash.casas.de.empeno.12km + 
                                seminarios.12km + servicio.tecnico.12km + spa.12km + teterias.12km + 
                                vegetariano.slash.naturista.12km + dist_plazarmas + dist_costanera, family = binomial(link = "logit"), 
                                data = train)

library(stargazer)
dep_labels = 'lugar de robo'
stargazer(model, modelo_sugerido_BWbinario,modelo_sugerido
          , title="Resultados", align=TRUE, type='html', no.space=TRUE, dep.var.labels = dep_labels
          , ci=TRUE, ci.level=0.95, single.row=TRUE)



