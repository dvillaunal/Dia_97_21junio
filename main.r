## ---- eval=FALSE, include=TRUE-------------------------------------------------------
## "Protocolo:
## 
##  1. Daniel Felipe Villa Rengifo
## 
##  2. Lenguaje: R
## 
##  3. Tema: Maquinas de Vector Soporte [Parte 2]
## 
##  4. Fuentes:
##     https://www.datacamp.com/community/tutorials/support-vector-machines-r"


## ------------------------------------------------------------------------------------

#Gracias a los archivos .RData sacaremos toda la información de los datos sin importar nada más:
#load("~/R/100DaysOfCode/Dia_97_21junio/.RData")

# Guardamos los OUTPUTS:
sink("OUTPUTS.txt")


## ------------------------------------------------------------------------------------
# Cargamos la libreria
library(e1071)

# Error de test
print("# Error de test")
predicciones = predict(modelo_svc, datosOJ_test)
table(prediccion = predicciones, real = datosOJ_test$Purchase)

library(dplyr)

paste("Observaciones de test mal clasificadas:", 
      100 * mean(datosOJ_test$Purchase != predicciones) %>% 
        round(digits = 4), "%")


## ------------------------------------------------------------------------------------
# AJUSTE DEL MODELO
# -----------------------------------------------------------------------------
# Configuración del proceso de selección del modelo
fitControl <- trainControl(method = "cv", 
                           number = 10, 
                           classProbs = TRUE, 
                           search = "grid")

# Parametros del modelo disponibles
getModelInfo(model = "svmLinear")[[2]]$parameters


library(caret)

# Valores del hiperparámetro C a evaluar
grid_C <- data.frame(C = c(0.001, 0.01, 0.1, 1, 5, 10, 15, 20))

# Entrenamiento del SVM con un kernel lineal y optimización del hiperparámetro C
set.seed(325)

modelo_svc <- train(Purchase ~ ., data = datosOJ_train, 
                    method = "svmLinear", 
                    trControl = fitControl, 
                    preProc = c("center", "scale"), #estandarizacion de los datos
                    tuneGrid = grid_C)

# Resultado del entrenamiento
print("# Resultado del entrenamiento")
print(modelo_svc)

png(filename = "modelo_svc.png")

# Evolución del accuracy en funcion del valor de coste en validacion cruzada
plot(modelo_svc)

dev.off()

# EVALUACIÓN DEL MODELO
# -----------------------------------------------------------------------------
print("# EVALUACIÓN DEL MODELO #-----------------------------------------------------------------------------")
confusionMatrix(predict(modelo_svc, datosOJ_test), datosOJ_test$Purchase)


## ------------------------------------------------------------------------------------
# Creamos una semilla:
set.seed(325)

## Ajuste del modelo:
tuning <- tune(svm, Purchase ~ ., data = datosOJ_train, 
               kernel = "polynomial",
               ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 15),degree = c(2, 3)),
               scale = TRUE)

# Resumen:
print("# Resumen:")
summary(tuning)


#Con un kernel polinómico, los hiperparámetros óptimos que reducen el error de validación son coste = 15, grado = 2.

png(filename = "ErrorVal.png")

# Error de validación
ggplot(data = tuning$performances, aes(x = cost, y = error, col = as.factor(degree)))+
  geom_line()+
  geom_point()+
  labs(title = "Error de validación ~ hiperparámetro C y polinomio")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme_bw()+ theme(legend.position = "bottom")

dev.off()


# Modelo SVM kernel polinómico
modelo_svmP <- svm(Purchase ~ ., data = datosOJ_train, kernel = "polynomial",
                   cost = 15, 
                   degree = 2, 
                   scale = TRUE)

print("# Modelo SVM kernel polinómico")
summary(modelo_svmP)