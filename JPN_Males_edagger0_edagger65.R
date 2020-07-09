### --------------------------------------------------------------------------
### R Code: univariate LSTM for e-dagger forecasting
### --------------------------------------------------------------------------
### 
### R version 3.6.3
### Platform: x86_64-w64-mingw32/x64 (64-bit)
### Running under: Windows 10 x64 (build 18362)
### Intel(R) Core(TM) i7 - 7500U CPU @ 2.70 GHz 2.90GHz - RAM 8 GB
###  
### --- Technical info about packages for LSTM implementation
###  #--- For e-dagger at x=0
### Keras version (R interface): 2.2.4.1
### Tensorflow version (R interface): 1.14.1 
### Tensorflow version (Python module for backend tensor operations): 1.10.1 (Python module)
### Python version : 3.6
### Conda version : 4.5.11
###
###  #--- For e-dagger at x=65
### Keras version (R interface): 2.2.5
### Tensorflow version (R interface): 2.0.0 
### Tensorflow version (Python module for backend tensor operations): 1.13.1 (Python module)
### Python version : 3.6
### Conda version : 4.8.3
###
### 
### --- e-dagger data
### 
### Data source : Mortality Rates from Human Mortality Database
### Period : 1947-2014
### Country : Japan
### Gender : Males
### Forecasting period : 2000-2014
### ---------------------------------------------------------------------------

library(tidyverse)

# Error metrics 
rmse = function (truth, prediction)  {
  sqrt(mean((prediction - truth)^2))
}
mae = function(truth, prediction){
  mean(abs(prediction-truth))
}

# Uploading Japan Males e-dagger time series, both for x=0 and x=65
#----------------------------------------------------------------------
data=read.csv2("Edagg_data.csv")


###---------------------------------------------------------------------------###
###           Working on e-dagger time series for x=0                         ###
###---------------------------------------------------------------------------###

# Neural Network supervisor definition 
# following a one-year autoregressive approach
# Splitting uploaded dataset in train-test set

y=as.vector(data$Years)
start = y[1]
finish = y[length(y)]
brek= 1999
n =(brek -start)+1 
L = finish -brek   

numberlag = 1

supervised_edagg_x0 = as.vector(data$x.0)
for(i in 1:numberlag) {
  lag_edagg_x0 = c(rep(as.vector(data$x.0)[1],i),as.vector(data$x.0)[1:(length(data$x.0)-i)])
  supervised_edagg_x0 = cbind(supervised_edagg_x0,lag_edagg_x0)}
train_edagg_x0 = supervised_edagg_x0[1:n, ]
test_edagg_x0 = supervised_edagg_x0[(n+1):nrow(supervised_edagg_x0),]
x_train_edagg_x0 = train_edagg_x0[,-1]
y_train_edagg_x0 = train_edagg_x0[,1]
x_test_edagg_x0 = test_edagg_x0[,-1]
y_test_edagg_x0 = test_edagg_x0[,1]


# Neural Network's tuning
#------------------------------------------------

# Tensor dimensionality for time series and input shaping
dim(x_train_edagg_x0) <- c(n, (numberlag), 1)
X_shape2 = numberlag
X_shape3 = 1
batch_size = 1

# Grid search: define the optimal NNs' hyper-parameter combination
# A priori, only 1 hidden layer (due to sample size) and the activation functions are selected
# The grid search is related to th number of neurons (unit) and to the number of epochs
tuning = expand.grid(epochs = seq(1,20,1), unit = seq(1,20,1)) 
tuning = cbind(tuning, performance = rep(0, nrow(tuning)))

set.seed(50)
seed= round(runif(1, min = 0, max = 1000))

system.time(for (g in 1:(nrow(tuning))){
    
    library(keras)
    use_session_with_seed(seed)
    build_model <- function() {
      model <- keras_model_sequential() 
      model%>%
        layer_lstm(tuning$unit[g],
                   batch_input_shape = c(batch_size, X_shape2, X_shape3),
                   activation='relu', recurrent_activation = "tanh")%>%
        layer_dense(units = 1)
      
      model %>% compile(
        loss = 'mean_squared_error',
        optimizer = optimizer_adadelta(),  
        )
      model
    }
    
    model <- build_model()
    model %>% summary()
    h <- model %>% fit(x_train_edagg_x0, 
                       y_train_edagg_x0,
                       epochs=tuning$epochs[g],
                       batch_size=batch_size, 
                       verbose=0, 
                       shuffle=FALSE)
    
    predict_test = numeric(L)
    X = tail(y_train_edagg_x0, n = 1)
    for(i in 1:L){
      X
      dim(X) = c(1,numberlag,1)
      yhat = model %>% predict(X, batch_size=batch_size)
      yhat
      predict_test[i] = yhat
      X = yhat
    }
    
  tuning$performance[g] = rmse(predict_test, y_test_edagg_x0)
})


# Get prediction with bet tuned NN
#--------------------------------------------------------
best_hyp = which.min(tuning$performance)

use_session_with_seed(seed)
  build_model <- function() {
    model <- keras_model_sequential() 
    model%>%
      layer_lstm(tuning$unit[best_hyp],
                 batch_input_shape = c(batch_size, X_shape2, X_shape3),
                 activation='relu', recurrent_activation = "tanh")%>%
      layer_dense(units = 1)
    
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = optimizer_adadelta(),  
      )
    model
  }
  
  model <- build_model()
  model %>% summary()
  h <- model %>% fit(x_train_edagg_x0, 
                     y_train_edagg_x0,
                     epochs=tuning$epochs[best_hyp],
                     batch_size=batch_size, 
                     verbose=0, 
                     shuffle=FALSE)
  
  predict_test = numeric(L)
  X = tail(y_train_edagg_x0, n = 1)
  for(i in 1:L){
    X
    dim(X) = c(1,numberlag,1)
    yhat = model %>% predict(X, batch_size=batch_size)
    yhat
    predict_test[i] = yhat
    X = yhat
  }
  
rmse(predict_test, y_test_edagg_x0)
mae(predict_test, y_test_edagg_x0)

# Plotting results
#---------------------------

lstm=c(c(rep(NA,(length(as.vector(data$x.0))-length(predict_test))),predict_test))
plot_data <- data.frame(y,as.vector(data$x.0),lstm)

theme_set(theme_bw(base_family = "mono",base_size = 20))

Sp <- plot_data %>%
  ggplot(aes(y,as.vector(data$x.0))) +
  geom_point( size= 3,alpha=0.55) +
  geom_line(aes(y,lstm),color="red",size=1.8,alpha=1)+
  labs( title = "Japan Male - 1947 to 1999 - Forecasting: 2000 to 2014",
        subtitle = "LSTM: red",
        y=expression("edagger"[0]),x="anno")+
  geom_vline(xintercept=brek, linetype="dotted",size=1)+
  theme(plot.title = element_text( colour="black", size=20),
        plot.subtitle = element_text( colour="black", size=18))+
  scale_x_continuous(limits=c(start,y[length(y)]))
Sp



###############################################################################

###---------------------------------------------------------------------------###
###           Working on e-dagger time series for x=65                        ###
###---------------------------------------------------------------------------###

supervised_edagg_x65 = as.vector(data$x.65)
for(i in 1:numberlag) {
  lag_edagg_x65 = c(rep(as.vector(data$x.65)[1],i),as.vector(data$x.65)[1:(length(data$x.65)-i)])
  supervised_edagg_x65 = cbind(supervised_edagg_x65,lag_edagg_x65)}
train_edagg_x65 = supervised_edagg_x65[1:n, ]
test_edagg_x65 = supervised_edagg_x65[(n+1):nrow(supervised_edagg_x65),]
x_train_edagg_x65 = train_edagg_x65[,-1]
y_train_edagg_x65 = train_edagg_x65[,1]
x_test_edagg_x65 = test_edagg_x65[,-1]
y_test_edagg_x65 = test_edagg_x65[,1]


# Neural Network's tuning
#------------------------------------------------

dim(x_train_edagg_x65) <- c(n, (numberlag), 1)
X_shape2 = numberlag
X_shape3 = 1
batch_size = 1

tuning = expand.grid(epochs = seq(1,50,1), unit = seq(1,50,1)) 
tuning = cbind(tuning, performance = rep(0, nrow(tuning)))

set.seed(50)
seed= round(runif(1, min = 0, max = 1000))

system.time(for (g in 1:(nrow(tuning))){
  
  library(keras)
  use_session_with_seed(seed)
  build_model <- function() {
    model <- keras_model_sequential() 
    model%>%
      layer_lstm(tuning$unit[g],
                 batch_input_shape = c(batch_size, X_shape2, X_shape3),
                 activation='relu', recurrent_activation = "tanh")%>%
      layer_dense(units = 1)
    
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = optimizer_adadelta(),  
    )
    model
  }
  
  model <- build_model()
  model %>% summary()
  h <- model %>% fit(x_train_edagg_x65, 
                     y_train_edagg_x65,
                     epochs=tuning$epochs[g],
                     batch_size=batch_size, 
                     verbose=0, 
                     shuffle=FALSE)
  
  predict_test = numeric(L)
  X = tail(y_train_edagg_x65, n = 1)
  for(i in 1:L){
    X
    dim(X) = c(1,numberlag,1)
    yhat = model %>% predict(X, batch_size=batch_size)
    yhat
    predict_test[i] = yhat
    X = yhat
  }
  
  tuning$performance[g] = rmse(predict_test, y_test_edagg_x65)
})


# Get prediction with best tuned NN
#---------------------------------------------------
best_hyp = which.min(tuning$performance)

use_session_with_seed(seed)
build_model <- function() {
  model <- keras_model_sequential() 
  model%>%
    layer_lstm(tuning$unit[best_hyp],
               batch_input_shape = c(batch_size, X_shape2, X_shape3),
               activation='relu', recurrent_activation = "tanh")%>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adadelta(),  
    )
  model
}

model <- build_model()
model %>% summary()
h <- model %>% fit(x_train_edagg_x65, 
                   y_train_edagg_x65,
                   epochs=tuning$epochs[best_hyp],
                   batch_size=batch_size, 
                   verbose=0, 
                   shuffle=FALSE)

predict_test = numeric(L)
X = tail(y_train_edagg_x65, n = 1)
for(i in 1:L){
  X
  dim(X) = c(1,numberlag,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  yhat
  predict_test[i] = yhat
  X = yhat
}

rmse(predict_test, y_test_edagg_x65)
mae(predict_test, y_test_edagg_x65)

# Plotting results
#---------------------------

lstm=c(c(rep(NA,(length(as.vector(data$x.65))-length(predict_test))),predict_test))
plot_data <- data.frame(y,as.vector(data$x.65),lstm)

theme_set(theme_bw(base_family = "mono",base_size = 20))

Sp <- plot_data %>%
  ggplot(aes(y,as.vector(data$x.65))) +
  geom_point( size= 3,alpha=0.55) +
  geom_line(aes(y,lstm),color="red",size=1.8,alpha=1)+
  labs( title = "Japan Male - 1947 to 1999 - Forecasting: 2000 to 2014",
        subtitle = "LSTM: red",
        y=expression("edagger"[65]),x="anno")+
  geom_vline(xintercept=brek, linetype="dotted",size=1)+
  theme(plot.title = element_text( colour="black", size=20),
        plot.subtitle = element_text( colour="black", size=18))+
  scale_x_continuous(limits=c(start,y[length(y)]))
Sp







