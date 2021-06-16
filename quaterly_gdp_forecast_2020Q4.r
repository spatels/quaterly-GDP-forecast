# Intital Setup -----------------------------------------------------------
# Clean environment
rm(list=ls(all=TRUE))
# ---SET INPUT VARIABLES---
begin_date = '1979-09-01'
backtest_date = '2007-01-01'
# Install needed packages
usePackage <- function(p) {
  if (!is.element(p, installed.packages()[,1]))
    install.packages(p, dep = TRUE)
  require(p, character.only = TRUE)
}
#initializeKeras()
initializeKeras<-function(first.time=F){
  if(Sys.info()["sysname"]=="Windows") {
    usePackage("reticulate")
  }
  usePackage("keras")
  if(first.time){
    library(keras);keras::install_keras()
  }
  py_discover_config()
  py_available(initialize = FALSE)
  import("numpy")
  import("keras")
  stopifnot(py_module_available('keras'))
} #initializeKeras
# Function to calculate MASE
getMASE<-function(y.xts, yhat.xts){
  #Mean Absolute Scaled Error
  comp<-na.omit(cbind(y.xts,yhat.xts))
  comp$AE<-abs(comp[,1]-comp[,2])
  sum.abs.diff.y<-sum(abs(diff(y.xts)),na.rm=T)
  TT<-nrow(comp)
  MASE=sum(comp$AE)/( (TT/(TT-1)) * sum.abs.diff.y)
  return(MASE)
}

# 1: Use Quandl to download...as reported by FRED.-----------------------------------------------------------
# Getting data from Quandl
getRawFred<-function(symbols,...){
  # get Raw (not transformed) data from quandl
  usePackage('Quandl')
  # set key
  Quandl::Quandl.api_key(
    readChar('C:/Users/Satya/Documents/quandl/key/api_key.txt',file.info('C:/Users/Satya/Documents/quandl/key/api_key.txt')$size)  
  )
  raw.fred=Quandl(paste('FRED/',symbols,sep=''), ...)
  names(raw.fred)<-symbols
  return(raw.fred)  
} # of getRawFred()
# assign data to a variable ---ADD the SYMBOLS---
time.series.xts=getRawFred(symbols=c('GDPC1','UNRATE','CPIAUCSL','INDPRO','FEDFUNDS','M1SL','M2SL'),start_date= begin_date,collapse='quarterly',type='xts')

# 2: Prepare a machine-learning function using keras in R -----------------------------------------------------------
#one step growth forecast
oneStepGDPGrowthForecast<-function(time.series.xts,seed=1){
  initializeKeras()
  #difference predictors
  differenced.xts<-diff(log(time.series.xts))
  #Do not lag GDP (y)
  y<-differenced.xts[,1]
  #Lag predictors (x)
  x<-lag(differenced.xts[,2:ncol(differenced.xts)])
  names(x)<-paste(names(x),'.lag1',sep='')
  #combine
  transformed.xts<-merge(y,x)
  #extract x.new
  x.new<-tail(transformed.xts,1); x.new<-x.new[,2:ncol(x.new)]
  #remove na's
  transformed.xts<-na.omit(transformed.xts)
  x.matrix<-as.matrix(transformed.xts[,2:ncol(transformed.xts)])
  y.vector<-as.vector(transformed.xts[,1])
  x.train<-x.matrix
  y.train<-y.vector
  # scale (normalize) data
  x.train.scaled<-scale(x.train)
  y.train.scaled<-scale(y.train)
  # scale (normalize) new x
  x.new.scaled <- scale(x.new, center = attr(x.train.scaled, "scaled:center") , scale = attr(x.train.scaled, "scaled:scale"))  
  #Challenge requirement
  #use_session_with_seed(seed,disable_gpu=T,disable_parallel_cpu = T,quiet=T)
  tensorflow::tf$random$set_seed(1)
  #---SETUP THE MODEL---
  model = keras_model_sequential() %>%
    layer_dense(units=1024
                ,input_shape= ncol(x)
                ,activation="relu"
    ) %>%
    layer_dense(units=512
                ,activation="relu"
    ) %>%
    layer_dense(units=256
                ,activation="relu"
    ) %>%
    layer_dense(units=128
                ,activation="relu"
    ) %>%
    layer_dense(units=1, activation = "linear")
  #regression-specific compilation
  model %>%
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error","MSE")
    )
  #---SETUP THE FITTING PARAMETERS---
  verbose=1
  validation.split=0.1
  epochs=100
  batch.size=8
  patience=10
  callbacks = list(
    callback_early_stopping(patience = 20, 
                            restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(factor = 0.01, 
                                  patience = 5)
  )
  fit = model %>%
    keras::fit(
      x = x.train.scaled,
      y = y.train.scaled,
      shuffle = T,
      verbose=verbose,
      validation_split = validation.split,
      epochs = epochs,
      callbacks=callbacks,
      batch_size = batch.size
    )
  #Prediction (scale)
  predictions <- model %>% predict(x.new.scaled)
  #Prediction (unscale/denormalize)
  predictions <- predictions * attr(y.train.scaled, "scaled:scale") + attr(y.train.scaled, "scaled:center")
  #recycle x.new (so we do not have to wory about the index)
  fcst.xts<-x.new
  fcst.xts$fcst<-predictions
  fcst.xts<-fcst.xts[,'fcst']
  return(fcst.xts)
} # of oneStepGDPGrowthForecast()

# 3: Use function to generate a backtest ---------------------------------------------------------------------
# Run forecast in a loop
#parameters
backtest.start=backtest_date
gdp.forecast.backtest<-NULL
#load
symbols=c('GDPC1','UNRATE','CPIAUCSL','INDPRO','FEDFUNDS','HOUST','M1SL','M2SL')
start.date=begin_date
time.series.xts=getRawFred(symbols=symbols,start_date=start.date,collapse='quarterly',type='xts')
#enumerator
backtest.quarters<-index(time.series.xts)
backtest.quarters<-backtest.quarters[which(backtest.quarters>=backtest.start)]
#backtest loop: subset and run
for(q in backtest.quarters){
  #q<-backtest.quarters[1]
  cat('Backtesting',q,'...\n')
  #subset
  time.series.xts.subset<-time.series.xts[which(index(time.series.xts)<=q),]
  #forecast
  one.step.fcst<-oneStepGDPGrowthForecast(time.series.xts=time.series.xts.subset)
  #append
  gdp.forecast.backtest<-rbind(gdp.forecast.backtest,one.step.fcst)
} # of for each quarter

# 4: Plot the actual and forecast quarterly GDP growth ---------------------------------------------------------------------
#attach actuals
gdp.forecast.backtest<-cbind(diff(log(time.series.xts[,1]))[which(index(time.series.xts)>=backtest.start),],gdp.forecast.backtest)
#col names
names(gdp.forecast.backtest)<-c('GDP.growth','GDP.growth.fcst')
#Graph
usePackage('PerformanceAnalytics')
chart.TimeSeries(gdp.forecast.backtest,legend.loc='topleft')
head(gdp.forecast.backtest)
tail(gdp.forecast.backtest)
#MASE
MASE=getMASE(gdp.forecast.backtest[,1],gdp.forecast.backtest[,2])
cat('MASE:',MASE,'\n')