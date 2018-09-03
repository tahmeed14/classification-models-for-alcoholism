setwd("C:/Users/Tahmeed/Desktop/Important/STATS_415/415_project")

## Project Members:
## Aidan Kearns, Statistics, University of Michigan
## Tahmeed Tureen, Statistics, University of Michigan
## Anthony Zheng, Statistics, University of Michigan

#Context: Our data set is courtesy of the UCI Machine Learning Website Repository
#Our data frame consists of sample size (n) = 1044 students and 33 original variables

student = read.csv("Student_Alcohol_Consumption_Merged.csv")

summary(student$Walc)
summary(student$Dalc)

#This for loop will create our response variable of interest, "heavy drinker"
heavydrinker = rep(0,1044) #We have 1044 students, so create a vector of length 1044
for(i in 1:nrow(student)){
  if(student$Dalc[i]>2 && student$Walc[i]>3){
    heavydrinker[i]= TRUE
  }
  else{
    heavydrinker[i] = FALSE
  }
}

#Create new var with all of the data
analysis_data <- read.csv("Student_Alcohol_Consumption_Merged.csv")
View(analysis_data)

analysis_data <- analysis_data[,-27] #Remove 27th column Dalc
View(analysis_data)
analysis_data <- analysis_data[,-27] #Remove new 27th column Walc

analysis_data$heavydrinker <- as.factor(heavydrinker) #Add our response column to the data frame
# We need this to be a factor with two levels so that we can create the error rate table in randomForest
View(analysis_data)


#************************************ [Training and Test Split] ************************************#
set.seed(15)
samp_size <- floor(0.70 * nrow(analysis_data)) #30% of analysis_data will be for our test set


train_split <- sample(seq_len(nrow(analysis_data)), size = samp_size)

train_analysis <- analysis_data[train_split, ] #training set will have 730 obs.
test_analysis <- analysis_data[-train_split, ] #test set will have 314 obs.



#******************************[Matrixify Train and Test Data]**********************************#

# FOR TRAIN
#X_Data <- analysis_data
X_Data <- train_analysis

#X_Data <- X_Data[,-32] #class = data frame
#X_matrix <- data.matrix(X_Data) #class = matrix, 1044 x 31

X_matrix <- model.matrix(heavydrinker ~ ., X_Data) #ideally we want this to make qualittative vars
X_matrix <- X_matrix[,-1] #Remove Intercept
X_interaction <- model.matrix(heavydrinker~.*., X_Data)
X_interaction <- X_interaction[,-1]
#into dummy variables because glmnet doesn't do well with qualitative vars

#Y_matrix <- analysis_data[["heavydrinker"]] #This is a vector
Y_matrix <- train_analysis[["heavydrinker"]] #This is a vector

# FOR TEST
X_testdata <- test_analysis
X_test <- model.matrix(heavydrinker ~., X_testdata)
X_test <- X_test[,-1]
X_test_interaction <- model.matrix(heavydrinker~.*., X_testdata)
X_test_interaction <- X_test_interaction[,-1]

require(glmnet)
#*******************************DIMENSION REDUCTION Approach #1 [ GLM LASSO ]**************************#

set.seed(125)
#Resource: Lab 7 with Jack Goetz, STATS 415:Data Mining
#Lasso_x <- glmnet(X_matrix, Y_matrix, alpha = 1, family = "binomial") #alpha = 1 is for lasso
#plot(Lasso_x, main = "Lasso Coefficient Plot for Heavy Drinker Prediction", sub = "Authors: TT, AZ, AK")
#dim(X_matrix)
#Lasso_interaction <- glmnet(X_interaction, Y_matrix, alpha = 1, family = "binomial")
#We can see that we have positive coefficients as well as negative coefficients based on our plot

# We will now look at cross validation to search for our best tuning parameters
Lasso_cvmod <- cv.glmnet(X_matrix,Y_matrix, alpha = 1, family = "binomial")
print(Lasso_cvmod)
plot(Lasso_cvmod, main = "Lasso Coefficient Plot for Heavy Drinker Prediction", sub = "Authors: TT, AZ, AK")
best_lambda <- Lasso_cvmod$lambda.min
# best lambda is equal to  based on our cross validation
Lasso_cv_interaction <- cv.glmnet(X_interaction,Y_matrix, alpha = 1, family = "binomial")
plot(Lasso_cv_interaction, main = "Lasso Coefficient Plot for Heavy Drinker Prediction with Interaction Terms", sub = "Authors: TT, AZ, AK")
#Use predict function with our best_lambda
lasso_coef <- predict(Lasso_x , type = "coefficients", s = best_lambda )
str_obj = paste(c("Best Lambda =", best_lambda), collapse = " ")
plot(lasso_coef, main = "Coefficient Plot for GLM Lasso", sub = str_obj, ylab = "Coefficient", xlab = "Variable Index", col = "BLUE")
print(lasso_coef)

# We will now create an classification table which will help us calculate error rate
# Do we use Lasso_x or Lasso_cvmod????
lasso_pred <- predict(Lasso_cvmod , s = best_lambda , newx = X_test, type = "class") #test_analysis must be made into Matrix
test_heavydrinker <- test_analysis$heavydrinker

lasso_table = table(lasso_pred, test_heavydrinker)
lasso_table = as.data.frame(lasso_table)
print(lasso_table)

set.seed(7)
for (i in Lasso_cvmod$lambda){
  #str_obj = paste(c("Lambda =", i), collapse = " ")
  #random_lasso_coef <- predict(Lasso_x, type = "coefficients", s = i)
  #plot(random_lasso_coef, main = "Coefficient Plot for GLM Lasso", sub = str_obj , ylab = "Coefficient", xlab = "Variable Index", col = "BLUE")
  
  
  lasso_pred <- predict(Lasso_cvmod , s = i , newx = X_test, type = "class") #test_analysis must be made into Matrix
  #test_heavydrinker <- test_analysis$heavydrinker
  
  x = table(lasso_pred, test_heavydrinker)
  x = as.data.frame(x)
  print((x$Freq[2]+x$Freq[3])/sum(x$Freq))
  #print(str_obj)
  #print(x)
  #print(random_lasso_coef)
}
set.seed(78)
for (i in Lasso_cv_interaction$lambda){
  
  lasso_pred_interact <- predict(Lasso_cv_interaction , s = i , newx = X_test_interaction, type = "class") #test_analysis must be made into Matrix
  test_heavydrinker <- test_analysis$heavydrinker
  
  x_interact = table(lasso_pred_interact, test_heavydrinker)
  x_interact = as.data.frame(x)
  print((x_interact$Freq[2]+x_interact$Freq[3])/sum(x_interact$Freq))
}

#Should we try test & train data or No??? If so we could do
#set.seed (1)
#lasso_pred <- predict(Lasso_cvmod , s = best_lambda , newx = X_test, type = "class") #test_analysis must be made into Matrix
#test_heavydrinker <- test_analysis$heavydrinker
#table(lasso_pred, test_heavydrinker)



#************************DIMENSION REDUCTION Approach #2 [ RIDGE REGRESSION ]**************************#

set.seed(420)
#Resource: Lab 7 with Jack Goetz, STATS 415:Data Mining
#Ridge_x <- glmnet(X_matrix, Y_matrix, alpha = 0, family = "binomial") #alpha = 0 is for lasso
#plot(Ridge_x, main = "Rigde Reg Coefficient Plot for Heavy Drinker Prediction", sub = "Authors: TT, AZ, AK")
#dim(X_matrix)

Ridge_cvmod <- cv.glmnet(X_matrix,Y_matrix, alpha = 0, family = "binomial")
print(Ridge_cvmod)
plot(Ridge_cvmod, main = "Rigde Reg Coefficient Plot for Heavy Drinker Prediction", sub = "Authors: TT, AZ, AK")
Ridge_cv_interaction <- cv.glmnet(X_interaction, Y_matrix, alpha = 0, family = "binomial")
plot(Ridge_cv_interaction, main = "Rigde Reg Coefficient Plot for Heavy Drinker Prediction with Interaction Terms", sub = "Authors: TT, AZ, AK")

best_lambda2 <- Ridge_cvmod$lambda.min

#Use predict function with our best_lambda
#ridge_coef <- predict(Ridge_x , type = "coefficients", s = best_lambda2 )
#str_obj = paste(c("Best Lambda =", best_lambda2), collapse = " ")
#plot(ridge_coef, main = "Coefficient Plot for Ridge Regression", sub = str_obj, ylab = "Coefficient", xlab = "Variable Index", col = "BLUE")
#print(ridge_coef)

# We will now create an classification table which will help us calculate error rate
# Do we use Lasso_x or Lasso_cvmod????
ridge_pred <- predict(Ridge_cvmod , s = best_lambda2 , newx = X_test, type = "class") #test_analysis must be made into Matrix
test_heavydrinker <- test_analysis$heavydrinker

ridge_table = table(ridge_pred, test_heavydrinker)
ridge_table = as.data.frame(ridge_table)
print(ridge_table)

set.seed(45)
for (i in Ridge_cvmod$lambda){
  #str_obj = paste(c("Lambda =", i), collapse = " ")
  #random_ridge_coef <- predict(Ridge_x, type = "coefficients", s = i)
  #plot(random_ridge_coef, main = "Coefficient Plot for Ridge Regression", sub = str_obj , ylab = "Coefficient", xlab = "Variable Index", col = "BLUE")
  
  
  ridge_pred <- predict(Ridge_cvmod , s = i , newx = X_test, type = "class") #test_analysis must be made into Matrix
  #test_heavydrinker <- test_analysis$heavydrinker
  
  x = table(ridge_pred, test_heavydrinker)
  x = as.data.frame(x)
  print((x$Freq[2]+x$Freq[3])/sum(x$Freq))
  #print(str_obj)
  #print(x)
  #print(random_ridge_coef)
}
set.seed(422)
for (i in Ridge_cv_interaction$lambda){
  #str_obj = paste(c("Lambda =", i), collapse = " ")
  #random_ridge_coef <- predict(Ridge_x, type = "coefficients", s = i)
  #plot(random_ridge_coef, main = "Coefficient Plot for Ridge Regression", sub = str_obj , ylab = "Coefficient", xlab = "Variable Index", col = "BLUE")
  
  
  ridge_pred_interaction <- predict(Ridge_cv_interaction , s = i , newx = X_test_interaction, type = "class") #test_analysis must be made into Matrix
  #test_heavydrinker <- test_analysis$heavydrinker
  
  x = table(ridge_pred_interaction, test_heavydrinker)
  x = as.data.frame(x)
  print((x$Freq[2]+x$Freq[3])/sum(x$Freq))
  #print(str_obj)
  #print(x)
  #print(random_ridge_coef)
}

#******************************************Regular Logistic Regression*********************************#

#Instead of reducing dimensions, we will first run a logistic regression and see what type of results we
# get for this fit

# Aidan and Anthony, can you try to figure this out?
set.seed(66)
glm_fit = glm(heavydrinker ~ . , data = analysis_data, family = "binomial")

glm_pred <- predict(glm_fit, type ="response")

glm.pred = rep(0,length(analysis_data))
for(i in 1:length(glm_pred)){
  if(glm_pred[i] > 0.5){
    glm.pred[i] = 1 
  }
  else {glm.pred[i] = 0
  }
}
y <- table(glm.pred, analysis_data[,32])
y = as.data.frame(y)
misclass_error_log = (y$Freq[2]+y$Freq[3])/1044 
print(misclass_error_log)

set.seed(72)
glm_fit_interaction = glm(heavydrinker~.*., data = analysis_data, family = "binomial")
glm_pred_interaction <- predict(glm_fit_interaction, type = "response")
glm.pred.interact = rep(0,length(analysis_data))
for(i in 1:length(glm_pred_interaction)){
  if(glm_pred_interaction[i] > 0.5){
    glm.pred.interact[i] = 1 
  }
  else {glm.pred.interact[i] = 0
  }
}
y_interaction <- table(glm.pred.interact, analysis_data[,32])
y_interaction = as.data.frame(y)
misclass_error_log_interact = (y$Freq[2]+y$Freq[3])/1044 
print(misclass_error_log_interact)


#***********************DIMENSION REDUCTION Approach #4 [ Random Forest Importance ]*******************#

#We will use the Importance() function from HW6
#Need to install these packages before running
#install.packages("rpart")
#install.packages("randomForest")
require(rpart)
require(randomForest)


#We will try different values of m
range_vector <- c(10, 15, 20, 25)
set.seed(57)
for (i in range_vector){
  bag_tree <- randomForest( heavydrinker ~ ., data = train_analysis, mtry = i, importance = TRUE)
  #print(bag_tree)
  
  predict_tree <- predict(bag_tree, test_analysis, method ="class" )
  test_x <- test_analysis$heavydrinker
  
  my_table <- table(predict_tree, test_x)
  my_table <- as.data.frame(my_table)
  print("table is for m = ")
  print(i)
  print(my_table)
  print((my_table$Freq[2]+my_table$Freq[3])/sum(my_table$Freq))

  
  varImpPlot(bag_tree,type = 2, main = "Plot for Random Forest", sub = i) #Dr. Park's hint
  x <- importance(bag_tree)
  print(x)
}


