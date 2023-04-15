# Paige Reynolds
# MATH 4020
# Lab 5
# SUBSET SELECTION #

rm(list= ls())
library(ISLR)
install.packages("leaps")
install.packages("glmnet")
library(leaps)
library(glmnet)

set.seed(314)
# Part 1 - Subset selection
sum(is.na(Hitters))

# Discard any rows that have NA values
hit.no.na <- na.omit(Hitters)

best.mods <- regsubsets(Salary ~ ., data= hit.no.na)
summary(best.mods)

# Look at ordinary R^2 vs model size
plot(summary(best.mods)$rsq, pch= 20, xlab= "No. Variables",
     ylab= expression(R^2))
lines(summary(best.mods)$rsq, type= "c", lwd= 2)

# Plot BIC vs model size
plot(summary(best.mods)$bic, pch= 20, xlab= "No. Variables",
     ylab= "BIC")
lines(summary(best.mods)$bic, type= "c", lwd= 2)

identify(1:length(summary(best.mods)$bic), summary(best.mods)$bic)

# Find the minimum with which.min()
which.min(summary(best.mods)$bic)

# Plot the regsubsets object directly
par(mfrow= c(1, 2))
plot(best.mods, scale= "adjr2")
plot(best.mods, scale= "bic")

# Look at the coefficients for the selected 6-variable model
coef(best.mods, 6)

# Compare forward/backward selection to b est subset selection
fwd.mods <- regsubsets(Salary ~ ., data= hit.no.na, method= "forward")
bkwd.mods <- regsubsets(Salary ~., data= hit.no.na, method= "backward")
summary(fwd.mods)

coef(best.mods, 7)
coef(fwd.mods, 7)
coef(bkwd.mods, 7)

# SELECTING AMONG THE COMPETITIVE MODELS via TEST SET or CROSS VALIDATION
# Test sets to validate the competing models
train <- sample(c(TRUE, FALSE), nrow(hit.no.na), rep= TRUE)
test <- !train

trn.best.mods <- regsubsets(Salary ~., data= hit.no.na[train, ], nvmax= 19)
summary(trn.best.mods)

# Pull out the test set design matrix. Then for a given model, we can peep 
# the corresponding columns of the test design matrix to as the design
# for that particular model.
test.mat <- model.matrix(Salary~ ., data= hit.no.na[test, ])
# Iterate through each model size and get the test predictions from the model:
mspes <- rep(NA, times= 19)
for(i in 1: length(mspes)){
  temp <- coef(trn.best.mods, id= i)
  pred <- test.mat[, names(temp)]%*%temp
  mspes[i] <- mean((hit.no.na$Salary[test]-pred)^2)
  
}

plot(mspes, pch= 20, xlab= "Model Size", ylab= "Predictive Error")
lines(mspes, type= "c", lwd= 2)

coef(trn.best.mods, 9)

# Best 9- variable model estimated using the first full set
mod9 <- lm(Salary ~ AtBat + Hits + HmRun + Runs + Walks +
           CRuns + CWalks + Division + PutOuts,
           data= hit.no.na)
coef(mod9)

coef(trn.best.mods, 9)

# 10 FOLD CROSS VALIDATION #
k <- 10
folds <- sample(1:k, nrow(hit.no.na), replace= TRUE)
head(folds)

cv.errors <- matrix(nrow= k, ncol= 19)

# Loop over each of k holdout sets
for(h.out in 1:k){
  
  best.fit <- regsubsets(Salary ~ ., data= hit.no.na[folds != h.out, ],
                         nvmax= 19)
  mod.mat <- model.matrix(Salary ~ ., data= hit.no.na[folds == h.out, ])
  
  for(i in 1:19){
    coefi <- coef(best.fit, id= i)
    pred <- mod.mat[, names(coefi)]%*%coefi
    cv.errors[h.out, i] <- mean((hit.no.na$Salary[folds == h.out] - pred)^2)
  }
  
}

head(cv.errors)

# Take the mean of each column to get the CV error for each model size
(mean.cv.errors <- apply(cv.errors, 2, mean))
plot(mean.cv.errors, type= "b", pch= 20)

full.subs <- regsubsets(Salary ~ ., data= hit.no.na, nvmax= 19)
coef(full.subs, 10)

# LAB 2: RIDGE REGRESSION AND THE LASSO #
x <- model.matrix(Salary ~ ., Hitters)[, -1]
y <- Hitters$Salary

library(glmnet)
grid <- 10^seq(10, -2, length= 100)
ridge.mod <- glmnet(x, y, alpha= 0, lambda= grid)

dim(coef(ridge.mod))

ridge.mod$lambda[50]
coef(ridge.mod)[, 50]

sqrt(sum(coef(ridge.mod)[-1, 50]^2))

ridge.mod$labmda[60]
coef(ridge.mod)[, 60]

predict(ridge.mod, s= 50, type= "coefficients")[1:20, ]

set.seed(1)
train <- sample(1: nrow(x), nrow(x)/2)
test <- (-train)
y.test <- ytest[test]

ridge.mod <- glmnet(x[train, ], y[train], alpha= 0, lambda= grid,
                    thresh= 1e-12)
ridge.pred <- predict(ridge.mod, s= 4, newx= x[test, ])
mean((ridge.pred - y.test)^2)

mean((mean(y[train])- y.test)^2)

ridge.pred <- predict(ridge.mod, s= 1e10, newx= x[test,])
mean((ridge.pred - y.test)^2)

ridge.pred <- predict(ridge.mod, s= 0, newx= x[test, ], exact= TRUE)
mean((ridge.pred - y.test)^2)

lm(y~x, subset= train)
predict(ridge.mod, s= 0, exact= TRUE, type= "coefficients")[1:20, ]

set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha= 0)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam

ridge.pred <- predict(ridge.mod, s= bestlam, newx= x[test, ])
mean((ridge.pred - y.test)^2)

out <- glm(x, y, alpha= 0)
predict(out, type="coefficients", s= bestlam)[1:20, ]

lasso.mod <- glmnet(x[train, ], y[train], alpha= 1, lambda= grid)
plot(lasso.mod)

set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha= 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s= bestlam, newx= x[test, ])
mean((lasso.pred - y.test)^2)

out <- glmnet(x, y, alpha= 1, lambda= grid)
lasso.coef <- predict(out, type= "coefficients", s= bestlam)[1:20, ]
lasso.coef

lasso.coef[lasso.coef != 0]


