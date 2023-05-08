    # Regressions 

setwd('D:\\HSG\\Research\\Social Learning and Market Structure\\replication code')

data <- read.csv(file = 'data.csv')

lm1 <- lm(p ~ K + N + N:K + beta.2. + beta.3. + beta.4., data = data)

lm2 <- lm(100*booked ~ K + N + N*K + p + beta.2. + beta.3. + beta.4., data = data)

library(stargazer)

stargazer(lm1,lm2,type="text")

stargazer(lm1,lm2,type="html")

