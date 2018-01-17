b <- 1
d <- 1
nu <- 0.5
la <- 0.1
tMax <- 500
iterMax <- tMax * 1000

nImm <- rpois(1, nu * TMax)
nImm <- nImm + 1
tImm <- runif(nImm, 0, tMax)

## function to simulate events starting with an immigration at t0
simEvents <- function(t0 = 0) {
    events <- sample(c(-1, 1), size = iterMax, replace = TRUE, prob = c(d, b))
    x <- cumsum(c(1, events))
    
    if(any(x == 0)) {
        x <- x[1:min(which(x == 0))]
    }
    
    tt <- cumsum(c(t0, rexp(length(x) - 1, x[-length(x)] * (b + d))))
    
    if(max(tt) > tMax) {
        x <- x[tt <= tMax]
        tt <- tt[tt <= tMax]
        x <- c(x, x[length(x)])
        tt <- c(tt, tMax)
    }
    
    return(list(t = tt, x = x))
}

## function to sample time trajectory at consistent times
sampX <- function(x, ntime = 500) {
    sampT <- seq(0, tMax, length.out = ntime)
    foo <- stepfun(x$t, c(0, x$x))
    
    return(foo(sampT))
}

x0 <- simEvents(0)

plot(x0$t, x0$x, type = 'l')
plot(stepfun(x0$t, c(0, x0$x)))
