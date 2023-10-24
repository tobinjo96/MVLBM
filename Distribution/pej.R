
pej <- function(ejVec, j, m, mu, p, z1tozjm1Vec){
  
  library(ordinalClust)

  ejVec <- unlist(ejVec)
  
  z1tozjm1Vec <- unlist(z1tozjm1Vec)
  
  mu <- as.integer(mu)
  
  #require(R.utils)

  ## test timeout
  #pejp <- withTimeout(ordinalClust:::pej(ejVec = ejVec, j = j, m = m, mu = mu, p = p, z1tozjm1Vec = z1tozjm1Vec), timeout = 150, onTimeout = "silent") # should be fine
  pejp <- ordinalClust:::pej(ejVec = ejVec, j = j, m = m, mu = mu, p = p, z1tozjm1Vec = z1tozjm1Vec)
  #pejp <- ordinalClust:::pej(ejVec = ejVec, j = j, m = m, mu = mu, p = p, z1tozjm1Vec = z1tozjm1Vec)

  return(pejp)
}

pejSim <- function(m, mu, p){
  library(ordinalClust)
  
  mu <- as.integer(mu)
  
  m <- as.integer(m)
  
  pejp <- c()
  
  for (ej in 1:m){
    pejp <- c(pejp, ordinalClust::pejSim(ej = ej, m = m, mu = mu, p = p))
    
  }
  
  return(pejp)
}

try_with_time_limit <- function(expr, cpu = Inf, elapsed = Inf)
{
  y <- try({setTimeLimit(cpu, elapsed); expr}, silent = TRUE) 
  if(inherits(y, "try-error")) NULL else y 
}
