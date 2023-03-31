# Remove genes with zero variance across measured timepoints
drop.constant.genes <- function(df.beta) {
  gene.vars <- apply(df.beta, 1, sd)
  return(df.beta[gene.vars != 0,])
}

# Create cell.type object for DIALOGUE
cell.type <- setClass(Class = "cell.type",
                      slots = c("name","cells","genes","cellQ",
                                "tpm","tpmAv","qcAv","zscores",
                                "X","samples",
                                "metadata",
                                "scores","scoresAv",
                                "tme","tme.qc","gene.pval",
                                "tme.OE",
                                "extra.scores",
                                "sig"))

csplotch.to.dialogue <- function(name, mean.betas) {
  exp.betas <- exp(mean.betas)
  tpm <- 1e6 * t(t(exp.betas)/colSums(exp.betas))
  tpmAv <- tpm
  #X <- t(mean.betas)
  X <- t(exp.betas)  # use exponentiated betas so MCPs are in units of counts.
  samples <- colnames(mean.betas)
  cellQ <- matrix(1000, dim(mean.betas)[2], 1)
  rownames(cellQ) <- colnames(mean.betas)
  metadata <- data.frame(cellQ=cellQ)
  
  r<-cell.type(name = gsub("_","",name),
               cells = colnames(tpm),
               genes = rownames(tpm),
               cellQ = cellQ,
               tpm = tpm,
               tpmAv = tpm,
               qcAv = aggregate(x = cellQ,by = list(samples),FUN = mean),
               X = X,
               samples = samples,
               metadata = metadata,
               extra.scores = list())
  print(paste("Cell type name:",r@name))
  if(!identical(r@cells,rownames(X))){
    print("Each row in X should include the original representation of the corresponding cell.")
    print("Error: Cell ids do not match the X input.")
    return("Error: Cell ids do not match the X input.")
  }
  rownames(r@qcAv)<-r@qcAv[,1]
  r@qcAv[,1]<-r@qcAv[,2]
  return(r)
}