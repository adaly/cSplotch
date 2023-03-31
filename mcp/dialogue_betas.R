library(DIALOGUE)
library(PMA)
library(stringr)
source("splotch_utils.R")
source("util.R")
source("DIALOGUE.R") # we need to change some things locally for our data

# Read in mean beta matrices
args = commandArgs(trailingOnly=TRUE)
beta.dir = args[1]
dest.dir = args[2]

for (d in list.dirs(beta.dir)){
  if (d == beta.dir) {next}
  aar = basename(d)
  rA = list()
  
  print(aar)
  
  for (beta.file in list.files(d)){
    # Read in mean betas (genes, timepoints) to DataFrame & clean
    df.beta <- read.csv(file.path(d, beta.file), row.names=1, header=TRUE)
    df.beta <- drop.constant.genes(df.beta)
    
    # Infer cell type from file name
    tokens = str_split(basename(beta.file), 'beta_means_', simplify=TRUE)
    ctype = str_split(tokens[2], '\\.', simplify=TRUE)[1]
    
    r <- csplotch.to.dialogue(ctype, df.beta)
    rA[[ctype]] = r
  }
  
  # Perform Multi-CCA
  full.version <- F
  names(rA)<-lapply(rA,function(r) r@name)
  
  # Can only infer as many MCPs as there are cell types
  k = length(rA)
  
  R<-DIALOGUE1(rA = rA,k = k,main = "test", results.dir = "DIALOGUE.results", 
                conf="cellQ", covar=c("cellQ"), n.genes=200, extra.sparse=T)
  if(R$message=="No programs"){summary(R)}
  
  # Write weights output by MultiCCA (as well as TPM matrices) to CSV files
  for (x in names(rA)) {
    ws = R$cca$ws[[x]]
    tpm = rA[[x]]@tpm
    y = rA[[x]]@X %*% ws
    
    # Create sub-directory for results, if it doesn't already exist.
    dir.create(file.path(dest.dir, aar), showWarnings = FALSE)

    ws_dest = file.path(dest.dir, aar, paste0('ws_', x, '.csv'))
    write.csv(ws, ws_dest)
    
    tpm_dest = file.path(dest.dir, aar, paste0('tpm_', x, '.csv'))
    write.csv(tpm, tpm_dest)
    
    mcp_dest = file.path(dest.dir, aar, paste0('mcp_', x, '.csv'))
    write.csv(y, mcp_dest)
  }
}