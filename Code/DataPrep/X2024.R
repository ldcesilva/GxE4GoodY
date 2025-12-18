# Modified from https://github.com/alenxav/Lectures/tree/master/MGC_2023/Script_CLAC_model.R 

root_path = "//tdl/Public2/G2F/Manuscript/" # Adjust to your path
prep_path = paste(root_path, "/Results/DataPrep/")
train_path = paste(root_path, "/Data/Challenge2024/Training_data");
test_path = paste(root_path, "/Data/Challenge2024/Testing_data");

# Read in genomic info
require(readr)
#gen = read_csv(paste(train_path, /"5_Genotype_Data_All_2014_2025_Hybrids_numerical.csv"))
gen = read.csv(paste(train_path, "/5_Genotype_Data_All_2014_2025_Hybrids_numerical.csv"), header = TRUE, row.names = 1)
dim(gen)
SnpID = c(dim(gen)[2], "")
for(i in 1:dim(gen)[2]){
  SnpID[i] = gen[1,i]
}
genID = rownames(gen)
gen2 = gen[-c(1:1),]
colnames(gen2) = SnpID
gen2[1:4,1:4]
gen[1:4,1:4]
dim(gen2)
#rm(gen)
gc()

# Formate genomic info
for(i in 1:ncol(gen2)){
  #cat(i,'\n')
  #x = gen2[,i][[1]]
  x = gen2[,i]
  x[x=='0']=0
  x[x=='1']=2
  x[x=='0.5']=1
  x = as.numeric(x)
  gen2[,i] = x}
rm(i,x)
gc()
gen2[1:4,1:4]

require(magrittr)
#gen2 %<>% as.data.frame()
#rownames(gen2) = genID
#gc()
#gen2[1:4,1:4]

for(i in 1:ncol(gen2)){ 
  #cat(i,'\n'); 
  gen2[,i] = as.integer(gen2[,i])
}
rm(i)
gc()
gen2[1:4,1:4]

# Format correctly
#gen2 %<>% as.matrix()
#gc()
#gen2 %<>% t()
#gc()
#genID = rownames(gen2)
#gen2[1:4,1:4]

# remove odd behaving SNPs
m = apply(gen2,2,function(x) max(x,na.rm = T))
m = which(m==1|m==2)
gen2 = gen2[,m]
rm(m)
gc()

# drop markers with too much missing
m = apply(gen2,2,function(x) mean(!is.na(x)) )
gen2 = gen2[,m>0.9]
dim(gen2)
gc()

# drop markers based on MAF
PS_dta = read.csv(paste(test_path, "Testing_Data/1_Submission_Template_2024.csv"))
pt = unique(PS_dta$Hybrid)
w = which(rownames(gen2)%in%pt)
m = colMeans(gen2[w,],na.rm = T) * 0.5 # PS MAF
gen2 = gen2[,m>0.1]
gc()
m = colMeans(gen2[-w,],na.rm = T) * 0.5  # ES MAF
gen2 = gen2[,m>0.1]
gc()
dim(gen2)


# imputation
m = c(table(gsub('_.+','',colnames(gen2))))
gen2 = bWGR::markov(as.matrix(gen2[1:dim(gen2)[1], 1:dim(gen2)[2]]), m)
gc()

if(0) {
# drop redundant markers (full LD or close to it)
set.seed(123)
m = matrix(rnorm(3*nrow(gen2)),ncol=3)
ev = t(gen2) %*% m
unique_ev = unique(round(ev))
gen2 = gen2[,rownames(unique_ev)]
rm(m,ev,unique_ev)

# multiple rounds of dropping neighbor SNPs based on LD
gen2 = NAM::snpQC(gen2,psy=0.999)          # 1st round
gen2 = NAM::snpQC(gen2,psy=0.999, MAF = 0) # 2nd round
gen2 = NAM::snpQC(gen2,psy=0.998, MAF = 0) # 3rd round
gen2 = NAM::snpQC(gen2,psy=0.998, MAF = 0) # 4th round
gen2 = NAM::snpQC(gen2,psy=0.997, MAF = 0) # 5th round
gen2 = NAM::snpQC(gen2,psy=0.997, MAF = 0) # 6th round
gen2 = apply(gen2,2,as.integer)
rownames(gen2) = genID
gc()
#save(gen2, file='gen.RData')
}

# Get kernel
K = bWGR::EigenARC(as.matrix(gen2),T,4)
dim(K)
K[1:4,1:4]
rownames(K) = colnames(K) = rownames(gen2)
#save(K, file = 'ArcKernel.RData')
rm(gen2); gc()

# Reparametrize
require(bWGR)
K2X = function(K){
  E = EigenEVD(K,cores=4)
  w = which(E$D>0.1)
  X = E$U[,w] %*% diag(sqrt(E$D[w]))
  rownames(X) = rownames(K)
  return(X)}
X = K2X(K)
dim(X)
X[1:4, 1:4]
write.csv(X, file = paste(prep_path, "/X2024n.csv"))
