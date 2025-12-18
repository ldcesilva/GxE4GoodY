# Modified from https://github.com/igorkf/Maize_GxE_Prediction/tree/main/src/kronecker.R
root_path = "\\tdl\Public2\G2F\Manuscript\"; # Adjust to your path
prep_path = root_path || "\Results\DataPrep\";
train_path = root_path || "\Data\Challenge2024\Training_data";
test_path = root_path || "\Data\Challenge2024\Testing_data";

library(data.table)

# you must run the code twice, first for additive than comment out additive and enable dominance
kinship_type <- "Additive"
#kinship_type <- "Dominance" 

kinship_path <- paste0(prep_path, kinship_type,  ".txt")
outfile <- paste0(prep_path, "/kronecker_", kinship_type, ".arrow")
outfile.csv <- paste0(prep_path, "/kronecker_", kinship_type, ".csv")
cat("Using", kinship_type, "matrix\n")

kinship_type_pca <- paste0("Subset of ",kinship_type," PCA")
kinship_path_pca <- paste0(prep_path, kinship_type_pca,  ".txt")

# read files
xfile <- "gxe_xtrain.csv"
xtrain <- fread(paste0(prep_path, '/', xfile), data.table = F)

xtestfile <- "gxe_xtest.csv"
xtest <- fread(paste0(prep_path, '/', xtestfile), data.table = F)

# bind files and aggregate
x <- rbind(xtrain, xtest)
rm(xtrain); rm(xtest); gc()
x <- x[, !grepl("yield_lag", colnames(x))]  # remove all yield-related features
x$Hybrid <- NULL
x <- aggregate(x[, -1], by = list(x$Env), FUN = mean)  # take mean within envs
rownames(x) <- x$Group.1
x$Group.1 <- NULL
x$Env <- NULL
x <- as.matrix(x)

# read phenotypes
yfile <- "gxe_ytrain.csv"
ytrain <- fread(paste0(prep_path, '/', yfile), data.table = F)

ytestfile <- "gxe_ytest.csv"
ytest <- fread(paste0(prep_path, '/', ytestfile), data.table = F)

# get unique combinations
y <- rbind(ytrain, ytest)
cat("y dim:", dim(y), "\n")
hybrids <- unique(y$Hybrid)
cat("hybrids dim:", length(hybrids), "\n")
env_hybrid <- unique(interaction(y$Env, y$Hybrid, sep = ':', drop = T))
rm(y); rm(ytrain); rm(ytest); gc()

# load kinship
kinship <- fread(file = kinship_path, data.table = F)
cat("kinship dim:", dim(kinship), "\n")
colnames(kinship) <- substr(colnames(kinship), 1, nchar(colnames(kinship)))  # fix column names
kinship <- as.matrix(kinship)
rownames(kinship) <- colnames(kinship)[1:nrow(kinship)]

kinship_pca <- fread(file = kinship_path_pca, data.table = F)
cat("kinship dim:", dim(kinship_pca), "\n")
kinship_pca <- as.matrix(kinship_pca)
rownames(kinship_pca) <- colnames(kinship)[1:nrow(kinship)]
kinship_pca <- kinship_pca[rownames(kinship_pca) %in% hybrids, ]
cat("kinship_pca dim:", dim(kinship_pca), "\n")

kinship <- kinship[rownames(kinship) %in% hybrids, colnames(kinship) %in% hybrids]
cat("kinship dim:", dim(kinship), "\n")

K <- kronecker(x, kinship_pca, make.dimnames = T)
cat("K dim:", dim(K), "\n")

# some Env:Hybrid combinations were not phenotyped
K <- K[rownames(K) %in% env_hybrid, ]
cat("K dim:", dim(K), "\n")

if (FALSE) {
	KAll <- c()
	j <- 1
	k <- 5 
	for(i in 1:10) {	
		print(c(i, j, k))
		K <- kronecker(x[,1:5], kinship, make.dimnames = T)
		rm(x); rm(kinship); gc()
		cat("K dim:", dim(K), "\n")

		# some Env:Hybrid combinations were not phenotyped
		K <- K[rownames(K) %in% env_hybrid, ]
		cat("K dim:", dim(K), "\n")
		KAll <- cbind(KAll, K)
		j <- j + 5
		k <- k + 5
	}
}

# write to feather for fast reading
arrow::write_feather(
  data.frame(id = rownames(K), K), 
  outfile
)

write.csv(data.frame(id = rownames(K), K), outfile.csv)

rm(K); gc()
cat("Writing file:", outfile, "\n\n")
Sys.sleep(5)
