# read tab-delimted input file with row feature and column samples
X <- read.table('path/to/file.txt', header = TRUE,
                row.names = 1,   sep = "\t", fill = FALSE, comment.char = "" , check.names = FALSE)

# tranpose for R to column be the features
X <- as.data.frame(t(X))
method = 'spearman'
x_corr <- rcorr(as.matrix(X), type=method)

# caluclate distance between features by 1 - abs(spearman correlation)
adist <- 1-abs(x_corr$r)
write.table(adist,  
            "path/to/adist.txt",
            sep = "\t", eol = "\n", col.names = NA, row.names = T)