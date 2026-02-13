#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(poolfstat)
  library(reticulate)
})

# --- Helper Functions (Identical to Observed) ---
hist2d_approx <- function(x, y, nbins=5) {
  breaks <- seq(0, 1.0001, length.out=nbins+1)
  cx <- cut(x, breaks=breaks, include.lowest=TRUE, labels=FALSE)
  cy <- cut(y, breaks=breaks, include.lowest=TRUE, labels=FALSE)
  mat <- matrix(0, nrow=nbins, ncol=nbins)
  if (length(cx) > 0 && length(cy) > 0) {
    counts <- table(factor(cx, levels=1:nbins), factor(cy, levels=1:nbins))
    mat <- as.matrix(counts)
  }
  if (sum(mat) > 0) mat / sum(mat) else mat
}

upper_flat <- function(M) {
  n <- nrow(M)
  if (n < 2) return(numeric(0))
  unlist(lapply(1:(n - 1), function(i) M[i, (i + 1):n]))
}

# --- Main Logic ---
args <- commandArgs(trailingOnly = TRUE)
counts_npz <- args[which(args == "--counts") + 1]
out_dir <- args[which(args == "--out") + 1]

np <- import("numpy", delay_load = TRUE)
data <- np$load(counts_npz, allow_pickle = TRUE)

alt_counts <- as.matrix(data[["alt_counts"]])
n_hap_per_pop <- as.integer(data[["n_hap_per_pop"]])
group_order <- as.character(data[["group_order"]])
target_cov <- as.integer(data[["target_cov"]])
pool_sizes <- as.numeric(data[["pool_sizes"]])
set.seed(as.integer(data[["seed"]]))

n.snp <- nrow(alt_counts)
K <- length(group_order)

# Optimized Vectorized Downsampling
p_alt_mat <- alt_counts / n_hap_per_pop
ac_down <- matrix(rbinom(n.snp * K, target_cov, as.numeric(p_alt_mat)), nrow = n.snp, ncol = K)
colnames(ac_down) <- group_order

# 1. 1D SFS (Transposed vector as in observed script)
half <- floor(target_cov / 2)
sfs1d <- matrix(0, nrow = K, ncol = half + 1)
for (i in seq_len(K)) {
  folded <- pmin(ac_down[, i], target_cov - ac_down[, i])
  tab <- table(factor(folded, levels = 0:half))
  sfs1d[i, ] <- as.numeric(tab) / sum(tab)
}

# 2. Diversity Stats
p_mat <- ac_down / target_cov
het_mat <- 2 * p_mat * (1 - p_mat)
het <- colMeans(het_mat, na.rm = TRUE)
seg_prop <- colMeans(ac_down > 0 & ac_down < target_cov, na.rm = TRUE)
tajima_proxy <- het - seg_prop
singleton_prop <- apply(ac_down, 2, function(ac) mean(pmin(ac, target_cov - ac) == 1))

# 3. Windowed Variance of Pi
n_windows <- min(20L, max(5L, floor(n.snp / 50L)))
window_id <- cut(seq_len(n.snp), breaks = n_windows, labels = FALSE)
pi_var_mean <- mean(apply(het_mat, 2, function(h) stats::var(tapply(h, window_id, mean), na.rm=TRUE)))
if (is.na(pi_var_mean) || is.nan(pi_var_mean)) pi_var_mean <- 0.0

# 4. Poolfstat Pairwise (Dxy and Fst)
pd_g <- new("pooldata")
pd_g@refallele.readcount <- as.matrix(target_cov - ac_down)
pd_g@readcoverage <- matrix(target_cov, nrow = n.snp, ncol = K)
pd_g@poolnames <- group_order
pd_g@poolsizes <- as.numeric(pool_sizes)
pd_g@npools <- as.integer(K)
pd_g@nsnp <- as.integer(n.snp)

res <- compute.fstats(pd_g, verbose = FALSE)
fst_flat <- upper_flat(res@pairwise.fst[group_order, group_order])
div_flat <- upper_flat(res@pairwise.div[group_order, group_order])

# 5. Pairwise Ratios (New)
pi_ratio_vals <- c()
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    dxy_val <- div_flat[length(pi_ratio_vals)+1]
    # FIX: Safe division to avoid Inf/NaN
    if (is.na(dxy_val) || abs(dxy_val) < 1e-9) {
      ratio <- 0.0
    } else {
      ratio <- ((het[a] + het[b])/2) / dxy_val
    }
    pi_ratio_vals <- c(pi_ratio_vals, ratio)
  }
}

# 6. 2D SFS (Coarse)
sfs_2d_coarse <- c()
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    sfs_2d_coarse <- c(sfs_2d_coarse, as.vector(hist2d_approx(p_mat[,a], p_mat[,b], nbins=5)))
  }
}

# 7. Variance of Fst Proxy
pair_fst_vars <- c()
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    num <- (p_mat[,a] - p_mat[,b])^2
    den <- p_mat[,a]*(1-p_mat[,b]) + p_mat[,b]*(1-p_mat[,a])
    fst_site <- ifelse(den > 0, num / den, NA_real_)
    # Calculate windowed Fst means
    w_means <- tapply(fst_site, window_id, mean, na.rm=TRUE)
    # Variance of these means
    v <- stats::var(as.numeric(w_means), na.rm=TRUE)
    pair_fst_vars <- c(pair_fst_vars, v)
  }
}
fst_var_mean <- mean(pair_fst_vars, na.rm = TRUE)

# FIX: Check for NaN/NA in final mean (This fixes Column 511)
if (is.na(fst_var_mean) || is.nan(fst_var_mean)) {
  fst_var_mean <- 0.0
}

# --- FINAL VECTOR ASSEMBLY (Identical Order to Observed) ---
x_sim <- c(
  as.vector(t(sfs1d)),
  as.numeric(het),
  as.numeric(seg_prop),
  as.numeric(tajima_proxy),
  as.numeric(singleton_prop),
  as.numeric(div_flat), # Observed uses dxy_flat then fst_flat
  as.numeric(fst_flat),
  as.numeric(pi_ratio_vals),
  as.numeric(sfs_2d_coarse),
  as.numeric(pi_var_mean),
  as.numeric(fst_var_mean)
)

np$savez_compressed(file.path(out_dir, "sim_summaries.npz"), x = x_sim)