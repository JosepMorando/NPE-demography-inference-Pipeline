#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(poolfstat)
  library(reticulate)
})

# --- Configuration ---
# CRITICAL: This must match the 'genome_length' from your config_pod.yaml (1e6)
# If your Pooldata covers a larger region (e.g. 500Mb), set this to that real total size.
GENOME_LENGTH <- 2000000 

# --- Helper Functions ---
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

# --- Main Script ---
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (!is.na(i) && i < length(args)) return(args[[i+1]])
  return(default)
}

pooldata_rdata <- get_arg("--pooldata", NULL)
pooled_object <- get_arg("--object", "filt.pooldata")
groups_csv <- get_arg("--groups", NULL)
target_cov <- as.integer(get_arg("--target_cov", "20"))
snp_start <- as.integer(get_arg("--snp_start", NA))
snp_end <- as.integer(get_arg("--snp_end", NA))
out_dir <- get_arg("--out", "observed_data")

if (is.null(pooldata_rdata) || is.null(groups_csv)) {
  stop("Usage: Rscript compute_observed_summaries.R --pooldata <.RData> --groups <groups.csv> --out observed_data")
}

np <- import("numpy", delay_load = TRUE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Load Pooldata
env <- new.env()
load(pooldata_rdata, envir = env)
pooldata <- get(pooled_object, envir = env)

# Subset if requested
if (!is.na(snp_start) && !is.na(snp_end)) {
  pooldata <- pooldata.subset(pooldata, snp.index = snp_start:snp_end)
}

ref.counts <- pooldata@refallele.readcount
alt.counts <- pooldata@readcoverage - ref.counts
n.snp <- nrow(ref.counts)

# Calculate Invariant Sites (Padding)
n_invariant <- max(0, GENOME_LENGTH - n.snp)
cat("SNPs:", n.snp, " Invariants Added:", n_invariant, " Total Length:", GENOME_LENGTH, "\n")

# Groups
map <- read.csv(groups_csv, stringsAsFactors = FALSE)
map$Group <- factor(map$Group, levels = unique(map$Group))
group_order <- levels(map$Group)

# Aggregate Counts
ref_grouped <- sapply(group_order, function(g) rowSums(ref.counts[, map$Pop[map$Group == g], drop=FALSE]))
alt_grouped <- sapply(group_order, function(g) rowSums(alt.counts[, map$Pop[map$Group == g], drop=FALSE]))

# Downsampling
K <- length(group_order)
ac_down <- matrix(NA_integer_, nrow = n.snp, ncol = K)
set.seed(42)

for (i in seq_len(K)) {
  tot <- ref_grouped[,i] + alt_grouped[,i]
  p <- ref_grouped[,i] / tot
  # Skip bad coverage
  valid <- !is.na(tot) & tot >= target_cov
  if (any(valid)) {
    ref_ds <- rbinom(sum(valid), target_cov, p[valid])
    ac_down[valid, i] <- target_cov - ref_ds
  }
}

# 1. 1D SFS (Corrected with Padding)
half <- floor(target_cov/2)
bins <- half + 1
sfs1d <- matrix(0, nrow=K, ncol=bins)

for (i in seq_len(K)) {
  ac <- ac_down[, i]
  ac <- ac[!is.na(ac)]
  folded <- pmin(ac, target_cov - ac)
  tab <- table(factor(folded, levels = 0:half))
  
  # --- PADDING FIX ---
  tab[1] <- tab[1] + n_invariant
  sfs1d[i, ] <- as.numeric(tab) / GENOME_LENGTH # Normalize by TOTAL length
}

# 2. Diversity (Corrected with Padding)
p_mat <- ac_down / target_cov
het_mat <- 2 * p_mat * (1 - p_mat)
het_per_snp <- colMeans(het_mat, na.rm = TRUE)
# Scale to per-bp
het <- het_per_snp * (n.snp / GENOME_LENGTH)

# 3. Proxies (Scaled)
seg_prop <- colMeans((ac_down > 0) & (ac_down < target_cov), na.rm = TRUE) * (n.snp / GENOME_LENGTH)
tajima_proxy <- het - seg_prop
singleton_prop <- sapply(seq_len(K), function(i) {
  ac <- ac_down[, i]; ac <- ac[!is.na(ac)]
  sum(pmin(ac, target_cov - ac) == 1) / GENOME_LENGTH
})

# 4. Pairwise Stats (Poolfstat)
# Recalculate pool sizes for poolfstat
pd_g <- pooldata
pd_g@refallele.readcount <- ref_grouped
pd_g@readcoverage <- ref_grouped + alt_grouped
pd_g@poolsizes <- sapply(group_order, function(g) sum(pooldata@poolsizes[pooldata@poolnames %in% map$Pop[map$Group == g]]))
pd_g@poolnames <- group_order
pd_g@npools <- K
pd_g@nsnp <- n.snp

res <- compute.fstats(pd_g, verbose = FALSE)
fst <- res@pairwise.fst[group_order, group_order]
div <- res@pairwise.div[group_order, group_order]

# Scale Dxy to per-bp (Fst is a ratio, so it stays the same)
div <- div * (n.snp / GENOME_LENGTH)

dxy_flat <- upper_flat(div)
fst_flat <- upper_flat(fst)

# 5. Pi / Dxy Ratios
pi_ratio_vals <- c()
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    pi_mean <- (het[a] + het[b]) / 2
    dxy_val <- div[a, b]
    ratio <- ifelse(dxy_val > 1e-9, pi_mean / dxy_val, 0)
    pi_ratio_vals <- c(pi_ratio_vals, ratio)
  }
}

# 6. Coarse 2D SFS (No padding needed, normalized by sum of segregating sites usually, or raw)
# But to match Sim, we typically just bin frequencies. 
# Sim SFS 2D might include invariants in bin (0,0).
# Let's check sim logic. If sim uses `hist2d_approx`, it normalizes by sum(mat).
# If `p_mat` in sim includes zeros, bin (0,0) is huge.
# We must replicate this manually.
sfs_2d_coarse <- c()
n_2d_bins <- 5
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    freq_a <- ac_down[, a] / target_cov
    freq_b <- ac_down[, b] / target_cov
    
    # Simple hist
    h <- hist2d_approx(freq_a, freq_b, nbins=n_2d_bins)
    
    # Re-normalize with invariants in (0,0)
    # Undo normalization
    counts <- h * n.snp 
    counts[1,1] <- counts[1,1] + n_invariant
    h_final <- counts / GENOME_LENGTH
    
    sfs_2d_coarse <- c(sfs_2d_coarse, as.vector(h_final))
  }
}

# 7. Variance Stats (No scaling needed for variance of Fst, maybe for Pi var?)
# Pi Variance: The variance of Pi across windows. 
# Since we only have SNPs, our "windows" are defined by SNP indices. 
# This matches the simulation logic (also defined by SNP indices). 
# So NO SCALING needed for 'pi_var_mean' or 'fst_var_mean'.
n_windows <- min(20L, max(5L, floor(n.snp / 50L)))
window_id <- cut(seq_len(n.snp), breaks = n_windows, labels = FALSE)

pi_var_mean <- mean(sapply(seq_len(K), function(i) {
  # Het per SNP
  vals <- tapply(het_mat[, i], window_id, mean, na.rm=TRUE) 
  stats::var(as.numeric(vals), na.rm=TRUE)
}), na.rm=TRUE)

pair_fst_vars <- c()
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    pa <- p_mat[, a]; pb <- p_mat[, b]
    num <- (pa - pb)^2
    den <- pa * (1 - pb) + pb * (1 - pa)
    fst_site <- ifelse(den > 0, num / den, NA_real_)
    fst_w <- tapply(fst_site, window_id, mean, na.rm=TRUE)
    pair_fst_vars <- c(pair_fst_vars, stats::var(as.numeric(fst_w), na.rm=TRUE))
  }
}
fst_var_mean <- mean(pair_fst_vars, na.rm=TRUE)
if(is.na(fst_var_mean)) fst_var_mean <- 0

# --- Output ---
x_obs <- c(
  as.vector(t(sfs1d)),
  as.numeric(het),
  as.numeric(seg_prop),
  as.numeric(tajima_proxy),
  as.numeric(singleton_prop),
  as.numeric(dxy_flat),
  as.numeric(fst_flat),
  as.numeric(pi_ratio_vals),
  as.numeric(sfs_2d_coarse),
  as.numeric(pi_var_mean),
  as.numeric(fst_var_mean)
)

np$savez_compressed(
  file.path(out_dir, "observed_summaries.npz"),
  x_obs = x_obs,
  group_order = group_order,
  target_cov = target_cov
)
cat("Done. Saved correct observed summaries for NPE.\n")