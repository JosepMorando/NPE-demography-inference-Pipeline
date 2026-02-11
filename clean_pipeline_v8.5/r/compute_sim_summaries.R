#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(poolfstat)
  library(reticulate)
})

# --- Helper Functions ---

# Helper to compute simple 2D hist without extra deps
# Defined early to ensure availability
hist2d_approx <- function(x, y, nbins=5) {
  breaks <- seq(0, 1.0001, length.out=nbins+1)
  # cut returns integer codes
  cx <- cut(x, breaks=breaks, include.lowest=TRUE, labels=FALSE)
  cy <- cut(y, breaks=breaks, include.lowest=TRUE, labels=FALSE)
  
  mat <- matrix(0, nrow=nbins, ncol=nbins)
  # simple loop to fill (vectorized index calculation is faster but this is safe)
  # 1-based indices
  if (length(cx) > 0 && length(cy) > 0) {
    counts <- table(factor(cx, levels=1:nbins), factor(cy, levels=1:nbins))
    mat <- as.matrix(counts)
  }
  
  # Normalize
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

counts_npz <- get_arg("--counts", NULL)
out_dir <- get_arg("--out", "sim_summaries")

if (is.null(counts_npz)) {
  stop("Usage: Rscript compute_sim_summaries.R --counts <counts.npz> --out <output_dir>")
}

np <- import("numpy", delay_load = TRUE)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data <- np$load(counts_npz, allow_pickle = TRUE)

alt_counts <- as.matrix(data[["alt_counts"]])
n_hap_per_pop <- as.integer(data[["n_hap_per_pop"]])
group_order <- as.character(data[["group_order"]])
target_cov <- as.integer(data[["target_cov"]])
pool_sizes <- as.numeric(data[["pool_sizes"]])
seed <- as.integer(data[["seed"]])

if (length(group_order) != ncol(alt_counts)) {
  stop("Group order length does not match alt_counts columns.")
}

set.seed(seed)

n.snp <- nrow(alt_counts)
K <- length(group_order)
if (n.snp == 0) {
  stop("No SNPs found in simulated counts; cannot compute poolfstat summaries.")
}

ac_down <- matrix(NA_integer_, nrow = n.snp, ncol = K)
colnames(ac_down) <- group_order

for (i in seq_len(K)) {
  for (j in seq_len(n.snp)) {
    alt <- alt_counts[j, i]
    if (is.na(alt)) next
    p_alt <- alt / n_hap_per_pop
    alt_ds <- rbinom(1, target_cov, p_alt)
    ac_down[j, i] <- alt_ds
  }
}

ref_grouped <- ifelse(is.na(ac_down), NA_integer_, target_cov - ac_down)
cov_grouped <- ifelse(is.na(ac_down), NA_integer_, target_cov)
dimnames(ref_grouped) <- dimnames(ac_down)
dimnames(cov_grouped) <- dimnames(ac_down)
rownames(ref_grouped) <- paste0("snp", seq_len(n.snp))
rownames(cov_grouped) <- rownames(ref_grouped)
if (any(colSums(!is.na(ref_grouped)) == 0)) {
  stop("One or more populations have no valid SNPs after downsampling.")
}
storage.mode(ref_grouped) <- "integer"
storage.mode(cov_grouped) <- "integer"

# 1D folded SFS per group
half <- floor(target_cov / 2)
bins <- half + 1
sfs1d <- matrix(0, nrow = K, ncol = bins)
rownames(sfs1d) <- group_order
colnames(sfs1d) <- paste0("bin", 0:half)

for (i in seq_len(K)) {
  ac <- ac_down[, i]
  ac <- ac[!is.na(ac)]
  folded <- pmin(ac, target_cov - ac)
  tab <- table(factor(folded, levels = 0:half))
  sfs1d[i, ] <- as.numeric(tab) / sum(tab)
}

# heterozygosity per group
p_mat <- ac_down / target_cov
het_mat <- 2 * p_mat * (1 - p_mat)
het <- colMeans(het_mat, na.rm = TRUE)

# Additional Ne-sensitive summaries
seg_prop <- colMeans((ac_down > 0) & (ac_down < target_cov), na.rm = TRUE)
tajima_proxy <- het - seg_prop
singleton_prop <- sapply(seq_len(K), function(i) {
  ac <- ac_down[, i]
  ac <- ac[!is.na(ac)]
  if (length(ac) == 0) return(NA_real_)
  folded <- pmin(ac, target_cov - ac)
  mean(folded == 1)
})

# Windowed variance of pi
n_windows <- min(20L, max(5L, floor(n.snp / 50L)))
window_id <- cut(seq_len(n.snp), breaks = n_windows, labels = FALSE)
pi_var_by_group <- sapply(seq_len(K), function(i) {
  vals <- tapply(het_mat[, i], window_id, function(v) mean(v, na.rm = TRUE))
  stats::var(as.numeric(vals), na.rm = TRUE)
})
pi_var_mean <- mean(pi_var_by_group, na.rm = TRUE)

# Build poolfstat object
pd_g <- methods::new("pooldata")
slot_names <- methods::slotNames(pd_g)

if ("refallele.readcount" %in% slot_names) methods::slot(pd_g, "refallele.readcount") <- ref_grouped
if ("readcoverage" %in% slot_names)        methods::slot(pd_g, "readcoverage")        <- cov_grouped

snp_info <- data.frame(
  chr = rep("chr1", n.snp),
  pos = seq_len(n.snp),
  ref = rep("A", n.snp),
  alt = rep("T", n.snp),
  stringsAsFactors = FALSE
)
rownames(snp_info) <- rownames(ref_grouped)
if ("snp.info" %in% slot_names) methods::slot(pd_g, "snp.info") <- snp_info

if ("poolnames" %in% slot_names) methods::slot(pd_g, "poolnames") <- group_order
if ("poolsizes" %in% slot_names) methods::slot(pd_g, "poolsizes") <- as.numeric(pool_sizes)
if ("npools" %in% slot_names)    methods::slot(pd_g, "npools")    <- length(group_order)
if ("nsnp" %in% slot_names)      methods::slot(pd_g, "nsnp")      <- n.snp

tryCatch(
  methods::validObject(pd_g),
  error = function(e) stop(e)
)

res <- compute.fstats(pd_g, verbose = FALSE)
fst <- res@pairwise.fst
div <- res@pairwise.div
fst <- fst[group_order, group_order]
div <- div[group_order, group_order]

dxy_flat <- upper_flat(div)
fst_flat <- upper_flat(fst)

# --- NEW: Pairwise Ratios (Pi_within / Dxy) ---
# Helps disentangle divergence time from ancestral Ne
pi_ratio_vals <- c()
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    pi_mean <- (het[a] + het[b]) / 2
    dxy_val <- div[a, b]
    ratio <- ifelse(dxy_val > 0, pi_mean / dxy_val, 0)
    pi_ratio_vals <- c(pi_ratio_vals, ratio)
  }
}

# --- NEW: Coarse 2D SFS (Block-wise) ---
# Extracts joint frequency information efficiently (low rank)
# Using a 5x5 grid (normalized to 0-1 range)
sfs_2d_coarse <- c()
n_2d_bins <- 5
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    # Get downsampled counts
    ac_a <- ac_down[, a]
    ac_b <- ac_down[, b]
    
    # Filter missing
    valid <- !is.na(ac_a) & !is.na(ac_b)
    ac_a <- ac_a[valid]
    ac_b <- ac_b[valid]
    
    # Normalize to [0, 1]
    freq_a <- ac_a / target_cov
    freq_b <- ac_b / target_cov
    
    # Bin
    h <- hist2d_approx(freq_a, freq_b, nbins=n_2d_bins)
    sfs_2d_coarse <- c(sfs_2d_coarse, as.vector(h))
  }
}

# --- Variance of Pairwise FST ---
pair_vals <- c()
for (a in 1:(K-1)) {
  for (b in (a+1):K) {
    pa <- p_mat[, a]
    pb <- p_mat[, b]
    num <- (pa - pb)^2
    den <- pa * (1 - pb) + pb * (1 - pa)
    fst_site <- ifelse(den > 0, num / den, NA_real_)
    fst_w <- tapply(fst_site, window_id, function(v) mean(v, na.rm = TRUE))
    pair_vals <- c(pair_vals, stats::var(as.numeric(fst_w), na.rm = TRUE))
  }
}
fst_var_mean <- mean(pair_vals, na.rm = TRUE)

x_sim <- c(
  as.vector(t(sfs1d)),
  as.numeric(het),
  as.numeric(seg_prop),
  as.numeric(tajima_proxy),
  as.numeric(singleton_prop),
  as.numeric(dxy_flat),
  as.numeric(fst_flat),
  as.numeric(pi_ratio_vals), # New
  as.numeric(sfs_2d_coarse), # New
  as.numeric(pi_var_mean),
  as.numeric(fst_var_mean)
)

np$savez_compressed(
  file.path(out_dir, "sim_summaries.npz"),
  x = x_sim,
  group_order = group_order,
  target_cov = target_cov,
  bins = bins
)

cat("Saved simulated summaries to:", out_dir, "\n")
cat("x length:", length(x_sim), "\n")