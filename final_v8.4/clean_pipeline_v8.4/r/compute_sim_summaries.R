#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(poolfstat)
  library(reticulate)
})

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
# Watterson-like proxy: proportion of segregating sites
seg_prop <- colMeans((ac_down > 0) & (ac_down < target_cov), na.rm = TRUE)
# Tajima-like proxy: pi - theta_W_proxy
tajima_proxy <- het - seg_prop
# Singleton / low-frequency proxy (folded count == 1)
singleton_prop <- sapply(seq_len(K), function(i) {
  ac <- ac_down[, i]
  ac <- ac[!is.na(ac)]
  if (length(ac) == 0) return(NA_real_)
  folded <- pmin(ac, target_cov - ac)
  mean(folded == 1)
})

# Windowed variance of pi (mean across groups)
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

# Required matrices (nsnp x npools)
if ("refallele.readcount" %in% slot_names) methods::slot(pd_g, "refallele.readcount") <- ref_grouped
if ("readcoverage" %in% slot_names)        methods::slot(pd_g, "readcoverage")        <- cov_grouped

# REQUIRED in poolfstat >= 3.0: snp.info (nsnp x 4)
# Create dummy but valid SNP metadata (names usually not enforced; order is)
snp_info <- data.frame(
  chr = rep("chr1", n.snp),
  pos = seq_len(n.snp),
  ref = rep("A", n.snp),
  alt = rep("T", n.snp),
  stringsAsFactors = FALSE
)
rownames(snp_info) <- rownames(ref_grouped)

if ("snp.info" %in% slot_names) methods::slot(pd_g, "snp.info") <- snp_info

# Pool meta
if ("poolnames" %in% slot_names) methods::slot(pd_g, "poolnames") <- group_order
if ("poolsizes" %in% slot_names) methods::slot(pd_g, "poolsizes") <- as.numeric(pool_sizes)
if ("npools" %in% slot_names)    methods::slot(pd_g, "npools")    <- length(group_order)
if ("nsnp" %in% slot_names)      methods::slot(pd_g, "nsnp")      <- n.snp

# Hard fail early with a useful message if still invalid
tryCatch(
  methods::validObject(pd_g),
  error = function(e) {
    cat("pooldata validity failed:\n")
    cat(conditionMessage(e), "\n")
    cat("Slot names:", paste(slot_names, collapse=", "), "\n")
    cat("Dims ref:", paste(dim(ref_grouped), collapse="x"), "\n")
    cat("Dims cov:", paste(dim(cov_grouped), collapse="x"), "\n")
    cat("pool_sizes length:", length(pool_sizes), "\n")
    stop(e)
  }
)

res <- compute.fstats(pd_g, verbose = FALSE)
fst <- res@pairwise.fst
div <- res@pairwise.div
# enforce group order
fst <- fst[group_order, group_order]
div <- div[group_order, group_order]

upper_flat <- function(M) {
  n <- nrow(M)
  unlist(lapply(1:(n - 1), function(i) M[i, (i + 1):n]))
}

dxy_flat <- upper_flat(div)
fst_flat <- upper_flat(fst)

# Windowed variance of pairwise Hudson-like FST proxy (mean across pairs)
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
cat("x length:", length(x_sim), " (K=", K, ", bins=", bins, ")\n", sep = "")
