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

# Build poolfstat object
pd_g <- getClass("pooldata")@prototype
slot_names <- slotNames(pd_g)

if ("refallele.readcount" %in% slot_names) slot(pd_g, "refallele.readcount") <- ref_grouped
if ("readcoverage" %in% slot_names) slot(pd_g, "readcoverage") <- cov_grouped
if ("poolnames" %in% slot_names) slot(pd_g, "poolnames") <- group_order
if ("poolsizes" %in% slot_names) slot(pd_g, "poolsizes") <- pool_sizes
if ("npools" %in% slot_names) slot(pd_g, "npools") <- length(group_order)
if ("nsnp" %in% slot_names) slot(pd_g, "nsnp") <- n.snp

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

x_sim <- c(as.vector(t(sfs1d)), as.numeric(het), as.numeric(dxy_flat), as.numeric(fst_flat))

np$savez_compressed(
  file.path(out_dir, "sim_summaries.npz"),
  x = x_sim,
  group_order = group_order,
  target_cov = target_cov,
  bins = bins
)

cat("Saved simulated summaries to:", out_dir, "\n")
cat("x length:", length(x_sim), " (K=", K, ", bins=", bins, ")\n", sep = "")
