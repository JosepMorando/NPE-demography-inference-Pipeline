#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(poolfstat)
  library(reticulate)
})

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
  stop("Usage: Rscript compute_observed_summaries.R --pooldata <.RData> --groups <groups.csv> [--object filt.pooldata] [--target_cov 20] [--snp_start X --snp_end Y] --out observed_data")
}

# Numpy via reticulate
np <- import("numpy", delay_load = TRUE)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Load pooldata
env <- new.env()
load(pooldata_rdata, envir = env)
if (!exists(pooled_object, envir = env)) {
  stop(paste0("Object '", pooled_object, "' not found in ", pooldata_rdata))
}
pooldata <- get(pooled_object, envir = env)

cat("Loaded pooldata:", pooldata@nsnp, "SNPs across", pooldata@npools, "pools\n")

# Optional subset SNPs (only if BOTH start and end are given)
if (!is.na(snp_start) && !is.na(snp_end)) {
  cat("Subsetting SNPs from index", snp_start, "to", snp_end, "\n")
  pooldata <- pooldata.subset(pooldata, snp.index = snp_start:snp_end)
} else {
  cat("Using ALL", pooldata@nsnp, "SNPs (no subset applied)\n")
}

ref.counts <- pooldata@refallele.readcount
alt.counts <- pooldata@readcoverage - ref.counts

# Groups
map <- read.csv(groups_csv, stringsAsFactors = FALSE)
stopifnot(all(c("Pop","Group") %in% colnames(map)))

# preserve group order as in CSV
map$Group <- factor(map$Group, levels = unique(map$Group))
group_order <- levels(map$Group)

# sanity: all pops exist
missing_pops <- setdiff(unique(map$Pop), colnames(ref.counts))
if (length(missing_pops) > 0) {
  stop(paste0("These pops in groups.csv are missing from pooldata: ", paste(missing_pops, collapse=", ")))
}

# aggregate counts per group
ref_grouped <- sapply(group_order, function(g) {
  pops <- map$Pop[map$Group == g]
  rowSums(ref.counts[, pops, drop=FALSE])
})
alt_grouped <- sapply(group_order, function(g) {
  pops <- map$Pop[map$Group == g]
  rowSums(alt.counts[, pops, drop=FALSE])
})

# filter by coverage in all groups
cov_grouped <- ref_grouped + alt_grouped
valid_snps <- apply(cov_grouped, 1, function(x) all(!is.na(x) & x >= target_cov))
ref_grouped <- ref_grouped[valid_snps, , drop=FALSE]
alt_grouped <- alt_grouped[valid_snps, , drop=FALSE]

n.snp <- nrow(ref_grouped)
K <- length(group_order)
cat("After coverage filtering:", n.snp, "SNPs retained across", K, "groups\n")

set.seed(42)
# downsample to target_cov in each group
ac_down <- matrix(NA_integer_, nrow = n.snp, ncol = K)
colnames(ac_down) <- group_order

for (i in seq_len(K)) {
  for (j in seq_len(n.snp)) {
    ref <- ref_grouped[j, i]
    alt <- alt_grouped[j, i]
    tot <- ref + alt
    if (is.na(tot) || tot < target_cov) next
    p <- ref / tot
    # downsample REF count, then ALT = cov - REF; we store ALT allele count
    ref_ds <- rbinom(1, target_cov, p)
    ac_down[j, i] <- target_cov - ref_ds
  }
}

# 1D folded SFS per group
half <- floor(target_cov/2)
bins <- half + 1
sfs1d <- matrix(0, nrow=K, ncol=bins)
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
het <- colMeans(het_mat, na.rm=TRUE)

# Prepare poolfstat object grouped by clusters
pd_g <- pooldata
pd_g@refallele.readcount <- ref_grouped
pd_g@readcoverage <- ref_grouped + alt_grouped

# pool sizes: sum across members
pool_sizes <- pooldata@poolsizes
pool_names <- pooldata@poolnames
sizes_by_group <- sapply(group_order, function(g) {
  pops <- map$Pop[map$Group == g]
  sum(pool_sizes[pool_names %in% pops])
})

pd_g@poolsizes <- sizes_by_group
pd_g@poolnames <- group_order
pd_g@npools <- length(group_order)
pd_g@nsnp <- nrow(ref_grouped)

res <- compute.fstats(pd_g, verbose = FALSE)
fst <- res@pairwise.fst
div <- res@pairwise.div

# enforce group order
fst <- fst[group_order, group_order]
div <- div[group_order, group_order]

upper_flat <- function(M) {
  n <- nrow(M)
  unlist(lapply(1:(n-1), function(i) M[i, (i+1):n]))
}

dxy_flat <- upper_flat(div)
fst_flat <- upper_flat(fst)

# Build final observed vector (must match python summaries order)
x_obs <- c(as.vector(t(sfs1d)), as.numeric(het), as.numeric(dxy_flat), as.numeric(fst_flat))

# Save
np$save(file.path(out_dir, "x_obs.npy"), x_obs)
np$save(file.path(out_dir, "obs_sfs1d.npy"), sfs1d)
np$save(file.path(out_dir, "obs_het.npy"), het)
np$save(file.path(out_dir, "obs_dxy_flat.npy"), dxy_flat)
np$save(file.path(out_dir, "obs_fst_flat.npy"), fst_flat)

saveRDS(list(
  group_order = group_order,
  target_cov = target_cov,
  n_snps = n.snp,
  bins = bins
), file.path(out_dir, "observed_meta.rds"))

# Also create a python-friendly NPZ for infer_posterior.py
np$savez_compressed(
  file.path(out_dir, "observed_summaries.npz"),
  x_obs = x_obs,
  group_order = group_order,
  target_cov = target_cov
)

cat("Saved observed summaries to:", out_dir, "\n")
cat("x_obs length:", length(x_obs), " (K=", K, ", bins=", bins, ")\n", sep="")
