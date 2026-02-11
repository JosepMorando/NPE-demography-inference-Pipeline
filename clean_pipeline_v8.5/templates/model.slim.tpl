// ============================================
// SLiM 5 forward simulation for Fagus sylvatica
// Catalonia phylogenetic structure
// ============================================

initialize() {
    initializeSLiMModelType("WF");

    // ---- Simulation constants ----
    defineConstant("TREE_OUT_slim", TREE_OUT);
    defineConstant("L", asInteger(LEN));    // Genome length
    defineConstant("R_slim", R);
    defineConstant("GENS_slim", GENS);
    defineConstant("BURNIN_slim", BURNIN);

    // ---- Mutation and recombination ----
    initializeTreeSeq(recordMutations=F, simplificationRatio=INF);
    initializeMutationRate(0);
    initializeMutationType("m1", 0.5, "f", 0.0);
    initializeGenomicElementType("g1", m1, 1.0);
    initializeGenomicElement(g1, 0, L - 1);
    initializeRecombinationRate(R_slim);
}

// ---- Initial ancestral population FR ----
1 early() {
    sim.addSubpop("p0", N0);   // ghost ancestor (not sampled)
}

// ---- BG01 splits early from the ancestor ----
BURNIN_slim + T_BG01 early() {
    sim.addSubpopSplit("p1", N_BG01, p0);  // BG01 population
}

// ---- set N=1 p0 for memory reduction ----
(BURNIN_slim + T_BG01 + 1) early() {
    p0.setSubpopulationSize(1);
}

// ---- Core Catalan lineage (Montnegre) ----
BURNIN_slim + T_CORE early() {
    sim.addSubpopSplit("p2", N_CORE, p1);  // unsampled lineage giving rise to others
}

// ---- SOUTH_LOW cluster (Sauva) ----
BURNIN_slim + T_SOUTH_LOW early() {
    sim.addSubpopSplit("p3", N_SOUTH_LOW, p2);
}

// ---- SOUTH_MID cluster (Montsenymid) ----
BURNIN_slim + T_SOUTH_MID early() {
    sim.addSubpopSplit("p4", N_SOUTH_MID, p2);
}

// ---- EAST cluster (BG04, BG05, BG07) ----
BURNIN_slim + T_EAST early() {
    sim.addSubpopSplit("p5", N_EAST, p3);
}

// ---- Pyreneean and Prepyrenean cluster (Pre-Litoral lineage BG13) ----
BURNIN_slim + T_INT early() {
    sim.addSubpopSplit("p6", N_INT, p2);  // unsampled lineage giving rise to Pyrenees
}

// ---- CENTRAL cluster (Coscollet, Cimadal) ----
BURNIN_slim + T_CENTRAL early() {
    sim.addSubpopSplit("p7", N_CENTRAL, p6);
}

// ---- PYRENEES cluster (Carlac, Conangles, Viros) ----
BURNIN_slim + T_PYRENEES early() {
    sim.addSubpopSplit("p8", N_PYRENEES, p6);
}


// ---- Optional demography extras (auto-injected) ----
//__BOTTLENECK_BLOCK__

//__EXPANSION_BLOCK__

//__MIGRATION_BLOCK__

// shrink one tick before output so the final extant individuals equal sample
(GENS_slim + BURNIN_slim) early() {
    p1.setSubpopulationSize(20);
    p2.setSubpopulationSize(1);
    p3.setSubpopulationSize(20);
    p4.setSubpopulationSize(20);
    p5.setSubpopulationSize(20);
    p7.setSubpopulationSize(20);
    p8.setSubpopulationSize(20);
}

// ---- Final sampling (exclude ghost ancestor) ----
(GENS_slim + BURNIN_slim + 1) late() {
    sim.treeSeqOutput(TREE_OUT_slim);
    sim.simulationFinished();
}
