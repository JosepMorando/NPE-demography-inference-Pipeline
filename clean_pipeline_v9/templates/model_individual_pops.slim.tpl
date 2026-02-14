// ============================================
// SLiM 5 forward simulation for Fagus sylvatica
// Individual populations phylogenetic structure
// Phylogeny: ((P001,(BG01,((((BG05,BG04),BG07),Sauva),(Montsenymid,((Carlac,(Conangles,Viros)),(Cimadal,Coscollet)))))))
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

// ---- Initial ancestral population ----
1 early() {
    sim.addSubpop("p0", N0);   // ghost ancestor (not sampled)
}

// ---- Split 1: P001 and Node1 split from p0 ----
BURNIN_slim + T_P001 early() {
    sim.addSubpopSplit("p1", N_P001, p0);      // P001 population (sampled)
    sim.addSubpopSplit("p2", N_Node1, p0);     // Node1 (ghost, ancestor of rest)
}

// ---- Set p0 to N=1 for memory reduction ----
(BURNIN_slim + T_P001 + 1) early() {
    p0.setSubpopulationSize(1);
}

// ---- Split 2: BG01 and Node2 split from Node1 ----
BURNIN_slim + T_BG01 early() {
    sim.addSubpopSplit("p3", N_BG01, p2);      // BG01 population (sampled)
    sim.addSubpopSplit("p4", N_Node2, p2);     // Node2 (ghost, ancestor of southern+northern)
}

// ---- Set p2 to N=1 ----
(BURNIN_slim + T_BG01 + 1) early() {
    p2.setSubpopulationSize(1);
}

// ---- Split 3: Southern (Node3) and Northern (Node6) clades split from Node2 ----
BURNIN_slim + T_MAJOR_SPLIT early() {
    sim.addSubpopSplit("p5", N_Node3, p4);     // Node3 (ghost, southern clade)
    sim.addSubpopSplit("p12", N_Node6, p4);    // Node6 (ghost, northern clade)
}

// ---- Set p4 to N=1 ----
(BURNIN_slim + T_MAJOR_SPLIT + 1) early() {
    p4.setSubpopulationSize(1);
}

// ---- Split 4: Sauva and Node4 (BG05/BG04/BG07) split from Node3 ----
BURNIN_slim + T_Sauva early() {
    sim.addSubpopSplit("p11", N_Sauva, p5);    // Sauva (sampled)
    sim.addSubpopSplit("p6", N_Node4, p5);     // Node4 (ghost, BG05/BG04/BG07)
}

// ---- Set p5 to N=1 ----
(BURNIN_slim + T_Sauva + 1) early() {
    p5.setSubpopulationSize(1);
}

// ---- Split 5: BG07 and Node5 (BG05/BG04) split from Node4 ----
BURNIN_slim + T_BG07 early() {
    sim.addSubpopSplit("p10", N_BG07, p6);     // BG07 (sampled)
    sim.addSubpopSplit("p7", N_Node5, p6);     // Node5 (ghost, BG05/BG04)
}

// ---- Set p6 to N=1 ----
(BURNIN_slim + T_BG07 + 1) early() {
    p6.setSubpopulationSize(1);
}

// ---- Split 6: BG05 and BG04 split from Node5 ----
BURNIN_slim + T_BG05_BG04 early() {
    sim.addSubpopSplit("p8", N_BG05, p7);      // BG05 (sampled)
    sim.addSubpopSplit("p9", N_BG04, p7);      // BG04 (sampled)
}

// ---- Set p7 to N=1 ----
(BURNIN_slim + T_BG05_BG04 + 1) early() {
    p7.setSubpopulationSize(1);
}

// ---- Split 7: Montsenymid and Node7 (Pyrenees) split from Node6 ----
BURNIN_slim + T_Montsenymid early() {
    sim.addSubpopSplit("p13", N_Montsenymid, p12);  // Montsenymid (sampled)
    sim.addSubpopSplit("p14", N_Node7, p12);        // Node7 (ghost, Pyrenees)
}

// ---- Set p12 to N=1 ----
(BURNIN_slim + T_Montsenymid + 1) early() {
    p12.setSubpopulationSize(1);
}

// ---- Split 8: Western (Node8) and Eastern (Node10) Pyrenees split from Node7 ----
BURNIN_slim + T_PYRENEES early() {
    sim.addSubpopSplit("p15", N_Node8, p14);   // Node8 (ghost, western Pyrenees)
    sim.addSubpopSplit("p20", N_Node10, p14);  // Node10 (ghost, eastern Pyrenees)
}

// ---- Set p14 to N=1 ----
(BURNIN_slim + T_PYRENEES + 1) early() {
    p14.setSubpopulationSize(1);
}

// ---- Split 9: Carlac and Node9 (Conangles/Viros) split from Node8 ----
BURNIN_slim + T_Carlac early() {
    sim.addSubpopSplit("p16", N_Carlac, p15);  // Carlac (sampled)
    sim.addSubpopSplit("p17", N_Node9, p15);   // Node9 (ghost, Conangles/Viros)
}

// ---- Set p15 to N=1 ----
(BURNIN_slim + T_Carlac + 1) early() {
    p15.setSubpopulationSize(1);
}

// ---- Split 10: Conangles and Viros split from Node9 ----
BURNIN_slim + T_Conangles_Viros early() {
    sim.addSubpopSplit("p18", N_Conangles, p17);  // Conangles (sampled)
    sim.addSubpopSplit("p19", N_Viros, p17);      // Viros (sampled)
}

// ---- Set p17 to N=1 ----
(BURNIN_slim + T_Conangles_Viros + 1) early() {
    p17.setSubpopulationSize(1);
}

// ---- Split 11: Cimadal and Coscollet split from Node10 ----
BURNIN_slim + T_Cimadal_Coscollet early() {
    sim.addSubpopSplit("p21", N_Cimadal, p20);    // Cimadal (sampled)
    sim.addSubpopSplit("p22", N_Coscollet, p20);  // Coscollet (sampled)
}

// ---- Set p20 to N=1 ----
(BURNIN_slim + T_Cimadal_Coscollet + 1) early() {
    p20.setSubpopulationSize(1);
}


// ---- Optional demography extras (auto-injected) ----
//__BOTTLENECK_BLOCK__

//__EXPANSION_BLOCK__

//__MIGRATION_BLOCK__

// shrink one tick before output so the final extant individuals equal sample
(GENS_slim + BURNIN_slim) early() {
    // Sampled populations: p1, p3, p8, p9, p10, p11, p13, p16, p18, p19, p21, p22
    p1.setSubpopulationSize(20);   // P001
    p3.setSubpopulationSize(20);   // BG01
    p8.setSubpopulationSize(20);   // BG05
    p9.setSubpopulationSize(20);   // BG04
    p10.setSubpopulationSize(20);  // BG07
    p11.setSubpopulationSize(20);  // Sauva
    p13.setSubpopulationSize(20);  // Montsenymid
    p16.setSubpopulationSize(20);  // Carlac
    p18.setSubpopulationSize(20);  // Conangles
    p19.setSubpopulationSize(20);  // Viros
    p21.setSubpopulationSize(20);  // Cimadal
    p22.setSubpopulationSize(20);  // Coscollet

    // Ghost populations set to 1
    p0.setSubpopulationSize(1);
    p2.setSubpopulationSize(1);
    p4.setSubpopulationSize(1);
    p5.setSubpopulationSize(1);
    p6.setSubpopulationSize(1);
    p7.setSubpopulationSize(1);
    p12.setSubpopulationSize(1);
    p14.setSubpopulationSize(1);
    p15.setSubpopulationSize(1);
    p17.setSubpopulationSize(1);
    p20.setSubpopulationSize(1);
}

// ---- Final sampling (exclude ghost populations) ----
(GENS_slim + BURNIN_slim + 1) late() {
    sim.treeSeqOutput(TREE_OUT_slim);
    sim.simulationFinished();
}
