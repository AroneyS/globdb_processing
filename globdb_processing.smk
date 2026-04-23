"""
Globdb processing pipeline

pixi run snakemake \
    --snakefile scripts/globdb_processing.smk \
    --directory results/globdb_processing/20260408 \
    --profile aqua --retries 3 \
    --cores 64 --local-cores 1 \
    --rerun-triggers mtime
"""

import os
import polars as pl

SNAKEFILE_DIR = workflow.basedir

GLOBDB_INFO = "/work/microbiome/ibis/public_genomes/globdb_r232/globdb_r232_domains.tsv"
GLOBDB_FASTA_PATHS = "/work/microbiome/ibis/public_genomes/globdb_r232/genome_fasta_paths.tsv"
DOMAINS = ["archaea", "bacteria"]
GTDB_ARC_TAX = "/work/microbiome/msingle/mess/214_R232_renew/ar53_taxonomy_r232.tsv"
GTDB_BAC_TAX = "/work/microbiome/msingle/mess/214_R232_renew/bac120_taxonomy_r232.tsv"
GTDB_ARC_RED = [0.2859284763413424, 0.42852344844350987, 0.5692329378979698, 0.758762614031466, 0.9250910192387802]
GTDB_BAC_RED = [0.3399759686386926, 0.4764466993167481, 0.6467334973890564, 0.7972969419385083, 0.9438360803986985]

globdb_paths = (
    pl.read_csv(GLOBDB_FASTA_PATHS, separator="\t", has_header=False, new_columns=["fasta"])
    .with_columns(
        f = pl.col("fasta").str.extract(r"([^/]+\.fa\.gz)$"),
    )
)

globdb = (
    pl.read_csv(GLOBDB_INFO, separator="\t")
    .select("ID", "domain", "checkm2_completeness", "checkm2_contamination")
    .with_columns(
        domain = 
            pl.when(pl.col("domain").str.contains("Archaea"))
            .then(pl.lit("archaea"))
            .when(pl.col("domain").str.contains("Bacteria"))
            .then(pl.lit("bacteria"))
            .otherwise(pl.lit("unknown")),
        f = pl.concat_str(pl.col("ID"), pl.lit(".fa.gz")),
        )
    .join(globdb_paths, on="f")
    .filter(pl.col("domain") != "unknown")
)

#################
### Functions ###
#################
def get_mem_mb(wildcards, threads):
    mem = 8 * 1000 * threads
    if mem == 512000:
        return 500000
    elif mem == 256000:
        return 250000
    else:
        return mem

####################
### Global rules ###
####################
rule all:
    input:
        expand("{domain}/name_clades", domain=DOMAINS),
        expand("{domain}/clade_summary.tsv", domain=DOMAINS),

rule gtdbtk_batch:
    output:
        "{domain}/genome_batch.tsv",
    threads: 1
    localrule: True
    run:
        (
            globdb
            .filter(pl.col("domain") == wildcards.domain)
            .select("fasta", "ID")
            .write_csv(output[0], separator="\t", include_header=False)
        )

rule gtdbtk_identify:
    input:
        genomes = "{domain}/genome_batch.tsv",
    output:
        directory("{domain}/gtdbtk_identify"),
    threads: 64
    resources:
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    benchmark:
        "{domain}/gtdbtk_identify/benchmark.txt"
    log:
        "{domain}/logs/gtdbtk_identify.log"
    shell:
        "pixi run -e gtdbtk gtdbtk identify "
        "--batchfile {input} "
        "--out_dir {output} "
        "--cpus {threads} "
        "&> {log} "

rule gtdbtk_align:
    input:
        identify = "{domain}/gtdbtk_identify",
    output:
        directory("{domain}/gtdbtk_align"),
    threads: 64
    resources:
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    benchmark:
        "{domain}/gtdbtk_align/benchmark.txt"
    log:
        "{domain}/logs/gtdbtk_align.log"
    shell:
        "pixi run -e gtdbtk gtdbtk align "
        "--skip_gtdb_refs "
        "--identify_dir {input.identify} "
        "--out_dir {output} "
        "--cpus {threads} "
        "&> {log} "

rule gtdbtk_infer:
    input:
        align = "{domain}/gtdbtk_align",
    output:
        directory("{domain}/gtdbtk_infer"),
    threads: 64
    resources:
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    benchmark:
        "{domain}/gtdbtk_infer/benchmark.txt"
    params:
        msa = lambda wildcards: "gtdbtk.ar53.user_msa.fasta.gz" if wildcards.domain == "archaea" else "gtdbtk.bac120.user_msa.fasta.gz",
    log:
        "{domain}/logs/gtdbtk_infer.log"
    shell:
        "pixi run -e gtdbtk gtdbtk infer "
        "--msa_file {input.align}/align/{params.msa} "
        "--out_dir {output} "
        "--cpus {threads} "
        "&> {log} "

rule preprocess_taxonomy:
    output:
        "{domain}/gtdb_taxonomy.tsv",
    threads: 1
    localrule: True
    params:
        taxonomy = lambda wildcards: GTDB_ARC_TAX if wildcards.domain == "archaea" else GTDB_BAC_TAX,
    run:
        (
            pl.read_csv(params.taxonomy, separator="\t", has_header=False, new_columns=["genome", "taxonomy"])
            .with_columns(pl.col("genome").str.extract(r"(GC[AF]_\d+)"))
            .write_csv(output[0], separator="\t", include_header=False)
        )

rule gtdbtk_root:
    input:
        infer = "{domain}/gtdbtk_infer",
        tax = "{domain}/gtdb_taxonomy.tsv",
    output:
        directory("{domain}/gtdbtk_root"),
    threads: 1
    resources:
        mem_mb = lambda wildcards: get_mem_mb(wildcards) * 4,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    params:
        outgroup = lambda wildcards: "p__Altiarchaeota" if wildcards.domain == "archaea" else "p__Fusobacteriota",
    benchmark:
        "{domain}/gtdbtk_root/benchmark.txt"
    log:
        "{domain}/logs/gtdbtk_root.log"
    shell:
        "pixi run -e gtdbtk gtdbtk root "
        "--input_tree {input.infer}/gtdbtk.unrooted.tree "
        "--outgroup_taxon {params.outgroup} "
        "--custom_taxonomy_file {input.tax} "
        "--output_tree {output}/gtdbtk.rooted.tree "
        "&> {log} "

rule gtdbtk_decorate:
    input:
        root = "{domain}/gtdbtk_root",
        tax = "{domain}/gtdb_taxonomy.tsv",
    output:
        directory("{domain}/gtdbtk_decorate"),
    threads: 1
    resources:
        mem_mb = lambda wildcards: get_mem_mb(wildcards) * 8,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    benchmark:
        "{domain}/gtdbtk_decorate/benchmark.txt"
    log:
        "{domain}/logs/gtdbtk_decorate.log"
    shell:
        "pixi run -e gtdbtk gtdbtk decorate "
        "--input_tree {input.root}/gtdbtk.rooted.tree "
        "--custom_taxonomy_file {input.tax} "
        "--output_tree {output}/gtdbtk.decorated.tree "
        "&> {log} "

rule phylorank_red:
    input:
        tree = "{domain}/gtdbtk_decorate",
        tax = "{domain}/gtdb_taxonomy.tsv",
    output:
        directory("{domain}/phylorank"),
    threads: 64
    resources:
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    benchmark:
        "{domain}/phylorank/benchmark.txt"
    log:
        "{domain}/logs/phylorank.log"
    shell:
        "pixi run -e phylorank phylorank outliers "
        "{input.tree}/gtdbtk.decorated.tree "
        "{input.tax} "
        "{output} "
        "&> {log} "

rule tree2tbl:
    input:
        dir = "{domain}/phylorank",
        tax = "{domain}/gtdb_taxonomy.tsv",
    output:
        tree = "{domain}/phylorank_tree.tsv",
    threads: 1
    resources:
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 24*60*attempt,
    params:
        red_cutoffs = lambda wildcards: GTDB_ARC_RED if wildcards.domain == "archaea" else GTDB_BAC_RED,
        input_tree = "gtdbtk.decorated.red_decorated.tree",
        script = os.path.join(SNAKEFILE_DIR, "tree2tbl.R"),
    benchmark:
        "{domain}/phylorank_tree_benchmark.txt"
    log:
        "{domain}/logs/tree2tbl.log"
    shell:
        "pixi run -e r-treedataverse Rscript {params.script} "
        "--input-tree {input.dir}/{params.input_tree} "
        "--taxonomy {input.tax} "
        "--output-tree {output.tree} "
        "--red-cutoffs '{params.red_cutoffs}' "
        "&> {log} "

rule name_clades:
    input:
        tree = "{domain}/phylorank_tree.tsv",
        tax = "{domain}/gtdb_taxonomy.tsv",
    output:
        directory("{domain}/name_clades"),
    threads: 32
    resources:
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    params:
        script = "/work/microbiome/ibis/public_genomes/scripts/name_clades.py",
        metadata = GLOBDB_INFO,
        red_cutoffs = lambda wildcards: " ".join([str(r) for r in GTDB_ARC_RED]) if wildcards.domain == "archaea" else " ".join([str(r) for r in GTDB_BAC_RED]),
        domain = lambda wildcards: "d__Archaea" if wildcards.domain == "archaea" else "d__Bacteria",
    benchmark:
        "{domain}/name_clades/benchmark.txt"
    log:
        "{domain}/logs/name_clades.log"
    shell:
        "pixi run -e name-clades python {params.script} "
        "--tree-df {input.tree} "
        "--metadata {params.metadata} "
        "--gtdb {input.tax} "
        "--domain {params.domain} "
        "--red-cutoffs {params.red_cutoffs} "
        "--output {output} "
        "&> {log} "

rule summarise_clades:
    input:
        name_clades = "{domain}/name_clades",
    output:
        "{domain}/clade_summary.tsv",
    threads: 1
    resources:
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 24*60*attempt,
    log:
        "{domain}/logs/clade_summary.log"
    run:
        placeholder = (
            pl.DataFrame({
                "magset": "placeholder",
                "taxon_novelty": ["p", "c", "o", "f", "g", "s"],
                })
        )

        magsets = (
            pl.concat([
                pl.read_csv(os.path.join(input.name_clades, "node_names.tsv"), separator="\t")
                    .with_columns(
                        taxon_novelty = pl.col("clade").str.head(1),
                        magset = pl.when(pl.col("genome_rep").str.contains(r"^MGYG"))
                            .then(pl.lit("MGYG"))
                            .when(pl.col("genome_rep").str.contains(r"^GWH"))
                            .then(pl.lit("GWH"))
                            .otherwise(pl.col("genome_rep").str.extract(r"^([^_]+)"))
                        )
                    .filter(pl.col("magset").is_in(["GCA", "GCF"]).not_())
                    .select("magset", "taxon_novelty"),
                placeholder
                ])
            .group_by("magset", "taxon_novelty")
            .len()
            .pivot(on="taxon_novelty", values="len")
            .fill_null(0)
            .select(
                magset = "magset",
                phyla = "p",
                classes = "c",
                orders = "o",
                families = "f",
                genera = "g",
                species = "s",
                )
            .filter(pl.col("magset") != "placeholder")
            .sort("phyla", "classes", "orders", "families", "genera", "species", descending=True)
        )

        (
            pl.concat([
                magsets,
                magsets
                    .drop("magset")
                    .group_by(1)
                    .agg(pl.all().sum())
                    .drop("literal")
                    .with_columns(magset = pl.lit("all"))
                ], how="diagonal")
            .write_csv(output[0], separator="\t", include_header=True)
        )
