"""
Globdb processing pipeline

conda activate globdb_processing

Local
snakemake \
    --snakefile scripts/globdb_processing.smk \
    --directory results/globdb_processing/20250402 \
    --software-deployment-method conda --conda-prefix /mnt/weka/pkg/cmr/aroneys/envs_aqua \
    --keep-going --rerun-triggers mtime --cores 32

Cluster submission
snakemake \
    --snakefile scripts/globdb_processing.smk \
    --directory results/globdb_processing/20250402 \
    --software-deployment-method conda --conda-prefix /mnt/weka/pkg/cmr/aroneys/envs_aqua \
    --profile aqua --retries 3 \
    --keep-going --rerun-triggers mtime --cores 64 --local-cores 32
"""

import os
import polars as pl

GLOBDB_FASTA_FOLDER = "/work/microbiome/ibis/public_genomes/globdb_r226/data/globdb_r226_genome_fasta_splits/"
GLOBDB_INFO = "/work/microbiome/ibis/public_genomes/globdb_r226/data/globdb_r226_tax_plus_stats.tsv"
DOMAINS = ["archaea", "bacteria"]
METAPACKAGE = "/work/microbiome/db/singlem/S4.3.0.GTDB_r220.metapackage_20240523.smpkg.zb"
GTDB_ARC_TAX = "/work/microbiome/ibis/public_genomes/data/taxonomy_files/ar53_taxonomy_r226.tsv"
GTDB_BAC_TAX = "/work/microbiome/ibis/public_genomes/data/taxonomy_files/bac120_taxonomy_r226.tsv"
GTDB_ARC_RED = [0.22112310781366834, 0.3878238738848844, 0.5300294639396943, 0.7336130438906969, 0.912225515599894]
GTDB_BAC_RED = [0.3226932930969102, 0.4596791881303727, 0.6142743482438423, 0.7621761461217935, 0.9212198070258302]

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
    .with_columns(
        fasta = pl.concat_str(
            pl.lit(GLOBDB_FASTA_FOLDER),
            pl.col("f").str.head(5), pl.lit("/"),
            pl.col("f").str.slice(5, 4), pl.lit("/"),
            pl.col("f").str.slice(9, 4), pl.lit("/"),
            pl.col("f")
            ),
        )
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
    conda:
        "gtdbtk.yaml"
    shell:
        "gtdbtk identify "
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
    conda:
        "gtdbtk.yaml"
    shell:
        "gtdbtk align "
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
    conda:
        "gtdbtk.yaml"
    shell:
        "gtdbtk infer "
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
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    params:
        outgroup = lambda wildcards: "p__Altiarchaeota" if wildcards.domain == "archaea" else "p__Fusobacteriota",
    benchmark:
        "{domain}/gtdbtk_root/benchmark.txt"
    log:
        "{domain}/logs/gtdbtk_root.log"
    conda:
        "gtdbtk.yaml"
    shell:
        "gtdbtk root "
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
        mem_mb=get_mem_mb,
        runtime = lambda wildcards, attempt: 48*60*attempt,
    benchmark:
        "{domain}/gtdbtk_decorate/benchmark.txt"
    log:
        "{domain}/logs/gtdbtk_decorate.log"
    conda:
        "gtdbtk.yaml"
    shell:
        "gtdbtk decorate "
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
    conda:
        "phylorank.yml"
    shell:
        "phylorank outliers "
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
    benchmark:
        "{domain}/phylorank_tree_benchmark.txt"
    log:
        "{domain}/logs/tree2tbl.log"
    conda:
        "r-treedataverse.yml"
    script:
        "tree2tbl.R"

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
    conda:
        "name_clades.yml"
    shell:
        "python {params.script} "
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
                    .group_by(1)
                    .agg(pl.all().sum())
                    .drop("literal")
                    .with_columns(magset = pl.lit("all"))
                ])
            .write_csv(output[0], separator="\t", include_header=True)
        )
