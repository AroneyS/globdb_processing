#!/usr/bin/env python3

"""
Author: Samuel Aroney
Name clades in tree
"""

import os
import sys
import argparse
import logging
import polars as pl
from tqdm import tqdm

# From GTDB r220
BAC_RED_CUTOFFS = [
        "0.3280941769231098",
        "0.449727838796469",
        "0.6083500718998613",
        "0.7576141066814935",
        "0.9220350796053899",
]
ARC_RED_CUTOFFS = [
    "0.2128708845277663",
    "0.35878546884559126",
    "0.5316295929627715",
    "0.7250725361353227",
    "0.9069458981600348",
]
MEDIAN_RED_TAXONS = ["phylum", "class", "order", "family", "genus"]

GENOMES_OUTPUT_COLUMNS = {
    "genome": str,
    "taxonomy": str,
}

NODES_OUTPUT_COLUMNS = {
    "node": int,
    "clade": str,
    "genome_rep": str,
}

GTDB_TAXONOMY_COLUMNS = {
    "genome": str,
    "taxonomy": str,
}

def name_clades(tree_df, genome_metadata, domain, red_cutoffs=None):
    """
    Traverse tree bottom up, naming nodes when novelty changes based on the earliest genome that is in the clade
    Record genome taxonomy as we traverse (naming taxa levels with skips)
    Record representative genome for each taxon as we traverse
    Start with list of non-GTDB genomes in order of quality (completeness - 5 * contamination)
    Stop once node reached with GTDB children - note node (fill out remaining taxonomy based on GTDB genomes in `fill_taxonomy`)
    Produce complete taxonomy string for all new genomes
    Produce node number - clade name - representative genome
    """

    if domain == "d__Bacteria":
        if not red_cutoffs: red_cutoffs = BAC_RED_CUTOFFS
    else:
        if not red_cutoffs: red_cutoffs = ARC_RED_CUTOFFS

    median_reds = {t:float(r) for t, r in zip(MEDIAN_RED_TAXONS, red_cutoffs)}
    median_reds["species"] = 1

    starting_counters = {
        "phylum": 1,
        "class": 1,
        "order": 1,
        "family": 1,
        "genus": 1,
        "species": 1,
    }

    genomes = (
        tree_df
        .filter(pl.col("magset").is_not_null())
        .filter(pl.col("magset") != "GTDB")
        .select("genome")
        .join(
            genome_metadata
            .select(
                genome = pl.col("Name"),
                completeness = pl.col("Completeness"),
                contamination = pl.col("Contamination"),
                ),
            on="genome"
            )
        .with_columns(quality = pl.col("completeness") - 5 * pl.col("contamination"))
        .sort(pl.col("quality"), pl.col("genome"), descending=[True, True])
        .get_column("genome")
        .to_list()
    )
    genomes = pl.DataFrame(
        enumerate(genomes),
        schema={"genome_order": int, "genome": str},
        orient="row"
        )

    parent_dict = {row[1]: row[0] for row in tree_df.iter_rows()}
    def find_parents(node, parent_dict = parent_dict):
        logging.debug(f"Finding parent for {node}")
        if node not in parent_dict:
            return []
        parent = parent_dict[node]
        if node == parent:
            return []
        return [parent] + find_parents(parent)

    tree_genomes = (
        tree_df
        .join(genomes, on="genome")
        .with_columns(
            parents = pl.col("node").map_elements(lambda x: find_parents(x), return_dtype=pl.List(pl.Int64))
            )
        .sort("genome_order")
    )

    def get_novelty_red(df):
        return (
            df
            .get_column("novelty_red")
            .to_list()[0]
            .split(" ")[0]
            .split("/")[0]
            .lower()
        )

    def get_node_red(node, df=tree_df):
        parent = (
            df
            .filter(pl.col("node") == node)
            .get_column("parent")
            .to_list()[0]
        )

        if parent == node:
            return 0
        else:
            return float(
                df
                .filter(pl.col("node") == parent)
                .get_column("RED")
                .to_list()[0]
            )

    def get_child_red(node, df=tree_df):
        return float(
            df
            .filter(pl.col("node") == node)
            .get_column("RED")
            .to_list()[0]
        )

    def child_red_closer(child_red, parent_red, median_red):
        return abs(child_red - median_red) < abs(parent_red - median_red)

    def get_info(node, df=tree_df):
        df_self = df.filter(pl.col("node") == node)
        parent = (
            df_self
            .get_column("parent")
            .to_list()[0]
        )
        df_parent = df.filter(pl.col("node") == parent)

        try:
            df_gtdb = df_self.filter(pl.col("nongtdb_group") != "gtdb")
            if df_gtdb.height > 0:
                novelty_red_self = df_self.pipe(get_novelty_red)
                novelty_red = df_parent.pipe(get_novelty_red)
            else:
                novelty_red_self = df_self.pipe(get_novelty_red)
                novelty_red = ""
        except (AttributeError, IndexError):
            novelty_red_self = ""
            novelty_red = ""

        return (novelty_red, novelty_red_self)

    def get_vying_children(parent, novelty_level, child, df=tree_df):
        if child:
            current_child = child[0]
        else:
            current_child = -1

        immediate_children = (
            df
            .filter(pl.col("genome").is_null())
            .filter(pl.col("parent") == parent)
            .filter(pl.col("node") != parent)
            .filter(pl.col("node") != current_child)
            .get_column("node")
            .to_list()
        )

        return [c for c in immediate_children if get_info(c)[0] == novelty_level]

    genomes_output = []
    nodes_output = []
    node_taxonomy = {}
    genome_index = tree_genomes.columns.index("genome")
    parents_index = tree_genomes.columns.index("parents")
    for row in tqdm(tree_genomes.iter_rows(), total=tree_genomes.height, desc="Naming clades", unit="genome"):
        genome = row[genome_index]
        parents = row[parents_index]
        logging.debug(f"Determining taxonomy for {genome}")
        genome_taxonomy = {t: "" for t, _ in starting_counters.items()}
        taxa = list(genome_taxonomy.keys())
        parent_novelty = {p: get_info(p) for p in parents}
        child = False

        for parent in parents:
            novelty_bounds = parent_novelty[parent]
            vying_children = []
            skip_parent = False
            if novelty_bounds[0] == novelty_bounds[1]:
                # Same taxon as child
                novelty_reds = [novelty_bounds[0]]

                # Check if any children are closer to median red than parent
                vying_children = get_vying_children(parent, novelty_reds[0], child)
                children_red = [get_node_red(c) for c in vying_children]
                parent_red = get_node_red(parent)
                median_red = median_reds[novelty_reds[0]]
                if any([child_red_closer(c, parent_red, median_red) for c in children_red]):
                    skip_parent = True
            elif novelty_bounds[0]:
                # Each node has taxon naming range between parent red (inc) and self red (exc)
                novelty_reds = taxa[taxa.index(novelty_bounds[0]):taxa.index(novelty_bounds[1])]
            else:
                # No novelty_bounds[0] due to being GTDB node
                novelty_reds = [""]

            if child:
                name_child = False
                if child[1][0] == novelty_reds[0] or (novelty_reds[0] == "" and child[1][0] == novelty_bounds[1]):
                    # If child novelty upper bound is within parent or GTDB-parent novelty bounds
                    child_red = get_node_red(child[0])
                    parent_red = get_node_red(parent)
                    median_red = median_reds[child[1][0]]
                    if child_red_closer(child_red, parent_red, median_red):
                        name_child = True
                        not_gtdb = novelty_reds[0] != ""
                        skip_parent = not_gtdb
                    else:
                        new_child = (child[0], child[1][1:])
                        child = new_child
                        name_child = len(child[1][1:]) > 0
                else:
                    name_child = True

                if name_child:
                    taxon_taxonomy = {}
                    for novelty_red in child[1]:
                        if not genome_taxonomy[novelty_red] and not novelty_red == "species":
                            clade_name = novelty_red[0] + "__" + genome
                            genome_taxonomy[novelty_red] = (child[0], clade_name)
                            nodes_output.append((child[0], clade_name, genome))
                            taxon_taxonomy[novelty_red] = clade_name
                    if taxon_taxonomy:
                        node_taxonomy[child[0]] = taxon_taxonomy

                child = False

            if skip_parent:
                continue

            try:
                parent_info = node_taxonomy[parent]
                for taxon, value in parent_info.items():
                    genome_taxonomy[taxon] = (parent, value)
                break
            except KeyError:
                if not novelty_reds[0]:
                    # If GTDB-containing node
                    novelty_index = taxa.index(novelty_bounds[1])
                    child_red = get_child_red(parent)
                    parent_red = get_node_red(parent)
                    median_red = median_reds[novelty_bounds[1]]
                    if not child_red_closer(child_red, parent_red, median_red):
                        novelty_index += 1

                    if novelty_index == 0:
                        # Reached GTDB taxa at phyla-novelty
                        break
                    gtdb_novelty = taxa[novelty_index-1]
                    genome_taxonomy[gtdb_novelty] = ("GTDB", parent)
                    break

                child = (parent, novelty_reds)

        if all([t == "" for _, t in genome_taxonomy.items()]):
            taxon_fill = {t: True for t, _ in starting_counters.items()}
        else:
            taxon_fill = {t: False for t, _ in starting_counters.items()}

            # Use previous parent taxonomy if a previously named node is reached
            highest_index, highest_tax = [(i, t) for i, (_, t) in enumerate(genome_taxonomy.items()) if t][0]
            if highest_tax and highest_index != 0 and highest_tax[0] != "GTDB":
                trailblazer_genome = [g for _,t,g in nodes_output if t == highest_tax[1]][0]
                trailblazer_taxonomy = [t for g,t in genomes_output if g == trailblazer_genome][0]
                ancestral_taxonomy = trailblazer_taxonomy.split(";" + highest_tax[1])[0]

                for i, taxon in enumerate(reversed(ancestral_taxonomy.split(";"))):
                    if taxon.startswith("d__"): continue
                    if taxon[0].isdigit():
                        node_info = "GTDB"
                        tax_info = int(taxon)
                    else:
                        node_info = None
                        tax_info = taxon

                    genome_taxonomy[[g for g in genome_taxonomy.keys()][highest_index-1-i]] = (node_info, tax_info)

        # Fill out remaining taxonomy
        taxon_levels = {t: i for i, t in enumerate(median_reds.keys())}
        def is_mid_taxa(prenamed, target_taxon):
            prenamed_levels = [taxon_levels[t] for t in prenamed]
            target_level = taxon_levels[target_taxon]
            return any([target_level < l for l in prenamed_levels])

        def not_gtdb(taxon_level):
            return taxon_level[0] != ""

        def within_taxon_bounds(taxon_level, target_taxon):
            target_level = taxon_levels[target_taxon]
            return taxon_levels[taxon_level[0]] <= target_level and taxon_levels[taxon_level[1]] >= target_level

        def is_lower_node(node, taxon):
            target_level = taxon_levels[taxon]
            try:
                node_levels = [taxon_levels[t] for t in node_taxonomy[node].keys()]
                return all([target_level < l for l in node_levels])
            except KeyError:
                return True

        species_name = ""
        prenamed_taxons = [t for t, v in genome_taxonomy.items() if v]
        for taxon, clade in genome_taxonomy.items():
            clade_name = ""
            if clade:
                if clade[0] == "GTDB":
                    taxon_fill = {t: True for t in taxon_fill}
                    continue
                taxon_fill = {t: t[0]+"__" not in clade[1] for t in taxon_fill}
                clade_name = clade[1]

            if not clade and taxon_fill[taxon]:
                if taxon == "species":
                    if species_name:
                        clade_name = species_name
                    else:
                        continue
                else:
                    clade_name = taxon[0] + "__" + genome

                # Ensure that filled clade names from in-between nodes are assigned to the next lower node
                # This should only happen when a GTDB node is closer to the median RED of a taxon than the next lower node
                # We can assign the clade to the highest parent with the current taxon within their novelty bounds
                # Though we also get here if we have passed the last named taxon - in this case we should not assign a node
                if taxon == "species" or not is_mid_taxa(prenamed_taxons, taxon):
                    viable_parents = None
                else:
                    viable_parents = [p for p, t in parent_novelty.items() if not_gtdb(t) and within_taxon_bounds(t, taxon) and is_lower_node(p, taxon)]

                if viable_parents:
                    clade_node = viable_parents[-1]
                    try:
                        node_taxonomy[clade_node][taxon] = clade_name
                    except KeyError:
                        node_taxonomy[clade_node] = {taxon: clade_name}
                else:
                    clade_node = None

                genome_taxonomy[taxon] = (clade_node, clade_name)
                nodes_output.append((clade_node, clade_name, genome))

            if taxon == "genus" and clade_name:
                genus_index = clade_name.index("g__")
                genus_name = clade_name[genus_index+1:]
                species_name = "s" + genus_name + " " + genome

        taxonomy = domain
        for taxon, clade in genome_taxonomy.items():
            try:
                taxonomy += ";" + clade[1]
            except TypeError:
                taxonomy = str(clade[1])
            except IndexError:
                continue

        genomes_output.append([genome, taxonomy])

    return (
        pl.DataFrame(genomes_output, schema=GENOMES_OUTPUT_COLUMNS, orient="row"),
        pl.DataFrame(nodes_output, schema=NODES_OUTPUT_COLUMNS, orient="row").sort("clade"),
    )

def fill_taxonomy(tree_df, genome_taxonomy, node_names, gtdb_taxonomy):
    def get_descendants(df, parent_node):
        children = df.filter(pl.col("parent") == parent_node).get_column("node").to_list()
        descendants = []
        for child in children:
            descendants.append(child)
            descendants.extend(get_descendants(df, child))
        return descendants

    def get_taxonomy(df, node, limit):
        descendants = get_descendants(df, node)
        descendant_genomes = (
            df
            .filter(pl.col("node").is_in(descendants))
            .join(gtdb_taxonomy, on="genome")
            .select(pl.col("taxonomy").str.extract(r"(.*);"+limit))
            .group_by("taxonomy")
            .count()
            .sort("count")
        )

        return (
            descendant_genomes
            .get_column("taxonomy")
            .to_list()[0]
        )

    genomes_output = []
    node_taxonomy = {}
    new_taxonomy = []
    for row in genome_taxonomy.iter_rows():
        genome = row[0]
        taxonomy = row[1]
        if not taxonomy.startswith("d__"):
            node = int(taxonomy.split(";")[0])
            named_taxonomy = ";".join(taxonomy.split(";")[1:])

            try:
                node_limit = named_taxonomy[0:3]
            except IndexError:
                node_limit = "g__"

            try:
                updated_taxonomy = node_taxonomy[node]
            except KeyError:
                updated_taxonomy = get_taxonomy(tree_df, node, node_limit)
                node_taxonomy[node] = updated_taxonomy

            if not named_taxonomy:
                genus_index = updated_taxonomy.index("g__")
                genus_name = updated_taxonomy[genus_index+1:]
                named_taxonomy = "s" + genus_name + " " + genome
                new_taxonomy.append((None, named_taxonomy, genome))

            taxonomy = updated_taxonomy + ";" + named_taxonomy

        genomes_output.append([genome, taxonomy])

    return(
        pl.DataFrame(genomes_output, schema=GENOMES_OUTPUT_COLUMNS, orient="row"),
        pl.concat([node_names, pl.DataFrame(new_taxonomy, schema=NODES_OUTPUT_COLUMNS, orient="row")]).sort("clade"),
    )

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--debug', help='output debug information', action="store_true")
    parser.add_argument('--quiet', help='only output errors', action="store_true")

    parser.add_argument('--tree-df', help='Tree as df with annotations', required=True)
    parser.add_argument('--metadata', help='Genome metadata', required=True)
    parser.add_argument('--gtdb', help='GTDB taxonomy file', required=True)
    default_domain = "d__Bacteria"
    parser.add_argument('--domain', help=f'Genome domain [default: {default_domain}]', default=default_domain)
    default_cutoffs = BAC_RED_CUTOFFS
    parser.add_argument('--red-cutoffs', nargs='+', help=f'RED cutoffs for phyla to genus, space separated in order. [default: {" ".join(default_cutoffs)}]', default=default_cutoffs)
    parser.add_argument('--output', help='Output folder', required=True)

    args = parser.parse_args(arguments)

    # Setup logging
    if args.debug:
        loglevel = logging.DEBUG
    elif args.quiet:
        loglevel = logging.ERROR
    else:
        loglevel = logging.INFO
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')

    logging.info("Create output folder")
    os.makedirs(args.output, exist_ok=True)

    logging.info("Loading inputs")
    tree_df = pl.read_csv(args.tree_df, separator="\t", null_values=["NA"], dtypes={"RED": float})
    genome_metadata = (
        pl.read_csv(args.metadata, separator="\t")
        .select(
            Name = pl.col("ID"),
            Completeness = pl.col("checkm2_completeness"),
            Contamination = pl.col("checkm2_contamination")
            )
    )
    gtdb_taxonomy = pl.read_csv(args.gtdb, separator="\t", has_header=False, new_columns=GTDB_TAXONOMY_COLUMNS.keys(), dtypes=GTDB_TAXONOMY_COLUMNS)

    logging.info("Naming nodes in tree")
    (genome_taxonomy, node_names) = name_clades(tree_df, genome_metadata, args.domain, args.red_cutoffs)

    logging.info("Filling taxonomy with GTDB values")
    (genome_taxonomy, node_names) = fill_taxonomy(tree_df, genome_taxonomy, node_names, gtdb_taxonomy)

    genome_taxonomy.write_csv(os.path.join(args.output, "genome_taxonomy.tsv"), separator="\t")
    node_names.write_csv(os.path.join(args.output, "node_names.tsv"), separator="\t")

    logging.info("Done")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
