##################
### tree2tbl.R ###
##################
library(ggtree)
library(ggtreeExtra)
library(tidytree)
library(treeio)
library(tidyverse)

# Snakemake inputs
input_tree <- str_c(snakemake@input[["dir"]], "/", snakemake@params[["input_tree"]])
taxonomy <- snakemake@input[["tax"]]
output_tree <- snakemake@output[["tree"]]
red_cutoffs <- snakemake@params[["red_cutoffs"]]

################################
### Extract info from labels ###
################################
COMBINED_LABEL_REGEX <- ":"
BOOTSTRAP_LABEL_REGEX <- "^[:digit:]{1}\\.[:digit:]{1,3}$"
BOOTSTRAP_EXTRACT_REGEX <- "(?<=')[:digit:]{1}\\.[:digit:]{1,3}(?=:)"
TAXA_REGEX <- "__"
TAXA_EXTRACT_REGEX <- "(?<=:).*(?=')"
GENOME_EXTRACT_REGEX <- "[^']+"
RED_COMBINED_LABEL_REGEX <- "\\|[^']+"
RED_EXTRACT_REGEX <- "(?<=\\|RED=)[:digit:]{1}\\.[:digit:]{1,3}"
remove_red <- function(string) {
  if (str_detect(string, RED_COMBINED_LABEL_REGEX)) {
    string <- str_remove(string, RED_COMBINED_LABEL_REGEX)
  }

  if (!str_detect(string, COMBINED_LABEL_REGEX)) {
    string <- str_remove_all(string, "'")
  }

  return(string)
}

extract_bootstrap <- function(string,
                              combined_label_regex = COMBINED_LABEL_REGEX,
                              bootstrap_label_regex = BOOTSTRAP_LABEL_REGEX,
                              bootstrap_extract_regex = BOOTSTRAP_EXTRACT_REGEX) {
  bootstrap <- NA

  string <- remove_red(string)

  if (str_detect(string, combined_label_regex)) {
    bootstrap <- str_extract(string, bootstrap_extract_regex)
  } else if (str_detect(string, bootstrap_label_regex)) {
    bootstrap <- string
  }

  return(as.numeric(bootstrap))
}

extract_taxa <- function(string,
                         combined_label_regex = COMBINED_LABEL_REGEX,
                         taxa_regex = TAXA_REGEX,
                         taxa_extract_regex = TAXA_EXTRACT_REGEX) {
  taxa <- NA_character_

  string <- remove_red(string)

  if (str_detect(string, taxa_regex)) {
    taxa <- str_extract(string, ".*")
  }

  if (str_detect(string, combined_label_regex)) {
    taxa <- str_extract(string, taxa_extract_regex)
  }

  return(taxa)
}

extract_genome <- function(string,
                        combined_label_regex = COMBINED_LABEL_REGEX,
                        bootstrap_label_regex = BOOTSTRAP_LABEL_REGEX,
                        taxa_regex = TAXA_REGEX,
                        genome_extract_regex = GENOME_EXTRACT_REGEX) {
  label <- NA_character_

  string <- remove_red(string)

  if (!str_detect(string, combined_label_regex) && !str_detect(string, bootstrap_label_regex) && !str_detect(string, taxa_regex)) {
    label <- str_extract(string, genome_extract_regex)
  }

  return(label)
}

extract_red <- function(string,
                        red_combined_label_regex = RED_COMBINED_LABEL_REGEX,
                        red_extract_regex = RED_EXTRACT_REGEX) {
  red <- NA_real_

  if (str_detect(string, red_combined_label_regex)) {
    red <- str_extract(string, red_extract_regex)
  }

  return(as.numeric(red))
}

# tribble(
#     ~label,
#     "spire_mag_01842612",
#     "3300027742_44",
#     "binchicken_co19_513",
#     "GB_GCA_018819225.1",
#     "SRR3989344rbin.455",
#     "d__Archaea",
#     "'1.0:p__Altiarchaeota; c__Altiarchaeia'",
#     "'0.373:g__CAIYYO01'",
#     "1.0",
#     "0.996",
#     "'1.0:s__UBA285 sp002495025'",
#     "'spire_mag_01041117|RED=1.000'",
#     "'1.0:g__Xenobium|RED=0.931'",
#     "'0.953|RED=0.332'",
#     "'0.999:c__UBA5301|RED=0.440'",
#     "'1.0|RED=0.927'",
#   ) %>%
#   mutate(
#     bootstrap = map_dbl(label, extract_bootstrap),
#     taxa = map_chr(label, extract_taxa),
#     genome = map_chr(label, extract_genome),
#     RED = map_dbl(label, extract_red),
#   )

########################
### Novelty from RED ###
########################
ROUND_DIGITS <- 2
produce_ranks <- function(median_reds) {
  ranks <- bind_rows(
      tibble(
        rank = c(""),
        median_red = c(0)
        ),
      tibble(
        rank = c("Phylum", "Class", "Order", "Family", "Genus"),
        median_red = median_reds
        ),
      tibble(
        rank = c("Species/Strain"),
        median_red = c(1)
        )
      ) %>%
    mutate(
      cutoff_red = case_when(
        rank == "" ~ 0,
        rank == "Species/Strain" ~ 1,
        TRUE ~ rowMeans(cbind(median_red, lead(median_red)))
        ),
      label = ifelse(rank == "", NA, str_c(rank, " (", round(lag(cutoff_red), ROUND_DIGITS), "-", round(cutoff_red, ROUND_DIGITS), "]")),
      )

  return(ranks)
}

rank_red <- tibble(rank = c("", "Phylum", "Class", "Order", "Family", "Genus", "Species/Strain")) %>%
  left_join(produce_ranks(red_cutoffs))

# tribble(
#     ~red, ~exp,
#     0.22, "phyla",
#     0.39, "class",
#     0.52, "class",
#     0.53, "order",
#     0.69, "family",
#     0.85, "genus",
#     0.90, "genus",
#     0.95, "genus",
#     0.98, "species",
#   ) %>%
#   mutate(
#     obs = cut(red, breaks = rank_red$cutoff_red, labels = rank_red$label[-1])
#     )

###########################
### Load tree functions ###
###########################
load_tree <- function(path, gtdb_genomes) {
  red_cutoffs <- rank_red$cutoff_red
  red_labels <- rank_red$label[-1]

  input_tree <- read.tree(path) %>%
    as_tibble() %>%
    mutate(RED = map_dbl(label, extract_red)) %>%
    mutate(
      bootstrap = map_dbl(label, extract_bootstrap),
      taxa = map_chr(label, extract_taxa),
      genome = map_chr(label, extract_genome),
      ) %>%
    left_join(gtdb_genomes, by = "genome") %>%
    replace_na(list(magset = "other"))

  grouping_nongtdb <- list(
    nongtdb = input_tree %>% filter(!is.na(genome), magset == "other") %>% pull(node),
    gtdb = input_tree %>% filter(!is.na(genome), magset == "GTDB") %>% pull(node)
    )

  novelty_tree <- input_tree %>%
    as.treedata() %>%
    groupOTU(grouping_nongtdb, "nongtdb_group") %>%
    as_tibble() %>%
    mutate(
      novelty_red = cut(RED, breaks = red_cutoffs, labels = red_labels)
      )

  processed_tree <- novelty_tree %>%
    as.treedata()

  return(processed_tree)
}

####################
### Process tree ###
####################
gtdb_genomes <- read_tsv(taxonomy, col_names = c("genome", "taxa")) %>%
  select(genome) %>%
  mutate(magset = "GTDB")

load_tree(input_tree, gtdb_genomes) %>% as_tibble() %>% write_tsv(output_tree)
