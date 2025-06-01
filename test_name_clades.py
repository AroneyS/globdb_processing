#!/usr/bin/env python3

import unittest
import polars as pl
from polars.testing import assert_frame_equal
from name_clades import name_clades, fill_taxonomy

TREE_INPUT_COLUMNS = {
    "parent": int,
    "node": int,
    # "branch.length",
    # "label",
    "nongtdb_group": str,
    # "bootstrap",
    # "taxa",
    "genome": str,
    "magset": str,
    "RED": float,
    "novelty_red": str,
}

GENOME_METADATA_COLUMNS = {
    "Name": str,
    "Completeness": float,
    "Contamination": float,
}

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

NOVELTY_RED_LEVELS = [
    "Species/Strain (0.95-1]",
    "Genus (0.82-0.95]",
    "Family (0.62-0.82]",
    "Order (0.43-0.62]",
    "Class (0.28-0.43]",
    "Phylum (0-0.28]",
]

class Tests(unittest.TestCase):
    def assertDataFrameEqual(self, a, b):
        assert_frame_equal(a, b, check_dtypes=False, check_row_order=False)

    def test_name_clades(self):
        # Label indicates novelty of that branch (i.e. novelty level of parent)

        # root           |(0)
        #                |
        # phylum         |(100000)
        #                |
        # order          |(1000)
        #           ___________
        #           |         |
        # genus     |(10)     |
        #        _______      |
        #        |     |      |
        #        |(1)  |(2)   |(3)
        #        sp1   bc1    bc2

        tree_df = pl.DataFrame([
            [10, 1, "nongtdb", "SPIREOTU_01842612", "other", None, None],
            [10, 2, "nongtdb", "BCRBG_01105", "other", None, None],
            [1000, 3, "nongtdb", "BCRBG_48201", "other", None, None],
            [1000, 10, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [100000, 1000, "nongtdb", None, None, 0.92, NOVELTY_RED_LEVELS[1]],
            [0, 100000, "nongtdb", None, None, 0.6, NOVELTY_RED_LEVELS[3]],
            [0, 0, "nongtdb", None, None, 0.3, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 90.0, 0.0],
            ["BCRBG_01105", 90.0, 1.0],
            ["BCRBG_48201", 90.0, 5.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 SPIREOTU_01842612"],
            ["BCRBG_01105", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 BCRBG_01105"],
            ["BCRBG_48201", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__BCRBG_48201;s__BCRBG_48201 BCRBG_48201"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [100000, "p__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "c__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [1000, "o__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [1000, "f__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [10, "g__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 BCRBG_01105", "BCRBG_01105"],
            [None, "g__BCRBG_48201", "BCRBG_48201"],
            [None, "s__BCRBG_48201 BCRBG_48201", "BCRBG_48201"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_all_magsets(self):
        # root                         |(0)
        #                              |
        # phlyum                       |(100000)
        #                  _________________________
        #                  |                       |
        # class            |(10000)                |(20000)
        #            _____________            ___________
        #            |           |            |         |
        # genus      |(10)       |(20)        |(30)     |
        #         _______     _______      _______      |
        #         |     |     |     |      |     |      |
        #         |(1)  |(2)  |(3)  |(4)   |(5)  |(6)   |(7)
        #         sp1   uh1   sm1   oc1    bc1   oc2    uh2

        tree_df = pl.DataFrame([
            [10, 1, "nongtdb", "SPIREOTU_01842612", "SPIRE", None, None],
            [10, 2, "nongtdb", "MGYG000003541", "UHGG", None, None],
            [20, 3, "nongtdb", "SRR11742948bin.25", "SMAG", None, None],
            [20, 4, "nongtdb", "TARA_SAMEA1", "OceanDNA", None, None],
            [30, 5, "nongtdb", "binchicken_co19_513", "binchicken", None, None],
            [30, 6, "nongtdb", "TARA_SAMEA2", "OceanDNA", None, None],
            [20000, 7, "nongtdb", "MGYG000003542", "UHGG", None, None],
            [10000, 10, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [10000, 20, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [20000, 30, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [100000, 10000, "nongtdb", None, None, 0.9, NOVELTY_RED_LEVELS[1]],
            [100000, 20000, "nongtdb", None, None, 0.9, NOVELTY_RED_LEVELS[1]],
            [0, 100000, "nongtdb", None, None, 0.35, NOVELTY_RED_LEVELS[4]],
            [0, 0, "nongtdb", None, None, 0.21, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 95.0, 0.0],
            ["MGYG000003541", 90.0, 0.0],
            ["SRR11742948bin.25", 90.0, 0.0],
            ["TARA_SAMEA1", 95.0, 0.0],
            ["binchicken_co19_513", 95.0, 0.0],
            ["TARA_SAMEA2", 90.0, 1.0],
            ["MGYG000003542", 90.0, 1.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "d__Archaea;p__binchicken_co19_513;c__TARA_SAMEA1;o__TARA_SAMEA1;f__TARA_SAMEA1;g__SPIREOTU_01842612;s__SPIREOTU_01842612 SPIREOTU_01842612"],
            ["MGYG000003541", "d__Archaea;p__binchicken_co19_513;c__TARA_SAMEA1;o__TARA_SAMEA1;f__TARA_SAMEA1;g__SPIREOTU_01842612;s__SPIREOTU_01842612 MGYG000003541"],
            ["SRR11742948bin.25", "d__Archaea;p__binchicken_co19_513;c__TARA_SAMEA1;o__TARA_SAMEA1;f__TARA_SAMEA1;g__TARA_SAMEA1;s__TARA_SAMEA1 SRR11742948bin.25"],
            ["TARA_SAMEA1", "d__Archaea;p__binchicken_co19_513;c__TARA_SAMEA1;o__TARA_SAMEA1;f__TARA_SAMEA1;g__TARA_SAMEA1;s__TARA_SAMEA1 TARA_SAMEA1"],
            ["binchicken_co19_513", "d__Archaea;p__binchicken_co19_513;c__binchicken_co19_513;o__binchicken_co19_513;f__binchicken_co19_513;g__binchicken_co19_513;s__binchicken_co19_513 binchicken_co19_513"],
            ["TARA_SAMEA2", "d__Archaea;p__binchicken_co19_513;c__binchicken_co19_513;o__binchicken_co19_513;f__binchicken_co19_513;g__binchicken_co19_513;s__binchicken_co19_513 TARA_SAMEA2"],
            ["MGYG000003542", "d__Archaea;p__binchicken_co19_513;c__binchicken_co19_513;o__binchicken_co19_513;f__binchicken_co19_513;g__MGYG000003542;s__MGYG000003542 MGYG000003542"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [100000, "p__binchicken_co19_513", "binchicken_co19_513"],
            [20000, "c__binchicken_co19_513", "binchicken_co19_513"],
            [20000, "o__binchicken_co19_513", "binchicken_co19_513"],
            [20000, "f__binchicken_co19_513", "binchicken_co19_513"],
            [10000, "c__TARA_SAMEA1", "TARA_SAMEA1"],
            [10000, "o__TARA_SAMEA1", "TARA_SAMEA1"],
            [10000, "f__TARA_SAMEA1", "TARA_SAMEA1"],
            [30, "g__binchicken_co19_513", "binchicken_co19_513"],
            [20, "g__TARA_SAMEA1", "TARA_SAMEA1"],
            [10, "g__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "g__MGYG000003542", "MGYG000003542"],
            [None, "s__binchicken_co19_513 TARA_SAMEA2", "TARA_SAMEA2"],
            [None, "s__binchicken_co19_513 binchicken_co19_513", "binchicken_co19_513"],
            [None, "s__TARA_SAMEA1 TARA_SAMEA1", "TARA_SAMEA1"],
            [None, "s__TARA_SAMEA1 SRR11742948bin.25", "SRR11742948bin.25"],
            [None, "s__SPIREOTU_01842612 MGYG000003541", "MGYG000003541"],
            [None, "s__SPIREOTU_01842612 SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__MGYG000003542 MGYG000003542", "MGYG000003542"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_gtdb_full(self):
        # root                 |(0)
        #                      |
        # phylum               |(100000)
        #                      |
        # class                |(10000)
        #             ___________________
        #             |                 |
        # order       |                 |(1000)
        #             |           _____________
        #             |           |           |
        # genus       |(10)       |(20)       |(30)
        #          _______     _______     _______
        #          |     |     |     |     |     |
        #          |(1)  |(2)  |(3)  |(4)  |(5)  |(6)
        #          gt1   sp1   gt2   bc1   sp2   bc2

        tree_df = pl.DataFrame([
            [10, 1, "gtdb", "GB_GCA_016935655.1", None, None, None],
            [10, 2, "nongtdb", "SPIREOTU_01842612", "SPIRE", None, None],
            [20, 3, "gtdb", "GB_GCA_016935655.2", None, None, None],
            [20, 4, "nongtdb", "binchicken_co19_1", "binchicken", None, None],
            [30, 5, "nongtdb", "spire_mag_2", "SPIRE", None, None],
            [30, 6, "nongtdb", "binchicken_co19_2", "binchicken", None, None],
            [10000, 10, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [1000, 20, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [1000, 30, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [10000, 1000, "gtdb", None, None, 0.92, NOVELTY_RED_LEVELS[1]],
            [100000, 10000, "gtdb", None, None, 0.61, NOVELTY_RED_LEVELS[3]],
            [0, 100000, "gtdb", None, None, 0.44, NOVELTY_RED_LEVELS[4]],
            [0, 0, "gtdb", None, None, 0.31, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 95.0, 0.0],
            ["spire_mag_2", 90.0, 0.0],
            ["binchicken_co19_1", 90.0, 0.0],
            ["binchicken_co19_2", 95.0, 0.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "10"],
            ["spire_mag_2", "1000;g__binchicken_co19_2;s__binchicken_co19_2 spire_mag_2"],
            ["binchicken_co19_1", "20"],
            ["binchicken_co19_2", "1000;g__binchicken_co19_2;s__binchicken_co19_2 binchicken_co19_2"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [30, "g__binchicken_co19_2", "binchicken_co19_2"],
            [None, "s__binchicken_co19_2 spire_mag_2", "spire_mag_2"],
            [None, "s__binchicken_co19_2 binchicken_co19_2", "binchicken_co19_2"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_gtdb_truncate(self):
        #                  |
        # class            |(10000)
        #             ___________
        #             |         |
        # genus       |(10)     |
        #          _______      |
        #          |     |      |
        #          |(1)  |(2)   |(3)
        #          gt1   gt2    sp1

        tree_df = pl.DataFrame([
            [10, 1, "gtdb", "GB_GCA_016935655.1", None, None, None],
            [10, 2, "gtdb", "GB_GCA_016935655.2", None, None, None],
            [10000, 3, "nongtdb", "SPIREOTU_01842612", "SPIRE", None, None],
            [10000, 10, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [0, 10000, "gtdb", None, None, 0.91, NOVELTY_RED_LEVELS[1]],
            [0, 0, "gtdb", None, None, 0.35, NOVELTY_RED_LEVELS[4]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 90.0, 0.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "10000;g__SPIREOTU_01842612;s__SPIREOTU_01842612 SPIREOTU_01842612"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [None, "g__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 SPIREOTU_01842612", "SPIREOTU_01842612"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_gtdb_truncate_early(self):
        #                  |
        # genus            |(20)
        #             ___________
        #             |         |
        # genus       |(10)     |
        #          _______      |
        #          |     |      |
        #          |(1)  |(2)   |(3)
        #          gt1   gt2    sp1

        tree_df = pl.DataFrame([
            [10, 1, "gtdb", "GB_GCA_016935655.1", None, None, None],
            [10, 2, "gtdb", "GB_GCA_016935655.2", None, None, None],
            [20, 3, "nongtdb", "SPIREOTU_01842612", "SPIRE", None, None],
            [20, 10, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [0, 20, "gtdb", None, None, 0.94, NOVELTY_RED_LEVELS[1]],
            [0, 0, "gtdb", None, None, 0.91, NOVELTY_RED_LEVELS[1]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 90.0, 0.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "20"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_gtdb_novel_phyla(self):
        #                  |
        #                  |(10000)
        #             ___________
        #             |         |
        # phyla       |(10)     |
        #          _______      |
        #          |     |      |
        #          |(1)  |(2)   |(3)
        #          gt1   gt2    sp1

        tree_df = pl.DataFrame([
            [10, 1, "gtdb", "GB_GCA_016935655.1", None, None, None],
            [10, 2, "gtdb", "GB_GCA_016935655.2", None, None, None],
            [10000, 3, "nongtdb", "BCRBG_01105", "binchicken", None, None],
            [10000, 10, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [0, 10000, "gtdb", None, None, 0.21, NOVELTY_RED_LEVELS[5]],
            [0, 0, "gtdb", None, None, 0.15, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["BCRBG_01105", 90.0, 0.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["BCRBG_01105", "d__Archaea;p__BCRBG_01105;c__BCRBG_01105;o__BCRBG_01105;f__BCRBG_01105;g__BCRBG_01105;s__BCRBG_01105 BCRBG_01105"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [None, "p__BCRBG_01105", "BCRBG_01105"],
            [None, "c__BCRBG_01105", "BCRBG_01105"],
            [None, "o__BCRBG_01105", "BCRBG_01105"],
            [None, "f__BCRBG_01105", "BCRBG_01105"],
            [None, "g__BCRBG_01105", "BCRBG_01105"],
            [None, "s__BCRBG_01105 BCRBG_01105", "BCRBG_01105"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_gtdb_almost_novel_phyla(self):
        #                  |
        # phylum           |(20)
        #             ___________
        #             |         |
        # phylum      |(10)     |
        #          _______      |
        #          |     |      |
        #          |(1)  |(2)   |(3)
        #          gt1   gt2    sp1

        tree_df = pl.DataFrame([
            [10, 1, "gtdb", "GB_GCA_016935655.1", None, None, None],
            [10, 2, "gtdb", "GB_GCA_016935655.2", None, None, None],
            [20, 3, "nongtdb", "BCRBG_01105", "binchicken", None, None],
            [20, 10, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [0, 20, "gtdb", None, None, 0.25, NOVELTY_RED_LEVELS[5]],
            [0, 0, "gtdb", None, None, 0.21, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["BCRBG_01105", 90.0, 0.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["BCRBG_01105", "20;c__BCRBG_01105;o__BCRBG_01105;f__BCRBG_01105;g__BCRBG_01105;s__BCRBG_01105 BCRBG_01105"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [None, "c__BCRBG_01105", "BCRBG_01105"],
            [None, "o__BCRBG_01105", "BCRBG_01105"],
            [None, "f__BCRBG_01105", "BCRBG_01105"],
            [None, "g__BCRBG_01105", "BCRBG_01105"],
            [None, "s__BCRBG_01105 BCRBG_01105", "BCRBG_01105"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_gtdb_almost_novel_phyla_further(self):
        #                  |
        # phylum           |(40)
        #             ___________
        #             |         |
        # phylum      |(30)     |
        #             |         |
        # phylum      |(20)     |
        #             |         |
        # phylum      |(10)     |
        #          _______      |
        #          |     |      |
        #          |(1)  |(2)   |(3)
        #          gt1   gt2    sp1

        tree_df = pl.DataFrame([
            [10, 1, "gtdb", "GB_GCA_016935655.1", None, None, None],
            [10, 2, "gtdb", "GB_GCA_016935655.2", None, None, None],
            [40, 3, "nongtdb", "BCRBG_01105", "binchicken", None, None],
            [20, 10, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [30, 20, "gtdb", None, None, 0.25, NOVELTY_RED_LEVELS[5]],
            [40, 30, "gtdb", None, None, 0.25, NOVELTY_RED_LEVELS[5]],
            [0, 40, "gtdb", None, None, 0.25, NOVELTY_RED_LEVELS[5]],
            [0, 0, "gtdb", None, None, 0.21, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["BCRBG_01105", 90.0, 0.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["BCRBG_01105", "40;c__BCRBG_01105;o__BCRBG_01105;f__BCRBG_01105;g__BCRBG_01105;s__BCRBG_01105 BCRBG_01105"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [None, "c__BCRBG_01105", "BCRBG_01105"],
            [None, "o__BCRBG_01105", "BCRBG_01105"],
            [None, "f__BCRBG_01105", "BCRBG_01105"],
            [None, "g__BCRBG_01105", "BCRBG_01105"],
            [None, "s__BCRBG_01105 BCRBG_01105", "BCRBG_01105"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_better_node(self):
        # root           |(0)
        #                |
        # phylum         |(100000)
        #                |
        # genus          |(20)
        #           ___________
        #           |         |
        # genus     |(10)     |
        #        _______      |
        #        |     |      |
        #        |(1)  |(2)   |(3)
        #        sp1   bc1    bc2

        tree_df = pl.DataFrame([
            [10, 1, "nongtdb", "SPIREOTU_01842612", "SPIRE", None, None],
            [10, 2, "nongtdb", "BCRBG_01105", "binchicken", None, None],
            [20, 3, "nongtdb", "BCRBG_48201", "binchicken", None, None],
            [20, 10, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [100000, 20, "nongtdb", None, None, 0.94, NOVELTY_RED_LEVELS[1]],
            [0, 100000, "nongtdb", None, None, 0.92, NOVELTY_RED_LEVELS[1]],
            [0, 0, "nongtdb", None, None, 0.31, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 90.0, 0.0],
            ["BCRBG_01105", 90.0, 0.0],
            ["BCRBG_48201", 90.0, 5.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 SPIREOTU_01842612"],
            ["BCRBG_01105", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 BCRBG_01105"],
            ["BCRBG_48201", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 BCRBG_48201"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [100000, "p__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "c__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "o__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "f__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [20, "g__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 BCRBG_01105", "BCRBG_01105"],
            [None, "s__SPIREOTU_01842612 BCRBG_48201", "BCRBG_48201"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_three_in_a_row(self):
        # root           |(0)
        #                |
        # phylum         |(100000)
        #                |
        # genus          |(30)
        #                |
        # genus          |(20)
        #           ___________
        #           |         |
        # genus     |(10)     |
        #        _______      |
        #        |     |      |
        #        |(1)  |(2)   |(3)
        #        sp1   bc1    bc2

        tree_df = pl.DataFrame([
            [10, 1, "nongtdb", "SPIREOTU_01842612", "other", None, None],
            [10, 2, "nongtdb", "BCRBG_01105", "other", None, None],
            [20, 3, "nongtdb", "BCRBG_48201", "other", None, None],
            [20, 10, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [30, 20, "nongtdb", None, None, 0.95, NOVELTY_RED_LEVELS[1]],
            [100000, 30, "nongtdb", None, None, 0.94, NOVELTY_RED_LEVELS[1]],
            [0, 100000, "nongtdb", None, None, 0.92, NOVELTY_RED_LEVELS[1]],
            [0, 0, "nongtdb", None, None, 0.31, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 90.0, 0.0],
            ["BCRBG_01105", 90.0, 0.0],
            ["BCRBG_48201", 90.0, 5.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 SPIREOTU_01842612"],
            ["BCRBG_01105", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 BCRBG_01105"],
            ["BCRBG_48201", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 BCRBG_48201"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [100000, "p__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "c__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "o__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "f__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [30, "g__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 BCRBG_01105", "BCRBG_01105"],
            [None, "s__SPIREOTU_01842612 BCRBG_48201", "BCRBG_48201"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_better_node_but_further(self):
        # root           |(0)
        #                |
        # phylum         |(100000)
        #                |
        # genus          |(20) - but further from median than 10
        #           ___________
        #           |         |
        # genus     |(10)     |
        #        _______      |
        #        |     |      |
        #        |(1)  |(2)   |(3)
        #        sp1   bc1    bc2

        tree_df = pl.DataFrame([
            [10, 1, "nongtdb", "SPIREOTU_01842612", "SPIRE", None, None],
            [10, 2, "nongtdb", "BCRBG_01105", "binchicken", None, None],
            [20, 3, "nongtdb", "BCRBG_48201", "binchicken", None, None],
            [20, 10, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [100000, 20, "nongtdb", None, None, 0.92, NOVELTY_RED_LEVELS[1]],
            [0, 100000, "nongtdb", None, None, 0.9, NOVELTY_RED_LEVELS[1]],
            [0, 0, "nongtdb", None, None, 0.31, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["SPIREOTU_01842612", 90.0, 0.0],
            ["BCRBG_01105", 90.0, 0.0],
            ["BCRBG_48201", 90.0, 5.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["SPIREOTU_01842612", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 SPIREOTU_01842612"],
            ["BCRBG_01105", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__SPIREOTU_01842612;s__SPIREOTU_01842612 BCRBG_01105"],
            ["BCRBG_48201", "d__Bacteria;p__SPIREOTU_01842612;c__SPIREOTU_01842612;o__SPIREOTU_01842612;f__SPIREOTU_01842612;g__BCRBG_48201;s__BCRBG_48201 BCRBG_48201"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [100000, "p__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "c__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "o__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [100000, "f__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [10, "g__SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 SPIREOTU_01842612", "SPIREOTU_01842612"],
            [None, "s__SPIREOTU_01842612 BCRBG_01105", "BCRBG_01105"],
            [None, "g__BCRBG_48201", "BCRBG_48201"],
            [None, "s__BCRBG_48201 BCRBG_48201", "BCRBG_48201"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_better_node_but_later(self):
        # root                   |(0)
        #                        |
        # phylum                 |(100000)
        #                        |
        # family                 |(200) - further from median than 100
        #           ___________________________
        #           |       |        |        |
        # family    |       |        |(100)   |
        #           |       |     _______     |
        #           |       |     |     |     |
        #           |(1)    |(2)  |(3)  |(4)  |(5)
        #           bc1     bc2   bc3   bc4   bc5

        tree_df = pl.DataFrame([
            [200, 1, "nongtdb", "BCRBG_01101", "binchicken", None, None],
            [200, 2, "nongtdb", "BCRBG_01102", "binchicken", None, None],
            [100, 3, "nongtdb", "BCRBG_01103", "binchicken", None, None],
            [100, 4, "nongtdb", "BCRBG_01104", "binchicken", None, None],
            [200, 5, "nongtdb", "BCRBG_01105", "binchicken", None, None],
            [200, 100, "nongtdb", None, None, 0.9, NOVELTY_RED_LEVELS[1]],
            [100000, 200, "nongtdb", None, None, 0.757, NOVELTY_RED_LEVELS[2]],
            [0, 100000, "nongtdb", None, None, 0.7, NOVELTY_RED_LEVELS[2]],
            [0, 0, "nongtdb", None, None, 0.31, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["BCRBG_01101", 95.0, 0.0],
            ["BCRBG_01102", 94.0, 0.0],
            ["BCRBG_01103", 93.0, 0.0],
            ["BCRBG_01104", 92.0, 0.0],
            ["BCRBG_01105", 91.0, 0.0],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["BCRBG_01101", "d__Bacteria;p__BCRBG_01101;c__BCRBG_01101;o__BCRBG_01101;f__BCRBG_01101;g__BCRBG_01101;s__BCRBG_01101 BCRBG_01101"],
            ["BCRBG_01102", "d__Bacteria;p__BCRBG_01101;c__BCRBG_01101;o__BCRBG_01101;f__BCRBG_01102;g__BCRBG_01102;s__BCRBG_01102 BCRBG_01102"],
            ["BCRBG_01103", "d__Bacteria;p__BCRBG_01101;c__BCRBG_01101;o__BCRBG_01101;f__BCRBG_01103;g__BCRBG_01103;s__BCRBG_01103 BCRBG_01103"],
            ["BCRBG_01104", "d__Bacteria;p__BCRBG_01101;c__BCRBG_01101;o__BCRBG_01101;f__BCRBG_01103;g__BCRBG_01104;s__BCRBG_01104 BCRBG_01104"],
            ["BCRBG_01105", "d__Bacteria;p__BCRBG_01101;c__BCRBG_01101;o__BCRBG_01101;f__BCRBG_01105;g__BCRBG_01105;s__BCRBG_01105 BCRBG_01105"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [100000, "p__BCRBG_01101", "BCRBG_01101"],
            [100000, "c__BCRBG_01101", "BCRBG_01101"],
            [100000, "o__BCRBG_01101", "BCRBG_01101"],
            [None, "f__BCRBG_01101", "BCRBG_01101"],
            [None, "f__BCRBG_01102", "BCRBG_01102"],
            [100, "f__BCRBG_01103", "BCRBG_01103"],
            [None, "f__BCRBG_01105", "BCRBG_01105"],
            [None, "g__BCRBG_01101", "BCRBG_01101"],
            [None, "g__BCRBG_01102", "BCRBG_01102"],
            [None, "g__BCRBG_01103", "BCRBG_01103"],
            [None, "g__BCRBG_01104", "BCRBG_01104"],
            [None, "g__BCRBG_01105", "BCRBG_01105"],
            [None, "s__BCRBG_01101 BCRBG_01101", "BCRBG_01101"],
            [None, "s__BCRBG_01102 BCRBG_01102", "BCRBG_01102"],
            [None, "s__BCRBG_01103 BCRBG_01103", "BCRBG_01103"],
            [None, "s__BCRBG_01104 BCRBG_01104", "BCRBG_01104"],
            [None, "s__BCRBG_01105 BCRBG_01105", "BCRBG_01105"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_extra_phyla(self):
        tree_df = pl.DataFrame([
            [275362, 83727, "nongtdb", "ERP107533_co1_57", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [275362, 83728, "nongtdb", "SRP141376_co3_207", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [275363, 83729, "gtdb", "GB_GCA_023135745.1", "GTDB", 1, NOVELTY_RED_LEVELS[0]],
            [275365, 83730, "nongtdb", "spire_mag_01765318", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [275366, 83731, "nongtdb", "binchicken_co117_471", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [275367, 83732, "nongtdb", "3300025882_39", "GEM", 1, NOVELTY_RED_LEVELS[0]],
            [275367, 83733, "nongtdb", "binchicken_co111_474", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [275364, 83734, "nongtdb", "spire_mag_02020990", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [275361, 275362, "nongtdb", None, None, 0.687, NOVELTY_RED_LEVELS[2]],
            [275361, 275363, "gtdb", None, None, 0.584, NOVELTY_RED_LEVELS[3]],
            [275363, 275364, "nongtdb", None, None, 0.654, NOVELTY_RED_LEVELS[3]],
            [275364, 275365, "nongtdb", None, None, 0.843, NOVELTY_RED_LEVELS[1]],
            [275365, 275366, "nongtdb", None, None, 0.945, NOVELTY_RED_LEVELS[1]],
            [275366, 275367, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [275360, 275361, "gtdb", None, None, 0.364, NOVELTY_RED_LEVELS[5]],
            [275359, 275360, "gtdb", None, None, 0.296, NOVELTY_RED_LEVELS[5]],
            [275358, 275359, "gtdb", None, None, 0.278, NOVELTY_RED_LEVELS[5]],
            # Fake root node
            [275358, 275358, "gtdb", None, None, 0.2, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["ERP107533_co1_57", 76.24, 0.7],
            ["SRP141376_co3_207", 90.29, 1.27],
            ["spire_mag_01765318", 87.75, 2.13],
            ["binchicken_co117_471", 93.5, 0.56],
            ["3300025882_39", 66.08, 2.27],
            ["binchicken_co111_474", 89.32, 1.57],
            ["spire_mag_02020990", 97.34, 0.58],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["ERP107533_co1_57", "275361;c__SRP141376_co3_207;o__SRP141376_co3_207;f__ERP107533_co1_57;g__ERP107533_co1_57;s__ERP107533_co1_57 ERP107533_co1_57"],
            ["SRP141376_co3_207", "275361;c__SRP141376_co3_207;o__SRP141376_co3_207;f__SRP141376_co3_207;g__SRP141376_co3_207;s__SRP141376_co3_207 SRP141376_co3_207"],
            ["spire_mag_01765318", "275363;o__spire_mag_02020990;f__binchicken_co111_474;g__spire_mag_01765318;s__spire_mag_01765318 spire_mag_01765318"],
            ["binchicken_co117_471", "275363;o__spire_mag_02020990;f__binchicken_co117_471;g__binchicken_co117_471;s__binchicken_co117_471 binchicken_co117_471"],
            ["3300025882_39", "275363;o__spire_mag_02020990;f__binchicken_co111_474;g__binchicken_co111_474;s__binchicken_co111_474 3300025882_39"],
            ["binchicken_co111_474", "275363;o__spire_mag_02020990;f__binchicken_co111_474;g__binchicken_co111_474;s__binchicken_co111_474 binchicken_co111_474"],
            ["spire_mag_02020990", "275363;o__spire_mag_02020990;f__spire_mag_02020990;g__spire_mag_02020990;s__spire_mag_02020990 spire_mag_02020990"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [275362, "c__SRP141376_co3_207", "SRP141376_co3_207"],
            [275364, "o__spire_mag_02020990", "spire_mag_02020990"],
            [275362, "o__SRP141376_co3_207", "SRP141376_co3_207"],
            [275365, "f__binchicken_co111_474", "binchicken_co111_474"],
            [None, "f__binchicken_co117_471", "binchicken_co117_471"],
            [None, "f__spire_mag_02020990", "spire_mag_02020990"],
            [None, "f__SRP141376_co3_207", "SRP141376_co3_207"],
            [None, "f__ERP107533_co1_57", "ERP107533_co1_57"],
            [275367, "g__binchicken_co111_474", "binchicken_co111_474"],
            [None, "g__spire_mag_02020990", "spire_mag_02020990"],
            [None, "g__spire_mag_01765318", "spire_mag_01765318"],
            [None, "g__binchicken_co117_471", "binchicken_co117_471"],
            [None, "g__SRP141376_co3_207", "SRP141376_co3_207"],
            [None, "g__ERP107533_co1_57", "ERP107533_co1_57"],
            [None, "s__binchicken_co111_474 3300025882_39", "3300025882_39"],
            [None, "s__spire_mag_02020990 spire_mag_02020990", "spire_mag_02020990"],
            [None, "s__spire_mag_01765318 spire_mag_01765318", "spire_mag_01765318"],
            [None, "s__binchicken_co111_474 binchicken_co111_474", "binchicken_co111_474"],
            [None, "s__binchicken_co117_471 binchicken_co117_471", "binchicken_co117_471"],
            [None, "s__SRP141376_co3_207 SRP141376_co3_207", "SRP141376_co3_207"],
            [None, "s__ERP107533_co1_57 ERP107533_co1_57", "ERP107533_co1_57"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_extra_family(self):
        # Later family-node is closer to family median RED than the earlier named node
        tree_df = pl.DataFrame([
            [312362, 120725, "nongtdb", "3300017485_6", "GEM", 1, NOVELTY_RED_LEVELS[0]],
            [312364, 120726, "nongtdb", "SRP124282_co5_49", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312364, 120727, "nongtdb", "binchicken_co203_446", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312365, 120728, "nongtdb", "spire_mag_01799808", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [312366, 120729, "nongtdb", "spire_mag_01799939", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [312367, 120730, "nongtdb", "spire_mag_01799662", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [312367, 120731, "nongtdb", "SRP124282_co6_56", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312368, 120732, "gtdb", "GB_GCA_005239745.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [312371, 120733, "nongtdb", "ERP119705_co25_475", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312373, 120734, "gtdb", "GB_GCA_016195485.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [312373, 120735, "nongtdb", "SRP269290_co5_116", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312376, 120736, "nongtdb", "ERP125453_co1_503", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312376, 120737, "nongtdb", "spire_mag_00098172", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [312375, 120738, "nongtdb", "spire_mag_01799640", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [312374, 120739, "nongtdb", "spire_mag_01799858", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [312378, 120740, "nongtdb", "SRP090828_co1_1", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312378, 120741, "nongtdb", "binchicken_co203_435", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312379, 120742, "gtdb", "GB_GCA_005239925.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [312379, 120743, "nongtdb", "spire_mag_00098246", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [312369, 120744, "nongtdb", "SRP124282_co5_22", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [312361, 312362, "nongtdb", None, None, 0.726, NOVELTY_RED_LEVELS[2]],
            [312362, 312363, "nongtdb", None, None, 0.775, NOVELTY_RED_LEVELS[2]],
            [312363, 312364, "nongtdb", None, None, 0.922, NOVELTY_RED_LEVELS[1]],
            [312363, 312365, "nongtdb", None, None, 0.816, NOVELTY_RED_LEVELS[2]],
            [312365, 312366, "nongtdb", None, None, 0.828, NOVELTY_RED_LEVELS[2]],
            [312366, 312367, "nongtdb", None, None, 0.863, NOVELTY_RED_LEVELS[1]],
            [312361, 312368, "gtdb", None, None, 0.735, NOVELTY_RED_LEVELS[2]],
            [312368, 312369, "gtdb", None, None, 0.83, NOVELTY_RED_LEVELS[2]],
            [312369, 312370, "gtdb", None, None, 0.849, NOVELTY_RED_LEVELS[1]],
            [312370, 312371, "gtdb", None, None, 0.896, NOVELTY_RED_LEVELS[1]],
            [312371, 312372, "gtdb", None, None, 0.93, NOVELTY_RED_LEVELS[1]],
            [312372, 312373, "gtdb", None, None, 0.946, NOVELTY_RED_LEVELS[1]],
            [312372, 312374, "nongtdb", None, None, 0.946, NOVELTY_RED_LEVELS[1]],
            [312374, 312375, "nongtdb", None, None, 0.954, NOVELTY_RED_LEVELS[1]],
            [312375, 312376, "nongtdb", None, None, 0.963, NOVELTY_RED_LEVELS[0]],
            [312370, 312377, "gtdb", None, None, 0.868, NOVELTY_RED_LEVELS[1]],
            [312377, 312378, "nongtdb", None, None, 0.908, NOVELTY_RED_LEVELS[1]],
            [312377, 312379, "gtdb", None, None, 0.906, NOVELTY_RED_LEVELS[1]],
            [312360, 312361, "gtdb", None, None, 0.705, NOVELTY_RED_LEVELS[2]],
            # Fake root node
            [312360, 312360, "gtdb", None, None, 0.2, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["spire_mag_00098172", 51.09, 0.02],
            ["spire_mag_00098246", 67.83, 8.36],
            ["spire_mag_01799640", 64.39, 0.08],
            ["spire_mag_01799662", 85.9, 0.12],
            ["spire_mag_01799808", 60.35, 0.63],
            ["spire_mag_01799858", 95.38, 1.23],
            ["spire_mag_01799939", 81.99, 0.15],
            ["3300017485_6", 94.03, 2.68],
            ["ERP119705_co25_475", 82.64, 4.34],
            ["ERP125453_co1_503", 86.02, 0.76],
            ["SRP090828_co1_1", 54.62, 0.2],
            ["SRP124282_co5_22", 91.66, 4.6],
            ["SRP124282_co5_49", 96.48, 0.15],
            ["SRP124282_co6_56", 82.77, 3.55],
            ["SRP269290_co5_116", 91.93, 2.75],
            ["binchicken_co203_435", 75.29, 1.58],
            ["binchicken_co203_446", 96.4, 0.16],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["spire_mag_00098172", "312372;g__spire_mag_01799858;s__spire_mag_01799858 spire_mag_00098172"],
            ["spire_mag_00098246", "312379;g__spire_mag_00098246;s__spire_mag_00098246 spire_mag_00098246"],
            ["spire_mag_01799640", "312372;g__spire_mag_01799858;s__spire_mag_01799858 spire_mag_01799640"],
            ["spire_mag_01799662", "312361;f__spire_mag_01799662;g__spire_mag_01799662;s__spire_mag_01799662 spire_mag_01799662"],
            ["spire_mag_01799808", "312361;f__spire_mag_01799662;g__spire_mag_01799808;s__spire_mag_01799808 spire_mag_01799808"],
            ["spire_mag_01799858", "312372;g__spire_mag_01799858;s__spire_mag_01799858 spire_mag_01799858"],
            ["spire_mag_01799939", "312361;f__spire_mag_01799662;g__spire_mag_01799939;s__spire_mag_01799939 spire_mag_01799939"],
            ["3300017485_6", "312361;f__3300017485_6;g__3300017485_6;s__3300017485_6 3300017485_6"],
            ["ERP119705_co25_475", "312371;g__ERP119705_co25_475;s__ERP119705_co25_475 ERP119705_co25_475"],
            ["ERP125453_co1_503", "312372;g__spire_mag_01799858;s__spire_mag_01799858 ERP125453_co1_503"],
            ["SRP090828_co1_1", "312377;g__binchicken_co203_435;s__binchicken_co203_435 SRP090828_co1_1"],
            ["SRP124282_co5_22", "312369;g__SRP124282_co5_22;s__SRP124282_co5_22 SRP124282_co5_22"],
            ["SRP124282_co5_49", "312361;f__SRP124282_co5_49;g__SRP124282_co5_49;s__SRP124282_co5_49 SRP124282_co5_49"],
            ["SRP124282_co6_56", "312361;f__spire_mag_01799662;g__SRP124282_co6_56;s__SRP124282_co6_56 SRP124282_co6_56"],
            ["SRP269290_co5_116", "312373"],
            ["binchicken_co203_435", "312377;g__binchicken_co203_435;s__binchicken_co203_435 binchicken_co203_435"],
            ["binchicken_co203_446", "312361;f__SRP124282_co5_49;g__binchicken_co203_446;s__binchicken_co203_446 binchicken_co203_446"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [None, "f__3300017485_6", "3300017485_6"],
            [312365, "f__spire_mag_01799662", "spire_mag_01799662"],
            [312364, "f__SRP124282_co5_49", "SRP124282_co5_49"],
            [312374, "g__spire_mag_01799858", "spire_mag_01799858"],
            [312378, "g__binchicken_co203_435", "binchicken_co203_435"],
            [None, "g__3300017485_6", "3300017485_6"],
            [None, "g__spire_mag_01799662", "spire_mag_01799662"],
            [None, "g__spire_mag_01799939", "spire_mag_01799939"],
            [None, "g__spire_mag_01799808", "spire_mag_01799808"],
            [None, "g__spire_mag_00098246", "spire_mag_00098246"],
            [None, "g__SRP124282_co5_49", "SRP124282_co5_49"],
            [None, "g__binchicken_co203_446", "binchicken_co203_446"],
            [None, "g__SRP124282_co5_22", "SRP124282_co5_22"],
            [None, "g__SRP124282_co6_56", "SRP124282_co6_56"],
            [None, "g__ERP119705_co25_475", "ERP119705_co25_475"],
            [None, "s__3300017485_6 3300017485_6", "3300017485_6"],
            [None, "s__spire_mag_01799858 ERP125453_co1_503", "ERP125453_co1_503"],
            [None, "s__spire_mag_01799858 spire_mag_00098172", "spire_mag_00098172"],
            [None, "s__spire_mag_01799858 spire_mag_01799640", "spire_mag_01799640"],
            [None, "s__spire_mag_01799858 spire_mag_01799858", "spire_mag_01799858"],
            [None, "s__spire_mag_01799662 spire_mag_01799662", "spire_mag_01799662"],
            [None, "s__spire_mag_01799939 spire_mag_01799939", "spire_mag_01799939"],
            [None, "s__spire_mag_01799808 spire_mag_01799808", "spire_mag_01799808"],
            [None, "s__spire_mag_00098246 spire_mag_00098246", "spire_mag_00098246"],
            [None, "s__SRP124282_co5_49 SRP124282_co5_49", "SRP124282_co5_49"],
            [None, "s__binchicken_co203_446 binchicken_co203_446", "binchicken_co203_446"],
            [None, "s__SRP124282_co5_22 SRP124282_co5_22", "SRP124282_co5_22"],
            [None, "s__binchicken_co203_435 SRP090828_co1_1", "SRP090828_co1_1"],
            [None, "s__binchicken_co203_435 binchicken_co203_435", "binchicken_co203_435"],
            [None, "s__SRP124282_co6_56 SRP124282_co6_56", "SRP124282_co6_56"],
            [None, "s__ERP119705_co25_475 ERP119705_co25_475", "ERP119705_co25_475"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Bacteria")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_many_phyla(self):
        tree_df = pl.DataFrame([
            [24277, 11877, "nongtdb", "binchicken_co122_247", "binchicken", 1.0, NOVELTY_RED_LEVELS[0]],
            [24278, 11878, "gtdb", "GB_GCA_023132255.1", None, 1.0, NOVELTY_RED_LEVELS[0]],
            [24280, 11879, "nongtdb", "binchicken_co181_128", "binchicken", 1.0, NOVELTY_RED_LEVELS[0]],
            [24280, 11880, "nongtdb", "spire_mag_00728314", "SPIRE", 1.0, NOVELTY_RED_LEVELS[0]],
            [24282, 11881, "nongtdb", "3300022834_11", "GEM", 1.0, NOVELTY_RED_LEVELS[0]],
            [24282, 11882, "nongtdb", "binchicken_co243_184", "binchicken", 1.0, NOVELTY_RED_LEVELS[0]],
            [24281, 11883, "nongtdb", "spire_mag_00727067", "SPIRE", 1.0, NOVELTY_RED_LEVELS[0]],
            [24276, 11884, "gtdb", "GB_GCA_018898425.1", None, 1.0, NOVELTY_RED_LEVELS[0]],
            [24269, 24276, "gtdb", None, None, 0.879, NOVELTY_RED_LEVELS[1]],
            [24276, 24277, "gtdb", None, None, 0.891, NOVELTY_RED_LEVELS[1]],
            [24277, 24278, "gtdb", None, None, 0.947, NOVELTY_RED_LEVELS[1]],
            [24278, 24279, "nongtdb", None, None, 0.971, NOVELTY_RED_LEVELS[0]],
            [24279, 24280, "nongtdb", None, None, 1.0, NOVELTY_RED_LEVELS[0]],
            [24279, 24281, "nongtdb", None, None, 0.978, NOVELTY_RED_LEVELS[0]],
            [24281, 24282, "nongtdb", None, None, 1.0, NOVELTY_RED_LEVELS[0]],
            # Fake root node
            [24269, 24269, "gtdb", None, None, 0.2, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["3300022834_11", 82.2, 0.22],
            ["spire_mag_00728314", 68.88, 0.67],
            ["spire_mag_00727067", 83.37, 4.46],
            ["binchicken_co122_247", 93.42, 1.1],
            ["binchicken_co243_184", 84.91, 0.21],
            ["binchicken_co181_128", 81.51, 0.8],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["3300022834_11", "24278"],
            ["spire_mag_00728314", "24278"],
            ["spire_mag_00727067", "24278"],
            ["binchicken_co122_247", "24277;g__binchicken_co122_247;s__binchicken_co122_247 binchicken_co122_247"],
            ["binchicken_co243_184", "24278"],
            ["binchicken_co181_128", "24278"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [None, "g__binchicken_co122_247", "binchicken_co122_247"],
            [None, "s__binchicken_co122_247 binchicken_co122_247", "binchicken_co122_247"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_name_clades_not_copied(self):
        tree_df = pl.DataFrame([
            [2663, 1, "gtdb", "GB_GCA_000008085.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2664, 2, "nongtdb", "spire_mag_00175299", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2666, 3, "nongtdb", "spire_mag_00186175", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2667, 4, "gtdb", "GB_GCA_020697515.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2668, 5, "nongtdb", "binchicken_co381_152", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [2668, 6, "nongtdb", "spire_mag_00175199", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2670, 7, "gtdb", "GB_GCA_003568775.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2671, 8, "nongtdb", "GCA_038735675.1_ASM3873567v1_genomic", "Tengchong", 1, NOVELTY_RED_LEVELS[0]],
            [2671, 9, "nongtdb", "GCA_038891875.1_ASM3889187v1_genomic", "Tengchong", 1, NOVELTY_RED_LEVELS[0]],
            [2672, 10, "nongtdb", "spire_mag_00097715", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2673, 11, "gtdb", "RS_GCF_023169545.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2678, 12, "gtdb", "GB_GCA_001552015.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2678, 13, "nongtdb", "3300025462_33", "GEM", 1, NOVELTY_RED_LEVELS[0]],
            [2679, 14, "nongtdb", "spire_mag_01105913", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2680, 15, "nongtdb", "spire_mag_00671272", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2681, 16, "gtdb", "GB_GCA_000387965.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2681, 17, "gtdb", "RS_GCF_003086415.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2676, 18, "gtdb", "GB_GCA_028275775.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2675, 19, "gtdb", "GB_GCA_028275885.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2682, 20, "gtdb", "GB_GCA_028276785.1", None, 1, NOVELTY_RED_LEVELS[0]],
            [2683, 21, "nongtdb", "spire_mag_01326119", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2683, 22, "nongtdb", "spire_mag_01109158", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2685, 23, "nongtdb", "SRP144503_co3_139", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [2685, 24, "nongtdb", "binchicken_co291_17", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [2686, 25, "nongtdb", "SRP144503_co2_142", "binchicken", 1, NOVELTY_RED_LEVELS[0]],
            [2686, 26, "nongtdb", "spire_mag_00707902", "SPIRE", 1, NOVELTY_RED_LEVELS[0]],
            [2661, 2662, "gtdb", None, None, 0.413, NOVELTY_RED_LEVELS[4]],
            [2662, 2663, "gtdb", None, None, 0.498, NOVELTY_RED_LEVELS[3]],
            [2663, 2664, "gtdb", None, None, 0.567, NOVELTY_RED_LEVELS[3]],
            [2664, 2665, "gtdb", None, None, 0.69, NOVELTY_RED_LEVELS[2]],
            [2665, 2666, "gtdb", None, None, 0.752, NOVELTY_RED_LEVELS[2]],
            [2666, 2667, "gtdb", None, None, 0.826, NOVELTY_RED_LEVELS[1]],
            [2667, 2668, "nongtdb", None, None, 0.92, NOVELTY_RED_LEVELS[1]],
            [2665, 2669, "gtdb", None, None, 0.757, NOVELTY_RED_LEVELS[2]],
            [2669, 2670, "gtdb", None, None, 0.863, NOVELTY_RED_LEVELS[1]],
            [2670, 2671, "nongtdb", None, None, 0.959, NOVELTY_RED_LEVELS[0]],
            [2669, 2672, "gtdb", None, None, 0.831, NOVELTY_RED_LEVELS[1]],
            [2672, 2673, "gtdb", None, None, 0.892, NOVELTY_RED_LEVELS[1]],
            [2673, 2674, "gtdb", None, None, 0.969, NOVELTY_RED_LEVELS[0]],
            [2674, 2675, "gtdb", None, None, 0.974, NOVELTY_RED_LEVELS[0]],
            [2675, 2676, "gtdb", None, None, 0.978, NOVELTY_RED_LEVELS[0]],
            [2676, 2677, "gtdb", None, None, 0.981, NOVELTY_RED_LEVELS[0]],
            [2677, 2678, "gtdb", None, None, 0.984, NOVELTY_RED_LEVELS[0]],
            [2677, 2679, "gtdb", None, None, 0.987, NOVELTY_RED_LEVELS[0]],
            [2679, 2680, "gtdb", None, None, 0.989, NOVELTY_RED_LEVELS[0]],
            [2680, 2681, "gtdb", None, None, 0.993, NOVELTY_RED_LEVELS[0]],
            [2674, 2682, "gtdb", None, None, 0.984, NOVELTY_RED_LEVELS[0]],
            [2682, 2683, "nongtdb", None, None, 0.986, NOVELTY_RED_LEVELS[0]],
            [2662, 2684, "nongtdb", None, None, 0.785, NOVELTY_RED_LEVELS[2]],
            [2684, 2685, "nongtdb", None, None, 0.957, NOVELTY_RED_LEVELS[0]],
            [2684, 2686, "nongtdb", None, None, 0.884, NOVELTY_RED_LEVELS[1]],
            # Fake root nodes
            [2660, 2661, "gtdb", None, None, 0.354, NOVELTY_RED_LEVELS[4]],
            [2660, 2660, "gtdb", None, None, 0.2, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        genome_metadata = pl.DataFrame([
            ["spire_mag_01105913", 88.5, 0.61],
            ["spire_mag_01109158", 83.8, 0.62],
            ["spire_mag_00671272", 54.0, 0.4],
            ["spire_mag_01326119", 89, 1.07],
            ["spire_mag_00097715", 75.6, 0.91],
            ["spire_mag_00175199", 69.4, 0.3],
            ["spire_mag_00175299", 88.0, 1.19],
            ["spire_mag_00186175", 85.9, 0.96],
            ["spire_mag_00707902", 86.6, 0.19],
            ["3300025462_33", 73.5, 0.76],
            ["GCA_038891875.1_ASM3889187v1_genomic", 76.8, 4.93],
            ["GCA_038735675.1_ASM3873567v1_genomic", 82.9, 2.72],
            ["SRP144503_co2_142", 82.4, 0.01],
            ["SRP144503_co3_139", 86.6, 0.29],
            ["binchicken_co291_17", 88.4, 0.22],
            ["binchicken_co381_152", 69.9, 0.32],
        ], schema=GENOME_METADATA_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["spire_mag_01105913", "2679"],
            ["spire_mag_01109158", "2682"],
            ["spire_mag_00671272", "2680"],
            ["spire_mag_01326119", "2682"],
            ["spire_mag_00097715", "2672;g__spire_mag_00097715;s__spire_mag_00097715 spire_mag_00097715"],
            ["spire_mag_00175199", "2667;g__binchicken_co381_152;s__binchicken_co381_152 spire_mag_00175199"],
            ["spire_mag_00175299", "2664;f__spire_mag_00175299;g__spire_mag_00175299;s__spire_mag_00175299 spire_mag_00175299"],
            ["spire_mag_00186175", "2666;f__spire_mag_00186175;g__spire_mag_00186175;s__spire_mag_00186175 spire_mag_00186175"],
            ["spire_mag_00707902", "2662;o__binchicken_co291_17;f__spire_mag_00707902;g__spire_mag_00707902;s__spire_mag_00707902 spire_mag_00707902"],
            ["3300025462_33", "2678"],
            ["GCA_038891875.1_ASM3889187v1_genomic", "2670;g__GCA_038735675.1_ASM3873567v1_genomic;s__GCA_038735675.1_ASM3873567v1_genomic GCA_038891875.1_ASM3889187v1_genomic"],
            ["GCA_038735675.1_ASM3873567v1_genomic", "2670;g__GCA_038735675.1_ASM3873567v1_genomic;s__GCA_038735675.1_ASM3873567v1_genomic GCA_038735675.1_ASM3873567v1_genomic"],
            ["SRP144503_co2_142", "2662;o__binchicken_co291_17;f__spire_mag_00707902;g__SRP144503_co2_142;s__SRP144503_co2_142 SRP144503_co2_142"],
            ["SRP144503_co3_139", "2662;o__binchicken_co291_17;f__binchicken_co291_17;g__binchicken_co291_17;s__binchicken_co291_17 SRP144503_co3_139"],
            ["binchicken_co291_17", "2662;o__binchicken_co291_17;f__binchicken_co291_17;g__binchicken_co291_17;s__binchicken_co291_17 binchicken_co291_17"],
            ["binchicken_co381_152", "2667;g__binchicken_co381_152;s__binchicken_co381_152 binchicken_co381_152"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [2686, "f__spire_mag_00707902", "spire_mag_00707902"],
            [None, "f__spire_mag_00175299", "spire_mag_00175299"],
            [None, "f__spire_mag_00186175", "spire_mag_00186175"],
            [2685, "f__binchicken_co291_17", "binchicken_co291_17"],
            [None, "g__spire_mag_00707902", "spire_mag_00707902"],
            [None, "g__spire_mag_00175299", "spire_mag_00175299"],
            [None, "g__spire_mag_00186175", "spire_mag_00186175"],
            [None, "g__spire_mag_00097715", "spire_mag_00097715"],
            [2668, "g__binchicken_co381_152", "binchicken_co381_152"],
            [2671, "g__GCA_038735675.1_ASM3873567v1_genomic", "GCA_038735675.1_ASM3873567v1_genomic"],
            [2685, "g__binchicken_co291_17", "binchicken_co291_17"],
            [None, "g__SRP144503_co2_142", "SRP144503_co2_142"],
            [2684, "o__binchicken_co291_17", "binchicken_co291_17"],
            [None, "s__spire_mag_00707902 spire_mag_00707902", "spire_mag_00707902"],
            [None, "s__spire_mag_00175299 spire_mag_00175299", "spire_mag_00175299"],
            [None, "s__spire_mag_00186175 spire_mag_00186175", "spire_mag_00186175"],
            [None, "s__spire_mag_00097715 spire_mag_00097715", "spire_mag_00097715"],
            [None, "s__binchicken_co381_152 binchicken_co381_152", "binchicken_co381_152"],
            [None, "s__binchicken_co381_152 spire_mag_00175199", "spire_mag_00175199"],
            [None, "s__GCA_038735675.1_ASM3873567v1_genomic GCA_038735675.1_ASM3873567v1_genomic", "GCA_038735675.1_ASM3873567v1_genomic"],
            [None, "s__GCA_038735675.1_ASM3873567v1_genomic GCA_038891875.1_ASM3889187v1_genomic", "GCA_038891875.1_ASM3889187v1_genomic"],
            [None, "s__binchicken_co291_17 SRP144503_co3_139", "SRP144503_co3_139"],
            [None, "s__binchicken_co291_17 binchicken_co291_17", "binchicken_co291_17"],
            [None, "s__SRP144503_co2_142 SRP144503_co2_142", "SRP144503_co2_142"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = name_clades(tree_df, genome_metadata, domain="d__Archaea")
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)

    def test_fill_taxonomy(self):
        # See test_name_clades_gtdb_full
        tree_df = pl.DataFrame([
            [10, 1, "gtdb", "GB_GCA_016935655.1", None, None, None],
            [10, 2, "nongtdb", "spire_mag_1", "SPIRE", None, None],
            [20, 3, "gtdb", "GB_GCA_016935655.2", None, None, None],
            [20, 4, "nongtdb", "binchicken_co19_1", "binchicken", None, None],
            [30, 5, "nongtdb", "spire_mag_2", "SPIRE", None, None],
            [30, 6, "nongtdb", "binchicken_co19_2", "binchicken", None, None],
            [10000, 10, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [1000, 20, "gtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [1000, 30, "nongtdb", None, None, 1, NOVELTY_RED_LEVELS[0]],
            [10000, 1000, "gtdb", None, None, 0.92, NOVELTY_RED_LEVELS[1]],
            [100000, 10000, "gtdb", None, None, 0.61, NOVELTY_RED_LEVELS[3]],
            [0, 100000, "gtdb", None, None, 0.44, NOVELTY_RED_LEVELS[4]],
            [0, 0, "gtdb", None, None, 0.31, NOVELTY_RED_LEVELS[5]],
        ], schema=TREE_INPUT_COLUMNS, orient="row")
        input_genomes = pl.DataFrame([
            ["spire_mag_1", "10"],
            ["spire_mag_2", "1000;g__SPIRE_b1;s__SPIRE_b1 spire_mag_2"],
            ["binchicken_co19_1", "20"],
            ["binchicken_co19_2", "1000;g__SPIRE_b1;s__SPIRE_b1 binchicken_co19_2"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        input_nodes = pl.DataFrame([
            [30, "g__SPIRE_b1", "spire_mag_2"],
            [None, "s__SPIRE_b1 spire_mag_2", "spire_mag_2"],
            [None, "s__SPIRE_b1 binchicken_co19_2", "binchicken_co19_2"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")
        gtdb_taxonomy = pl.DataFrame([
            ["GB_GCA_016935655.1", "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Azobacteroidaceae;g__Azobacteroides;s__Azobacteroides pseudotrichonymphae_A"],
            ["GB_GCA_016935655.2", "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Chitinophagales;f__Chitinophagaceae;g__Agriterribacter;s__Agriterribacter sp001899685"],
            ["GB_GCA_016935655.3", "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Chitinophagales;f__Saprospiraceae;g__Aureispira;s__Aureispira sp000724545"],
        ], schema=GTDB_TAXONOMY_COLUMNS, orient="row")

        expected_genomes = pl.DataFrame([
            ["spire_mag_1", "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Azobacteroidaceae;g__Azobacteroides;s__Azobacteroides spire_mag_1"],
            ["spire_mag_2", "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Chitinophagales;f__Chitinophagaceae;g__SPIRE_b1;s__SPIRE_b1 spire_mag_2"],
            ["binchicken_co19_1", "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Chitinophagales;f__Chitinophagaceae;g__Agriterribacter;s__Agriterribacter binchicken_co19_1"],
            ["binchicken_co19_2", "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Chitinophagales;f__Chitinophagaceae;g__SPIRE_b1;s__SPIRE_b1 binchicken_co19_2"],
        ], schema=GENOMES_OUTPUT_COLUMNS, orient="row")
        expected_nodes = pl.DataFrame([
            [30, "g__SPIRE_b1", "spire_mag_2"],
            [None, "s__SPIRE_b1 spire_mag_2", "spire_mag_2"],
            [None, "s__SPIRE_b1 binchicken_co19_2", "binchicken_co19_2"],
            [None, "s__Azobacteroides spire_mag_1", "spire_mag_1"],
            [None, "s__Agriterribacter binchicken_co19_1", "binchicken_co19_1"],
        ], schema=NODES_OUTPUT_COLUMNS, orient="row")

        (observed_genomes, observed_nodes) = fill_taxonomy(tree_df, input_genomes, input_nodes, gtdb_taxonomy)
        self.assertDataFrameEqual(expected_genomes, observed_genomes)
        self.assertDataFrameEqual(expected_nodes, observed_nodes)


if __name__ == "__main__":
    unittest.main()
