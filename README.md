# Globdb processing pipeline

```bash
conda activate globdb_processing
```

Run locally

```bash
snakemake \
    --snakefile scripts/globdb_processing.smk \
    --directory results/globdb_processing/20250402 \
    --software-deployment-method conda --conda-prefix /mnt/weka/pkg/cmr/aroneys/envs_aqua \
    --keep-going --rerun-triggers mtime --cores 32
```

Run with cluster submission

```bash
snakemake \
    --snakefile scripts/globdb_processing.smk \
    --directory results/globdb_processing/20250402 \
    --software-deployment-method conda --conda-prefix /mnt/weka/pkg/cmr/aroneys/envs_aqua \
    --profile aqua --retries 3 \
    --keep-going --rerun-triggers mtime --cores 64 --local-cores 32
```
