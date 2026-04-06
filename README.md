# bam2zarr

A Python library and CLI to convert a BAM file into a VCF Zarr-compatible dataset,
where each read is represented as one sample.

## Install

Install directly from GitHub:

```bash
pip install git+https://github.com/hyanwong/bam2zarr
```

For local development:

```bash
uv venv .venv
uv sync --dev
```

Activate the virtual environment if desired:

```bash
source .venv/bin/activate
```

## Usage

```bash
uv run bam2zarr input.bam human.fa output.zarr --known-sites known_sites.csv
```

If `output` is omitted, it defaults to the BAM path with a `.vcz` suffix:

```bash
uv run bam2zarr input.bam human.fa --known-sites known_sites.csv
# writes input.vcz
```

Tree-sequence known sites can be provided per contig:

```bash
uv run bam2zarr input.bam human.fa output.zarr \
    --tree-sequence chr1=chr1.trees \
    --tree-sequence chr2=chr2.trees \
    --tree-sequence-position-base 1
```

To save output variant positions that were not present in the known-sites input:

```bash
uv run bam2zarr input.bam human.fa output.zarr \
    --known-sites known_sites.vcf.gz \
    --save-extra-positions extra_positions.tsv
```

Optional flags:

- `--min-mapq INT`: filter reads by minimum mapping quality
- `--include-secondary`: include secondary/supplementary alignments
- `--known-sites PATH`: known-sites source path (CSV/TSV, VCF/BCF, or VCF Zarr)
- `--tree-sequence CONTIG=PATH`: map a contig to a tree-sequence file (repeat per contig)
- `--tree-sequence-position-base {0,1}`: interpret tree-sequence positions (default `1`)
- `--save-extra-positions PATH`: write output variant positions absent from the known-sites input as `contig<TAB>position`
- `--no-progress`: disable progress and diagnostics on stderr
- `reference_fasta` is required and provides the REF allele plus mismatch-driven site discovery
- Gzipped reference FASTA files must be BGZF-compressed (`.fa.gz` with faidx support); plain gzip FASTA is not supported.

## Python API

```python
summary = convert_bam_to_vcfzarr(
    bam_path="input.bam",
    output_zarr_path="output.zarr",
    known_sites="known_sites.vcf.gz",
    reference_fasta_path="human_g1k_v37.fa",
)
print(summary)
```

## Tests

Tests are in the top-level `tests/` directory.

```bash
python -m pytest tests -q
```

## Notes

- This tool models each read as an individual sample.
- Known sites input is required as the second CLI argument.
- Reference FASTA input is required.
- Input format is auto-detected (CSV/TSV, VCF/BCF, or VCF Zarr).
- `known_sites` in the Python API can be a known-sites path or a tree-sequence dict keyed by chromosome.
- When `known_sites` is provided as tree sequences in the Python API, bam2zarr reports the ancestral/reference match and mismatch percentages before scanning the BAM. A mismatch around `25%` often indicates an off-by-one error in `tree_sequence_position_base`.
- In the CLI, use either `--known-sites PATH` or one or more `--tree-sequence CONTIG=PATH` values.
- The output also includes sites where any fragment differs from the reference FASTA.
- `--save-extra-positions` writes the output sites that were created only because one or more BAM reads disagreed with the reference and the site was not already present in the known-sites input.
- The allele list for each output site is copied from the known-sites input in the same order.
- Genotypes are haploid-encoded (`0`, `1`, or missing `.` represented as `-1`).
- The output dataset and returned summary include `out_of_known_alleles_site_proportion`,
  the proportion of known sites where at least one observed base is not in the known allele set.
