# Raw Data Directory

## E. coli Genome Dataset

**Source**: NCBI RefSeq  
**Accession**: GCF_000005845.2_ASM584v2  
**Size**: ~4.6 million base pairs

### Download Instructions

The E. coli genome is automatically downloaded by the notebook, but you can also manually download:
```bash
# Using wget
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz

# Or using curl
curl -O https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz
```

### File Structure
- `ecoli_genome.fna.gz` - Compressed FASTA format
- File is in `.gitignore` (too large for git)

### Citation
If using this data, cite:
```
Blattner FR, et al. (1997) The complete genome sequence of Escherichia coli K-12. 
Science 277(5331):1453-62.
```