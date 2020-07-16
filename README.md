# GWASmetric

Comparison of the GWAS association powers computed by

- Hail (wald, score and lrt tests)
- Plink (simple --assoc)
- Gini
- Entropy

The GWASmetric plain notebook is provided in [Notebook](Notebook) directory.

An example without homozygous alternate genotype ("1/1") is provided [NoHom](NoHom) directory.

An example with homozygous alternate genotype ("1/1") is provided [WithHom](WithHom) directory.

The summary stat for both above examples are provided in [Summary](Summary) directory.

One you have a look at the above exaples (NoHom first) consider the following questions:

### Biologically Proven Associated SNP

If we comput association power (using all different methods) is there a way to say one method better detect true biologifcal signals?

The question is what it mean "to better detect"?

### Real Genotype

What if we perform the analysis with real case/control genotype data but not simulated contingency table?

Would it increase the correlation between different GWAS metrics.

### Hardy-Weinberg

What if we color points in the plot based on Hardy-Weinberg pvalue. Would be any correlation between Hardy-Weinberg pvalue and being an outlier?

### Gini in RandomForest

How Gini is used in RandomForest to compute _Variable Importance_?
