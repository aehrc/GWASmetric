# GWASmetric
There are number of ways to measure association power of a SNP with respect to a binary phenotype. 


```python
import csv
import numpy as np
import pandas as pd
```

# Simulation Parameter
in our setup, Hail read/write data from/to AWS S3 storage. If this is not yor setup you should manually change part of this notebook. The easeiest way is to use [VariantSpark on the AWS MarketPlace](https://aws.amazon.com/marketplace/pp/AEHRC-VariantSpark-Notebook/B07YVND4TD) described above.

Note that this notebook also use the local storage and write some files.

| step | noHom=True | noHom=False |
|:----:|:----------:|:-----------:|
|  0.5 |          9 |          36 |
|  0.1 |        121 |        4356 |
| 0.05 |        441 |       53361 |
| 0.01 |      10201 |    26532801 |


```python
numCase=5000 # total number of cases
numCtrl=5000 # total number of controls
step=0.1     # Granuality of the simulation, more variants are simulated with lower step (0<step<1)
             # For the number of simulated variants see above table

# Output file prefix
ofn="output"

# noHom=True keep the number of Homozygeour Alternate Genotype (1/1) 0.
# So that each SNP can take only 2 values 0/0 or 0/1
# This is simpler case where you can decrease the step to a smaller value
noHom=False 

```

# Set Purity
These functions compute the set purity using Gini and Entropy. Set purity is simply 1-(set impurity).

Asume A$$ and B represent the number of cases and controls in a set where T=A+B is the number of all samples in the set. Then PA and PB represent the probability of cases and controls.

$$P_A=\frac{A}{T}$$

$$P_B=\frac{B}{T}$$

$$GiniPurity={P_A}^2+{P_B}^2$$

$$EntropyPurity=1-(P_A\times \log_2(P_A) + P_B\times \log_2(P_B))$$

While entropy range between 0 and 1, Gini range from 0.5 to 1. Note that if there are more than two clasess (case, control) the mimumum Gini would be smaller. 



```python
def GiniPurity(A, B):
    T=A+B
    GP = 0
    if T:
        PA = A/T
        PB = B/T
        GP = (PA**2)+(PB**2)
    return GP

def EntropyPurity(A, B):
    T=A+B
    EP = 0
    if T:
        PA = A/T
        PB = B/T
        EP = 1
        EP += (PA*np.log2(PA)) if PA else 0
        EP += (PB*np.log2(PB)) if PB else 0
    return EP

#### Test Code
# A=10
# for B in range(0,11):
#     print(GiniPurity(A,B), EntropyPurity(A,B))
```

# Information Gained (IG)
When a variable splits a set into multiple subsets, the information gained is the difference between weither average purity of subsets and the purity of the parent set. In the average, the weight is the fraction of samples in the subset compared to the parent set.

In case of a SNP, the parent set is all of the samples (all cases and all controls) which is divided into three groups base on their genotype for the given SNP (0/0, 0/1, 1/1) or (Ref, Het, Hom).


```python
# IG: Information Gained
def GiniIG(ParentSetPurity, CaseRef, CaseHet, CaseHom, CtrlRef, CtrlHet, CtrlHom):
    total = float(CaseRef + CaseHet + CaseHom + CtrlRef + CtrlHet + CtrlHom)
    InformationGained = 0
    if total:
        ######         Set Purity                  Weight
        Ref = GiniPurity(CaseRef, CtrlRef) * ((CaseRef + CtrlRef) / total)
        Het = GiniPurity(CaseHet, CtrlHet) * ((CaseHet + CtrlHet) / total)
        Hom = GiniPurity(CaseHom, CtrlHom) * ((CaseHom + CtrlHom) / total)
        WeightedAverageSetPurity = Ref + Het + Hom
        InformationGained = WeightedAverageSetPurity - ParentSetPurity
    return InformationGained


def EntropyIG(ParentSetPurity, CaseRef, CaseHet, CaseHom, CtrlRef, CtrlHet, CtrlHom):
    total = float(CaseRef + CaseHet + CaseHom + CtrlRef + CtrlHet + CtrlHom)
    InformationGained = 0
    if total:
        ######         Set Purity                  Weight
        Ref = EntropyPurity(CaseRef, CtrlRef) * ((CaseRef + CtrlRef) / total)
        Het = EntropyPurity(CaseHet, CtrlHet) * ((CaseHet + CtrlHet) / total)
        Hom = EntropyPurity(CaseHom, CtrlHom) * ((CaseHom + CtrlHom) / total)
        WeightedAverageSetPurity = Ref + Het + Hom
        InformationGained = WeightedAverageSetPurity - ParentSetPurity
    return InformationGained

### Simple test for information gaine function
# print(GiniIG   (0.5, 3, 5, 2, 7, 2, 1))
# print(EntropyIG(0.0, 3, 5, 2, 7, 2, 1))
# print("==============================")
# print(GiniIG   (0.5, 8, 1, 1, 7, 2, 1))
# print(EntropyIG(0.0, 8, 1, 1, 7, 2, 1))
# print("==============================")
# print(GiniIG   (0.5, 1, 1, 8, 7, 2, 1))
# print(EntropyIG(0.0, 1, 1, 8, 7, 2, 1))
# print("==============================")
```

## User Define Association Power Test


```python
##USER-DEF##
# Define your association power test below and return the association power.
# Do not change the function name (or change it in the entire notebook)
def UserDef(CaseRef, CaseHet, CaseHom, CtrlRef, CtrlHet, CtrlHom):
    return 0
```

# SNP simulation
## Also compute Gini and Entropy Information Gained for SNPs
All different possibility of the contingency table (given number of cases and controls) are considered and the corresponding SNP genotype is produced for population of cases and controls.

The contingency table is described below and the nested loop in the function generate all possible value with granuality of the step.

|  Genotype |   Case  |   Ctrl  |
|:---------:|:-------:|:-------:|
| 0/0 (Ref) | refCase | refCtrl |
| 0/1 (Het) | hetCase | hetCtrl |
| 1/1 (Hom) | homCase | homCtrl |

SimulateSNPs_noHom consider situation where homCase and homCtrl are always 0.

For the below contingency table, it does not matter the genotype are produced in formA, formB, or any other form, the association power should remains the same (i.e. the order of sample in the input data does not affect association power). Note that hear we consider individual SNP association power and does not consider any covariate.

|  Genotype | Case | Ctrl |
|:---------:|:----:|:----:|
| 0/0 (Ref) |   3  |   1  |
| 0/1 (Het) |   0  |   2  |
| 1/1 (Hom) |   1  |   1  |

 | Form | case1 | case2 | case3 | case4 | ctrl1 | ctrl2 | ctrl3 | ctrl4 |
|------|-------|-------|-------|-------|-------|-------|-------|-------|
| A    | 0/0   | 0/0   | 0/0   | 1/1   | 0/0   | 0/1   | 0/1   | 1/1   |
| B    | 0/0   | 0/1   | 0/0   | 0/0   | 0/1   | 1/1   | 0/0   | 0/1   |

**Hardy-Weinberg is not considered in this simulation**


```python
def SimulateSNPs(writer, giniParentSetPurity, entropyParentSetPurity, numCase, numCtrl, step):
    
    stepCtrl = int(numCase * step)
    stepCase = int(numCtrl * step)
    
    
    pos=0
    for refCase in range(0, numCase+1, stepCase):  
        for hetCase in range(0, (numCase-refCase)+1, stepCase):
            homCase = numCase - (refCase + hetCase)
            for refCtrl in range(0, numCtrl+1, stepCtrl):
                for hetCtrl in range(0, (numCtrl-refCtrl)+1, stepCtrl):
                    homCtrl = numCtrl - (refCtrl + hetCtrl)
                    
                    pos+=1
                    
                    giniIG    = GiniIG   (giniParentSetPurity   , refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl)
                    entropyIG = EntropyIG(entropyParentSetPurity, refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl)
                    ##USER-DEF##
                    userDef   = UserDef(refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl)
                    
                    writer.writerow([refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl, giniIG, entropyIG, userDef])
    print("SNPs simulated: ", pos)

def SimulateSNPs_noHom(writer, giniParentSetPurity, entropyParentSetPurity, numCase, numCtrl, step):
    
    stepCtrl = int(numCase * step)
    stepCase = int(numCtrl * step)
    
    
    pos=0
    for refCase in range(0, numCase+1, stepCase):  
        hetCase = numCase-refCase
        homCase = 0
        for refCtrl in range(0, numCtrl+1, stepCtrl):
            hetCtrl = numCtrl-refCtrl
            homCtrl = 0

            pos+=1

            giniIG    = GiniIG   (giniParentSetPurity   , refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl)
            entropyIG = EntropyIG(entropyParentSetPurity, refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl)
            ##USER-DEF##
            userDef   = UserDef(refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl)

            writer.writerow([refCase, hetCase, homCase, refCtrl, hetCtrl, homCtrl, giniIG, entropyIG, userDef])
    print("SNPs simulated: ", pos)

```

# VCF File Simulation
To compute association power using plink, Hail and other GWAS software
we cannot directly input the contingency table.
Thus we generate VCF file where the SNP genotype reflect the contingency table that we produce.


```python
def Simulate(ofn="test", numCase=10, numCtrl=10, step=0.1):
    
    # Compute Parent set purity
    giniParentSetPurity    = GiniPurity(numCase, numCtrl)
    entropyParentSetPurity = EntropyPurity(numCase, numCtrl)
    print("Gini    Parent Set Purity", giniParentSetPurity)
    print("Entropy Parent Set Purity", entropyParentSetPurity)

    # Simulte VCF file
    with open(ofn+'.all.tsv','w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['refCase', 'hetCase', 'homCase', 'refCtrl', 'hetCtrl', 'homCtrl', 'GiniIG', 'EntropyIG', 'UserDef'])

        if noHom:
            SimulateSNPs_noHom(writer, giniParentSetPurity, entropyParentSetPurity, numCase, numCtrl, step)
        else:
            SimulateSNPs(writer, giniParentSetPurity, entropyParentSetPurity, numCase, numCtrl, step)

```

### Perform the Simulation


```python
Simulate(ofn=ofn, numCase=numCase, numCtrl=numCtrl, step=step)
```

# Prepare for Plots
Read the data and compute -log10 of all pvalues


```python
pdf = pd.read_csv(ofn+'.all.tsv', sep='\t')
```


```python
pdf.head()
```

# Gini vs Entropy


```python
pdf.plot(x='EntropyIG', y='GiniIG', style='o')
```

# UserDef vs Gini


```python
##USER-DEF##
pdf.plot(x='UserDef', y='GiniIG', style='o')
```
