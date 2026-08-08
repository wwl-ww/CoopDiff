# DeepTernary 论文全文

> **SE(3)-Equivariant Ternary Complex Prediction Towards Target Protein Degradation**

> 原文: arXiv:2502.18875 | Nature Communications 16, 5514 (2025)
> 作者: Fanglei Xue, Meihan Zhang, Shuqi Li, Xinyu Gao, James A. Wohlschlegel, Wenbing Huang, Yi Yang, Weixian Deng
> DOI: 10.1038/s41467-025-61272-5
> PDF 下载自: https://arxiv.org/pdf/2502.18875.pdf

---

## Page 1

SE(3)-Equivariant Ternary Complex Prediction

Towards Target Protein Degradation

Fanglei Xue1, Meihan Zhang2, Shuqi Li3, Xinyu Gao4,
James A. Wohlschlegel6, Wenbing Huang3*, Yi Yang5*, Weixian Deng6*

1ReLER Lab, AAII, University of Technology Sydney, Sydney, NSW, 2007,
Australia.
2College of Life Sciences, Nankai University, Tianjin, China.
3Gaoling School of Artificial Intelligence, Renmin University of China, Beijing,
China.
4University of Chinese Academy of Sciences, Beijing, China.
5ReLER Lab, CCAI, Zhejiang University, Hangzhou, China.
6Department of Biological Chemistry at David Geffen School of Medicine,
University of California, Los Angeles, Los Angeles, CA, 90095, U.S.A..

*Corresponding author(s). E-mail(s): hwenbing@ruc.edu.cn;
yangyics@zju.edu.cn; weixiandeng@ucla.edu;
Contributing authors: faxue@uw.edu; zmh1023@126.com; shuqili@ruc.edu.cn;

gzgaoxinyu@163.com; jwohl@mednet.ucla.edu;

Abstract

Targeted protein degradation (TPD) induced by small molecules has emerged as a rapidly
evolving modality in drug discovery, targeting proteins traditionally considered ”undrug-
gable.” This strategy induces the degradation of target proteins rather than inhibiting
their activity, achieving desirable therapeutic outcomes. Proteolysis-targeting chimeras
(PROTACs) and molecular glue degraders (MGDs) are the primary small molecules that
induce TPD. Both types of molecules form a ternary complex linking an E3 ubiquitin
ligase with a target protein, a crucial step for drug discovery. While significant advances
have been made in in-silico binary structure prediction for proteins and small molecules,
ternary structure prediction remains challenging due to obscure interaction mechanisms
and insufficient training data. Traditional methods relying on manually assigned rules per-
form poorly and are computationally demanding due to extensive random sampling. In
this work, we introduce DeepTernary, a novel deep learning-based approach that directly
predicts ternary structures in an end-to-end manner using an encoder-decoder architec-
ture. DeepTernary leverages an SE(3)-equivariant graph neural network (GNN) with both
intra-graph and ternary inter-graph attention mechanisms to capture intricate ternary
interactions from our collected high-quality training dataset, TernaryDB. The proposed
query-based Pocket Points Decoder extracts the 3D structure of the final binding ternary
complex from learned ternary embeddings, demonstrating state-of-the-art accuracy and
speed in existing PROTAC benchmarks without prior knowledge from known PROTACs.
It also achieves notable accuracy on the more challenging MGD benchmark under the
blind docking protocol. Remarkably, our experiments reveal that the buried surface area
calculated from DeepTernary-predicted structures correlates with experimentally obtained
degradation potency-related metrics. Consequently, DeepTernary shows potential in effec-
tively assisting and accelerating the development of TPDs for previously undruggable
targets.


## Page 2

Deep
Ternary

training data

Protein Data Bank

collect & clean

X-ray
R value
resolution

similarity

cluster

disassemble
b

c

Cα coordinate
amino acid type

x M

2 proteins + 1 ligand

monomer

inter-graph
message passing

atom coordinate 
atomic number
chirality
formal charge
implicit Hs
aromatic
...

Cα coordinate
amino acid type

intra-graph
message passing

complex

POI
E3

Ub

E2

POI

Ub

Ub

Ub Ub

E3

Ub

E2

POI
PROTAC

recycle

E3

Ub

E2

POI

E3

Ub

E2

MG

Ub

Ub Ub Ub

recycle

Proteasome

Molecular Glue
PROTAC

ubiquitination

degradation

ubiquitination

a

POI

POI

SE(3)-Equivariant Encoder

Ubiquitin-Proteasome System

discard test-
similar samples
known MGs
and PROTACs

p1

lig

p2

(1)

(2)

rigid align to a complex 
according to predicted four

groups of pocket points

pocket points *
PAE

Pocket Points Decoder

pocket points

queries

predict aligned 
error (PAE) query

coord.

feature

coord.

feature

3.27

self-attention

cross-attention

cross-
attention

MLP

MLP

cross-attention

coord-
attention

x N

feature

coord.

align p2 to lig **

align p2-lig  to p1

(1)

(2)

parts disassemble from ternary complexes

were directly used for training

random
conformation

sampled
using RDKit

Fig. 1
DeepTernary is a deep learning model for predicting the structure of the ternary complex
induced by PROTACs and MG(D)s. a, The MOA of PROTACs and MGDs. The protein of interest (POI)
and the E3 ligase are recruited to proximity by PROTACs or MGDs to form a ternary complex, which then
the Ubiquitin-Proteasome System (UPS) is employed to transfer the ubiquitin and degrade the POI. b, To
mitigate the scarcity of known PROTACs and MG(D)s structures, a large-scale ternary complex dataset (named
TernaryDB) was collected by searching and cleaning complexes from the Protein Data Bank (PDB) archive. The
collected samples were then grouped into clusters by similarity. Any complex that is similar to known PROTAC
and MG(D) induced complexes was excluded from the training set. DeepTernary was trained on this filtered
database by predicting the original complex structure using dissembled monomers. c, DeepTernary is an SE(3)-
equivalent graph neural network equipped with attention blocks to facilitate efficient information exchange. It
begins by representing two proteins and a small molecule as three graphs, encoding node coordinates, diverse
amino acid or atom characteristics as node features, edge types and distances as edge features. The three graphs
are fed into an encoder consisting of a series of SE(3)-equivariant blocks, enabling both intra- and inter-graph
learning to capture interactions effectively. The encoder will predict the conformation of the small molecule and
output the refined node features/coordinates of the two proteins. Subsequently, a decoder comprising several
attention-based blocks employs these refined features/coordinates to generate two pairs of pocket points and a
predicted aligned error (PAE). The pocket points are then used to align the small molecule and protein 2 to
protein 1. * For PROTAC, the pocket points are taken from unbound structures, don’t need to predict. ** For
MG(D), the ligand and protein 2 are simultaneously aligned to protein 1.

1 Introduction

Targeted protein degradation (TPD) is a rapidly evolving field in drug discovery, representing
a promising therapeutic approach to degrade target proteins via harnessing the ubiquitin-
proteasome system and autophagy-lysosome system [1–4]. Traditional drug discovery mainly
focuses on inhibiting the activity of target proteins, which may not always be effective, espe-
cially in cases where the target protein is ‘undruggable’ by occupancy-driven inhibitors like
small molecules [5]. These ‘undruggable’ proteins include oncology targets in the SWI/SNF
complex [6, 7] and many kinases [8] which share high homology active domain with their
essential non-disease related family members, and transcriptional factors [9] that are highly
unstructured until they form active conformations. TPD presents an alternative strategy, which
is to induce the degradation of target proteins rather than inhibit their activity to achieve
desirable therapeutic outcomes. The mode of action (MOA) for TPD offers several advantages:


## Page 3

Firstly, TPD molecules do not require targeting ‘active site’, allowing them to selectively tar-
get disease-driver proteins without affecting other essential homologous proteins that often
share conserved active sites, and exert potential to engage highly-unstructured transcriptional
factors [10] and other scaffolding targets that do not depend on active sites [10]. Secondly,
its transient protein interaction via event-driven mechanism reduces the reliance on strong
binding affinity, in contrast to inhibitor drugs [11]. Furthermore, its catalytic nature miti-
gates the requirement for high dosages and the subsequent challenges associated with off-target
effects [12]. Lastly, even for existing targetable proteins by inhibitors, it still offers alternative
therapeutic options to fight against drug resistance caused by active site mutations [13].

Proteolysis-targeting chimeras (PROTACs) and molecular glue degraders (MGDs) are two
main modes of TPD [11]. As shown in Fig. 1a, PROTACs are hetero-bifunctional small
molecules consisting of three moieties, including a warhead, which is the ligand of the protein
of interest (POI), an anchor, which is the ligand of an E3 ubiquitin ligase being employed, and
a linker linking the warhead and anchor. With the hetero-bifunctional structure, PROTACs
recruit the POI to an E3 ubiquitin ligase, leading to the ubiquitination of the POI and its sub-
sequent degradation process by UPS [2, 14]. As of January 2023, there have been 18 PROTACs
under evaluation by regulatory authorities, targeting different malignant cancer diseases [15].
MGDs, in contrast, are small molecules that facilitate the interaction between the POI and
an E3 ubiquitin ligase, enabling the ubiquitination and degradation processes of the POI [16].
Unlike PROTACs, they do not require a bifunctional structure but act by stabilizing exist-
ing protein-protein interactions or inducing new interactions [17]. Despite their distinct modes
of action, both PROTACs and MG(D)s share a common feature: the induction of a ternary
complex structure that is crucial for their respective mechanisms.

Understanding the ternary structure induced by PROTACs or MGDs provides crucial
insights into the molecular basis of induced protein degradation. In the context of PROTACs,
the ternary structure elucidates how the PROTAC molecule facilitates the connection between
the POI and the E3 ligase, demonstrating the interacting poses, properties of the contact inter-
face, and solvent-exposed amino acid residues essential for efficient ubiquitination. For instance,
the buried surface area (BSA) of the ternary structure [18] is a critical parameter indicating
the extent of interaction surface between the PROTAC, the POI, and the E3 ligase, directly
correlating with the stability and efficacy of the induced degradation [17, 19]. The ternary
structure can also suggest possible modifications in terms of the length and the composition of
the PROTAC linker in order to improve selectivity and reduce off-target effects [19]. Similar to
PROTACs, the BSA of the MGD-induced ternary structure is a crucial determinant of their
functional impact, influencing both the strength and specificity of the interaction between the
POI and the E3 ligase [17], which can also provide clues about the molecular features that are
crucial for the molecular glue’s activity [20].

Existing experimental approaches to obtain the PROTAC- or MGD-induced ternary struc-
tures, such as X-ray crystallography and cryo-EM, often depend on costly instrumentation
and intricate reagents and remain a formidable challenge for seasoned structural biologists
due to the necessity of high-purity proteins and precise buffer conditions. Instead, in silico
approaches have been proposed to predict ternary structures that primarily using various dock-
ing methods (such as PatchDock [21, 22], FRODock [23, 24], RosettaDock [22, 24, 25], and
PIPER [19, 26, 27]) to generate big pools of structures and then to rank, filter, and refine
the docked ternary structures by minimizing free energy [19, 22, 27], atom clash [27–29], con-
straining distance to E2 ligase [27], and molecular dynamics simulations [19]. In spite of the
encouraging progress, the structures predicted by existing docking methods still deviate greatly
from experimentally determined ones, and the docking process is usually time-consuming.
Recently, deep learning technologies such as AlphaFold2 [30] and RosettaFold [31] have shown
promising prediction accuracy for protein structure prediction by making use of deep and
sophisticated neural networks to distill crucial features from extensive training datasets. These
remarkable achievements have attracted significant scientific interest in extending deep learning
to other related tasks, including protein-protein [32, 33] and protein-ligand complex structure
prediction [34, 35]. However, to our best knowledge, there were no reported research on pre-
dicting PROTAC- or MGD-induced ternary structures by using deep learning approaches. This
can be attributed to the heightened complexity of modeling ternary structures compared to
the unitary or binary structures tackled in prior studies. Additionally, the scarcity of training
data presents a significant obstacle to training deep learning models, as there are only a few
resolved ternary structures for both PROTACs and MGDs [17, 36], making it impractical to
train such models with such limited data.


## Page 4

In this work, we introduce a novel deep learning-based framework for predicting ternary
complexes induced by PROTACs and MG(D)s1. This represents the first attempt to apply deep
learning to PROTAC structure prediction and the first in silico approach for MG(D) struc-
ture prediction. To achieve this, we curated TernaryDB, a large-scale dataset comprising over
20,000 ternary complexes from the Protein Data Bank (PDB). The dataset focuses on high-
quality complexes that include a small molecule and two proteins while deliberately excluding
known PROTACs and MG(D)s from the training list. Fig. 1b outlines the construction process
of the dataset. Leveraging TernaryDB, we trained DeepTernary, an SE(3)-equivariant graph
neural network specifically designed for ternary structure prediction (Fig. 1c). In this model,
the ternary complexes were disassembled into three components—p1 (protein1), lig (ligand),
and p2 (protein2)—each modeled as a graph. Graph neural networks(GNN) [37] offer a pow-
erful framework for processing graph-structured data through message passing between nodes
and edges. To improve data efficiency, we employed an SE(3)-equivariant GNN, leveraging
the symmetry properties of SE(3) to ensure invariance to the translation and rotation of 3D
structures. Additionally, we introduced a novel ternary inter-graph attention mechanism to cap-
ture the intricate relationships between ternary components, along with a query-based pocket
points decoder to predict the final complex structure. With these innovations, DeepTernary
effectively predicts both the conformation of the small molecule and the docking poses of the
ternary complex. DeepTernary was evaluated against existing PROTAC and MG(D) bench-
marks, achieving state-of-the-art performance with DockQ scores of 0.65 and 0.21, with average
inference times of approximately 7 seconds and 1 second, respectively. The model’s ability
to generalize from a non-PROTAC/MG(D) PDB dataset to PROTAC/MG(D) ternary struc-
tures highlights its capacity to capture the fundamental interaction patterns governing ternary
complex formation, rather than relying on memorization. Moreover, the predicted buried sur-
face area (BSA) of the PROTAC complexes aligns closely with values reported in existing
literature, with BSA ranging from 1100 to 1500, indicating high degradation potential. These
results demonstrate DeepTernary’s potential to advance our understanding and manipulation
of protein degradation mechanisms.

2 Results

The construction of TernaryDB

There are only a few dozen experimentally determined PROTAC- and MG(D)-involved ternary
complexes in the PDB [39]. Despite the remarkable success of deep learning in protein struc-
ture prediction [30, 31], protein-protein docking [32], and protein-ligand interactions [34, 40],
its application to targeted protein degradation (TPD) remains underdeveloped, primarily due
to the scarcity of training data. We hypothesized that TPD complexes adhere to the same
fundamental atom-interaction principles as other tripartite complexes. To test this hypothesis
and enable deep learning-based prediction of TPD complex structures, we curated a compre-
hensive dataset of ternary complexes from the PDB. After stringent data filtering (details are
provided in Methods 4), the final dataset comprised 22,303 complexes, with their key attributes
illustrated in Fig. 2. The distribution of ligand atom counts, excluding hydrogens, is shown in
Fig. 2b, revealing that the majority of ligands contain fewer than 60 heavy atoms, with only
a small subset exceeding 100. The chemical diversity of these ligands, represented by Mor-
gan fingerprints (Fig. 2f), highlights the broad chemical space and drug-like properties of the
dataset. Proteins from 363 species, ranging from bacteria to humans, are included in the dataset
(Fig. 2d). Although the protein space is relatively sparse, it adequately covers PROTAC- and
MG(D)-induced proteins (Fig. 2e).

To rigorously assess our method, we integrated known PROTAC and MG(D) ternary com-
plexes into the test sets. To prevent data leakage, we utilized MMseqs2 [41] to cluster the
dataset based on protein sequence similarity. Clusters containing known PROTAC or MG(D)
complexes were excluded from the training set and served as a validation set, ensuring no
overlap between training and test data. This clustering approach yielded 16,203 complexes
distributed across 1,398 clusters for PROTACs and 22,046 complexes across 1,982 clusters for
MG(D)s. The distribution of cluster sizes is shown in Fig. 2c, where most clusters are small,
although a few contain over 100 complexes.

To mitigate potential biases during training, we adopted a cluster-wise sampling strategy.
Traditional uniform sampling within batches could result in the selection of highly similar

1We use the term MG(D) to denote both degraders and non-degraders, as the formation of MGD ternary complexes
can be generalized to non-degraders.


## Page 5

46,797 PDB IDs
including two proteins and one 
small molecule

22,221 PDB IDs 
42,441 complexes
are determined by X-ray structures 
with a resolution of ≤ 3.5 Å and an 
R free value of ≤ 0.26

25,756 complexes
consist of proteins with more than 
3 amino acids, where the small 
molecule shares the same chain 
ID as the proteins

22,303 complexes
are retained where each small 
molecule has at least three contacts 
with both of the two proteins

PROTAC
MG(D)

Training

PROTAC
MG(D)

Training

a
c
b

d
e
f

0
20
40
60
80
100
120
140

Ligand atom number (without hydrogens)

Count

0
100
200
300
400
500
600
Cluster size

Count

Homo sapiens

Bos taurus

Thermosynechococcus vestitus BP

Thermostichus vulcanus
Mus musculus

Escherichia coli

Mycobacterium tuberculosis

Saccharomyces cerevisiae

Cereibacter sphaeroides

Sus scrofa

Rattus norvegicus
Thermus thermophi...
Pseudomonas putida
Chlamydomonas re...

Azotobacter vinel...
Methanothermoba...

synthetic construct

Gallus gallus

HIV-1

Other

Atom Number (exclude Hs)

Fig. 2
TernaryDB construction and visualization. a, The process of collecting and cleaning the ternary
complexes dataset. Initially, a search of ternary structures from the PDB yielded 46,797 PDB IDs, each of which
contains at least two proteins and one small molecule. High-quality PDB IDs were retained based on criteria such
as X-ray crystallography data, resolution, and R-free value. From this subset, 42,441 complexes were extracted,
each comprising just two proteins and one small molecule. These complexes underwent further refinement based
on peptide chain length and the number of contacts. Ultimately, 22,303 complexes met our stringent criteria and
were used to train our model. b, Histogram of the ligand atom number (excluding hydrogens) within the dataset.
c, Histogram of cluster sizes within the dataset according to the protein sequence similarity. d, The distribution
of protein source organisms in the dataset. e, Proteome-wide view of the collected dataset. ESM-1b [38] sequence
embeddings for the two proteins in each complex are calculated and concatenated. This is followed by two-
dimensional (2D) Uniform Manifold Approximation and Projection (UMAP). Similar complexes to PROTACs-
and MG(D)s-involved ternary structures are denoted as red and green square points, respectively. f, Chemical
space covered by the dataset. Morgan fingerprints are converted to 1024-length vectors and visualized through
a 2D UMAP. The points on the map are differentiated and colored by molecular weight (hydrogen excluded).
PROTACs- and MG(D)s-like molecules are highlighted as red and green square points, respectively.

complexes, thereby skewing the training process. Instead, we first randomly sampled a cluster
with equal probability and then selected the representative complex with a 20% likelihood;
otherwise, a random complex from the cluster was chosen. The representative complex was
determined using the MMseqs2 toolkit during clustering. This approach ensures a diverse and
representative sampling of the training data, enhancing the model’s ability to generalize across
complex structures.

The architecture of DeepTernary

DeepTernary is designed to predict the structures of small molecule-induced ternary com-
plexes, such as those formed by PROTACs and MG(D)s-induced E3 ligase with POI complexes.
Unlike existing methods that rely on standard protein-protein docking programs to approxi-
mate the interaction between two proteins—often neglecting the presence of small molecules,
DeepTernary employs a deep neural network to directly learn the intricate dynamics of
protein-protein and protein-ligand interactions within ternary complexes. For predicting
PROTAC-induced complexes, DeepTernary takes as inputs the respective mono forms of the
two protein structures (E3 ligase and target protein) along with docked warheads and anchors
from other PDB entries (unbound structures), in addition to the PROTAC Simplified Molec-
ular Input Line Entry System (SMILES) strings. In the case of MG(D)-induced complexes,
since the unbound structures are hard to find, we adopt the respective in-complex form of
the two protein structures (randomly rotated and transformed) and the corresponding MG(D)
SMILES strings as input.

In general, DeepTernary consists of an encoder and a decoder. The process begins by
generating a random conformation of the small molecule using RDKiT [42] and randomly
displacing the small molecule and protein 2 (p2) away from protein 1 (p1). This serves as


## Page 6

IEGMN
PPPD (Ours)

Decoder Type

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

****

1
2
4
8

Head Number

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

64
128
256

Embedding Dim

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

DockQ

0.6774
0.6874
0.7062

0.1736
0.1924

0.2493

PROTAC
MG(D)

PROTAC
MG(D)
Noise Level

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

1
2
3

1
5
10
20
30
40

Number of Sampled Random Conformations

0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8

DockQ

PROTAC
MG(D)

0%

20%

40%

60%

80%

100%

Accept Rate

a
b
c

d
e

Fig. 3
Effectiveness of DeepTernary designs on PROTAC and MG(D) test benchmarks. All results
are based on test sets comprising 22 PROTAC complexes or 94 MG(D) complexes. Statistical significance was
determined using an independent t-test: * p ≤0.05, ** p ≤0.01, and ***p ≤0.001, similarly hereinafter. a,
Comparison of decoder types: our proposed Pocket Points Decoder outperforms IEGMN in predicting medium-
to high-quality binding poses (DockQ > 0.49). b, Impact of multi-head attention on coordination prediction:
increasing the number of heads results in a slight decrease in DockQ scores. c, Effect of latent embedding
dimension on model performance: larger dimensions yield improved learning, especially for MG(D) complexes.
d, Influence of noise level on model robustness: elevating the noise level from 1 to 2 enhances performance on
both PROTAC and MG(D) benchmarks. e, Effect of number of sampled random conformations: more sampled
conformations lead to higher DockQ scores and acceptance rates (DockQ > 0.23) for PROTACs, while MG(D)
remains largely unaffected.

the starting point for learning the interactions between the two proteins and the ligand. As
illustrated in Fig. 1c, these three monomers are encoded as graphs and processed through an
SE(3)-equivariant encoder. This encoder facilitates the interaction of the encoded entities in a
geometrically consistent manner. Multiple blocks of alternating intra- and inter-graph message
passing are employed to update the coordinates and latent features of the three monomers. To
efficiently capture the symmetry in their interactions, the parameters in the encoders for p1 and
p2 are weight-shared (Methods 4). Following the encoding stage, the conformation of the small
molecule is utilized as the final conformation. The final ternary structure is generated based on
this predicted conformation and pocket points. For PROTACs, the pocket points are derived
from unbound structures, while for MG(D)s, these points are predicted by the proposed query-
based pocket points decoder (PPPD). With these information, we can rigid align the ligand
and p2 back to p1 for form the final structure. The PPPD will also predict an alignment error
for this predicted structure. Notably, benefiting from the Transformer architecture’s inherent
ability to handle variable numbers of input queries without architectural modifications, the
proposed PPPD architecture is unified for both PROTAC and MG(D). This simplifies the
model design and implementation. For PROTACs, only PAE queries are input to the decoder,
while for MG(D)s, both pocket point and PAE queries are used.

Effectiveness of model designs

Based on the binary interaction prediction model [32, 34], we had explored various choices of
model designs and hyper-parameters for DeepTernary. To ensure robust model selection, we
employed a validation set consisting of curated structures that are dissimilar to the training
set and also not in the test set. Model performance on this validation set was assessed using a
simple score calculated as the average of the DockQ scores for the top-ranked prediction and
the best overall prediction (detailed information is provided in Supplementary Section A.1).
We now present the results obtained on the test set.


## Page 7

First, DeepTernary incorporates a ternary inter-graph attention mechanism in the encoder,
enabling it to capture more complex ternary interactions. However, the initial decoder design,
denoted as IEGMN, struggled to effectively translate the encoded information into accurate
binding poses (Fig. 3a). By introducing the newly developed Prompt-based Pocket Points
Decoder (PPPD) (detailed in Methods 4), we significantly enhanced performance, with many
samples achieving medium to high quality (DockQ > 0.49). Additionally, we found that, while
multi-head attention is beneficial in natural language processing, it is less effective for pre-
dicting pocket point coordinates in this context. Specifically, as the number of attention heads
increases, the DockQ score decreases slightly (Fig. 3b). Consequently, we employed single-head
attention in the PPPD to extract coordinates accurately. Transitioning from binary to ternary
interaction prediction posed additional challenges. We discovered that increasing the latent
embedding space improved the model’s capacity to learn complex triplet interactions, particu-
larly for MG(D) complexes, which exhibit greater structural complexity (Fig. 3c). Besides, to
avoid the risk of overfitting, we increased the noise added to both the coordinates and latent
features, from 1 to 2, which improved performance across both PROTAC and MG(D) bench-
marks (Fig. 3d). Nevertheless, adding too much noise (noise level from 2 to 3) will hinder the
performance.

In line with previous studies [24, 27, 36], DeepTernary utilizes RDKit [42] to generate
initial conformations for small molecules, sampling multiple conformations with different seed
numbers during inference. Our ablation studies (Fig. 3e) demonstrated that both the DockQ
score and accept rate (DockQ > 0.23) increased as the number of sampled conformations for
PROTAC grew. Conversely, MG(D) complexes showed little change. This discrepancy can be
attributed to the fact that PROTACs have more atoms and exhibit greater structural flexibility,
while MG(D)s have a smaller conformation space. Based on these findings, we sample 40
initial random conformations for each PROTAC and rank the predicted results using the PAE
score. For MG(D) predictions, we use a single initial conformation to conserve computational
resources.

DeepTernary achieves the highest accuracy in PROTACs-induced
ternary structure prediction

To evaluate our method, we utilized the PROTAC benchmark compiled by Rao et al. [36], which
consists of 22 known PROTAC-induced ternary structures serving as the test set. The unbound
protocol adopted in this benchmark emulates the real-world scenario encountered during drug
discovery, where the experimental structure of the ternary complex is often unavailable. In this
protocol, an unbound complex refers to a protein with a bounded ligand similar to the warhead
or anchor of the PROTAC, but not co-crystallized with the entire PROTAC molecule. To align
with the rational design process of PROTACs, we followed this unbound protocol to evaluate
DeepTernary.

To mitigate data leakage, we excluded any similar protein pairs from the dataset used
to train our model. Unlike previous methods, which rely on human-defined heuristics – such
as manually set thresholds for free energy [19, 22, 27], atom clashes [27–29], or linker ends
distances [27] – to filter ternary conformations, we leveraged deep learning to automatically
capture high-dimensional interactions between PROTACs and proteins. Furthermore, in con-
trast to Drummond et al. [28, 29], our model was trained without any PROTAC-involved
structures and directly evaluated on the PROTAC benchmark, using a zero-shot protocol. This
approach tests the model’s ability to learn general interaction rules applicable to any ternary
structure, not just those induced by PROTACs, offering a stringent measure of how well the
model generalizes from non-PROTAC to PROTAC data.

For a comprehensive comparison, we employed several evaluation metrics, including DockQ
scores, rank of the first prediction achieving a DockQ score greater than 0.23, the percentage
of CAPRI high/medium/acceptable predictions, and the percentage of predictions with RMSD
< 10 ˚A (metrics detailed in Evaluation metrics). As shown in Fig. 4a, b, and c, DeepTernary
consistently produces higher DockQ scores and higher rates of acceptable predictions (both
in terms of High/Medium/Acceptable predictions and < 10 ˚A) compared to other published
methods, including FRODock- and RosettaDock-Based methods [24], BOTCP [36], Method
4 [28], Method 4B [29], PRosettaC [22], and most recently published AlphaFold 3 [43] and Chai-
1 [44]. Specifically, it achieved an average DockQ score of 0.65 across the test set, significantly
outperforming the recently proposed BOTCP [36], which scored 0.44. Although other methods
were only evaluated on subsets of the benchmark, DeepTernary demonstrated superior per-
formance across overlapping tested structures. Notably, as illustrated in the lower portion of


## Page 8

FERDock-

Based

RosettaDock-

Based

Method 4B
DeepTernary

Method

0.0

0.2

0.4

0.6

0.8

1.0

% High/Medium/Acceptable

**

**

PRosettaC
Method 4
Method 4B
DeepTernary
Method

0.0

0.2

0.4

0.6

0.8

1.0

1.2

1.4

% RMSD < 10Å

**

**

***

VHL
CRBN
BIRC2
E3

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

FRODock-Based
BOTCP
RosettaDock-Based
DeepTernary

Chai-1
AlphaFold3FRODock-

Based

BOTCP RosettaDock-

Based

DeepTernary

Acceptable Rank

*

*

Chai-1
AlphaFold3 FRODock-

Based

BOTCP RosettaDock-

Based

DeepTernary

Method

0.0

0.2

0.4

0.6

0.8

1.0

1.2

DockQ

*

***

**

0.0
0.2
0.4
0.6
0.8
1.0
RosettaDock-Based

0.0

0.2

0.4

0.6

0.8

1.0

DeepTernary

DockQ

0.0
0.2
0.4
0.6
0.8
1.0
Method 4B

0.0

0.2

0.4

0.6

0.8

1.0

DeepTernary

% High/Medium/Acceptable

0.0
0.2
0.4
0.6
0.8
1.0
Method 4B

0.0

0.2

0.4

0.6

0.8

1.0

DeepTernary

% RMSD < 10Å

5T35_H_E (VHL - BRD4-2)
DockQ: 0.90

6W7O_C_A (BIRC2 - BTK)
DockQ: 0.93

6W8I_D_A (BIRC2 - BTK)
DockQ: 0.91

DeepTernary Predicted
Co-crystalized

a
c
b

d
e

f

Fig. 4
DeepTernary achieves the highest accuracy in PROTACs-induced ternary structure pre-
diction. a, DeepTernary outperforms existing methods in predicting 22 existing PROTAC-induced ternary
structures regarding the metrics of the DockQ score (top panel) and the first acceptable rank containing at
least one prediction with DockQ ≥0.23 (middle panel). For those that failed to generate a acceptable result, we
manually set the rank value to 41 for a fair comparison. Besides, DeepTernary achieves better DockQ perfor-
mance on most complexes compared to the current best model, the RosettaDock-Based model (bottom panel).
b, DeepTernary outperforms existing methods on the percentage of High/Medium/Acceptable. c, DeepTernary
has a higher potential to generate decent (RMSD < 10 ˚A) results. d, DockQ performance comparison among
different E3 ligases. e, Surface illustration of the predicted structure of PDB ID 5T35. f, Three examples of
predicted ternary structures (teal and orange for the protein and ligand, respectively) overlaid with the exper-
imental structures (gray and green for the protein and ligand). The receptor protein is colored in red, and the
chemical structure diagrams of PROTAC molecules are illustrated at the bottom.

Fig. 4a, DeepTernary surpasses the top-performing RosettaDock-based method [24] for most
testing structures.

PROTAC molecules, with their larger atom counts compared to natural small molecules,
exhibit diverse conformations due to their significant degrees of freedom. To model this flexi-
bility, we employed the RDKit toolkit to generate multiple initial conformations of the ligand
using different random seeds, each of which was input into our model. To estimate the predic-
tion quality, we introduced a predicted aligned error (PAE), allowing us to rank the predicted
results and select the most confident output. With an average rank of 4.06 under 40 seeds
(Fig. 4a, middle panel), DeepTernary reliably generated acceptable predictions (DockQ > 0.23).
In another words, there is generally at least one acceptable prediction within the top four
results. To compare with existing methods, we also calculate the prediction success rate for
each complex based on another two criteria: CAPRI criteria and RMSD < 10 ˚A (Fig. 4b and
c). As we can see, DeepTernary significantly improve the success rate to around 50%, which
means for most of the test complexes, more than half of the predictions are above acceptable


## Page 9

EquiDock
DeepTernary

0.0

0.2

0.4

0.6

0.8

DockQ

****

Group 1
Group 2

0.0

0.2

0.4

0.6

0.8

DockQ

****

****

EquiDock
DeepTernary

PDB ID: 1NSG_A_B

DockQ: 0.56

PDB ID: 5MN0_A_B

DockQ: 0.62

PDB ID: 4JDD_A_B

DockQ: 0.92
PDB ID: 7BQU_A_B

DockQ: 0.38

DeepTernary
Co-crystalized

EquiDock
AlphaFold3

p1

p2

p1
p2

0
10
20
30

Expected Position Error

a
c
b

d
e

DeepTernary Predicted
Co-crystalized

DeepTernary Predicted
Co-crystalized

EquiDock

AlphaFold3

Fig. 5
DeepTernary reaches acceptable accuracy for MG(D)-induced ternary complex structure
prediction. a, Since no MG(D)-induced complex prediction method exists, we compared DeepTernary with
the traditional protein-protein interaction (PPI) prediction method EquiDock. DeepTernary significantly out-
performs EquiDock by precisely modeling ternary interactions. b, Comparison with EquiDock and the recently
realeased AlphaFold3 (AF3) on PDB ID 7BQU. The predicted aligned error (PAE) matrix from AF3 is shown
in the top right corner. AF3 predicts the structure with plDDT values between 50 and 90 but shows low confi-
dence for docking, with PAE values exceeding 20 between p1 and p2. c, Visualization of the predicted ternary
structure for PDB ID 4JDD (Group 2), displayed using both cartoon and surface illustrations. d, Two predicted
results from Group 1. e, Performance comparison across different interaction modes: domain-domain (Group
1) and sequence motif-domain (Group 2).

quality.
Since the ground-truth structure is typically unavailable in practice, distinguishing
between higher- and lower-quality output structures remains challenging without of a reliable
scoring or ranking system. Our DeepTernary addresses this challenge by incorporating a PAE
predictor, where lower PAE values indicate higher confidence in predictions (Fig. A2). The
mean Top-1 DockQ based on PAE reaches up to 0.4 (Tab. A5), surpassing the acceptable cutoff
of 0.23, which enhances its utility in real-world drug discovery applications.

Finally, we examined the performance of DeepTernary across the three distinct E3 lig-
ases present in the 22 benchmark complexes. As shown in Fig. 4d, DeepTernary consistently
achieved desirable DockQ scores across all ligases, highlighting its robustness and generalizabil-
ity. Visual comparisons of the predicted and experimentally determined structures (Fig. 4e–f)
demonstrate that our model can generate high-quality predictions, with DockQ values exceed-
ing 0.9. Notably, for PDB IDs 6W7O and 6W8I, which share the same E3 ligase and POI
pair but differ in their PROTACs, DeepTernary accurately captured the structural differences,
producing predictions aligned with experimental expectations.

DeepTernary reaches acceptable accuracy in MG(D)-induced ternary
complex structure prediction

Molecular glue degraders (MGDs) represent a novel class of TPD drugs, distinct from PRO-
TACs due to their lower molecular weight and alternative MOA. These characteristics often
result in an advantageous starting point for medicinal chemistry optimization, as well as
enhanced drug-like physicochemical properties [45]. Their simplicity in structure further
facilitates later-stage drug development. The rising interest in MGDs has prompted signifi-
cant research efforts and corporate investments focused on this new modality. In particular,
structure-based rational design plays a crucial role in maximizing the chances of successful
drug discovery. For instance, the crystal structure of the β-TrCP, β-catenin, and NRX-1933
ternary complex has been instrumental in developing MGDs with improved mutant selectiv-
ity [46]. Similarly, the discovery of ALV2, a mutant-specific Ikaros degrader, relied on known
crystal structures for guidance [47, 48].

MG(D)s can either stabilize endogenous protein-protein interactions or induce non-native
ones [49]. However, predicting MG(D)-induced ternary complex structures poses a challenge


## Page 10

due to the often weak binding affinity between the small molecule and one of the proteins. With
no existing in silico method specifically designed for MG(D)-induced complexes, we employed
EquiDock [32], a protein-protein docking approach, to test whether weak interactions between
two proteins could approximate MG(D)-induced binding features. Using MG(D)-induced com-
plexes collected by Rui et al.[17] as a test set, we evaluated the models’ performance using
DockQ scores, similar to our approach with PROTAC experiments. The results, shown in
Fig. 5a, reveal that EquiDock achieves an average DockQ score of only 0.04. In contrast,
DeepTernary significantly improves the score to 0.21, demonstrating the advantage of incor-
porating small molecule information and modeling ternary interactions within the model
architecture.

Recently, DeepMind introduced AlphaFold3 (AF3) [43], which is able to predict complexes
involving nearly all molecular types in the PDB, including proteins and small molecules. How-
ever, since the code has not yet been released and the AlphaFold Server does not currently
allow customization of small molecules, we used AF3 solely for protein-protein binding pre-
dictions, as we did with EquiDock. Fig. 5b illustrates a prediction for PDB ID 7BQU, where
AF3, thought better than EquiDock, performs significantly worse than DeepTernary, which
predicts a structure closest to the co-crystallized ground truth (green vs. gray). We show the
PAE matrix in the top right corner. The PAE in AlphaFold, measured in ˚Angstr¨oms (˚A), repre-
sents the expected positional error between two residues in the predicted structure. Typically,
an AF PAE value exceeding 15 ˚A is considered indicative of a less confident prediction. In this
prediction, the PAE values are approximately 20 ˚A, highlights AF3’s lower confidence in this
predicted interaction.

Rui et al. categorized the collected MG(D)-induced complexes into two groups based on the
nature of their protein-protein interface: Group 1 involves domain-domain interactions, where
two proteins bind through well-structured domains (as shown in Fig. 5d), and Group 2 involves
sequence motif-domain interactions, where a protein sequence motif binds to a structured
domain (illustrated in Fig. 5c). Our results indicate that both EquiDock and DeepTernary
perform better on Group 2 complexes than on Group 1, as shown in Fig. 5e. This suggests
that the large, well-folded domains in Group 1 complexes involve more complex binding rules,
which may not be adequately covered by the training set (Supplementary Fig. A3). In contrast,
the interactions involving small recognition motifs in Group 2 are better captured, leading to
improved predictions.

The total buried surface area (BSA) from our predicted structures
strongly correlates with degradation potency

Experimental work by Wurz et al. [19] has demonstrated a strong correlation between the total
buried surface area (BSA) of PROTAC-mediated ternary complexes and the equilibrium dis-
sociation constant (KLP T ) for VHL-PROTAC-SMARCA2. Their findings revealed that BSA
has a negative correlation with ln(KLP T ), while a lower KLP T corresponds to higher degra-
dation potency. In other words, a higher BSA corresponds to higher degradation potency. To
test whether our predicted ternary structures could reflect this relationship, we calculated the
total BSA for the predicted VHL-PROTAC-SMARCA2 complexes ( Fig. 6b). Consistent with
the experimental data, our predictions also show a generally negative correlation between total
BSA and ln(KLP T ), supporting the findings of Wurz et al.

In a separate study, Zorba et al. [50] investigated the effect of PROTAC linker length
on degradation potency, using cereblon (CRBN) as the E3 ligase to induce degradation of
Bruton’s tyrosine kinase (BTK). They synthesized 11 PROTACs with varying linker lengths
(PROTACs 1-11) and found that longer PROTACs (6-11) yield detectable ternary complex
formation via fluorescence resonance energy transfer (FRET) and demonstrated potent cellular
BTK degradation. In contrast, shorter PROTACs (1-4) showed weak or no FRET signals and
were ineffective in cells. PROTAC (5) displayed intermediate behavior.

To further explore the relationship between degradation potency and ternary structure,
we use DeepTernary to predict the ternary structures induced by these 11 PROTACs and
computed their total BSA. The results, illustrated in Fig. 6c, indicate that as linker length
increases, total BSA decreases sharply at first before plateauing. This trend correlates nega-
tively with degradation potency, consistent with the findings from Zorba et al. For the predicted
structures of PROTACs (1-4), severe atom clashes between proteins lead to higher BSAs (left
side of Fig. 6d), which explains their inability to form stable ternary complexes and induce


## Page 11

1
2
3
4
5
6
7
8
9
10
11
PROTAC ID

Linker Length

2
3
4
5
6

ln(K       )
LPT

BSA

PROTAC ID: 1
PROTAC ID: 10

FERDOCK-based protocol

RosettaDock-based refinement

PROTACability

BOTCP

DeepTernary
DeepTernary (w GPU)

DeepTernary
 DeepTernary (w GPU)

Methods

10
0

10
1

10
2

10
3

10
4

Time (s)

PROTAC
MG(D)

100x faster

a
c
b

d
e

Fig. 6
The buried surface area (BSA) based on our predicted structures correlates with degra-
dation potency. a, The BSA of 22 known PROTAC-induced ternary complexes from experimental structures.
b, Correlation analyses between BSA and ln(KLP T ) for the predicted BRD4-VHL ternary complex with vari-
ous PROTACs. The quantity ln(KLP T ) has been proven in previous work [19] to have a positive relationship
with ln(DC50). The red line represents a linear regression of the points. c, Correlation analyses between BSA
and linker length for predicted CRBN-BTK complexes induced by 11 different PROTACs. As PROTAC ID
increases, linker length increases. The red line shows a second-order polynomial regression. Notably, PROTACs
(6-11) are associated with significant cellular knockdown, while shorter PROTACs(1-4) exhibit weak/no degra-
dation capability. d, Predicted ternary structures for CRBN-BTK complexes induced by PROTAC (1) and
PROTAC (10), illustrating severe atom clashes PROTAC (1) and increased flexibility to avoid clashes in PRO-
TAC (10). e, DeepTernary predicts ternary structures approximately 100 times faster than existing methods.

degradation. In contrast, for PROTACs (5-11), the increased linker length allows for more flexi-
bility, reducing atomic clashes (right side of Fig. 6d) and facilitating productive protein-protein
interactions, which correlate with effective degradation.

Although both Wurz et al. and Zorba et al. demonstrated strong correlations between PRO-
TAC degradation potency and factors like BSA and linker length, it remained unclear whether
the observed relationships for VHL-SMARCA2 PROTACs could be generalized to CRBN-BTK
PROTACs. By employing DeepTernary to model the ternary structures and calculate BSA
for all PROTACs whose degradation potency was experimentally validated, we were able to
compare their results and examine these conclusions more thoroughly. In the VHL-SMARCA2
system (Fig. 6b), higher BSA correlates with higher degradation potency (lower log(KLP T )),
whereas in the CRBN-BTK system (Fig. 6c), higher BSA–indicative of shorter linker lengths–
is associated with lower degradation potency, highlighting conflicting trends (Supplementary
Fig. A4).

By synthesizing the findings from both studies (Fig. 6b and c), we conclude that their
conclusions do not inherently conflict. This is because the correlation between BSA and degra-
dation potency appears to be more nuanced than a simple linear correlation. In Fig. 6a, we
analyzed the BSA range for 22 known PROTAC-induced complexes with experimentally deter-
mined structures, highlighting a range of 1,175 to 1,422 ˚A2 (shaded gray). When comparing the
BSA values of PROTAC-induced complexes in Fig. 6b and c, we found that PROTACs tend
to exhibit higher degradation potency when their total BSA falls within the 1,100 to 1,500 ˚A2

range. This suggests that BSA could be a useful metric for virtual screening and inform future
PROTAC design.


## Page 12

DeepTernary is significantly faster than existing methods

Existing methods for predicting ternary structures often require generating numerous candi-
date structures and applying multiple filtering criteria to identify the most viable options.
For instance, Weng et al. utilized a multi-step protocol involving FRODOCK for local dock-
ing, followed by energy scoring with Open Babel Obenergy and AutoDock Vina, and further
refinement using RosettaDock [24]. This approach, while effective, is time-consuming, taking
approximately one hour on an 18-core CPU for the FRODOCK-based process alone, with
RosettaDock-based refinement adding another nine hours. More recent methods like BOTCP,
which employ Bayesian optimization to expedite candidate sampling, have reduced the process
to around two hours [36].

In contrast, DeepTernary introduces a substantial leap in efficiency by leveraging an end-
to-end neural network that embeds learned knowledge directly into its parameters. Unlike
traditional docking-based techniques that rely on iterative candidate generation and refine-
ment, DeepTernary predicts PROTAC ternary structures in a fraction of the time. Using 40
seeds, it can predict a ternary complex in just 12.37 seconds on a 15-core CPU, and as lit-
tle as 6.48 seconds with GPU acceleration. For MG(D) complexes, the process is even faster,
requiring only a single forward pass of the embedded graphs, yielding results in under 1 sec-
ond (Fig. 6e). It is worth noting that this time includes both the model’s forward time and
the data preprocessing time (such as using RDKit to generate initial conformations and file
operations), making it instructive for real-world applications. The model-only forward time is
reported in Supplementary Table A7.

This dramatic improvement in prediction speed has the potential to revolutionize drug dis-
covery by facilitating the rapid in-silico screening of a significantly large number of candidates,
making it feasible to explore a broader range of compounds in less time.

3 Discussion

In this study, we introduced DeepTernary, a novel deep learning framework consisting of an
SE(3)-equivariant graph neural network and a pocket point decoder to predict ternary com-
plex structures induced by PROTACs and MG(D)s. DeepTernary offers a powerful tool for
drug discovery by modeling complex interactions within ternary complexes, enabling the opti-
mization of key drug characteristics such as selectivity and potency. Unlike traditional docking
methods, which rely on predefined strategies, DeepTernary learns the underlying physical-
chemical rules governing ternary complex formation, resulting in both improved prediction
accuracy and significant reductions in computational time. This allows for rapid screening of
PROTAC libraries across different E3 ligases and protein targets, providing structure-guided
insights for drug development. The model’s ability to correlate buried surface area (BSA)
with degradation potency further enhances its utility in designing more potent degraders.
Additionally, DeepTernary excels in predicting low-affinity, transient interactions for MG(D)s,
overcoming limitations of traditional methods and supporting the growing interest in MG(D)s
as therapeutics with distinct mechanisms of action.

While DeepTernary is a significant advance, it shares a common limitation with data-driven
approaches: a dependence on large datasets and a susceptibility to biases from the training set.
Despite we have collected a broad dataset from the PDB, there remain room for improvement.
Expanding the training data and incorporating lower-resolution experimental datasets could
further enhance the model’s accuracy and applicability. Future developments in this direction
will likely extend DeepTernary’s impact, enabling broader application in drug discovery.

In conclusion, DeepTernary offers a fast and accurate approach for predicting ternary com-
plexes, representing a valuable tool in the development of TPD therapeutics. In addition, the
BSA calculated from generated complexes by DeepTernary may offer valuable insights into the
degraded potency, potentially facilitating the structure-guided TPD design. By refining this
framework and integrating additional structural data, we anticipate even greater contributions
to the field of targeted protein degradation.

4 Methods

Data collection and filtering

In our quest to identify potential ternary complexes, We searched the Protein Data Bank
(PDB) to extract potential ternary complexes, applying filters to select structures with at least


## Page 13

two proteins and more than one small molecule. This initial filtration process yielded 46,797
potential PDB entries. Subsequently, the filtered candidates were further refined by selecting
only X-ray structures that met our high-quality standards—specifically, those with a resolution
of 3.5 ˚A or better and an R-free value of 0.26 or lower—thereby refining our dataset to 22,221
PDB IDs.

From these entries, we extracted 42,441 ternary complexes, some of which included multiple
complexes within a single entry, such as assemblies (e.g., 5T35 D A 759 and 5T35 H E 759) and
instances where different ligands interacted with the same protein pair (e.g., 6ZO8 B C LPX
and 6ZO8 B C PTY). To ensure meaningful protein-small molecule interactions, we imposed
additional criteria: the small molecule must share a chain ID with one of the proteins, and the
protein components must meet a minimum length requirement—seven amino acids for PRO-
TACs and three for MG(D)s. This was exemplified by the TRAP motif in PDB ID 4TR9 [51].
Such stringent criteria effectively pruned nearly half of the initial complexes, leaving us with
25,756 viable candidates for further analysis.
In our final step to validate meaningful protein-ligand interactions, we implemented a two-
tiered filtering approach. First, we excluded complexes where the ligand established fewer than
three contacts with the protein, defined as ligand atoms positioned within 4 ˚A of any protein
atom. Second, we removed complexes exhibiting steric clashes, identified as any heavy atom
pair (one from the ligand and one from the protein) separated by less than 2 ˚A. While this
stringent criteria led to the exclusion of some well-characterized PROTACs and MG(D)s, such
as PDB ID 6HAX [52] (Rfree = 0.268), which marginally exceeded our 0.26 threshold, and
PDB ID 6BN7 [53] (chain B ligand clash of 1.97 ˚A), these structures were manually curated
and retained in our database due to their established significance. Additionally, many ligands
in the PDB are crystallization buffers that frequently appear across numerous PDB entries
and are not functionally relevant. To address this, we manually exclude commonly occurring
ligands such as ACT, GOL, PEG, SO4, TRS, XYP, BME, EDO, PG4, and PG5 from the
dataset. The culmination of our efforts resulted in a comprehensive structure collection dataset
comprising 22,303 complexes.

Similarity-based dataset splitting

To mitigate the risk of test data leakage and to prevent model overfitting, we adopted a
similarity-based dataset splitting strategy. This approach was designed to rigorously evaluate
the model’s generalization capabilities by ensuring that training complexes were not similar to
those in the test set.

We utilized MMseqs2 [41], a highly efficient toolkit for sequence clustering, to group proteins
based on a minimum sequence identity threshold of 50%. This involved clustering proteins with
similar sequences. Any cluster containing a test set complex was designated as a test cluster.

To maintain the integrity of our training set, we excluded all complexes within the test
clusters from the training set. Specifically, any protein complex with a sequence similarity
exceeding 50% to any complex in the test set was removed from the training data and treated
as a validation sample. For PROTACs, the 22 known-structure test complexes were clustered
into 7 groups, resulting in the exclusion of 16 test-similar complexes from the training set, all
of which were PROTAC-induced, ensuring the validation set’s relevance.

For MG(D)s, the 94 test complexes clustered into 44 groups, resulting in 182 excluded
training set complexes forming the validation set. This rigorous approach ensured the test set’s
novelty and provided a robust evaluation of the model’s generalization ability. The performance
of DeepTernary on the corresponding validation and test set is detailed in Supplementary
Table A.1.

Featurization

Following the EquiBind approach [34], both the ligand and proteins were encoded as geometric
graphs using the k-nearest neighbor method. In the ligand graph Glig = (Vlig, Elig), each node
(representing an atom) vi ∈Vlig was characterized by atom attributes fi (a list feature of
atomic number, chirality, total degree, formal charge, number of implicit hydrogens, number
of hydrogen, radical electrons, hybridization state, aromaticity, and ring participation) and a
3D position vector xi ∈R3. Edges Elig were defined between atoms within a distance of less
than 4 ˚A, determined by relative Euclidean distances and bond angles. For the protein graphs
Gp1 = (Vp1, Ep1) and Gp2 = (Vp2, Ep2), nodes were defined as amino acid type and edges were
defined similarly as ligand.


## Page 14

For PROTACs, we utilized the known-pocket unbound evaluation protocol from previous
studies [24, 27–29, 36]. This protocol requires prior knowledge of the unbound structures of both
anchor-E3 ligase and warhead-POI binary complexes during inference—a standard practice in
PROTAC discovery. To integrate pocket information into DeepTernary, we introduced pocket
embeddings for graph nodes associated with pockets. These embeddings were integrated into
the node features by summation. During training, pocket node coordinates were replaced with
their actual values (after random rotation and transformation), while during inference, pocket
coordinates from unbound pockets were used. Note that we had ensured that the atom indexes
of the unbound pockets and the candidate complexes were well aligned beforehand.

Model architecture

DeepTernary leverages an SE(3)-equivariant graph neural network along with the attention
mechanism, allowing invariant message passing regarding the atom attributes and equivariant
message passing regarding the atom coordinates. The model accepts inputs in various formats:
the structures of two proteins (E3 ligase and PoI) in PDB or CIF format and the 2D geometry
of the small molecule derived from SMILES strings or files in PDB, mol2, or structure-data
file (SDF) format. Initially, the RDKit tool was employed to generate possible coordinates of
the small molecule. Subsequently, the proteins and the ligand were represented as geometric
graphs. The model is fundamentally composed of two primary components: the encoder and the
decoder. The encoder learns SE(3)-invariant semantic features and SE(3)-equivariant coordi-
nates, while the decoder outputs pocket points and predicted aligned errors. The comprehensive
network architecture of DeepTernary was already depicted in Fig. 1c.

Encoder. After obtaining the graph representations of the proteins and the ligand, we
employed the Independent E(3)-Equivariant Graph Matching Network (IEGMN) [32] by
extending its input from binary complex to ternary complex, in order to facilitate interactions
among triplets. This extension involves a series of layers where node coordinates and feature
embeddings were updated through both in-graph and cross-graph message passing. Unlike the
original IEGMN, our extension allowed for feature updates in a triplet-wise fashion, enabling
each monomer to update its features with the awareness of the other two monomers. The
update of the coordinates maintains E(3)-equivariance, ensuring that the output faithfully mir-
rors any independent rotations and translations applied to the input. Formally, there are totally
M encoder layers and the latent embedding hl+1

i
and node coordinate xl+1

i
at the (l + 1)-th
layer were computed as follows:

(1) Intra-graph message passing, which updates edge and node latent embeddings:

mj→i = ϕe(h(l)

i , h(l)

j , ||x(l)

i
−x(l)

j ||2, fj→i), ∀(i, j) ∈Elig ∪Ep1 ∪Ep2,
(1)

mi =
1
|N(i)|

X

j∈N(i)

mj→i, ∀i ∈Vlig ∪Vp1 ∪Vp2,
(2)

(2) Ternary inter-graph message passing: For the nodes of the ligand, the message from the
nodes of the other two graphs Vp1 ∪Vp2 was computed by

aj→i =

exp(< ϕq(h(l)

i ), ϕk(h(l)

j ) >)
P

j′∈Vp1∪Vp2 exp(< ϕq(h(l)

i ), ϕk(h(l)

j′ ) >)

, ∀i ∈Vlig,
(3)

µj→i = aj→iW h(l)

j , µi =

X

j∈Vp1∪Vp2

µj→i, ∀i ∈Vlig,
(4)

We also derived cross-graph message µi for the nodes Vp1 and Vp2 similar to the above processes.

(3) Calculation of the new node coordinates and embeddings:

x(l+1)

i
= Ψ



x(l)

i
+

X

j∈N(i)

x(l)

i
−x(l)

j
∥x(l)

i
−x(l)

j ∥

ϕx(mj→i)



,
(5)

h(l+1)

i
= (1 −β) · h(l)

i
+ β · ϕh(h(l)

i , mi, µi, fi), ∀i ∈Vlig ∪Vp1 ∪Vp2.
(6)
Here, ϕe, ϕx, ϕh, ϕq, ϕk denote multi-layer perceptrons (MLPs), fj→i and fi represents the
initial edge and node features (prior to processing through the IEGMN layers), separately,


## Page 15

N(i) collects all the neighbors of node i, aj→i indicates the SE(3)-invariant cross-attention
coefficient (ai→i indicates the self-attention coefficient), < · > computes the inner product of
two vectors, W is a learnable matrix that transforms latent embeddings according to the cross-
attention coefficients, Ψ is a function that imposes distance geometric constraints [34], and
β is a trade-off parameter. After the encoder process, the latent embeddings and coordinates
of all nodes across the three graphs were updated to reflect the intricate interactions among
the triple molecules. The predicted coordinates of the ligand (xl

i, ∀i ∈Vlig) were assumed to
represent the ligand’s final conformation within the predicted ternary complex.

Given a ternary complex composed of protein1, a ligand, and protein 2 (p1-lig-p2), the pre-
dicted structure should be invariant to the order of the proteins. In other words, the predicted
structure should be the same regardless of whether the input is (p1-lig-p2) or (p2-lig-p1). To
learn this symmetry, the two protein encoders are share parameters to learn generalize protein
features. During training, the p1 and p2 were randomly swapped for data augmentation.

Decoder. For the prediction of ternary structures, we use the ligand conformation derived
from IEGMN and require two pairs of pocket points to rigidly align the second protein (pro-
tein2) and the ligand with the first protein (protein1), forming a complex. Additionally, the
model must predict the predicted alignment error (PAE) for protein2 to assess the quality
of the prediction. To this end, we designed a Transformer-based decoder to extract neces-
sary information from graph embeddings. We designed two different decoders for MG(D)s and
PROTACs owing to their different MOAs.

Specifically, for MG(D)s, we defined two pairs of pocket points: ( Plig, Pp1→lig) and
(Pp2, Pp1→p2). The first pair represents the pocket points between the ligand and protein1,
where Plig denotes the ligand pocket bound to protein1, and Pp1→lig denotes the protein1
pocket bound to the ligand. Similarly, the second pair represents the pocket points between
protein2 and protein1, with Pp2 denoting the protein2 pocket bond to protein1, and Pp1→p2
denoting the protein1 pocket bond to protein2. Their corresponding queries are matrices
Qlig, Qp1→lig, Qp2, Qp1→p2, each row of which denotes the query of each node. In addition,
we denote the PAE query as qPAE. All these values were initialized randomly and processed
through an N-layer decoder. Each layer requires to compute the attention function, represented
as Attn(Q, K, V ):

Attn(Q, K, V ) = a(Q, K)W V ,
(7)
where Q, K and V represent the querie, key, and value matrices, respectively; a(Q, K) returns
the attention matrix, and its element of the i-th row and j-th column was given by the attention
coefficient aj→i defined in Eq. 3. When Q, K and V become the same, we call it self-attention,
otherwise, we call it cross-attention.

We now introduce how to process the queries Qlig, Qp1→lig, Qp2, Qp1→p2 and qPAE, with
the information of the hidden embeddings obtained from the encoder before. We first conducted
column-wise concatenation:

Q = Qlig∥Qp1→lig∥Qp2∥Qp1→p2∥qPAE,
(8)

where ∥denotes column-wise concatenation. For conciseness, we collect the updated coordinates
and embeddings of the final layer in the encoder over all nodes as X and H henceforth.
Specifically for H we further involved the graph embedding features e in order to distinguish
the graph identity:

H =

 

∥i∈Vlig(hi + elig)



∥

 

∥j∈Vp1(hj + ep1)



∥

 

∥k∈Vp2(hk + ep2)



,
(9)

Then the pocket queries Q were updates with the following attention layer:

Q = Attn(Q, Q, Q),
(10)

Q′ = ϕ(Attn(Q, H, H)),
(11)
H′ = Attn(H, Q′, Q′).
(12)
where ϕ is a learnable MLP. We repeated the above attention layer several times. The final
queries and embeddings were unfolded as:

Q′′ = ϕ(Attn(Q′, H′, H′)).
(13)

Qlig, Qp1→lig, Qp2, Qp1→p2, qPAE = unfold(Q′′),
(14)


## Page 16

Hlig, Hp1, Hp2 = unfold(H′).
(15)

For the pocket coordinates, we first computed the attention values between the queries
of each local pocket and the embeddings of the corresponding global graph. We then derived
the coordinate of each pocket atom as a weighted sum of the coordinates of the entire graph.
Specifically, we computed:

Plig = a(Qlig, Hlig)Xlig,
(16)

Pp1→lig = a(Qp1→lig, Hp1)Xp1,
(17)

Pp2 = a(Qp2, Hp2)Xp2,
(18)

Pp1→p2 = a(Qp1→p2, Hp1)Xp1,
(19)

where the matrices P denote the predicted pocket point coordinates.

The PAE qPAE was estimated using an MLP, reflecting the prediction confidence. Given
the computational intensity of real-time DockQ score calculations, we use the Root-Mean-
Square Deviation (RMSD) between predicted and actual coordinates of protein2 as a training
surrogate for PAE. With predicted pocket points and the ligand conformation, the final ternary
complex structure is assembled, which will be detailed in the next subsection.

For PROTACs, we directly bound the two proteins at their two ends, and designed two
protein-ligand pocket coordinates: ( Pp1, Plig→p1) and (Pp2, Plig→p2), representing pockets of
(E3, anchor) and (POI, warhead), respectively. Different from MG(D)s, these pocket points are
already known from the unbound structures. Thus, without the need of the computations above
derived for MG(D)s, Pp1 and Pp2 were directly taken from the unbound protein structures,
Plig→p1 and Plig→p2 were taken from the predicted ligand coordinates X from the encoder
according to unbounded pocket masks. The decoder predicts the PAE for PROTACs using
the same architecture but with only the PAE query reserved. In other words, we conducted
Equation 10 - Equation 13 by setting Q = qPAE.

Transformation to generate the final output

Considering the different modes of action of PROTAC and MGD, we adopted two slightly
different ways to construct the final complex structure. PROTAC molecules comprise three
elements: the anchor, warhead, and connecting linker. The anchor and warhead are typically
selected from known bounded ligands to E3 ligase and the POI, respectively. This selection
facilitates rational design, leveraging existing unbound binding data between the anchor and
E3, as well as the warhead and PoI, to construct the complex structure. Following this process,
the PROTAC was first aligned with the unbound pocket of E3 (protein1) based on the predicted
pocket points for the anchor. The linker and warhead coordinates were determined according
to the conformation of the PROTAC. Subsequently, the coordinates of POI (protein2) were
determined by aligning its unbound structures to the aligned warhead positions according to
predicted protein2 pocket points:

Rlig, tlig = kabsch(Plig→p1, Pp1),
(20)

xlig = (Rlig x⊤

lig)⊤+ tlig,
(21)

P

′
lig→p2 = (R P ⊤

lig→p2)⊤+ t,
(22)

Rp2, tp2 = kabsch(Pp2, P

′
lig→p2),
(23)

xp2 = (Rp2 x⊤

p2)⊤+ tp2,
(24)

where kabsch denotes the Kabsch algorithm [54], ⊤denotes matrix transpose.

A more direct alignment approach was employed for MG(D)s. Both the ligand and pro-
tein2 were aligned directly to protein1. This was achieved by predicting the pocket points
of interaction between protein1 and the ligand, as well as between protein1 and protein2.
The decoder’s predicted pocket points facilitated the alignment of the ligand and protein2 to
protein1, resulting in the final ternary complex structure:

Rlig, tlig = kabsch(Plig, Pp1→lig),
(25)

xlig = (Rlig x⊤

lig)⊤+ tlig,
(26)


## Page 17

Rp2, tp2 = kabsch(Pp2, Pp1→p2),
(27)

xp2 = (Rp2 x⊤

p2)⊤+ tp2.
(28)

Training and inference

During the training process, protein structures were derived from bound structures and ligand
conformations generated by the RDKit toolkit [42]. For each training ligand, we pre-generated
a pool of 50 random conformations. In each training iteration, protein1 or protein2 was ran-
domly fixed, while the other protein and a randomly selected conformation from the ligand’s
50-conformation pool were subjected to random rotations and translations from their orig-
inal positions. Coordinates were normalized before being input into the model to stabilize
the training process, with random noise added to graph features and coordinates to avoid
overfitting.

The model was trained with six losses to guide it towards generating accurate outputs. The
total loss is formulated as follows:

L = Llig + Lkabsch lig + Lot1 + Lot2 + Lintersection + LPAE.
(29)

where Llig indicated the mean squared error (MSE) loss between the predicted and ground-
truth ligand coordinates, and Lkabsch lig denoted the MSE loss after rigid alignment of the
predicted ligand to the ground truth using Kabsch algorithm [54]. Lot1 and Lot2 corresponded
to the optimal transport loss [55] between the predicted pocket points and target pocket coor-
dinates. Lintersection represented intersection punishment between proteins and the ligand, and
LPAE indicated the predicted aligned error of protein2, calculated using the L1 loss between
the predicted and ground-truth RMSD of protein2.

During inference, unbound structures were used for PROTACs and bound structures for
MG(D)s. The initial ligand conformations were randomly generated by RDKit using different
seeds. For each PROTAC, we performed 40 samplings and ranked results based on predicted
PAEs. For molecule glues, only one sampling was performed due to their limited atom numbers
and conformational flexibility.

DeepTernary contains 16.73 million parameters and was trained for about five hours on
four Nvidia V100 GPUs.

Calculation of buried surface area (BSA)

The buried surface area (BSA) was calculated using ChimeraX [56]. Ligands were assigned
unique sequence IDs separate from proteins, and the ”interfaces” command computed solvent-
accessible surface area (SASA) for each interacting chain pair within the complexes. The total
BSA was determined by summing the SASA values across all protein-protein and protein-
ligand interactions. The BSA represented in Fig. 6 is the average BSA of the top five most
confident (lower PAE) predictions from DeepTernary.

Evaluation metrics

Following recent studies [24, 36], we adopted the DockQ score [57] as a quantitative measure
to evaluate prediction quality. The DockQ score is a continuous metric ranging from 0 to 1,
calculated based on three components: Fnat, LRMS, and iRMS. Fnat represents the fraction
of native contacts maintained in the predicted complexes. LRMS is the root mean square
deviation (RMSD) between backbone atoms after aligning the predicted structure to the native
one. iRMS is the RMSD of backbone atoms of the interface residues. By integrating these three
criteria, the DockQ score provides a comprehensive measure of prediction quality, with higher
values indicating higher-quality predictions.

To compare our methods with previously published approaches, we also calculated the
fraction of acceptable predictions and compared them with other methods. It is worth noting
that the criteria for an “acceptable” prediction vary across different studies. We categorize
these criteria as follows:

DockQ > 0.23: This threshold indicates a quality prediction based on the DockQ scoring
system.

CAPRI criterion: Derived from the Critical Assessment of Predicted Interaction
(CAPRI) [58]), predictions are classified into High, Medium, or Acceptable. This criterion has
been employed to assess the quality of PROTAC-induced complex predictions, as used by
Drummond et al. [29].


## Page 18

RMSD < 10 ˚
A: This criterion involves calculating the Cα RMSD and is commonly
used as the upper limit for an ”acceptable” pose in protein-protein docking contexts. It is
straightforward and easy to compute.

For PROTACs, given the model’s generation of multiple predictions from varying initial
conformations, we employ the Acceptable Rank metric, following existing methods. This
metric is determined by sorting predictions based on their predicted alignment error (PAE)
and identifying the rank of the first prediction achieving a DockQ score greater than 0.23.

By applying these metrics, we ensure a robust evaluation of our model’s performance in
predicting ternary complex structures.

Declarations

Data Available.
The TernaryDB complex list and cluster results of the training dataset,
training log, pre-trained model weights and unbound structures for PROTAC testing are avail-
able at https://github.com/youqingxiaozhua/DeepTernary. The complex crystal structures are
downloaded from the Protein Data Bank at https://www.rcsb.org/. Source Data are provided
with this paper.

Code Available.
Codes for running DeepTernary have been released on GitHub and are
free for academic, personal, and commercial use at https://github.com/youqingxiaozhua/
DeepTernary.

Acknowledgements.
We thank Yaoyukun Jiang and Yifan Sun for their helpful discussions.
W.H. was jointly supported by the following projects: the National Science and Technology
Major Project under Grant 2020AAA0107300, the National Natural Science Foundation of
China (No. 62376276); Beijing Nova Program (No. 20230484278); Beijing Outstanding Young
Scientist Program (No. BJJWZYJH012019100020098); the Fundamental Research Funds for
the Central Universities, and the Research Funds of Renmin University of China (23XNKJ19);
Public Computing Cloud, Renmin University of China. W.D. was supported by the Whitcome
fellowship and the Molecular Biology Interdepartmental graduate program at UCLA. J.A.W.
was supported by National Institutes of Health grant R01 GM089778 and GM112763 and the
David Geffen School of Medicine at UCLA. Computational facilities were provided by the UTS
Interactive High-Performance Computer Cluster.

Author contributions statement.
F.X., W.H., and W.D. conceived the study. F.X. wrote
the code and trained the model. F.X. and M.Z. collected and processed training data. All
authors contributed to the analysis of the results. F.X., M.Z., W.H., and W.D. wrote the
manuscript. W.H., Y.Y., and W.D. offered supervision throughout the project. All authors
revised the manuscript.

References

[1] Sakamoto, K.M., Kim, K.B., Kumagai, A., Mercurio, F., Crews, C.M., Deshaies, R.J.:

Protacs: Chimeric molecules that target proteins to the skp1–cullin–f box complex for
ubiquitination and degradation. Proceedings of the National Academy of Sciences 98(15),
8554–8559 (2001)

[2] Buckley, D.L., Van Molle, I., Gareiss, P.C., Tae, H.S., Michel, J., Noblin, D.J., Jorgensen,

W.L., Ciulli, A., Crews, C.M.: Targeting the von hippel–lindau e3 ubiquitin ligase using
small molecules to disrupt the vhl/hif-1α interaction. Journal of the American Chemical
Society 134(10), 4465–4468 (2012)

[3] Kr¨onke, J., Udeshi, N.D., Narla, A., Grauman, P., Hurst, S.N., McConkey, M., Svinkina,

T., Heckl, D., Comer, E., Li, X., et al.: Lenalidomide causes selective degradation of ikzf1
and ikzf3 in multiple myeloma cells. Science 343(6168), 301–305 (2014)

[4] Banik, S.M., Pedram, K., Wisnovsky, S., Ahn, G., Riley, N.M., Bertozzi, C.R.: Lysosome-

targeting chimaeras for degradation of extracellular proteins. Nature 584(7820), 291–297
(2020)

[5] Hammoudeh, D.I., Follis, A.V., Prochownik, E.V., Metallo, S.J.: Multiple independent

binding sites for small-molecule inhibitors on the oncoprotein c-myc. Journal of the
American Chemical Society 131(21), 7390–7401 (2009)


## Page 19

[6] Mittal, P., Roberts, C.W.: The swi/snf complex in cancer—biology, biomarkers and

therapy. Nature reviews Clinical oncology 17(7), 435–448 (2020)

[7] Willis, M.S., Homeister, J.W., Rosson, G.B., Annayev, Y., Holley, D., Holly, S.P., Madden,

V.J., Godfrey, V., Parise, L.V., Bultman, S.J.: Functional redundancy of swi/snf catalytic
subunits in maintaining vascular endothelial cells in the adult heart. Circulation research
111(5), 111–122 (2012)

[8] Huang, H.-T., Dobrovolsky, D., Paulk, J., Yang, G., Weisberg, E.L., Doctor, Z.M., Buck-

ley, D.L., Cho, J.-H., Ko, E., Jang, J., et al.: A chemoproteomic approach to query the
degradable kinome using a multi-kinase degrader. Cell chemical biology 25(1), 88–99
(2018)

[9] Vairy, S., Tran, T.H.: Ikzf1 alterations in acute lymphoblastic leukemia: the good, the bad

and the ugly. Blood Reviews 44, 100677 (2020)

[10] Henley, M.J., Koehler, A.N.: Advances in targeting ‘undruggable’transcription factors with

small molecules. Nature Reviews Drug Discovery 20(9), 669–688 (2021)

[11] Teng, M., Gray, N.S.: The rise of degrader drugs. Cell Chemical Biology (2023)

[12] Bondeson, D.P., Mares, A., Smith, I.E., Ko, E., Campos, S., Miah, A.H., Mulholland, K.E.,

Routly, N., Buckley, D.L., Gustafson, J.L., et al.: Catalytic in vivo protein knockdown by
small-molecule protacs. Nature chemical biology 11(8), 611–617 (2015)

[13] Barouch-Bentov, R., Sauer, K.: Mechanisms of drug resistance in kinases. Expert opinion

on investigational drugs 20(2), 153–208 (2011)

[14] Li, K., Crews, C.M.: Protacs: past, present and future. Chemical Society Reviews 51(12),

5214–5236 (2022)

[15] Chirnomas, D., Hornberger, K.R., Crews, C.M.: Protein degraders enter the clinic—a new

approach to cancer therapy. Nature Reviews Clinical Oncology 20(4), 265–278 (2023)

[16] Schreiber, S.L.: The rise of molecular glues. Cell 184(1), 3–9 (2021)

[17] Rui, H., Ashton, K.S., Min, J., Wang, C., Potts, P.R.: Protein–protein interfaces in molec-

ular glue-induced ternary complexes: classification, characterization, and prediction. RSC
Chemical Biology 4(3), 192–215 (2023)

[18] Yamamoto, J., Ito, T., Yamaguchi, Y., Handa, H.: Discovery of crbn as a target of thalido-

mide: a breakthrough for progress in the development of protein degraders. Chemical
Society Reviews 51(15), 6234–6250 (2022)

[19] Wurz, R.P., Rui, H., Dellamaggiore, K., Ghimire-Rijal, S., Choi, K., Smither, K.,

Amegadzie, A., Chen, N., Li, X., Banerjee, A., Chen, Q., Mohl, D., Vaish, A.: Affinity and
cooperativity modulate ternary complex formation to drive targeted protein degradation.
Nature Communications 14(1), 4177 (2023) https://doi.org/10.1038/s41467-023-39904-5

[20] Oleinikovas, V., Gainza, P., Ryckmans, T., Fasching, B., Thom¨a, N.H.: From Thalido-

mide to Rational Molecular Glue Design for Targeted Protein Degradation. Annual
Review of Pharmacology and Toxicology 64, 291–312 (2024) https://doi.org/10.1146/
annurev-pharmtox-022123-104147

[21] Duhovny, D., Nussinov, R., Wolfson, H.J.: Efficient unbound docking of rigid molecules. In:

Algorithms in Bioinformatics: Second International Workshop, WABI 2002 Rome, Italy,
September 17–21, 2002 Proceedings 2, pp. 185–200 (2002). Springer

[22] Zaidman, D., Prilusky, J., London, N.: PRosettaC: Rosetta Based Modeling of PROTAC

Mediated Ternary Complexes. Journal of Chemical Information and Modeling 60(10),
4894–4903 (2020) https://doi.org/10.1021/acs.jcim.0c00589

[23] Garzon, J.I., Lop´ez-Blanco, J.R., Pons, C., Kovacs, J., Abagyan, R., Fernandez-Recio,


## Page 20

J., Chacon, P.: FRODOCK: a new approach for fast rotational protein–protein docking.
Bioinformatics 25(19), 2544–2551 (2009) https://doi.org/10.1093/bioinformatics/btp447

[24] Weng, G., Li, D., Kang, Y., Hou, T.: Integrative Modeling of PROTAC-Mediated Ternary

Complexes. Journal of Medicinal Chemistry 64(21), 16271–16281 (2021) https://doi.org/
10.1021/acs.jmedchem.1c01576

[25] Lyskov, S., Gray, J.J.: The rosettadock server for local protein–protein docking. Nucleic

Acids Research 36, 233–238 (2008)

[26] Kozakov, D., Brenke, R., Comeau, S.R., Vajda, S.: Piper: An fft-based protein docking

program with pairwise potentials. Proteins: Structure 65 (2006)

[27] Ignatov, M., Jindal, A., Kotelnikov, S., Beglov, D., Posternak, G., Tang, X., Maisonneuve,

P., Poda, G., Batey, R.A., Sicheri, F., Whitty, A., Tonge, P.J., Vajda, S., Kozakov, D.: High
Accuracy Prediction of PROTAC Complex Structures. Journal of the American Chemical
Society 145(13), 7123–7135 (2023) https://doi.org/10.1021/jacs.2c09387

[28] Drummond, M.L., Williams, C.I.: In Silico Modeling of PROTAC-Mediated Ternary Com-

plexes: Validation and Application. Journal of Chemical Information and Modeling 59(4),
1634–1644 (2019) https://doi.org/10.1021/acs.jcim.8b00872

[29] Drummond, M.L., Henry, A., Li, H., Williams, C.I.: Improved Accuracy for Modeling

PROTAC-Mediated Ternary Complex Formation and Targeted Protein Degradation via
New In Silico Methodologies. Journal of Chemical Information and Modeling 60(10),
5234–5254 (2020) https://doi.org/10.1021/acs.jcim.0c00897

[30] Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunya-

suvunakool, K., Bates, R., ˇZ´ıdek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl,
S.A.A., Ballard, A.J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J.,
Back, T., Petersen, S., Reiman, D., Clancy, E., Zielinski, M., Steinegger, M., Pacholska,
M., Berghammer, T., Bodenstein, S., Silver, D., Vinyals, O., Senior, A.W., Kavukcuoglu,
K., Kohli, P., Hassabis, D.: Highly accurate protein structure prediction with AlphaFold.
Nature 596(7873), 583–589 (2021) https://doi.org/10.1038/s41586-021-03819-2

[31] Baek, M., DiMaio, F., Anishchenko, I., Dauparas, J., Ovchinnikov, S., Lee, G.R., Wang, J.,

Cong, Q., Kinch, L.N., Schaeffer, R.D., Mill´an, C., Park, H., Adams, C., Glassman, C.R.,
DeGiovanni, A., Pereira, J.H., Rodrigues, A.V., van Dijk, A.A., Ebrecht, A.C., Opperman,
D.J., Sagmeister, T., Buhlheller, C., Pavkov-Keller, T., Rathinaswamy, M.K., Dalwadi,
U., Yip, C.K., Burke, J.E., Garcia, K.C., Grishin, N.V., Adams, P.D., Read, R.J., Baker,
D.: Accurate prediction of protein structures and interactions using a three-track neural
network. Science 373(6557), 871–876 (2021) https://doi.org/10.1126/science.abj8754

[32] Ganea, O.-E., Huang, X., Bunne, C., Bian, Y., Barzilay, R., Jaakkola, T., Krause, A.:

Independent SE(3)-Equivariant Models for End-to-End Rigid Protein Docking. In: ICLR
(2022). https://doi.org/10.48550/arXiv.2111.07786

[33] Evans, R., O’Neill, M., Pritzel, A., Antropova, N., Senior, A., Green, T., ˇZ´ıdek, A., Bates,

R., Blackwell, S., Yim, J., Ronneberger, O., Bodenstein, S., Zielinski, M., Bridgland, A.,
Potapenko, A., Cowie, A., Tunyasuvunakool, K., Jain, R., Clancy, E., Kohli, P., Jumper,
J., Hassabis, D.: Protein Complex Prediction with AlphaFold-Multimer. bioRxiv (2021).
https://doi.org/10.1101/2021.10.04.463034

[34] St¨ark, H., Ganea, O., Pattanaik, L., Barzilay, D.R., Jaakkola, T.: EquiBind: Geometric

Deep Learning for Drug Binding Structure Prediction. In: ICML, pp. 20503–20521 (2022)

[35] Corso, G., St¨ark, H., Jing, B., Barzilay, R., Jaakkola, T.S.: DiffDock: Diffusion Steps,

Twists, and Turns for Molecular Docking. In: The Eleventh International Conference on
Learning Representations (2022)

[36] Rao, A., Tunjic, T.M., Brunsteiner, M., M¨uller, M., Fooladi, H., Gasbarri, C., Weber, N.:

Bayesian optimization for ternary complex prediction (BOTCP). Artificial Intelligence in


## Page 21

the Life Sciences 3, 100072 (2023) https://doi.org/10.1016/j.ailsci.2023.100072

[37] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S.: A Comprehensive Survey on

Graph Neural Networks. IEEE Transactions on Neural Networks and Learning Systems
32(1), 4–24 (2021) https://doi.org/10.1109/TNNLS.2020.2978386

[38] Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C.L.,

Ma, J., Fergus, R.: Biological structure and function emerge from scaling unsupervised
learning to 250 million protein sequences. Proceedings of the National Academy of Sciences
118(15), 2016239118 (2021) https://doi.org/10.1073/pnas.2016239118

[39] Berman, H.M., Westbrook, J., Feng, Z., Gilliland, G., Bhat, T.N., Weissig, H., Shindyalov,

I.N., Bourne, P.E.: The Protein Data Bank. Nucleic Acids Research 28(1), 235–242 (2000)
https://doi.org/10.1093/nar/28.1.235

[40] Lu, W., Zhang, J., Huang, W., Zhang, Z., Jia, X., Wang, Z., Shi, L., Li, C., Wolynes,

P.G., Zheng, S.: DynamicBind: Predicting ligand-specific protein-ligand complex structure
with a deep equivariant generative model. Nature Communications 15(1), 1071 (2024)
https://doi.org/10.1038/s41467-024-45461-2

[41] Steinegger, M., S¨oding, J.: MMseqs2 enables sensitive protein sequence searching for the

analysis of massive data sets. Nature Biotechnology 35(11), 1026–1028 (2017) https://
doi.org/10.1038/nbt.3988

[42] Landrum, G., et al.: Rdkit: A software suite for cheminformatics, computational chemistry,

and predictive modeling. Greg Landrum 8(31.10), 5281 (2013)

[43] Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O.,

Willmore, L., Ballard, A.J., Bambrick, J., Bodenstein, S.W., Evans, D.A., Hung, C.-C.,
O’Neill, M., Reiman, D., Tunyasuvunakool, K., Wu, Z., ˇZemgulyt˙e, A., Arvaniti, E., Beat-
tie, C., Bertolli, O., Bridgland, A., Cherepanov, A., Congreve, M., Cowen-Rivers, A.I.,
Cowie, A., Figurnov, M., Fuchs, F.B., Gladman, H., Jain, R., Khan, Y.A., Low, C.M.R.,
Perlin, K., Potapenko, A., Savy, P., Singh, S., Stecula, A., Thillaisundaram, A., Tong, C.,
Yakneen, S., Zhong, E.D., Zielinski, M., ˇZ´ıdek, A., Bapst, V., Kohli, P., Jaderberg, M.,
Hassabis, D., Jumper, J.M.: Accurate structure prediction of biomolecular interactions
with AlphaFold 3. Nature, 1–3 (2024) https://doi.org/10.1038/s41586-024-07487-w

[44] Discovery, C., Boitreaud, J., Dent, J., McPartlon, M., Meier, J., Reis, V., Rogozhnikov,

A., Wu, K.: Chai-1: Decoding the Molecular Interactions of Life. bioRxiv (2024). https:
//doi.org/10.1101/2024.10.10.615955

[45] Tsai, J.M., Nowak, R.P., Ebert, B.L., Fischer, E.S.: Targeted protein degradation: from

mechanisms to clinic. Nature Reviews Molecular Cell Biology, 1–18 (2024)

[46] Simonetta, K.R., Taygerly, J., Boyle, K., Basham, S.E., Padovani, C., Lou, Y., Cum-

mins, T.J., Yung, S.L., von Soly, S.K., Kayser, F., Kuriyan, J., Rape, M., Cardozo,
M., Gallop, M.A., Bence, N.F., Barsanti, P.A., Saha, A.: Prospective discovery of small
molecule enhancers of an E3 ligase-substrate interaction. Nature Communications 10(1),
1402 (2019) https://doi.org/10.1038/s41467-019-09358-9

[47] Matyskiela, M.E., Lu, G., Ito, T., Pagarigan, B., Lu, C.-C., Miller, K., Fang, W., Wang,

N.-Y., Nguyen, D., Houston, J., Carmel, G., Tran, T., Riley, M., Nosaka, L., Lander,
G.C., Gaidarova, S., Xu, S., Ruchelman, A.L., Handa, H., Carmichael, J., Daniel, T.O.,
Cathers, B.E., Lopez-Girona, A., Chamberlain, P.P.: A novel cereblon modulator recruits
GSPT1 to the CRL4CRBN ubiquitin ligase. Nature 535(7611), 252–257 (2016) https:
//doi.org/10.1038/nature18611

[48] Wang, E.S., Verano, A.L., Nowak, R.P., Yuan, J.C., Donovan, K.A., Eleuteri, N.A., Yue,

H., Ngo, K.H., Lizotte, P.H., Gokhale, P.C., Gray, N.S., Fischer, E.S.: Acute pharmacolog-
ical degradation of Helios destabilizes regulatory T cells. Nature Chemical Biology 17(6),
711–717 (2021) https://doi.org/10.1038/s41589-021-00802-w


## Page 22

[49] Dewey, J.A., Delalande, C., Azizi, S.-A., Lu, V., Antonopoulos, D., Babnigg, G.: Molecular

Glue Discovery: Current and Future Approaches. Journal of Medicinal Chemistry 66(14),
9278–9296 (2023) https://doi.org/10.1021/acs.jmedchem.3c00449

[50] Zorba, A., Nguyen, C., Xu, Y., Starr, J., Borzilleri, K., Smith, J., Zhu, H., Farley, K.A.,

Ding, W., Schiemer, J., Feng, X., Chang, J.S., Uccello, D.P., Young, J.A., Garcia-Irrizary,
C.N., Czabaniuk, L., Schuff, B., Oliver, R., Montgomery, J., Hayward, M.M., Coe, J.,
Chen, J., Niosi, M., Luthra, S., Shah, J.C., El-Kattan, A., Qiu, X., West, G.M., Noe,
M.C., Shanmugasundaram, V., Gilbert, A.M., Brown, M.F., Calabrese, M.F.: Delineating
the role of cooperativity in the design of potent PROTACs for BTK. Proceedings of the
National Academy of Sciences 115(31), 7285–7292 (2018) https://doi.org/10.1073/pnas.
1803662115

[51] Nemetski, S.M., Cardozo, T.J., Bosch, G., Weltzer, R., O’Malley, K., Ejigiri, I., Kumar,

K.A., Buscaglia, C.A., Nussenzweig, V., Sinnis, P., et al.: Inhibition by stabilization:
targeting the plasmodium falciparum aldolase–trap complex. Malaria journal 14, 1–18
(2015)

[52] Farnaby, W., Koegl, M., Roy, M.J., Whitworth, C., Diers, E., Trainor, N., Zollman,

D., Steurer, S., Karolyi-Oezguer, J., Riedmueller, C., Gmaschitz, T., Wachter, J., Dank,
C., Galant, M., Sharps, B., Rumpel, K., Traxler, E., Gerstberger, T., Schnitzer, R.,
Petermann, O., Greb, P., Weinstabl, H., Bader, G., Zoephel, A., Weiss-Puxbaum, A.,
Ehrenh¨ofer-W¨olfer, K., W¨ohrle, S., Boehmelt, G., Rinnenthal, J., Arnhof, H., Wiechens,
N., Wu, M., Owen-Hughes, T., Ettmayer, P., Pearson, M., McConnell, D.B., Ciulli, A.: Baf
complex vulnerabilities in cancer demonstrated via structure-based protac design. Nature
Chemical Biology 15(7), 672–680 (2019) https://doi.org/10.1038/s41589-019-0294-6

[53] Nowak, R.P., DeAngelo, S.L., Buckley, D., He, Z., Donovan, K.A., An, J., Safaee, N.,

Jedrychowski, M.P., Ponthier, C.M., Ishoey, M., et al.: Plasticity in binding confers selec-
tivity in ligand-induced protein degradation. Nature chemical biology 14(7), 706–714
(2018)

[54] Kabsch, W.: A solution for the best rotation to relate two sets of vectors. Acta Crystallo-

graphica Section A: Crystal Physics, Diffraction, Theoretical and General Crystallography
32(5), 922–923 (1976)

[55] Flamary, R., Courty, N., Gramfort, A., Alaya, M.Z., Boisbunon, A., Chambon, S., Chapel,

L., Corenflos, A., Fatras, K., Fournier, N., et al.: Pot: Python optimal transport. Journal
of Machine Learning Research 22(78), 1–8 (2021)

[56] Meng, E.C., Goddard, T.D., Pettersen, E.F., Couch, G.S., Pearson, Z.J., Morris, J.H.,

Ferrin, T.E.: Ucsf chimerax: Tools for structure building and analysis. Protein Science
32(11) (2023) https://doi.org/10.1002/pro.4792

[57] Basu, S., Wallner, B.: DockQ: A Quality Measure for Protein-Protein Docking Models.

PLOS ONE 11(8), 0161879 (2016) https://doi.org/10.1371/journal.pone.0161879

[58] M´endez, R., Leplae, R., Lensink, M.F., Wodak, S.J.: Assessment of capri predictions in

rounds 3–5 shows progress in docking procedures. Proteins 60(2), 150–169 (2005) https:
//doi.org/10.1002/prot.20551


## Page 23

Appendix A
Supplementary Materials

A.1
Model selection on the validation set

For model selection during training, we evaluated model performance on a validation set com-
prising curated structures dissimilar to the training set but also not in the test set. After
training, a single scalar Validation Performance Score (VPS) is calculated via a arithmetic
mean of the DockQ scores for the top-ranked prediction based on PAE (Dtop-1) and the best
overall prediction (Dbest):

VPS = (Dtop-1 + Dbest)/2
(A1)
where D represents the DockQ score between the predicted and ground-truth crystal struc-
tures. Despite its simple definition, VPS assesses both structure and confidence accuracy.

The model’s performance on the validation set is summarized in Supplementary Tab. A.1.
Unlike the PROTAC, the performance on MG(D) plateaued with increasing seed numbers.
To balance performance and computational cost, we used a single seed for MG(D) evaluation,
resulting in VPS = Dtop-1 = Dbest.

Supplementary Table A1 Results on the validation set for model selection.
The highest VPS for each hyper-parameter is marked in bold. ‘=’ denotes
Dbest = Dtop-1 for MG(D)s, because only one seed is used for evaluation.

PROTAC
MG(D)
Mean VPS
Dtop-1
Dbest
VPS
Dtop-1
Dbest
VPS
dim64
29.99
68.17
49.08
10.97
=
10.97
30.03
dim128
31.49
66.60
49.05
10.14
=
10.14
29.59
dim256
39.97
68.40
54.19
22.56
=
22.56
38.37
noise1
25.32
70.19
47.76
22.01
=
22.01
34.88
noise2
39.97
68.40
54.19
22.56
=
22.56
38.37
noise3
41.43
67.46
54.45
19.72
=
19.72
37.08
seed1
33.93
33.93
33.93
22.56
=
22.56
28.25
seed5
29.09
45.58
37.34
22.71
23.3
23.30
30.32
seed10
37.02
56.86
46.94
22.83
23.63
23.63
35.29
seed20
39.42
61.12
50.27
22.89
23.82
23.82
37.05
seed30
38.93
68.19
53.56
22.86
23.89
23.89
38.73
seed40
39.97
68.40
54.19
22.73
23.94
23.94
39.06

A.2
Reproducibility study

To assess the reproducibility of our method, we retrained the model five times using different
random seeds (0, 1, 2, 3, and 4) for both model initialization and data sampling. Tab. A.3
summarizes the results, including mean, standard deviation, and range for each evaluated
metric. Notably, the standard deviations are 3.50 for PROTAC top-1 DockQ, 1.36 for PROTAC
best DockQ, and 1.47 for MG(D) top-1 DockQ. The largest variation was observed in the
PROTAC top-1 DockQ metric, which ranged from 28.02 to 38.05. We also found that the
RDKit version has a significant impact on performance; the results in Tab. A.3 were obtained
using rdkit==2023.9.3.

A.3
Ligand pose accuracy

In the main manuscript, we reported the DockQ score for ternary complexes following estab-
lished studies. Although the DockQ score only assesses protein-protein docking performance,
it depends on accurate ligand positioning for the ternary complexes. For example, as shown in
the manuscript Fig. 4f, complexes with the same protein pairs (e.g. PDB IDs 6W7O and 6W8I)
can exhibit notable structural differences due to the variations in the PROTAC molecules. This
observation is further supported by our analysis of BSAs in the BRD4-VHL and CRBN-BTK
systems.

To directly evaluate ligand docking accuracy, we calculated the RMSD between the pre-
dicted ligand positions and their corresponding crystal structures. We report both the RMSD
for the top-ranked prediction (based on predicted alignment error, PAE) and the best RMSD
observed across all generated conformations. For PROTACs, the mean top-ranked RMSD is
3.43 ˚A, with 43% of predictions achieving an RMSD below 2 ˚A. By contrast, MG(D) pre-
dictions exhibit a significantly higher mean top-ranked RMSD of 13.14 ˚A, with only 1.9% of


## Page 24

Supplementary Table A2 Reproducibility study of
model performance across five training runs with
different random seeds. Reported are the mean, standard
deviation, minimum, and maximum values for key
performance metrics.

PROTAC
MGD

Top-1 DockQ
Best DockQ
Top-1 DockQ
Seed0
29.54
67.21
22.34
Seed1
28.02
65.73
18.89
Seed2
38.05
64.74
22.61
Seed3
31.22
66.23
21.85
Seed4
33.5
63.22
19.85
Mean
32.07
65.43
21.11
Std
3.50
1.36
1.47
Min
28.02
63.22
18.89
Max
38.05
67.21
22.61

a
b

RMSD
RMSD < 2Å (%)

RMSD

MG(D)

RMSD < 2Å (%)

Top-1

RMSD
RMSD < 2Å (%)

1.5

2.0

2.5

3.0

3.5

4.0

4.5

5.0

RMSD

PROTAC

RMSD < 2Å (%)

Top-1
Best

Supplementary Figure A1 Ligand pose accuracy on the test set of (a) PROTAC and (b) MG(D).

predictions falling below 2 ˚A. These results underscore the substantial challenge in accurately
predicting MGD binding poses and highlight the need for further methodological development
in this area.

A.4
Comparison with AlphaFold3 and Chai-1

During the submission period of this work, AlphaFold3 (AF3) [43] and Chai-1 [44] were pro-
posed to tackle joint structure prediction, with both models capable of handling complexes
involving proteins and small molecules. However, their training datasets differ from ours—AF3
was trained on PDB entries released before September 30, 2021, while Chai-1 used a cutoff of
January 12, 2021—resulting in the inclusion of some of our test samples. To ensure a fair com-
parison, we filtered out PROTAC and MG(D) test clusters that were present in the training
data of AF3 and Chai-1.

For
PROTACs,
three
test
complexes
(7JTO L B VKA,
7JTP L A X6M,
and
7Q2J C D 8KH) were not part of AF3’s training set. We then performed MSA and template
searches (to align with our unbound prediction setting as much as possible) and generated 40
predictions for each complex using different random seeds. The best DockQ scores for these
three complexes were 0.12, 0.25, and 0.48 with AF3, compared to 0.56, 0.67, and 0.53 with our
model (Tab. A3), indicating a clear decrease in performance of AF3. It is also worth noting
that AF3 was trained on other 19 test complexes whereas our model never seen PROTAC-like
structures during training. Furthermore, one complex (7KHH C D WEP) was not included
in Chai-1’s training data; although Chai-1 achieved a high DockQ score of up to 0.81 on this
complex, its performance on other PROTAC complexes was inferior.

For MG(D)s, all test set clusters were included in AF3’s training data—either directly or
through highly similar samples. As a result, AF3 achieved a high mean DockQ score of 0.52
on the MG(D) test set, based on one sampled random ligand conformation. There was only
one MG(D) complex 7LRD B A X5M that is not trained by Chai-1. For this complex, Chai-1
produced a DockQ score of just 0.009 compared to 0.014 from our method.


## Page 25

Supplementary Table A3 Comparison of DockQ with Chai-1 and
AlphaFold3 (AF3) on the PROTAC test set. Our DeepTernary illustrates
great generalization to unseen PROTAC and MG(D). – indicates that
AF3 is trained on this complex cluster.

Type
Complex ID
Chai-1
AF3
DeepTernary

PROTAC

7JTP L A X6M
0.40
0.25
0.67
7JTO L B VKA
0.04
0.12
0.56
7Q2J C D 8KH
0.06
0.48
0.53
7KHH C D WEP
0.81
-
0.38

MG(D)
7LRD B A X5
0.009
-
0.014

Supplementary Table A4 Comparison of SmRMSD on
PROTACs. None indicates the item failed to generate an result.

PDB ID
Ignatov et al. [27]
Ours

SmRMSD
Best Rank
SmRMSD
Best Rank

5T35 D A
2.01
1
0.61
3
5T35 H E
0.70
1
6BN7 B C
2.19
3
1.02
5
6BN8
2.21
1
6BN9
1.35
1
6BNB
1.63
5
6BOY B C
5.24
3
2.94
4
6HAX B A
1.56
2
1.83
3
6HAX F E
1.68
3
6HAY B A
1.29
5
1.60
2
6HAY F E
1.35
5
6HR2 B A
1.52
6
1.27
4
6HR2 F E
1.68
3
6XHC
None
None
6SIS D A
0.67
5
6SIS H E
0.84
2
6W7O C A
2.70
2
6W7O D B
1.12
2
6W8I F C
3.67
3
6W8I D A
1.34
2
6W8I E B
1.62
4
6ZHC A D
4.21
3
7JTO L B
2.62
5
7JTP L A
1.86
6
1.34
5
7KHH C D
2.41
4
4.63
3
7KHH 2
7Q2J C D
2.59
1
7PI4
1.97
4.00
Mean
2.12
3.36
1.91
3.18

These results indicate that despite the greater resources used to train Chai-1 and AF3,
their poor performance on unseen PROTAC and MG(D) complexes highlights the superior
generalization capability of our model.

A.5
PAE can be used as a confidence score for screening

Fig. A2 illustrates the correlation between DockQ scores and PAE values for both intra-
and inter-complex predictions of PROTACs and MG(D)s. Overall, our results reveal a
strong negative correlation—complexes with lower PAE scores tend to exhibit higher DockQ
scores, reflecting more accurate predictions. Although several cases (e.g., 6BN7 B C RN3,
6BOY B C RN6, 6HAX B A FWZ, 6HAY F E FX8, 6W7O C A TL7, and 6ZHC A D QL8)
do not show a clear trend, the overall pattern supports the use of PAE scores as a reliable
confidence metric and an effective filter for selecting high-quality predictions in drug discovery
applications.


## Page 26

1
2
3
4
5
6
7
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

5T35_D_A_759

1.5
2.0
2.5
3.0
3.5
4.0
4.5
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

5T35_H_E_759

1.6
1.7
1.8
1.9
2.0
2.1
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6BN7_B_C_RN3

1.4
1.5
1.6
1.7
1.8
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6BOY_B_C_RN6

2.5
3.0
3.5
4.0
4.5
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6HAX_B_A_FWZ

2.0
2.5
3.0
3.5
4.0
4.5
5.0
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6HAX_F_E_FWZ

2.0
2.5
3.0
3.5
4.0
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6HAY_B_A_FX8

2.0
2.5
3.0
3.5
4.0
4.5
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6HAY_F_E_FX8

2.0
2.5
3.0
3.5
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6HR2_B_A_FWZ

2.00
2.25
2.50
2.75
3.00
3.25
3.50
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6HR2_F_E_FWZ

1.5
2.0
2.5
3.0
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6SIS_D_A_LFE

1.5
2.0
2.5
3.0
3.5
4.0
4.5
5.0
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6SIS_H_E_LFE

4
6
8
10
12
14
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6W7O_C_A_TL7

4
6
8
10
12
14
16
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6W7O_D_B_TL7

4
5
6
7
8
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6W8I_F_C_TKY

4
6
8
10
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6W8I_D_A_TKY

2.5
5.0
7.5
10.0
12.5
15.0
17.5
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6W8I_E_B_TKY

4.40
4.45
4.50
4.55
4.60
4.65
4.70
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

6ZHC_A_D_QL8

3
4
5
6
7
8
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

7JTO_L_B_VKA

2.00
2.25
2.50
2.75
3.00
3.25
3.50
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

7JTP_L_A_X6M

1.8
2.0
2.2
2.4
2.6
2.8
3.0
3.2
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

7KHH_C_D_WEP

3.0
3.5
4.0
4.5
5.0
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

7Q2J_C_D_8KH

2
4
6
8
10
12
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

Best Result

1.5
2.0
2.5
3.0
3.5
4.0
4.5
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

Top1 Result

2
4
6
8
10
12
14
16
PAE

0.0

0.2

0.4

0.6

0.8

1.0

DockQ

Top1 Result

a

c
b
d

Supplementary Figure A2 Correlation between PAE and DockQ scores for predicted ternary
complexes. a, Intra-complex correlation for PROTACs. Scatter plots illustrate the relationship between PAE
and DockQ scores across 40 initial conformations for each PROTAC test complex. Each point represents a
single conformation. Most complexes display a negative correlation, indicating that lower PAE values generally
correspond to higher DockQ scores. This suggests that PAE can serve as a useful indicator of prediction
accuracy within a given complex. b, Scatter plot for the best-predicted conformation (i.e., the one with the
highest DockQ) for each PROTAC test complex. The plot demonstrates a clear trend: complexes with PAE
scores below 4 tend to have higher DockQ scores (> 0.5), further supporting the use of PAE as a confidence
metric. c, Across-Complex Correlation for top-1 PROTAC predictions (i.e., the predictions with the lowest
PAE). Despite some false positives, the overall trend remains negatively correlated. d, Correlation for MG(D)
predictions. Similar to PROTACs, a clear negative correlation is observed, with lower PAE values associated
with higher DockQ scores, suggesting that PAE is also an effective confidence metric for MG(D) predictions.

Supplementary Table A5 Top-1
DockQ scores with different
numbers of sampled random
conformations on the PROTAC test
set. The results are tested on the
same checkpoint.

Sample Number
Top-1 DockQ

1
0.33
10
0.37
20
0.40
40
0.40


## Page 27

0
200
400
600
800
1000
1200
1400

Protein length

Frequency

Protein length distribution in the training set

0
200
400
600
800
1000
1200
1400
Protein 1 length

Protein 2 length

a
b

Supplementary Figure A3 Protein length distribution in the MG(D) training set. a, The joint
distribution of sequence lengths for interacting chain pairs. b, The overall distribution of sequence lengths across
all chains. The training set includes many proteins with sequence lengths between 200 to 600 residues, alongside
a notable fraction of shorter proteins (20 - 40 residues). Additionally, the joint distribution reveals that the
most common complex involves a longer protein interacting with a shorter protein–mirroring the characteristics
of Group 2 complexes in the test set.

0
2
4
6
8
10
ln(DC50)

BSA

Source

Zorba et al.
Wurz et al.

training set
test set

10000

BSA

a
c
b

Supplementary Figure A4 Buried surface area (BSA) analysis of PROTAC-induced ternary
complexes. a, BSA distribution in the training dataset, showing a peak around 1000 ˚A2. b, BSA distribu-
tion for known PROTAC-induced complexes. c, Correlation between BSA and ln(DC50) for BRD4-VHL and
CRBN-BTK complexes, revealing an optimal BSA range for high degradation potential, rather than a linear
relationship. The distinct BSA distribution in the training set compared to known complexes indicates the
model learns generalizable principles beyond training set bias, suggesting its ability to predict degradation
potential from predicted structures.

Supplementary Table A6 Hyperparameters of
the model.

Hyperparameter
PROTAC
MG(D)

feature dim
256
256
encoder depth
8
8
decoder depth
1
4
number of pocket points
-
40
noise initial
2
2
batch size
64
64
optimizer
AdamW
AdamW
learning rate
1e-4
1e-4
weight decay
1e-4
1e-4
gradient clip
9
9
number of epochs
1000
1000


## Page 28

Supplementary Figure A5
Predicted structure of 7BQU A B from AlphaFold3, shown with pLDDT scores
from two viewing angles. The Predicted Aligned Error (PAE) is displayed on the right.

Supplementary Table A7
DeepTernary Forward Pass Time.
The execution times reported in the
main manuscript include both the
model’s forward pass time and the
time required for data preprocessing.
This table specifically presents the
forward pass time to allow for a more
detailed analysis of model
performance. *For PROTAC
calculations, 40 conformations were
used for each test sample.

Time (s)
w/o GPU
w/ GPU

PROTAC*
4.79
1.85
MG(D)
0.15
0.05


