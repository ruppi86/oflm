Ref. Ares(2021)7378458 - 30/11/2021

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Horizon 2020

Deliverable D5.2
Report on preliminary characterisation and modelling

Date of preparation: 2021/11/30

Revision: 1

Start date of project: 2019/12/01 Duration: 36 months

Project coordinator: UWE

Classiﬁcation: public

Partners:
lead: CITA

contribution: CITA, Mogu

Project website:

http://fungar.eu/

H2020-FETopen-2019

Deliverable D5.2

Page 1 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

DELIVERABLE SUMMARY SHEET

Grant agreement number:

858132

Project acronym:

FUNGAR

Deliverable No:

Deliverable D5.2

Due date:

M24

Delivery date:

2021/11/30

Name:

Report on preliminary characterisation and modelling

Description:

In this deliverable, we report on activities and results related to
T5.2 Preliminary Characterisation and Modelling as deﬁned in
the scope of works. This includes preliminary characterisation of
mechanical properties of mycelium-based composites (MBC) in
compression, tension and bending according to relevant standard
tests. Characterisation and modelling of mechanical properties
of Kagome weaves is also reported, with particular attention on
examining diﬀerent material compositions, including hybrids of
synthetic and organic materials. Thermal characterisation is also
conducted on MBC panel materials, and then extended to make
preliminary characterisations of a notional cavity wall construc-
tion. The mechanical and thermal characterisation work presen-
ted here is contextualised against the state-of-the-art, with novel
contributions clearly identiﬁed. These preliminary characterisa-
tion results provide the basis for numerical models which will be
integrated in digital design workﬂows (T5.3 Design rules for fungal
architecture) to facilitate the speciﬁcation and prediction of be-
haviour of novel architectural designs from MBC, and setting the
stage for the construction of an ambitious 1:1 demonstrator (T5.4
Persistent modelling).

Partners owning:

CITA

Partners contributed:

CITA, Mogu

Made available to:

public

Page 2 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Table of contents

1 Introduction

2 Mycelium-Based Composite model

2.1 Enzymatic activity in ligninolytic fungi . . . . . . . . . . . . . . . . . . . . . . . .
2.2 Lignocellulosic substrates
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.3 Water in fungal decay . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.4 Mycelium mechanical properties
. . . . . . . . . . . . . . . . . . . . . . . . . . .
2.5 Substrate mechanical properties . . . . . . . . . . . . . . . . . . . . . . . . . . . .
2.6 Composite mechanical properties . . . . . . . . . . . . . . . . . . . . . . . . . . .

3 Materials and methods

. . . . . . . . . . . . . . . . . . . . .
3.1 CITA cultivation method: mechanical tests
3.2 CITA cultivation method: thermal test . . . . . . . . . . . . . . . . . . . . . . . .
3.3 Mogu cultivation method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.4 Chemical analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.5 Compression series . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.6 Flexion series . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.7 Tension series . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.8 Simulation model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.9 Thermal series

4 Chemical characterisation

5 Mechanical characterisation

5.1 Compression: eﬀect of particle size & reinforcements . . . . . . . . . . . . . . . .
5.2 Compression: eﬀect of species . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5.3 Flexion: eﬀect of reinforcements . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5.4 Tension: eﬀect of particle size . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5.5 Kagome structural analysis
5.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

6 Thermal characterisation

6.1 Material focused U-value measurement results . . . . . . . . . . . . . . . . . . . .
6.2 Assembly focused U-value predictions
. . . . . . . . . . . . . . . . . . . . . . . .
6.3 Assembly focused U-value measurement results . . . . . . . . . . . . . . . . . . .
6.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

7 Conclusions and perspectives

References

Appendices

4

7
7
9
12
15
16
17

19
19
23
24
24
25
26
26
27
28

33

36
36
45
46
49
52
55

64
64
64
64
70

71

75

89

Deliverable D5.2

Page 3 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

1 Introduction

Mycelium-Based Composites (MBC) are being investigated in design and materials engineering
by leveraging the saprotrophic lifestyle of ligninolytic fungi, taking inspiration in the XIXth to
early XXth century method of fungal strain transfer by lignocellulosic solid-state cultivation
[34]. Because MBC cultivation protocols can be based on virtually any substrate containing
organic polymers such as lignin, hemicellulose, and cellulose, and as they instrumentalise a range
of widely available basidiomycota, this class of composite shows potential in obtaining viable
products for a variety of uses. Furthermore, MBC conform to circular economy production
principles, appear to be biodegradable [118], and are assumed to have a low environmental
impact in regards to Life-Cycle Assessment (LCA) – although no peer-reviewed studies have
been published to date on this aspect. Lignocellulosic substrates cover a variety of aggregate
geometries and chemical proﬁles, from industrial grade dusts and particles to supplies of irregular
shavings, from grain husks to straws; this variety of supplies has led to the emergence of a rich
craft in MBC production. However, this poses a challenge in systematically understanding
the behaviour of this new class of materials. We argue that rationalising and systematising
approaches to analysing their complexity is necessary to actualise their potential and facilitate
market readiness.
As composites, MBC are made of a matrix phase that corresponds to a continuous and foam-
like or elastomer-like mycelium, and a dispersed phase composed of particles and/or ﬁbres.
Additionally, surfacic or volumetric components can be used to compose the substrate of a
composite. Fungal decay is functionalised as a binding method between elements of the dispersed
phase. From this generic deﬁnition we can appreciate the sheer breadth of the design space, and
we can identify the main material characteristics for deﬁning the cultivation protocol and specify
composite properties:

• Fungal species enzyme array,

• Substrate chemical proﬁle,

• Substrate aggregate characteristics,

• Substrate density and composition,

• Substrate wetting characteristics.

The resulting composite performances depend on the solid-state fermentation (SSF) model that
is adopted. While tacit knowledge of MBC craft is suﬃcient for reaching a satisfactory out-
put, the explicitation and modelling of the driving parameters of an MBC system can foster an
engineering practice for it, thus extending the reach and eﬀectiveness of material research and
improving market readiness. 3D printing strategies of substrate forming for fungal colonisation
are not studied in this report as this technique was not investigated in the context of Fungal
Architectures. The inner structure of a typical MBC is presented in Fig.1.

We start by presenting a review of the MBC model from ligninolytic fungi enzymatic activities
perspective (section 2.1), and proceed to describe the biochemistry of lignocellulosic substrates
while correlating it to fungi enzymatic abilities (section 2.5). Building upon studies of fungal
niches in the wild and their parasitic or saprotrophic behaviour in trees, we come to under-
stand the hydrolitic abilities of wood in relation to their chemical composition and evolutionary
stage as a primary parameter for predicting fungal colonisation (section 2.3). Furthermore, in
describing the behaviour of MBC from this perspective, we review studies having shown clear

Page 4 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

correlation between the mycelium mechanical properties and their cultivation substrates chemical
composition (section 2.4). While these properties have been demonstrated to be modiﬁable to
result in foam-like or elastomer-like behaviour, the contribution of substrate aggregates nature,
size, distribution, and shape are the principal contributors of strength and stiﬀness (section 2.5).
Outstanding mechanical results from the MBC state-of-the-art are then discussed with regards
to these insights, in an eﬀort to describe the composite behaviour. We identify three main
substrate-based exploration strategies for the MBC system: densiﬁcation (by dense packing,
cold or hot-pressing), composition (by introducing structuring elements), and supplementation
(targeting mycelium properties, and based on chemical tuning of the substrate)(section 2.6).
We are reporting on the experimental investigation of the eﬀect of composition over the mechan-
ical behaviour in compression and ﬂexion (sections 5.1 and 5.3). Mogu contributed to this report
by supplying a series of specimens for the ﬂexural characterisation series. Because the eﬀect of
substrate particles size has not yet been investigated, we tested its impact in compression and
tension (sections 5.1 and 5.4). With a standard protocol, we tested a change in strains on the
best performing granulate size (section 5.2). Similarly, and as a way to verify the impact of
the nature of lignins on white-rot fungi colonisation from a performative perspective, we tested
in compression specimens of pine wood cultivated G. lucidum (section 5.2). In the context of
the extended compression series, the principal substrate and composition materials have been
qualiﬁed by Fourier-Transform Infrared (FTIR) spectrometry (section 4). The decay of G. lu-
cidum on beech wood is also quantiﬁed by FTIR. Additionally, and in the continuity of the
works developed in D5.1, we present a structural analysis of a kagome gridshell supporting a
wet mycelium composite load. In this study we report on the eﬀectiveness of designing hybrid
kagome gridshells, made of bamboo and carbon ﬁbre members (section 5.5). Finally, we evaluate
the impact of substrate nature and aggregate geometry on thermal performances with varying
thicknesses. Moreover, we report on the evaluation of two insulation buildups (section 6). Future
research perspectives for the continuation of the Fungal Architectures project and beyond are
presented in the concluding section.

Deliverable D5.2

Page 5 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 1: Typical inner structure of a MBC.

Page 6 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

2 Mycelium-Based Composite model

2.1 Enzymatic activity in ligninolytic fungi

Trametes spp., Ganoderma spp., and Pleurotus spp. are among the most frequently cited famil-
ies in MBC design [9]. Schizophyllum commune is a less investigated species but ﬁnds a growing
interest [5], especially because it is one of the few mushroom-forming fungi for which genes have
been inactivated by homologous recombination [87]. This strategy has proved useful for genetic
studies and functional genomics [106]. Gene inactivation articulates the principles of homolog-
ous recombination for replacing an existing autologous gene with a designed heterogenous gene.
Furthermore, the recombinant DNA constructs of S. commune has been reported to express
in other mushroom-forming fungi, such as with Pycnoporus cinnabarinus, which supports this
species importance for fungal biology [87, 2]. Irpex lacteus has been used previously in MBC
development [138], or even Agaricus bisporus [123], and Fomes fomentarius [84]. Other species
in punctual use have been reported [9].
Wood-rot fungi can be classiﬁed by their decay activity, which cover three main modes: white-
rot (speciﬁed as selective deligniﬁcation, and simultaneous rot), brown-rot, and soft-rot. Across
the surveyed species, all of them are white-rot, ligninolytic fungi. They have the enzymatic
ability to decompose all three principal polymer classes found in wood:
lignin, hemicellulose,
and cellulose, but most often degrade primarily lignin and hemicellulose such as reported for
G. lucidum [26]. In such case, they qualify as selective deligniﬁers, as opposed to simultaneous
decomposers where all three polymer classes endure similar decay rates. Brown-rot fungi uses a
two-step oxidative-enzymatic mechanism for the breakdown of cellulose and hemicellulose [19].
Their lignin degradation abilities are relatively little investigated yet, and although brown-rot
fungi is a polyphyletic group evolved from at least seven white-rot lineages [43], they reportedly
lack more than 60 % of the genes known to be involved in white rot. Brown-rot species degrade
cellulose and hemicellulose at a higher rate than white-rot ones in laboratory isolation [23]. This
decay mode is most often encountered in coniferous trees. Soft-rot fungi commonly share the
ability to demethylate. This mode is associated with ascomycota and asexual forms of dica-
ryomycota, and is distinctive for its tunnelling decay pattern through lignin layers to cellulose
and hemicellulose richer areas [111]. Lignicolous fungi can beneﬁt from a variety of enzymatic
strategies to ultimately fully decompose lignocellulosic substrates. The environmental conditions
for enzyme activation help deﬁne the species niche and their subsequent distribution. The prin-
cipal wood-rot enzyme families are presented in Table 1.
Lignin linkages result from radical reactions, contrarily to polysaccharides polymerisation result-
ing from water removal. Lignin therefore does not involve hydrolytic enzymes, and cannot be
used as a source of carbon or energy for most wood-rot fungi. Its decomposition serves primarily
for the fungus to gain access to hemicellulose, pectin, and cellulose compounds. Lignin depoly-
merisation is mostly initiated by aromatic ring oxidation by peroxidases (by use of H2O2 or
R-OOH) [62], while other aromatic compounds in the cell structure are mostly catabolised by
monooxygenases and dioxygenases. Peroxidases are majoritarily heme proteins, such as lignin
peroxidase (LiP) and manganese peroxidase (MnP). LiP are known to oxidise phenolic aromatic
substrates along with various nonphenolic lignin compounds, and other organic compounds with
a high redox potential. MnP leads to H2O2 oxidising Mn2+ to Mn3+, an oxidising compound
to monomeric phenols. Mn3+ chelates permeate through wood cells and decay lignin selectively.
These chelates cannot oxidise nonphenolic compounds, but radicals formed by Mn3+ oxidation
can in turn oxidise benzyl alcohols and other diarylpropane structures. MnP can also peroxidise
lipids [131]. Versatile peroxidase (VP), additionally to MnP and LiP, are enzymes that are Mn2+
speciﬁc but that can oxidise phenolic and nonphenolic compounds in its absence (as with LiP).

Deliverable D5.2

Page 7 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

They are involved in oxidation of high redox potential aromatic substrates, thus reducing Mn2+-
independent oxidation but not aﬀecting Mn2+-dependent one [127]. Mn2+-independent oxidation
capabilities for phenols, small dye compounds, and amines has also been described for MnP [62],
and it is predicted from sequencing that similar hybrid MnP may be present in Trametes spp.,
Pleurotus spp., and A. bisporus among other species [79, 62]. Hydrogen peroxide, necessary
for LiP and MnP activities, is the product of fungal enzymatic activity. A peroxidase family
with wide substrate speciﬁcity, active at pH of approximately 3, has been identiﬁed in fungi:
dye-oxidising peroxidase (DyP). Additionally to oxidising peroxidase substrates, it can perform
for anthraquinones. Laccases then are a very versatile family of enzymes: they are found in
nearly all fungi including ones without lignin degrading abilities [12] and perform functions ran-
ging from synthesis of melanin and other pigments, to conidia and fruiting bodies formation,
and lignin decomposition indeed [36]. These belong to a multicopper enzyme family and cata-
lyse the removal of an electron from phenolic hydroxylic groups to form phenoxy radicals which
polymerise then via radical coupling. Ultimately, the reactions lead to the presence of oxidised
quinones and couples oligomeric products. Laccases also have a wide substrate speciﬁcity and
can act upon an extended range of phenols, diphenols, aminophenols, and aromatic compounds.
When their catalytic reaction is accompanied by demethylation it may lead to ring cleavage. Ty-
ing lignin decomposition to cellulose and hemicellulose decay, cellobiose dehydrogenase (CDH)
has the ability to oxidise cellobiose, cellodextrins, and lactose [139]. Hemicellulose degrading
enzymes can have the ability to hydrolyse various polymers, such as (cid:12)-1,4-galactosidase that
degrades xyloglucan, xylan, galactomannan, and pectin [73]. More speciﬁc xyloglucanases, xy-
lanases, mannan-degrading enzymes, and pectinases can be found in all fungi to a various extent.
Typically, pectin can be hydrolysed by polygalacturonidases, but also cleaved by (cid:12)-elimination
by pectin and pectate lyases. Cellulose hydrolysis by fungal enzymes employs glyco-hydrolases
(GH) for glycosic bonds cleaving. Among GH families, cellobiohydrolase I (CBH I) is the major
cellulase being produced in white and soft rot; it binds to crystalline cellulose and hydrolyses it
into cellobiose. CBH II, found in all but brown-rot fungi, binds to amorphous cellulose and sep-
arates cellobiose from the non-reducing end of cellulose using acid catalysis of the (cid:12)-1,4-glycosic
bond. Endoglucanases are a major enzyme group able to hydrolyse cellulose polymer from within
the molecule rather than its end. The variety encountered in fungi refers to four GH families:
GH5, GH7, GH12, and GH45. GH5 endoglucanase is a less studied family as per fungi. Because
it belongs to the same GH family, GH7 endoglucanase displays a similar mechanism to CBH
I, with the diﬀerence that it can hydrolyse from the middle of cellulose molecules thanks to its
structure. GH12 endoglucanase lacks a carbohydrate-binding module (CBM), they are therefore
unable to degrade crystalline cellulose but can hydrolyse it in its amorphous form. GH45 endo-
glucanase beneﬁts from a CBM. (cid:12)-glucosidases, produced by all fungi, belong to GH1 and GH3
families. GH1 (cid:12)-glucosidases are able to cleave soluble (cid:12)-linked oligosaccharides from up to nine
glucose residues chains and aglycone-linked (cid:12)-glucosides, and can be competitively inhibited by
the products of CDH activity. GH3 (cid:12)-glucosidases can remove single glucosyl residues from the
non-reducing ends of oligo- and polysaccharides such as (cid:12)-D-glucans, (cid:12)-1,3-D-glucans, (cid:12)-1,4-D-
glucans, to name a few [64]. Lytic polysaccharide monooxygenases (LPMO) have been shown to
enhance cellulose degradation [57, 28].
Carbohydrate-active enzymes (CAZymes) have been classiﬁed in six principal families: Auxili-
ary Activities (AA), Glycoside Hydrolase (GH), Glycosyltransferase (GT), Polysaccharide Lyase
(PL), Carbohydrate Esterase (CE), and Carbohydrate-Binding Module (CBM). We summarised
the genome distribution of T. versicolor, P. ostreatus PC15, and G. lucidum G0119 according to
this nomenclature in Table 2. The G. lucidum species beneﬁts from the widest known CAZymes
array with 565 genes having been assigned to these functions [141, 26]. This species is a well
known versatile fungus capable of selective deligniﬁcation – we conﬁrm this experimentally in

Page 8 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

section 4. The actualisation of decay strategies in selective deligniﬁcation capable fungi has been
correlated to wood species, lignin contents, temperature, pH, and moisture content [44, 92, 112,
1, 15].

2.2 Lignocellulosic substrates

Lignin is the most eﬀective barrier to fungal decay as it prevents migration of low molecular
weight diﬀusible agents for decomposing cellulose and hemicellulose. The nature of lignins in
presence is of considerable inﬂuence on their sensitivity to enzymatic activity. Lignins are poly-
mers of phenylpropene units, they comprise: guaiacyl, syringyl, and p-hydroxyphenyl units.
Gymnospermous woods (conifers) are almost exclusively composed of guaiacyl monomers, while
angiospermous trees (hardwood) contain approximately equal shares of guaiacyl and syringyl
monomers [132]. Similarly, diﬀerent distributions of hemicellulose compounds are found in gym-
nospermous trees (where galactoglucomannan, glucomannan, and arabinoglucuronxylan are the
principal ones) and in angiospermous trees (where arabinogalactan, xyloglucan, and various glu-
cans are the principal compounds) [111]. Pectins can be accounted for too, and are primarily
made of (cid:12)-1,4-D-galacturonase acid units and their methyl esters, interrupted locally by 1,2-
linked L-rhamnose units [53]. And the primary constituent of wood is cellulose, constituted by
repeating glucose units joined by hydroxyl linkages. Cellulose represents about 50 % of the wood
cell wall, lignin 25 %, hemicellulose 20 to 25 %, and pectin 1 to 4 %. Broad-leaf plants such
as hemp are often in use in MBC practice. Their tow lignocellulosic compounds ratio C:HC:L
is close to 15:2:1, when their shives display a ratio closer to 2:1:1. The cultivation period and
weather has been reported to induce diﬀerences in non-cellulosic composition [129].
The variety of wood structures and chemical compositions pairs with various decay strategies
driven from the set of fungi CAZymes expression. These wood speciﬁcs can be understood from
their evolutionary history and subsequent specialised cell contents. Principle wood structures
are presented in Fig.3. Wood cells can be made of various specialised cells: tracheids, xylem ray
parenchyma, axial parenchyma, and vessels. Tracheids are found down to the ﬁrst evolutionary
stage of trees in gymnospermous woods, where they perform both water transport and structural
stiﬀness. Gymnospermous woods typically lack vessels and rarely have axial parenchyma, their
tracheids to xylem ray parenchyma ratio is most often in the vicinity of 9:1. They contain prin-
cipally guaiacyl lignin, a product of coniferyl alcohol [40]. In more recent angiospermous woods,
water is transported by vessels while tracheids perform structurally. The second evolutionary
stage of water transport system in trees can be exempliﬁed by European beech (Fagus sylvat-
ica) and birch (Betula pendula). Their xylem includes vessels along with tracheids, the former
being the only distinct location of guaiacyl lignin. Tracheids and parenchyma contain a fraction
of syringyl lignin spread across the cell wall layers, making this secondary evolutionary wood
structure attractive for white-rot fungi [111]. The third evolutionary stage can be illustrated by
European oak (Quercus robur ), for which the xylem is ring-porous containing early and late wood
vessels in bands within a thin-walled tracheids matrix. The strength of the wood comes from
libriform ﬁbres containing higher syringyl lignin, while tracheids have a higher guaiacyl content
[108]. The latest evolutionary stage is exempliﬁed by sycamore (Acer pseudoplanatus) and ash
(Fraxinus spp.), where syringyl dominates [109]. Vessels are dedicated to water transport here
thanks to the presence of highly specialised cell functions, and are jacketed in living libriform
ﬁbres that are included in dead ones. Guaiacyl lignin has been reported to have a higher resist-
ance to fungal decay [111], which has been corroborated by phylogenetic and experimental study
showing that catalytic tryptophan presence in more recently evolved peroxidases (located in LiP
and VP families) was more eﬃcient at oxidising angiospermous woods while others performed
better in gymnospermous ones [10]. Early stage parenchyma decomposition and a resistance

Deliverable D5.2

Page 9 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

l
e
d
o
m
n
i
n
g
i
l

c
i
l
o
n
e
h
p
n
o
n

.
s
d
n
u
o
p
m
o
c

0
.
5

–

0
.
4

C

°

0
5

–

5
3

,
l
o
n
e
h
p
i
d

,
l
o
n
e
h
P

n
o
i
t
a
d
i
x
O

e
s
a
c
c
a
L

0
.
7

–

0
.
3

C

°

0
7

–

8
2

,
s
e
y
d

l
a
i
t
n
e
t
o
p
-
x
o
d
e
r

h
g
i
H

n
o
i
t
a
d
i
x
O

e
s
a
d
i
x
o
r
e
p

g
n
i
s
i
d
i
x
o
-
e
y
D

0
.
7

C

°

5
3

,
l
o
n
e
h
p

,
s
n
o
i

e
s
e
n
a
g
n
a
M

.
s
c
i
t
a
m
o
r
a

c
i
l
o
n
e
h
p
n
o
n

,
s
d
n
u
o
p
m
o
c

e
y
d

l
l
a
m

s

.
s
e
n
i
m
a

n
o
i
t
a
d
i
x
O

e
s
a
d
i
x
o
r
e
p

e
l
i
t
a
s
r
e
V

0
.
7

–

0
.
5

C

°

0
4

–

3
2

,
l
o
n
e
h
p

,
s
n
o
i

e
s
e
n
a
g
n
a
M

n
o
i
t
a
d
i
x
O

e
s
a
d
i
x
o
r
e
p

e
s
e
n
a
g
n
a
M

.
s
c
i
t
a
m
o
r
a

5
.
4

–

0
.
3

C

°

0
4

–

8
2

c
i
l
o
n
e
h
p
n
o
n

,
l
o
n
e
h
P

H
p

e
r
u
t
a
r
e
p
m
e
T

e
t
a
r
t
s
b
u
S

n
o
i
t
c
a
e
R

n
o
i
t
a
d
i
x
O

y
l
i

m
a
f

e
m
y
z
n
E

r
e
m
y
l
o
p

t
e
g
r
a
T

e
s
a
d
i
x
o
r
e
p

n
i
n
g
i
L

i

n
n
g
i
L

4
.
9

–

0
.
3

C

°

5
7

–

7
3

.
s
e
d
i
r
a
h
c
c
a
s
o
g
i
l
o

n
a
c
u
l
g
o
l
y
x

,
e
s
o
l
u
l
l
e
c

l
y
h
t
e
m
y
x
o
b
r
a
c

c
i
t
a
m
o
r
a

,
l
o
n
e
h
p
o
n
i
m
a

,
s
n
a
c
u
l
g
-
3
,
1
-
(cid:12)

.
s
d
n
u
o
p
m
o
c

s
i
s
y
l
o
r
d
y
H

e
s
a
n
a
c
u
l
g
o
l
y
X

e
s
o
l
u

l
l
e
c
i
m
e
H

0
.
9

–

0
.
3

C

°

0
5

–

8
2

n
a
l
y
x

n
i

s
d
n
o
b

c
i
s
o
c
y
l
G

s
i
s
y
l
o
r
d
y
H

e
s
a
n
a
l
y
X

.
e
n
o
b
k
c
a
b

5
.
1
1

–

0
.
8

0
.
5

–

C

°

0
9

–

C

°

0
5

0
3

–

.
d
e
ﬁ
i
r
e
t
s
e
-
l
y
h
t
e
m
y
l
i
v
a
e
H

d
n
a

n
o
i
t
a
c
ﬁ
i
r
e
t
s
e

r
e
w
o
L

.
n
i
t
c
e
P

5
.
7

–

4
.
2

C

°

2
9

–

5
4

.
n
a
n
n
a
m
o
t
c
a
l
a
g

,
n
a
n
n
a
M

n
o
i
t
a
n
i
m

n
o
i
t
a
n
i
m

i
l
e
-
(cid:12)

i
l
e
-
(cid:12)

s
i
s
y
l
o
r
d
y
H

s
i
s
y
l
o
r
d
y
H

e
s
a
d
i
n
o
r
u
t
c
a
l
a
g
y
l
o
P

g
n
i
d
a
r
g
e
d
-
n
a
n
n
a
M

e
s
a
y
l

e
t
a
t
c
e
P

e
s
a
y
l

n
i
t
c
e
P

.

+
2
a
C

0
.
0
1

–

0
.
3

C

°

0
6

,
s
n
i
r
t
x
e
d
o
l
l
e
c

,
e
s
o
i
b
o
l
l
e
C

n
o
i
t
a
d
i
x
O

e
s
a
n
e
g
o
r
d
y
h
e
d

e
s
o
i
b
o
l
l
e
C

0
.
7

–

0
.
5

C

°

0
5

–

5
2

.
n
a
c
u
l
g
-
(cid:12)

s
i
s
y
l
o
r
d
y
H

e
s
a
d
i
s
o
c
u
l
g
-
(cid:12)

3
H
G

0
.
6

–

0
.
4

–

0
.
7

0
.
7

0
.
7

–

–

–

0
.
7

–

5
.
4

5
.
4

5
.
4

0
.
7

0
.
5

C

°

C

°

0
8

3
6

C

°

C

°

C

°

C

°

C

°

5
6

5
6

5
6

5
6

0
5

–

–

–

–

–

–

–

0
4

0
5

5
4

5
4

5
4

5
4

5
2

.
s
e
d
i
s
o
c
u
l
g
-
(cid:12)

d
e
k
n
i
l
-
e
n
o
c
y
l
g
a

,
s
e
d
i
r
a
h
c
c
a
s
o
g
i
l
o

d
e
k
n
i
l
-
(cid:12)

.
e
s
o
l
u
l
l
e
c

e
n
i
l
l
a
t
s
y
r
C

.
e
s
o
l
u
l
l
e
c

s
u
o
h
p
r
o
m
A

.
e
s
o
l
u
l
l
e
c

e
n
i
l
l
a
t
s
y
r
C

.
e
s
o
l
u
l
l
e
C

.
n
a
l
y
x
-
(cid:12)

s
i
s
y
l
o
r
d
y
H

s
i
s
y
l
o
r
d
y
H

s
i
s
y
l
o
r
d
y
H

s
i
s
y
l
o
r
d
y
H

s
i
s
y
l
o
r
d
y
H

e
s
a
n
a
c
u
l
g
o
d
n
e

5
H
G

e
s
a
n
a
c
u
l
g
o
d
n
e

7
H
G

e
s
a
n
a
c
u
l
g
o
d
n
e

2
1
H
G

e
s
a
n
a
c
u
l
g
o
d
n
e

5
4
H
G

e
s
a
d
i
s
o
c
u
l
g
-
(cid:12)

1
H
G

.
e
s
o
t
c
a
l

.
e
s
o
l
u
l
l
e
c

e
n
i
l
l
a
t
s
y
r
C

,
e
s
o
l
u
l
l
e
c

s
u
o
h
p
r
o
m
A

s
i
s
y
l
o
r
d
y
H

s
i
s
y
l
o
r
d
y
H

I

e
s
a
l
o
r
d
y
h
o
i
b
o
l
l
e
C

I
I

e
s
a
l
o
r
d
y
h
o
i
b
o
l
l
e
C

e
s
o
l

u

l
l
e
C

Table 1: Extracellular enzymes in lignicolous fungi and their optimal activation temperature
and medium pH (adapted from [73, 139, 127, 49, 91, 130, 71, 98, 74, 94, 30, 96, 85, 83, 137, 135,
22, 86, 59, 124]).

Page 10 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

CAZy family Enzymes
AA
CBM
GH

Laccase, LiP, MnP, VP, DyP, CDH, LPMO.
GH5-GH7-GH45 endoglucanase, CBH I
GH12 endoglucanase, (cid:12)-glucosidase,
xyloglucanase, xylanase, polygalacturonidase,
mannan-degrading, CBH II.
Glycosyltransferases.
Pectin lyase, pectate lyase.
Carbohydrate esterases.

TV PO GL
96
114
89
33
82
47
262
235
222

85
9
19

67
23
28

72
11
91

GT
PL
CE

Table 2: Genome distribution of CAZymes from selected species: T. versicolor (TV) [117], P.
ostreatus PC15 (PO) [117], G. lucidum G0119 (GL) [141].

from libriform ﬁbres to white-rot decay has been conﬁrmed experimentally [111].
The structure of the wood cell wall composing the xylem divides between the middle lamella
(ML), primary wall (P), and three layers of secondary wall (S1, S2, S3). They are represented
with their ﬁbre orientation and indicative thicknesses in Fig.2. The structuring middle lamella
is composed of lignin, calcium, and pectic compounds that are amorphous. Crystalline cellulose
is most prominent in younger wood, so it dominates S3, and is found in lowering percentages
moving towards the P. Inversely, hemicellulose can be found in greater extend in the P and in
reducing shares towards S3, while lignin is quite evenly spread across the cell wall, with higher
concentrations in the ML. The distribution of lignocellulosic compounds in woods is presented
in Table 3, and an illustration of wood cell wall speciﬁcation in arboreal plants is presented in
Fig.4. Colonisation happens through pit apertures present in wood, or the inducing of boreholes
by enzymatic activity. Although pectin contents are marginal in wood, pectinases expression is
critical to decomposing the pit membrane (Fig.2), which also contains cellulose, and allow the
fungus to proceed in colonising [113]. Calcium is associated with pectin in wood, which can be
hydrolysed by fungal oxalic acid, leading to Ca2+ chelates [53]. The main plant ﬁbre minerals in-
clude calcium, potassium, phosphors, and magnesium. The mineral content (ash) in softwood is
0.02 – 1 wt% and 0 – 5 wt% in hardwood [128]. We can note that large concentrations of calcium
oxalate and black residues of MnO2 accumulate during white-rot decay [15]. Wood can moreover
contain a variety of toxic compounds, mainly polyphenols or tannins in angiospermous trees and
phenolic compounds in gymnospermous ones. Selective deligniﬁcation capable fungi can degrade
polyphenols thanks to their chemical proximity to lignin. Ganoderma spp. are remarkable not
only for having this ability, but also because they have been reported to be attracted to these
compounds [110]. Nitrogen being necessary to fungi for enzyme synthesis, the limited amounts
in wood makes it resilient to decay (C:N of birch is typically of 55, and sycamore 401). This
compound is mostly found in parenchyma and is known to migrate towards the bark surface
during log desiccation [16]. To ﬁll in for a lack of available nitrogen, white-rot fungi such as
P. ostreatus have been shown to be able to trap and feed from nematodes, while the latter are
usually ones to feed from their mycelia [13].
As previously suggested, the wide distribution of wood-decaying fungi suggests with increasing
evidences that environmental factors and substrate speciﬁcs drive and regulate fungal colonisa-
tion between various fungal species in presence. Furthermore, while the pH of wood is most
often in the 4.0 – 5.5 range (beech: 5.11, birch: 5.29, for instance) [46], it has been demonstrated
that substrate pH is also inﬂuenced by the action of decay [44]. It has been reported by compar-
ative transcriptonics that white-rot related genes of a Phanerochaete carnosa species would all
express in ﬁr (Abies balsamea), pine (Pinus contorta), spruce (Picea glauca), and maple (Acer

Deliverable D5.2

Page 11 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

saccharum) media, but to a diﬀerent extent. For instance, MnP genes read was 2.42 times higher
in ﬁr compared to maple within the 30 most expressed genes, but LiP related genes read was 4.7
folds higher in maple [80]. This indicates both a resilience of species to the use of a variety of
vernacular substrates, but also a precariousness considering the sheer distribution of competitors
of an individual species. In its most extreme cases this leads to intense fungus chemical wrestling
for resource [61], a battle in which Ganoderma spp. has been recognised the advantage of being
able to degrade reaction zones [110]. Studies of lignicolous fungi ecological distribution point
to water content and temperature variations being the principal drivers of selection [17]. Cli-
mate change has then been related to shifts in fungal niches across hosts [18]. In the wild, host
tree species and their evolutionary stage can have an inﬂuence on patterns of decay: the latest
evolutionary stage that has the most diﬀuse xylem vessel system may result in a slower initial
colonisation. Similarly, gymnospermous wood having 3 – 4 mm long tracheids it beneﬁts from a
cellular compartmentalisation strategy to oppose fungal decay, additionally to its guaiacyl lignin
content. These topological eﬀects are prominent at the early stage of colonisation [16].

Figure 2: Wood cell wall structure: middle lamella (ML), primary wall (P), secondary wall (S1,
S2, S3) [133].

2.3 Water in fungal decay

The presence of capillary water in the substrate is a prerequisite to allow extracellular trans-
port of metabolites in fungi, and to establish osmotic potential for allowing turgor pressure and
subsequent hyphal growth. Turgor is also thought to enable fungi to penetrate solid substrates
[82]. Moreover, the activity of water, because it allows the maintaining of turgidity of fungal
cells at hyphal tip, is a critical parameter of growth. To illustrate the extent of the inﬂuence,
it has been reported that the white-rot species Physisporinus vitreus radial growth at pH 5 and
25 ºC increased from just below 0.5 mm/d at aw (cid:20) 0.974, to 4.5 mm/d at aw = 0.998 [45].
Furthermore, the production of enzymes has been related to water activity [47].
SSF requiring the absence of solution in between particles for a mycelium to develop, the sub-
strate water holding capacity is principal parameter for ensuring an optimal fungal colonisation
and predicting fungal growth [19, 39, 47]. In timber, moisture can exist as bound water within
cell walls below ﬁbre saturation point (FSP), free liquid water in cell cavities above FSP, and

Page 12 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 3: Principle structures of soft and hardwood. Illustration from [69].

Wood type

Cell type Layer

Gymnospermous
early wood

Tracheid

S1 – S3

Share of total
lignin content
(%)
65

Lignin
concentration
(%)
24

Gymnospermous
late wood

Tracheid

Angiospermous
wood

Fibre

Fibre
Fibre
Vessel
Vessel
Ray cells

ML, P
Cell corners
S1 – S3

ML, P
Cell corners
S1 – S2

ML, P
Cell corners
S1 – S3
ML, P
S1 – S3

21
14
75

14
11
60

9
9
9
2
11

49
64
22

51
78
19

40
85
25
40
25

Table 3: Approximated lignin distribution in the wood cell wall. Adapted from [41].

Deliverable D5.2

Page 13 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 4: Structure of arboreal plants. Microscopic tissues from left to right: adult spruce wood
(bar 50 µm), spruce compression wood (bar 20 µm), juvenile spruce wood (bar 50 µm), poplar
tension wood (50 µm), adult oak wood (bar 200 µm), and vascular bundle of bamboo (bar 200
µm). Last row showing schematics of cell wall structures. Light gray indicates the compound
middle lamella (0.5 – 1.5 µm) and primary cell wall (approximately 0.1 µm), yellow S1 layer with
a thickness of 0.1 – 0.35 µm, orange S2 layer as 1 – 10 µm thick; inner light gray layer shows the
0.5 – 1.1 µm thin S3 layer; the green G-layer in tension wood ﬁbres can ﬁll the whole lumen and
the cellulose ﬁbrils are oriented along the longitudinal cell axis. Illustration from [35].

Page 14 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

water vapour. Relative humidity (RH) of up to 97 – 98 % in the environment of timber is gen-
erally considered to correspond to cell wall intake (hygroscopic range), and as RH moves toward
saturation, wood cell voids intake tend to become more signiﬁcant in the moisture distribution
(over-hygroscopic range) [126]. The latter is conventionally considered as the ﬁttest range for
optimal fungal growth in lignocellulosic SSF. The validity of this convention is nonetheless dis-
cussed as it was shown that T. versicolor would colonise European beech wood at a minimum
moisture content (MC) of 15 % for a mass loss (ML) over 2 % [81], that brown-rot fungi grow at
92 – 97 % RH, and that instrumental limitations are encountered in the over-hygroscopic range
thus complicating measures and experimental control [126]. The FSP of European beech wood
has previously been determined to be 24.02 % MC at 97 % RH and 20 ºC [70]. The minimum
moisture threshold has been introduced as the metric having the most impact on predicting
fungal growth [20, 19]. It represents the MC below which fungal decay cannot be initiated, and
is typically given at a ML (cid:21) 2%.
Hydroxyl groups are the predominant sorption sites for water molecules in wood. Their amount
in hemicellulose is twice as high as in lignin, and four times higher than that of cellulose [125]. It
has been reported that cellulose can endure structural changes while wood is desiccated [8, 75].
It was hypothesised that lower desiccation rates would give enough time for amorphous cellulose
to crystallise, rendering the substrate less hydrophilic. On the contrary, higher rates of desicca-
tion may lead to more prominent amorphous cellulose and more hydroxyl availability. Lignin,
hemicellulose, and pectin polymers act as a barrier to cellulose crystallisation, which reﬂects on
the higher crystalline content towards the S3 cell wall layer [104]. Because hemicellulose have
a softening point at room temperature and 18 – 20 % MC [39], while lignin softens above 60
°C when saturated by water, and cellulose is hardly softened in its crystalline conﬁguration ((cid:21)
200 ºC), higher thermal treatments may results in hemicellulose degradation thus rendering the
substrate less hydrophilic [95].
Finally, the amount and quality of water in the SSF system varies during fermentation. We can
note that fungal decay by oxidation produces water along with CO2:

(C6H10O5)n + 6n(O2) = 6n(CO2) + 5n(H2O)

(1)

C6H10O5 being the repeating unit of glucose polymers, such as found in starch or cellulose. As
bound water is freed too during decay, and while substrate dry ML is occurring, the MC is
increasing if the system is closed. The quality and location of water in SSF systems are then
a prime set of information to predict and then foster or prevent fungal development. They are
of primary interest to design and produce MBC, while to the best of our knowledge no study
investigating the role of water in MBC production has been reported yet. The optimal moisture
content for laccase, LiP, and MnP activation is consistently reported to be 60 – 75 % [56, 66, 65,
7].

2.4 Mycelium mechanical properties

It is clear that mycelium acts as a binder between wood particles: it does so by degrading ligno-
cellulosic compounds and producing a dense hyphal network that interlocks dispersed particles.
The binding between two particles is not distinctly strong at the unit scale, but the very high
redundancy in the composite can result in a mechanically valuable material. It has been reported
that a P. ostreatus mycelium would be more than two times stiﬀer than a G. lucidum one when
both cultivated on a cellulose medium [55]. Authors comment this diﬀerence with regards to
the higher protein and lipids content in G. lucidum that can act as plasticisers. Furthermore,
in this study a set of experiments grown on a PDB-cellulose medium has been compared to
the previously mentioned results, and it was found to decrease the Young’s modulus of the G.

Deliverable D5.2

Page 15 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Tensile behaviour
Young’s modulus (MPa)
Ultimate strength (MPa)
Elongation at brake (%)

PO* PO** GL* GL** NDy
12
28
1.1
0.7
14
4

0.6 – 2.0
0.1 – 0.3
–

4
0.8
33

17
1.1
9

Table 4: Tensile behaviour of mycelia from P. ostreatus (PO) [55], G. lucidum (GL) [55], and
a non-disclosed species (ND) [67]. *Cellulose medium. **Cellulose and PDB medium. yNon-
disclosed medium.

lucidum mycelium ﬁlm by a factor of 3 while increasing its elongation at brake by 2.3 folds. This
corroborates their previous comment on plasticisers, and suggests that nutrition medium tuning
can help specify composites behaviour further than solely seeking stiﬀness. This study suggests
that the matrix phase of the composite can be made closer to an elastomer or stiﬀened by varying
protein, lipids, or alcohol contents. Similar ﬁndings have been reported [3]. Isolated mycelium
mechanical behaviour has been investigated so as to model it [68]. At the resolution of hyphae,
researchers have identiﬁed three main stages of response of a ligninolytic fungi mycelium: linear
elastic behaviour at small strains (hyphae bending), ﬁbre buckling and local structural collapse
or densiﬁcation at larger strains, and rapid stiﬀening associated with full compaction and large
number of inter-ﬁbre contacts. This model reﬂects upon the open-cell foam model. To ﬁt their
FEA simulation model, an hyperelastic model was used in this study as a mycelium undergoes
large displacements in its elastic behaviour and results in a non-linear stress-strain response. This
study relies on the characterisation in tension and compression of an aerial mycelium. Young’s
modulus situate in the 0.6 – 2.0 MPa range for a density in 30 – 50 kg/m3, the yield stress in
40 – 80 kPa, and ultimate tensile strength in 100 – 300 kPa [67]. The results are presented in
Table 4.

2.5 Substrate mechanical properties

Most commercially interesting woods have a Young’s modulus situated in 5.5 – 15.7 GPa at 12
% MC, and 4.4 – 12.3 GPa when green [52]. The distribution of these performances is similar
in gymnospermous and angiospermous woods. Wood particles are therefore important load-
carrying members of the composite system and reduce the magnitude of stress experienced by
the mycelial matrix. The plastic strain of the composite is contributed to only by particles in
such composite. The dewetting behaviour of the larger particles present in the composite is a
principal contributor to damage nucleation in such two-phase particulate composite. Further-
more, the shape, nature, and distribution of particles in two-phase particulate composites has
been shown to have a substantial inﬂuence over the load transfer between members and hence
their overall stiﬀness [115]. Moreover, while lignin is a primary contributor of strength parallel
to grain, hemicellulose supports compression strength perpendicular to grain. Its decay aﬀects
greatly the structural integrity of wood and its hardness [29].
In particle studies of granular materials, shape parameters such as ﬂakiness/ﬂatness (ratio of
particle thickness to width), elongation (length to width ratio), sphericity (deviation from a
sphere geometry), and roundness/angularity (measure of angles sharpness) have been investig-
ated [114]. Without matrix phase, studies have reported that a 3:1 ratio of ﬂaky particles content
would be an approximate optimal for shear strength (depending on the system of study). This
was related to cohesion increase under stress related to particles interlocking. Furthermore, it
was reported that a higher particle angularity leads to a decrease in elastic modulus, while ulti-
mate strength increases. The shear strength was reported to follow the same dynamic, increasing

Page 16 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

with particle angularity. Increasing ﬂakiness and angularity leading to increased cohesion, it is
also subjected to a higher abrasion and leads to damage accumulation under repeated loads, thus
leading to strain accumulation [114].
While wood particles are of common use in MBC, other more elastic substrates have been used
such as non woven cotton ﬁbres [4]. In commercially available hemp shives too, ﬂexible residues
of hemp tow processing can be found. In such cases, the particle may not contribute structurally
and the resulting composite behaviour will be driven by the mycelium and its chemical expres-
sion, they nonetheless contribute to the composite strength in tension.
Few researches have also been investigating the addition of non-organic aggregates to a lignocel-
lulosic substrate for improving its stiﬀness, such as with carbonate sand [50] (which also acts as
a pH buﬀer being mainly composed of CaCO3), and sand and gravel [84]. It was reported that
supplementation of a cotton and wheat bran medium with 37.5 wt% carbonate sand resulted
in a 1.6 fold stiﬀening, and a factor 4 increase in ultimate compressive strength, while density
increased by 27 %.

2.6 Composite mechanical properties

The large number of inﬂuential variables on the mechanical properties leads to the emergence of a
variety of methods for tuning them. While common practice leads to a maximum of compressive
modulus of maximum 2 MPa, post-cultivation heat-pressing has been reported to lead to Young’s
modulus of 35 – 97 MPa and ﬂexural modulus of 34 – 80 MPa [4]. Use of particles-additives
strategy additionally to a wheat bran supplemented medium made of cotton stalk has proved
signiﬁcant: an elastic modulus of 48.5 MPa was reported with 37.5 % carbonate sand addition
with P. ostreatus, while a non supplemented control in this study reached 30.3 MPa [50]. A
second study reported on an elastic modulus situated in the 39 – 60 MPa range for a density in
240 – 265 kg/m3 [138]. Authors of the latter are elusive about production details, but nonethe-
less indicates the use of Alaska birch shavings (Betula neoalaskana), millet grain, wheat bran,
calcium sulphate, and the addition of a natural ﬁbre. The speciﬁcs of the performative batch are
not disclosed, but a pre-colonisation was performed prior to moulding. The performance does
not come from a single factor considering the experimental plan. The considerable use of ﬁbres
(of undisclosed quality), grains, brans, and CaSO4, has us understand this as an hybrid improve-
ment using both substrate based reinforcement with the ﬁbres, and additives. Other stiﬀening
strategies have been investigated, such as with the introduction of SBR Latex, a bonding agent
for the construction industry. With a 5 % addition of SBR Latex and 0.5 % silane coupling agent
to a cotton seed hull medium, the cultivation of a P. ostreatus species displayed a 2.39 factor
increase in strength with a 27.6 % density increase [58]. The resulting environmental impact was
not discussed in the study. As another high impact material [6], cellulose nanofribrils (CNF)
addition has been reported [121]. Between two specimen groups, the optimal supplementation of
CNF was reportedly of 2.5 – 5 wt% with a contribution in Young’s modulus of a factor 2.56 – 5.6
and 2.8 – 7 in ultimate strength. In this study, the eﬀect of densiﬁcation on composites embed-
ding a 2.5 wt% CNF part was reported: densifying from 300 kg/m3 to 600 kg/m3 contributed
to increase exponentially the strength to a factor 17, while the stiﬀness increase by 14.38 folds.
The elastic modulus in tension of MBC is reportedly situated in 3.0 – 13.0 MPa, these values
resulting from a densiﬁed 24 days cultivated T. versicolor mycelium on wheat bran supplemented
beech sawdust [4]. For this specimen group, a higher ﬂexural modulus of 9.0 MPa was reported.
The state-of-the-art in MBC research largely considers monolithic and homogeneous composites,
besides a few study groups investigating jute type materials in sandwich composite reinforce-
ment, and wood panels introduction [72, 103, 142].
Across the state-of-the-art we can identify three main research strategies: densiﬁcation (by com-

Deliverable D5.2

Page 17 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

posite packing, cold or hot-pressing), composition (by introducing structuring elements such as
ﬁbres and aggregates), and supplementation (chemical tuning of the substrate). The contribu-
tion of these strategies on the compressive elastic modulus are illustrated in Fig.5. These are
substrate focused strategies, targeting mycelium properties tuning in the case of supplementa-
tion, while the very choice of the fungal species can result in radically diﬀerent cultivation lead
time and mechanical properties too. Genetic modiﬁcation of fungi could promote speciﬁc en-
zymatic activity or modify the composition of its cell wall to help improve performances, while
the investment cost and ethics of this practice may lead to prioritising more frugal and aﬀordable
techniques.

Figure 5: Eﬀect of exploration strategies on the compressive Young’s modulus. Data retrieved
from [142, 138, 100, 50, 38, 60]; cita_UnpublishedData_2021 refers to values currently only
reported in this document. Larger points correspond to structured composites (such as textile
or ﬁbre reinforced).

Page 18 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

3 Materials and methods

Across the three strategies (densiﬁcation, composition, and supplementation), we are investigat-
ing the eﬀect of composition over the mechanical behaviour in compression and ﬂexion (sections
3.5 and 3.6). Because the eﬀect of substrate particles size has not yet been investigated, we tested
its impact in compression and tension (sections 3.5 and 3.7). With a standard protocol, we tested
a change in strains on the best performing granulate size (section 3.5). Similarly, and as a way
to verify the impact of the nature of lignins on white-rot fungi colonisation from a performative
perspective, we tested in compression specimens of pine wood (Pinus sylvestris) cultivated G.
lucidum (section 3.5). In the context of the extended compression series, the principal substrate
and composition materials have been qualiﬁed by Fourier-Transform Infrared (FTIR) spectro-
metry (section 3.4). The decay of G. lucidum on beech wood is also quantiﬁed by FTIR.
A variety of experimental designs are being used in the ﬁeld of MBC research to evaluate mech-
anical characteristics. Few studies consider material evaluation standards, and systematically
derive from the standards guidelines (regarding specimen or experimental design). The refer-
enced standards in the state-of-the-art are presented in Table 5. Among the three strategies
introduced previously, we have decided to investigate both the eﬀect of composition strategies
over the behaviour in compression and ﬂexion, and the eﬀect of aggregate size over compression,
tension, and ﬂexion. The ASTM that is the ﬁttest for evaluating these properties with adopting
the two-phase particulate composite model is ASTM D1037. It has the beneﬁt of being the most
referred set of guidelines in MBC development and covers various tests. The experimental plan
is presented in Fig.6, and its colour legend is reported in Table 6.
In addition to the investigation of mechanical properties of MBC, we have conducted studies on
thermal performance of MBC with a view towards building insulation application. We extend
the scope of the studies reported in the literature (see Table 7), which focus on material prop-
erties, to consider hypothetical building envelopes composed of MBC. This eﬀort will directly
inform design approaches being developed as part of T5.3 - Design Rules for Fungal Architecture.
As such, we follow ISO 9869 for determining thermal resistance and thermal transmittance via
in-situ measurement. We design a two stage experimental plan that determines thermal proper-
ties of two material series, and then tests two conﬁgurations of a hypothetical building envelope
build-up (section 3.9).

3.1 CITA cultivation method: mechanical tests

Materials

Millet-grown spawns of species G. lucidum (reference M9726) and G. resinaceum (reference
M9732) were acquired from Mycelia BVBA (Nevele, Belgium). Spawns were stored at a constant
4 ºC and 65 % RH. The principal substrate of the specimens is European beech wood (Fagus
sylvatica). We used three granulations (small, medium, large): 0.5 – 1.0 mm (Räuchergold
type HB 500/1000, J. Rettenmaier & Söhne GmbH + Co KG, Rosenberg, Germany), 0.75 – 3.0
mm (Räuchergold type HB 750/2000, J. Rettenmaier & Söhne GmbH + Co KG, Rosenberg,
Germany), and 4.0 – 12.0 mm (Räuchergold type KL 2/16, J. Rettenmaier & Söhne GmbH +
Co KG, Rosenberg, Germany). A fourth particle type was added to the experimental plan, as
a 1:1:1 volume ratio mix of the three granulations. Longitudinal ﬁbres were introduced in a
specimen series by using common reed ﬁbres (Phragmites australis; Tækkemand Chresten Finn
Guld, Køge, Denmark). 6 mm diameter rattan ﬁbres have been used for compression and ﬂexion
testing (Calamus manan; B.V. INAPO, Bloemendaal, Netherland), such as hemp-based hessian
sativa; NEMO Hemp jam web 370 g/m2, Naturellement Chanvre,
(Cannabis sativa subsp.

Deliverable D5.2

Page 19 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Test
Compression ASTM D3501

Standard

ASTM D695

ASTM C67

ASTM D2166

ASTM D1037

ASTM C165

EN 826

ASTM D3574

Flexion

ASTM C203

ASTM C393

ASTM D7250

ASTM D1037

ISO 16978

ISO 12344

Tension

ASTM D1037

Designation
Standard Test Methods for Wood-Based
Structural Panels in Compression.
Standard Test Method for Compressive
Properties of Rigid Plastics.
Standard Test Methods for Sampling and
Testing Brick and Structural Clay Tile.
Standard Test Method for Unconﬁned
Compressive Strength of Cohesive Soil.
Standard Test Methods for Evaluating
Properties of Wood-Base Fiber and Particle
Panel Materials.
Standard Test Method for Measuring
Compressive Properties for Thermal
Insulations.
Thermal insulating products for building
applications – Determination of compression
behavior
Standard Test Methods for Flexible Cellular
Materials—Slab, Bonded, and Molded
Urethane Foams
Standard Test Methods for Breaking Load
and Flexural Properties of Block-Type
Thermal Insulation.
Standard Test Method for Core Shear
Properties of Sandwich Constructions by
Beam Flexure
Standard Practice for Determining Sandwich
Beam Flexural and Shear Stiﬀness
Standard Test Methods for Evaluating
Properties of Wood-Base Fiber and Particle
Panel Materials.
Wood-based panels – Determination of
modulus of elasticity in bending and of
bending strength.
Thermal insulating products for building
applications – Determination of bending
behavior.
Standard Test Methods for Evaluating
Properties of Wood-Base Fiber and Particle
Panel Materials.

Ref.
[38]

[119]

[88]

[138]

[142, 25]

[63]

[37]

[60]

[63]

[72]

[72]

[77, 122, 25]

[37]

[37]

[142, 25]

Table 5: Referenced standards for mechanical property determination in the MBC state-of-the-
art.

Page 20 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 6: Distribution of experimental characterisation series across fungal species (GL: G.
lucidum, GR: G. resinaceum, GRm: G. resinaceum from Mogu records), substrate types (PM:
pine shavings, HC: hemp ﬁbres supplemented with coﬀee ground, BS: beech wood small, BM:
beech wood medium, BL: beech wood large, BSML: 1:1:1 volume ratio mix of BS, BM, and BL),
and ﬁbre composition strategies (planes). The colours legend is reported in Table 6.

Colour
Magenta
Black
Cyan
Red
Green
Yellow

Flexion Compression Tension

X

X

X

X

X
X
X

X

X
X

Table 6: Experimental plan test types correspondence table.

Deliverable D5.2

Page 21 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Test
Thermal
Conductivity

Standard
ASTM D5334-14

ASTM D5334-00

Designation
Standard Test Method for Determination
of Thermal Conductivity of Soil and Soft
Rock by Thermal Needle Probe
Procedure.
Standard Test Method for Determination
of Thermal Conductivity of Soil and Soft
Rock by Thermal Needle Probe
Procedure.

Ref.
[138]

[38]

ISO 8302/EN 1946–3 Thermal insulation — Determination of

[31]

DIN EN 12667

ASTM C518-17

None referenced

None referenced

None referenced

None referenced

None referenced

steady-state thermal resistance and
related properties — Guarded hot plate
apparatus.
Thermal performance of building
materials and products - Determination of
thermal resistance by means of guarded
hot plate and heat ﬂow meter methods -
Products of high and medium thermal
resistance.
Standard Test Method for Steady-State
Thermal Transmission Properties by
Means of the Heat Flow Meter Apparatus.
Study conducted using EP500 hot plate
apparatus (Lambda–Messtechnik).
Study conducted using KD-2 Pro thermal
analyser.
Study conducted using TPS 500 thermal
conductivity instrument.
Study conducted using dynamic hot-wire
method.
Study conducted using KD-2 Pro thermal
analyser.

[102]

[134]

[97]

[136]

[63]

[107]

[136]

Speciﬁc Heat
Capacity

Table 7: Overview of literature covering thermal property determination of MBC for thermal
insulation applications, together with referenced standards where declared.

Page 22 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Echandelys, France). The materials used by CITA for the mechanical and thermal studies are
illustrated in Fig.7.

Figure 7: Materials used by CITA for the mechanical and thermal studies of this report (left to
right, top to bottom): beech large (BL), beech medium (BM), beech small (BS), birch shavings,
hemp shives, rattan ﬁbres, hessian, and common reed ﬁbres.

Protocol

The principal substrates, ﬁbres, and hessian were prepared at 70 % MC with mineralized water,
and sterilised at 121 ºC for 15 min. The principal substrates were then mixed with 16 wt% spawn
and incubated in polypropylene ﬁltered bags (SacO2, Deinze, Belgium) for 7 days at 25 ºC in the
dark. Once colonised, the principal substrates were broken down and formed with the sterile ﬁbre
and hessian into alcohol cleaned aerated PETG moulds. The formed specimens were incubated
for 21 days at 25 ºC in the dark, then oven dried for 48 hours at 60 ºC. The dried specimens
were stored at 4 ºC and 65 % RH prior to testing. No external mycelium was cultivated on the
external boundaries of the specimens. No additive was used. The same cultivation protocol was
used for all mechanical characterisation test related specimens (compression, ﬂexion, tension).

3.2 CITA cultivation method: thermal test

A millet-grown spawn of species G. lucidum (reference M9726) was acquired from Mycelia BVBA
(Nevele, Belgium). The spawn was stored at a constant 4 ºC and 65 % RH. Two principle
substrates were selected for the specimens based on their nutritional compatibility with the
selected mycelium species, and diﬀerence in dry density (see section 3.9). These were: hemp

Deliverable D5.2

Page 23 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

shives (Cannabis sativa subsp. sativa) sourced from Zooplus AG, Germany; spent birch shavings
(Betula spp.) sourced from a local distillery, Copenhagen.

Protocol

The principal substrates were prepared at 70 % MC with mineralized water, and sterilised at
121 ºC for 15 min. The principal substrates were then mixed with 16 wt% spawn and incubated
in polypropylene ﬁltered bags (SacO2, Deinze, Belgium) for 7 days at 25 ºC in the dark. Once
colonised, the principal substrates were broken down and added to alcohol cleaned polypropylene
microboxes with HEPA-ﬁlters integrated into the lids (SacO2, Deinze, Belgium). Microboxes of
dimension 185x185mm at base were used, with three height variations (78 mm: model TP2000;
112 mm: model TP3000; 191 mm: model TP5000) to produce MBC panels of 40 mm, 80 mm,
and 120 mm thickness respectively. The formed specimens were incubated for 21 days at 25 ºC
in the dark. An external mycelium was cultivated on the external boundaries of the specimens
by demoulding and incubating them for another 5 days in a moisturised container. No additive
was used. The specimens were oven dried for 48 hours at 60 ºC. The dried specimens were stored
in ambient room conditions prior to testing.

3.3 Mogu cultivation method

Materials

A G. resinaceum species from Mogu record 19-18 was used (GRm). Hemp shives (cellulose: 35
%; hemicellulose: 18 %; lignin: 21 %; protein, pectin, and other products: 18 %) were acquired
from HempFlax BV. (ref. 7000277; Oude Pekela, The Netherlands). Coﬀee ground was acquired
from a local grocery store, ﬁltered for 5 min with boiling water, and the spent coﬀee ground was
collected.

Protocol

The principal substrates, ﬁbres, and hessian were prepared at 60 % MC with mineralized water,
and sterilised at 121 ºC for 90 min. The substrates were then mixed with 3 wt% spawn and
incubated for 21 days at 25 ºC in the dark. Once colonised, the principal substrates were broke
down and formed with the sterile ﬁbre and hessian into aerated moulds. The formed specimens
were incubated for 7 days at 25 ºC in the dark, then oven dried for 72 hours at 70 ºC.

3.4 Chemical analysis

Fourier-Transform Infrared (FTIR) spectrometry has been used previously for analysing the
lignocellulosic proﬁles of substrates and their relation to fungal degradation patterns [90, 89, 55],
with the beneﬁt of requiring a limited specimen preparation, and spectra shape and frequencies
being directly related to microscopical physical quantities and hence prepared for interpretation
[51]. FTIR spectrometry was conducted in this study on a single reﬂection diamond Attenuated
Total Reﬂectance (ATR) Agilent 4500a FTIR (Santa Clara, USA). The acquisition resolution
was 4 cm(cid:0)1 with 16 scans per specimen, for a band between 4000 cm(cid:0)1 and 650 cm(cid:0)1. We
corrected the baseline of FTIR spectra following the adaptive iteratively reweighted Penalised
Least Squares (airPLS) method [140], and spectra normalisation was done with amide I/II band
envelopes [11]. Four samples were isolated from G. lucidum colonised beech wood specimens,
their spectra were averaged for analysis. The other materials were scanned with one replicate.

Page 24 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Chemical analysis of G. lucidum colonised beech was performed on specimens having been tested
in compression previously.

3.5 Compression series

Following ASTM D1037, we report compression parallel to surface evaluation, for which the
short-column method has been chosen as the specimens have of a nominal thickness above 25
mm. They are parallelepipeds of 1:1:4 ratio, their dry dimensions are 34 x 34 x 140 mm.

Eﬀect of particle size & reinforcements

Four specimen groups were designed and prepared by CITA:

• Principal substrate as small granulation beech wood,

• Principal substrate as medium granulation beech wood,

• Principal substrate as large granulation beech wood,

• Principal substrate as mixed granulation beech wood,

Additionally, reinforcement strategies were investigated in four groups:

• Control: principal substrate only,

• Hessian jacketing: a hessian jacketing was introduced in the length,

• Fibres perpendicular to load: 32 mm long rattan ﬁbres were positioned regularly within the
principal substrate as two layers of ﬁbres, centred in the specimen thickness and separated
by a 10 mm layer of principal substrate,

• Fibres coaxial to load: eight to ten ﬁbres of common reed of 1 mm ± 0.5 diameter were
chosen so as to balance their dimensional variability and positioned in two layers separated
by 10 mm of principal substrate.

No external mycelium was left to grow on the external boundaries of the specimens so as to
observe the eﬀect of experimental variables without introducing a specimen geometry bias. We
identify this bias as critical for the reproducibility of experiments as the characteristics of the ex-
ternal mycelium mat is never found to be reported in the state of the art. A jacketing strategy has
been integrated to study the eﬀect of boundary reinforcement. Six replicates were produced and
tested for each specimen type. These specimens were tested at a loading speed of 1.0 mm/min.

Eﬀect of species

A medium granulation of beech wood cultivated G. resinaceum was tested with an identical
cultivation protocol (BM_GR group). Pine wood sawdust was collected from a CITA milling
workshop and was prepared following the previously described protocol with a G. lucidum species.
These results are discussed with regards to G. lucidum grown medium beech wood (BM). 5
replicates have been tested for the BM_GR group, and only two could be tested for the PM_-
GL group. Load testing was performed on a Mecmesin MultiTest-dV testing bench equipped
with a 2500 N load sensor. These specimens were tested at a loading speed of 4.0 mm/min.
Young’s modulus and ultimate compressive strength were calculated following ASTM D1037.

Deliverable D5.2

Page 25 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

3.6 Flexion series

Our experimental plan investigates the eﬀect of reinforcement strategies over the ﬂexural modulus
and ultimate ﬂexural strength. Following ASTM D1037, we report on three points bending. To
this end, four specimen groups were designed:

• Control: no ﬁbre,

• Inner hessian: a ﬂat layer of hessian was introduced at mid-thickness,

• Hessian jacketing: a hessian jacketing was introduced in the length,

• Rattan: ﬁve parallel rattan ﬁbres of 6 mm diameter by 500 mm, separated by 8 mm, were

introduced in the length and at mid-thickness.

Six replicates were produced and tested for each of the specimen types. Load testing was per-
formed on a Mecmesin MultiTest-dV testing bench equipped with a 2500 N load sensor, with a
loading speed of 10.0 mm/min. Flexural modulus and ultimate ﬂexural strength were calculated
following ASTM D1037. Two specimen series were produced, one by CITA following the protocol
in section 3.1, and one by Mogu following the protocol in section 3.3. The wet specimens cultiv-
ated by CITA are parallelepipeds of 520 x 72 mm, with a nominal thickness of 20 mm. The width
and thickness were not aﬀected by the desiccation, but the length of the dry specimens varied
between reinforcement strategies and shrank on average by 3.5 % in the control group, 2.7 % in
the inner hessian group, 1.7 % in the hessian jacketing group, and 0.8 % in the rattan group.
Mogu cultivated specimens exhibit a volumetric variability of up to 23.8 %. Calculations are
taking the dimensional variability into account as averaged per specimen group. Detail pictures
of a comparable specimen of each series are presented in Fig.8 and Fig.9.

Figure 8: Detail in top view of CITA cultivated specimen (left), and Mogu cultivated specimen
(right).

3.7 Tension series

Following ASTM D1037 dog-bone specimens of a nominal thickness of 20 mm were cultivated by
CITA. We report on tension parallel to surface. Our experimental plan investigates the eﬀect of
granulate sizes over the tensile modulus and ultimate tensile strength. To this end, two specimen
groups were designed:

• Small granulation of beech wood,

Page 26 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 9: Detail in side view of CITA cultivated specimen (left), and Mogu cultivated specimen
(right).

• Medium granulation of beech wood.

Six replicates were produced and tested for each of the specimen types. Load testing was per-
formed on a Mecmesin MultiTest-dV testing bench equipped with a 2500 N load sensor, with
a loading speed of 4.0 mm/min. Tensile modulus and ultimate tensile strength were calculated
following ASTM D1037.

3.8 Simulation model

Simulation of a kagome weave model has been investigated in particle-based simulation in report
D5.1. Here, we extend the scope of this study with structural analysis. The analysis was primarily
performed with geometrically nonlinear analysis in SOFiSTiK simulation software (SOFiSTiK
AG, Nuremberg, Germany). The analysis will use the 3rd principle in SOFiSTiK, it applies all
geometric analysis in the setup. The role model is based on the valences-four mixed valences-ﬁve
carbon ﬁbres demonstration. Our experimental plan investigates the eﬀect of diﬀerent sizes and
diﬀerent materials for weave members, a the resulting bending strength in the kagome weaves.
The material group as follow:

• Carbon ﬁbres strips: Young’s modulus of 1:7e5 for a 20 mm width and 1.5 mm thick

member,

• Bamboo strips: Young’s modulus of 5e4 for a 20 mm width and 4 mm thick member.

Thus, we export the hybrid strategy of kagome weaves, from minimum bending radius and
analysed the principal stress line of the mycelium shell. Stress-line analysis provide an idealization
of material continuity and encode the optimal topology. The strategy is to place the high bending
radius material cross singularity. Subsequently, placing high bending strength at high-density
region in stress line (Fig.10). Six models of three weaves only models and three integrated
mycelium assembly, and simulated in its dead load and 0.2 kN/m2 on the top. We evaluated the
result of the displacement from the analysis report.

Deliverable D5.2

Page 27 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 10: The image of (a) minimum bending radius analysis, (b) principal stress line, and (c)
the suggestion of hybrid weaves

3.9 Thermal series

Thermal properties of MBC have been investigated and reported in the literature (see Table
7) with a predominant focus of the measurement of thermal conductivity ((cid:21)). A consolidation
of results from this literature yields thermal conductivity values in the range of 0.029 – 0.180
W/mK for a wide variety of MBC compositions. Here, we extend the scope of the state-of-the-
art by: 1) measuring thermal transmittance (U-value); 2) making preliminary investigations into
the thermal properties of material assemblies that suggest construction approaches with MBC.
Speciﬁcally, we consider a cavity wall typology, with MBC material utilised for both the outer
and inner walls. Our experimental plan is designed to ﬁrstly assess the thermal properties of
MBC panels (approx. 165 x 165 mm after denaturing) of varying thicknesses prepared using G.
lucidum with two diﬀerent densities of substrate (Fig.11). This part of the experimental plan
comprises the following series:

• Substrate of hemp shives ((cid:26) = c.120 kg/m3); panel thicknesses: 40 mm, 80 mm, 120 mm;

six replicates,

• Substrate of birch shavings ((cid:26) = c.50 kg/m3); panel thicknesses: 40 mm, 80 mm; six

replicates.

From these results, we calculated the median thermal conductivity of each substrate type and
used this to determine the anticipated U-Value of various MBC assemblies. We empirically tested
one assembly arrangement (40 mm panel; 20 mm air cavity; 40 mm panel) for both substrate
types, with three replicates. We evaluated these results against the predicted performance.
Thermal testing was performed using the gSKIN® U-Value Kit (KIT-3610C) which consists of 1
heat ﬂux sensor (XO 67 7C), 1 calibrated datalogger (DLOG-4243) and 2 temperature sensors, to
deliver quantitative insulation data (U-Value) according to ISO 9869, ASTM C1046 and ASTM
C1155 (Fig.12). These standards cover in-situ thermal transmittance and thermal resistance
measurement of building envelopes and components, and are therefore suited to our aim of
making preliminary, thermal insulation application focused, evaluations of MBC assemblies for
building enclosures.
A thermal chamber was constructed from 100mm rigid insulation panels (Jackodur KF 300
Standard SF100, thermal conductivity ((cid:21)D) 0.035 W/mK) with internal dimensions 500 x 500
x 400 mm (W:D:H). Four sample ports were cut – two ports per face on opposing faces of the
thermal chamber – to perform sample tests in parallel, as shown in Fig.13. Each sample required

Page 28 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

its own gSKIN® U-Value Kit for performing the measurement. The sample ports were cut
centred on the mid-point of the vertical dimension of the interior face and distributed with equal
spacing across the horizontal dimension of the internal face. The sample port openings were
cut at 220 x 220 mm (relative to sample sizes c.175 x 175 mm) to permit packing of insulation
between the chamber and the sample.
Thermography was used to verify that there were no thermal leaks at the interfaces between
samples and the thermal test chamber. Thermography was conducted using a FLIR One Pro
iOS thermal camera (1440 x 1080 pixels, 0.1 ° resolution across a temperature range of (cid:0)20 to 400
°C). The internal temperature of the thermal chamber was regulated using a heating mat set to 35
°c (Digital Thermostat Thermo 2, Bio Green GmbH, Germany, dimensions 400 x 750 mm, power
operation 65 W, max power 3000 W, operating temp. ﬁeld -50 - 99 °c) in order to guarantee the
required minimum temperature delta of 5 °c between internal and external ambient conditions.
The heating mat was located in the chamber vertically and equidistant from both sample port
faces. The test chamber was located in an unheated basement room with ambient temperature
ﬂuctuation between 15-18 °c. Using the proprietary gSKIN® software, sample measurements
were conducted for a minimum of 72 hours and validated that the U-Value did not deviate more
than 5 % from the value 24 h earlier. To make measurements of the 120 mm deep specimens, the
thermal chamber was modiﬁed with an additional 50 mm of rigid insulation on the faces with
port cutouts to mitigate against heat transfer to external ambient conditions from the side of
the specimen.

Figure 11: The two substrates used in the production of MBC panels for thermal characterisation
– (a) hemp shives, (b) birch shavings

Deliverable D5.2

Page 29 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 12: The gSKIN® U-Value Kit (KIT-3610C) used for the thermal transmittance measure-
ments reported here. The kit consists of 1 heat ﬂux sensor (XO 67 7C), 1 calibrated datalogger
(DLOG-4243) and 2 temperature sensors. Image source: https://www.greenteg.com/U-Value/

Figure 13: Thermal test chamber constructed to test four samples simultaneously.

Page 30 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a) assessment of two specimens

(b) detail assessment of single specimen

Figure 14: Thermographic veriﬁcation of the thermal characterisation setup. Two specimens
mounted and seen as yellow areas (a), and detail of a single specimen (b), displaying good
thermal sealing with only marginal temperature diﬀerence between the exposed surfaces and
their periphery.

Deliverable D5.2

Page 31 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a) Section showing mounting of 40mm and 80mm panel specimens

(b) Section showing mounting of 120mm panel specimen and cavity wall assembly

Figure 15: Section through the thermal test chamber setup (a) (b).

Page 32 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

4 Chemical characterisation

We used FTIR spectrometry to characterise the four raw materials used in the Compression:
eﬀect of particle size & reinforcements series (Fig.16, section 5.1). The materials were hessian,
beech wood, rattan, and common reed. Beech wood and rattan spectra display a chemical
proﬁle that is very similar, with the exception of peaks at 1123 cm(cid:0)1 and 1160 cm(cid:0)1, and the
1300 – 1500 cm(cid:0)1 region. This indicates a slightly higher content of cellulose, hemicellulose and
lignin in our tested beech wood specimen (C–O stretching, C–O–C asymmetrical stretching, C–H
deformation, COOH groups symmetrical stretching, symmetric C–H bending, CH2 deformation
stretching, CH3 asymmetrical angular vibration, vibrational mode of amide C–O stretching)
[14, 76, 116]. Common reed displays a minimal amount of lignin and hemicellulose compared
to our other samples, while the peak at 890 cm(cid:0)1 is associated with C–O–C stretching at the
(cid:12)-(1→4)-glycosidic linkages of amorphous cellulose [27]. The hessian displays distinctive peaks
at 707 cm(cid:0)1, 890 cm(cid:0)1, 1060 cm(cid:0)1, 1316 cm(cid:0)1, and 1430 cm(cid:0)1 in the ﬁngerprint region, and
1640 cm(cid:0)1, and 2921 cm(cid:0)1. The 750 – 680 cm(cid:0)1 and 1680 – 1630 cm(cid:0)1 regions (C=O streching)
are associated with primary and secondary amides in hemp (amide V: C–N and N–H vibrations)
[120]. Primary amides in hemp are amino acids, fatty acids, and steroids, which contribute to
the 3500 – 3000 cm(cid:0)1 region. The 1310 – 1230 cm(cid:0)1 region (C–N stretching) is associated
to secondary amides, such as cannabinoids, ﬂavonoids, stilbenoids, terpenoids, alkaloids, and
lignans [42]. The peak at 2921 cm(cid:0)1 is associated with alkyl C–H groups [33].
Four samples were isolated from G. lucidum colonised beech wood specimens after they were
used for load testing. Their spectra were averaged and are presented on Fig.17 along with the
beech wood spectrum, and a sample scan of G. lucidum mycelium. Peaks at 886 cm(cid:0)1, 1075
cm(cid:0)1 and 1160 cm(cid:0)1 are characteristic of (1→3)- and (1→6)-(cid:12)-glucans that are present in the
fungal cell wall (identiﬁed as [2], and [4] on Fig.17). The peaks [1] and [3] at 780 cm(cid:0)1 and
1043 cm(cid:0)1 are also associated with (cid:12)-glucans [105]. Chitin is identiﬁed at peak [5] 1313 cm(cid:0)1
(amide III: C–N stretching), which also aﬀects the 1640 cm(cid:0)1 region [6] alongside the presence
of peptides and secondary metabolites (aromatic rings and conjugated alkenes). The peak [7]
at 2922 cm(cid:0)1 is representative of chitin and ergosterol (C–H stretching) [48]. The 3600 – 3000
cm(cid:0)1 region (peak [8]) is considered to be inﬂuenced by residual water and entrapped CO2 (O–H
and N–H stretching). Finally, peaks [9] to [12] represent decreases at 1231 cm(cid:0)1, 1425 cm(cid:0)1,
1506 cm(cid:0)1, and 1733 cm(cid:0)1. They are associated with lignin and xylan breakdown (syringyl ring
breathing and C–O stretching, C=C stretching vibration in aromatic ring), and cellulose (peak
[11]) and hemicellulose (peak [11] and [12]) breakdown is observed (CH2 scissor vibration, C=O
stretching) [21]. To evaluate the lignocellulosic changes undertaken during G. lucidum activity
quantitatively, the band ratio indices at 1231 cm(cid:0)1, 1425 cm(cid:0)1, and 1506 cm(cid:0)1 were calculated
from the 2921 cm(cid:0)1 band [32] for beech wood and G. lucidum colonised beech wood as:

;

In
I2921
Where In is the speciﬁc band intensity and I2921 the band intensity at 2921 cm(cid:0)1. The band
ratio at 1231 cm(cid:0)1 went from 1.58 in beech wood to 0.61 in G. lucidum colonised beech wood;
the band ratio at 1425 cm(cid:0)1 went from 0.96 in beech wood to 0.63 in G. lucidum colonised
beech wood; the band ratio at 1506 cm(cid:0)1 went from 0.71 in beech wood to 0.24 in G. lucidum
colonised beech wood; the band ratio at 1733 cm(cid:0)1 went from 1.19 in beech wood to 0.41 in G.
lucidum colonised beech wood. We can therefore observe that G. lucidum had a preference in
breaking down lignin and xylan at 1231 cm(cid:0)1 compared to cellulose and hemicellulose at 1425
cm(cid:0)1 (2.94:1), which is conﬁrmed by the ratios at 1506 cm(cid:0)1 for lignin (1.42:1), and 1733 cm(cid:0)1
for hemicellulose (2.36:1). The CH2 scissor vibration corresponding to the peak at 1425 cm(cid:0)1

(2)

Deliverable D5.2

Page 33 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

reﬂecting both cellulose and hemicellulose, the present decrease might be primarily related to
hemicellulose breakdown. This preference of G. lucidum for lignin and hemicellulose is consistent
with ﬁndings reported in the literature [141].

Figure 16: FTIR spectra of hemp-based hessian, beech wood, rattan, and common reed ﬁbres.

Page 34 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 17: FTIR spectra of G. lucidum colonised beech wood, G. lucidum mycelium, and beech
wood. Green areas represent increased values in mycelium-colonised specimens (peaks 1 to 8),
red areas are decreased values in mycelium-colonised specimens (peaks 9 to 12).

Deliverable D5.2

Page 35 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

5 Mechanical characterisation

5.1 Compression: eﬀect of particle size & reinforcements

We investigated the eﬀect of particle sizes on the mechanical behaviour in compression of MBC
using four levels of granulation: small (BS family), medium (BM family), large particles (BL
family), and a 1:1:1 volume ratio mix of the three previous granulations (BSML family). A
second parameter was introduced to investigate the anisotrope modiﬁcation of MBC. Three
typologies of ﬁbre composition were implemented in the experimental plan: hessian jacketing
coaxial to the load case (H), unidirectional rattan ﬁbres perpendicular to the load case (R), and
unidirectional common reed ﬁbres coaxial to the load case (V). Isotropic controls were added for
each level of granulation (BS, BM, BL, BSML specimen types in the ﬁgures). Fig.18 illustrates
the three typologies alongside the control. Experimental parameters per specimen type and
resulting mean density, mean Young’s modulus and mean ultimate strength are presented in
Table 8. Box plots of the results for Young’s modulus and ultimate strength are presented in
Fig.19. A picture of a specimen from the series is presented in Fig.20, a group of specimens in
Fig.21, and specimens after having been tested in Fig.23 and Fig.24.

Figure 18: Fibre placement strategies and their sectional CT scan (left to right): control (BS),
jacketing coaxial to load (BM_H), ﬁbres perpendicular to load (BS_R), ﬁbres coaxial to load
(BS_V).

Jacketing coaxial to load

The introduction of the hessian jacket oﬀers a contrasting illustration of the eﬀect of the mycelial
mat usually grown on the external boundary of MBC. We observe that the dispersion of Young’s
modulus results across all specimen families is reduced compared to their controls with the
exception of the BL family. The jacketing also aﬀects the dispersion of results in ultimate

Page 36 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

e
t
a
m

i
t
l
u

n
a
e
M

)
.
d
.
s
(

h
t
g
n
e
r
t
s

)
4
5
.
6
3
(

a
P
k

6
8
.
1
7
1

)
8
3
.
4
3
(

a
P
k

9
7
.
5
7
1

)
9
4
.
8
5
(

a
P
k

6
0
.
9
8

)
6
2
.
9
3
(

a
P
k

5
8
.
6
4
1

)
4
6
.
7
5
(

a
P
k

8
3
.
6
0
3

)
7
4
.
5
3
(

a
P
k

5
8
.
8
9
2

)
5
8
.
2
6
(

a
P
k

0
3
.
2
3
2

)
6
7
.
9
7
(

a
P
k

3
9
.
0
7
2

)
1
3
.
0
3
(

a
P
k

0
6
.
5
4
2

)
8
8
.
5
2
(

a
P
k

3
9
.
3
2
2

)
5
7
.
4
6
(

a
P
k

8
8
.
0
8
1

)
3
8
.
0
0
1
(

a
P
k

6
8
.
0
9
2

)
3
7
.
1
3
(

a
P
k

9
0
.
7
3
2

)
7
4
.
8
4
(

a
P
k

8
4
.
4
9
1

)
3
1
.
6
2
(

a
P
k

4
4
.
1
7
1

)
9
3
.
5
6
(

a
P
k

5
7
.
8
3
3

s
’
g
n
u
o
Y
n
a
e
M

)
.
d
.
s
(

s
u
l
u
d
o
m

)
1
4
.
0
(

a
P
M
9
7
.
1

)
2
4
.
0
(

a
P
M
8
5
.
1

)
2
4
.
0
(

a
P
M
6
6
.
0

)
1
0
.
9
1
(

3

m
/
g
k

9
5
.
6
9
1

r
a
l
u
c
i
d
n
e
p
r
e
p

n
a
t
t
a
R

)
1
5
.
2
(

a
P
M
8
8
.
3

)
0
8
.
0
(

a
P
M
2
3
.
3

)
9
0
.
5
(

3

m
/
g
k

2
1
.
4
9
1

l
a
i
x
a
o
c

d
e
e
r

n
o
m
m
o
C

)
4
0
.
9
(

3

m
/
g
k

7
8
.
3
3
2

l
o
r
t
n
o
C

)
4
5
.
0
(

a
P
M
9
9
.
2

)
0
2
.
2
1
(

3

m
/
g
k

0
7
.
8
4
2

g
n
i
t
e
k
c
a
j

n
a
i
s
s
e
H

)
5
4
.
4
(

a
P
M
2
0
.
4

)
2
4
.
6
(

a
P
M
1
2
.
9

)
9
1
.
6
(

3

m
/
g
k

7
7
.
6
2
2

r
a
l
u
c
i
d
n
e
p
r
e
p

n
a
t
t
a
R

)
9
8
.
2
(

3

m
/
g
k

4
1
.
8
9
1

l
a
i
x
a
o
c

d
e
e
r

n
o
m
m
o
C

)
4
0
.
1
(

a
P
M
6
9
.
2

)
8
5
.
0
1
(

3

m
/
g
k

0
6
.
7
1
2

l
o
r
t
n
o
C

)
6
4
.
0
(

a
P
M
1
0
.
3

)
7
9
.
1
1
(

3

m
/
g
k

5
0
.
4
6
2

g
n
i
t
e
k
c
a
j

n
a
i
s
s
e
H

)
8
5
.
0
(

a
P
M
4
2
.
2

)
6
5
.
4
(

a
P
M
0
5
.
8

)
6
3
.
0
(

a
P
M
7
1
.
2

)
1
9
.
3
(

3

m
/
g
k

8
9
.
0
4
2

r
a
l
u
c
i
d
n
e
p
r
e
p

n
a
t
t
a
R

)
0
9
.
7
(

3

m
/
g
k

7
4
.
9
0
2

l
a
i
x
a
o
c

d
e
e
r

n
o
m
m
o
C

)
2
1
.
8
(

3

m
/
g
k

9
5
.
0
2
2

l
o
r
t
n
o
C

)
4
0
.
1
(

a
P
M
0
2
.
2

)
9
2
.
1
1
(

3

m
/
g
k

5
8
.
6
4
2

g
n
i
t
e
k
c
a
j

n
a
i
s
s
e
H

)
0
3
.
0
(

a
P
M
7
8
.
1

)
1
4
.
2
(

a
P
M
9
8
.
7

)
5
5
.
3
(

3

m
/
g
k

9
0
.
4
2
2

r
a
l
u
c
i
d
n
e
p
r
e
p

n
a
t
t
a
R

)
4
2
.
6
(

3

m
/
g
k

8
0
.
3
0
2

l
a
i
x
a
o
c

d
e
e
r

n
o
m
m
o
C

)
7
4
.
6
(

3

m
/
g
k

7
6
.
9
0
2

)
8
8
.
9
(

3

m
/
g
k

8
4
.
0
3
2

g
n
i
t
e
k
c
a
j

n
a
i
s
s
e
H

l
o
r
t
n
o
C

m
m
0
.
1

m
m
0
.
1

m
m
0
.
1

m
m
0
.
1

-

-

-

-

5
.
0

5
.
0

5
.
0

5
.
0

m
m
0
.
3

m
m
0
.
3

m
m
0
.
3

m
m
0
.
3

-

-

-

-

5
7
.
0

5
7
.
0

5
7
.
0

5
7
.
0

m
m
0
.
2
1

m
m
0
.
2
1

m
m
0
.
2
1

m
m
0
.
2
1

m
m
0
.
2
1

m
m
0
.
2
1

m
m
0
.
2
1

m
m
0
.
2
1

-

-

-

-

-

-

-

-

0
.
4

0
.
4

0
.
4

0
.
4

5
.
0

5
.
0

5
.
0

5
.
0

H
_
S
B

R
_
S
B

V
_
S
B

M
B

S
B

H
_
M
B

R
_
M
B

V
_
M
B

H
_
L
B

R
_
L
B

V
_
L
B

L
M
S
B

L
B

H
_
L
M
S
B

R
_
L
M
S
B

V
_
L
M
S
B

)
.
d
.
s
(

y
t
i
s
n
e
d

n
a
e
M

n
o
i
t
i
s
o
p
m
o
c

e
r
b
i
F

e
z
i
s

e
t
a
l
u
n
a
r
G

e
p
y
T

Table 8: Summary of specimen types parameters, resulting dried densities, and compressive
properties.

Deliverable D5.2

Page 37 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 19: Box plots for compressive Young’s modulus results (a) and ultimate strength results
(b).

Page 38 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 20: One dried specimen from the Compression: eﬀect of particle size & reinforcement
series.

Deliverable D5.2

Page 39 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 21: Specimens from the Compression: eﬀect of particle size & reinforcement series.

Page 40 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

strength in the case of the BS, BM, and BSML families, with a reduction of the deviation between
the ﬁrst and third quartiles. The containment of stress applied to the specimens within tight
boundaries forces the arrangement of the particles within, restricting the ability for particles to
arrange freely. Jacketed specimens have an average reduction of 0.12 MPa to the controls as per
Young’s modulus (s.d. 0.19), and an average decrease of 16.97 kPa to the controls as per ultimate
strength (s.d. 20.05). The jacket has two important advantages: it oﬀers a durable alternative to
low-ductility mycelial mats usually grown on the external boundary of MBC, and we hypothesise
that it can substantially contribute to an increase in fracture resistance performance in shearing
and bending load cases.

Fibres perpendicular to load

Specimens supplemented with rattan ﬁbres display a lower performance across particle sizes
considering their median in Young’s modulus and ultimate strength. The mean ultimate strength
follows the performance of the mean of the controls (Fig.22) with an average reduction of 71.81
kPa (s.d. 8.45). This suggests that, should the production conditions of such MBC improve to
reduce the dispersion of results and increase the material behaviour predictability, introducing
strategically parsed weakness points in composites could ﬁnd a use with calibrated materials by
tuning their failure mode.

Figure 22: Parameters interaction graph for mean Young’s modulus (a) and mean ultimate
strength (b).

Fibres coaxial to load

Common reed ﬁbre reinforced specimens resulted in the largest standard deviations in Young’s
moduli (reported in Table 8), especially in the BM and BL families. This is due to the ﬁbres
having partially misaligned to the load case axis during specimen production. Nevertheless,
results suggest that MBC can be successfully stiﬀened with regards to their use case. The eﬀect
of this stiﬀening on the ultimate strength is less obvious as we note that the smaller particles (BS
and BM families) tend to perform better without reinforcement coaxial to load. This is a result
of the inherent large displacement of the ﬁbres within the specimens under stress due to their

Deliverable D5.2

Page 41 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

stiﬀness, thus initiating an early critical failure. The mean Young’s moduli (Table 8) display a
clear improvement compared to the controls: we observe an average increase of a factor 2.86 (s.d.
0.6) between the mean of the controls and the mean of the ﬁbre coaxial to load specimens. As
per mean ultimate strengths, they improved in the BL and BSML families when compared to
controls (respectively by a factor 1.18 and 1.43), but decreased in the smaller particles families
BS and BM (respectively by a factor 0.86 and 0.88).

Principal substrate particles

The use of smaller particles in MBC increases the surface area to volume ratio of what serves
as a nutrient for the fungus, hence facilitating its access to it. Filamentous fungi also need air
access to develop a mycelium, thus space between particles, if one desires to have it synthesise
a biomass that has a considerable eﬀect over its mechanical properties. The small granulation
essentially qualiﬁes as a dust with particles size in the 0.5 – 1.0 mm interval, leaving minimal
amounts of air between particles within the constrained boundaries of the specimen mould. The
best performing BM family (as per ultimate strength) is composed of 0.75 – 3.0 mm particles,
thus embedding particles of a comparable size to the BS dust, while containing particles that are
three times as long.
The plastic strain of the composite is contributed to only by particles in such composite, which
is clearly exhibited in the range of results in Fig.19. As introduced with the common reed and
rattan containing specimens, the dewetting behaviour of the larger particles or reinforcements
present in the composite is a principal contributor to damage nucleation. Furthermore, the
shape, nature, and distribution of particles in a two-phase composite has been shown to have
a substantial inﬂuence over the load transfer between members and hence their overall stiﬀness
[115].

Statistical analysis

The result distributions are two-tailed. The mean of Fisher’s deﬁned kurtosis for Young’s mod-
ulus series is (cid:0)0:3328 (s.d. 0.9353) and (cid:0)1:0564 for ultimate strength (s.d. 0.5343). Fisher-
Pearson’s skewness coeﬃcient mean for Young’s modulus is 0.5834 (s.d. 0.8540), and 0.1733
for ultimate strength (s.d. 0.5137). The distributions are considered normal [54], which was
veriﬁed for ultimate strength and Young’s modulus results with the Shapiro-Wilk test (respect-
ively p=0.9224 and p=0.0030, (cid:11)=0.001). Equality of variances was controlled with the Levene
test; Young’s modulus result variances are not equal (p=1.3940e-05, (cid:11)=0.05), neither are ul-
timate strength ones (p=0.0459, (cid:11)=0.05). Welch’s ANOVA was conducted for the two para-
meters: ﬁbre placement for Young’s modulus and ultimate strength (respectively p=0.0001 and
p=0.0013), and particle size for Young’s modulus and ultimate strength (respectively p=0.0030
and p=4:6462e (cid:0) 09). The mean values of specimen groups are signiﬁcantly diﬀerent ((cid:11)=0.005).
Using the pairwise Games-Howell test we identiﬁed the most signiﬁcant reinforcement to be the
ﬁbre coaxial to load against ﬁbre perpendicular to load, the control, and hessian jacketing (all
p=0.001 as per Young’s modulus; respectively p=0.030, p=0.004, and p=0.004 as per ultimate
strength; (cid:11)=0.05). Continuing this test, we identiﬁed the most signiﬁcant aggregate size to be
the 0.5 - 1.0 mm interval (BS family) against the BM and BL families (respectively p=0.029 and
p=0.036 as per Young’s modulus; all p=0.001 as per ultimate strength; (cid:11)=0.05). The BS family
had a signiﬁcant diﬀerence to the BSML family as per aggregate size over ultimate strength
(p=0.001, (cid:11)=0.05), but not over Young’s modulus (p=0.106, (cid:11)=0.05).

Page 42 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 23: Rattan embedding specimen used for compressive characterisation, after having been
tested.

Deliverable D5.2

Page 43 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 24: Hessian jacketed specimen used for compressive characterisation, after having been
tested.

Page 44 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Type

Composition

BM_GR G. resinaceum, BM
PM_GL G. lucidum, pine wood
BS
BM
BL
BSML

G. lucidum, BS
G. lucidum, BM
G. lucidum, BL
G. lucidum, BSML

n Mean density (s.d.) Mean Young’s
modulus (s.d.)
1.75 MPa (0.17)
0.34 MPa (0.00)
1.79 MPa (0.41)
3.32 MPa (0.80)
2.96 MPa (1.04)
2.17 MPa (0.36)

207.49 kg/m3 (5.79)
114.00 kg/m3 (2.19)
209.67 kg/m3 (6.47)
233.87 kg/m3 (9.04)
217.60 kg/m3 (10.58)
220.59 kg/m3 (8.12)

5
2
6
6
6
6

Mean ultimate
strength (s.d.)
0.11 MPa (0.02)
0.04 MPa (0.00)
0.17 MPa (0.04)
0.31 MPa (0.06)
0.25 MPa (0.03)
0.24 MPa (0.03)

Table 9: Summary of specimen types parameters, resulting dried densities, and compressive
properties. Values in the lower section of the table are reported from section 5.1.

5.2 Compression: eﬀect of species

We investigated the eﬀect of species on the mechanical behaviour in compression of MBC, with a
G. resinaceum cultivated on medium granulate size beech wood (Fig.27), and comparing results
to the BM group of the experimental series presented in section 5.1. Five replicates could
be tested in this group. So as to conﬁrm the lower softwood decaying potential of the G.
lucidum species, a pine wood specimen group was tested too. Because of a low colonisation ratio
as compared to hardwood cultivated composites, only two replicates could be tested for this
conﬁguration (Fig.26). Even with a lower number of replicates, it appears that the pine substrate
is not a relevant substrate for the G. lucidum species. This limited screening is nonetheless
consistent with section 2.5 regarding guaiacyl lignins presence in gymnospermous wood and their
higher resistance to fungal decay. G. resinaceum cultivated specimens resulted in a compressive
behaviour similar to small granulation beech wood colonised by G. lucidum. The cultivation
protocols were identical between these two series. Experimental parameters per specimen type
and resulting mean density, mean Young’s modulus and mean ultimate strength are presented
in Table 9. Box plots of the results for Young’s modulus and ultimate strength are presented in
Fig.25.

Statistical analysis

Because only two replicates could be tested for PM_GL, this series results are not considered
signiﬁcant and were not statistically analysed. The BM_GR series was tested against the BM
series from section 5.1. The result distributions are two-tailed. The mean of Fisher’s deﬁned
kurtosis for BM_GR Young’s modulus is (cid:0)1:8933 and (cid:0)1:3775 for ultimate strength. Fisher-
Pearson’s skewness coeﬃcient mean for BM_GR Young’s modulus is 0.0806, and 0.4800 for
ultimate strength. The distributions are considered normal [54], which was veriﬁed for ulti-
mate strength and Young’s modulus results with the Shapiro-Wilk test (respectively: p=0.6605,
p=0.9611; (cid:11)=0.5). Equality of variances between the BM_GR and BM groups was controlled
with the Levene test; Young’s modulus and ultimate strength results variances are considered
equal (respectively p=0.0627, p=0.0799; (cid:11)=0.05). Welch’s ANOVA was conducted for Young’s
modulus and ultimate strength (respectively p=0.0037 and p=0.0054). The mean values of
specimen groups are signiﬁcantly diﬀerent ((cid:11)=0.01). Using the pairwise Games-Howell test we
conﬁrm the signiﬁcant diﬀerence between groups for Young’s modulus and ultimate strength
(respectively p=0.004, p=0.005; (cid:11)=0.005).

Deliverable D5.2

Page 45 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 25: Box plots for Young’s modulus results (a) and ultimate compressive strength results
(b).

5.3 Flexion: eﬀect of reinforcements

We investigated the eﬀect of diverse reinforcements on the mechanical behaviour in ﬂexion of
MBC, using three levels: inner hessian (I), hessian jacketing (H), and rattan ﬁbres in the length
(R). These strategies are illustrated in Fig.28, and a picture of specimens is presented in Fig.29.
The principal substrate of the specimen is the medium granulation of beech wood (0.75 - 3.0
mm). The BM series was made following protocol CITA, and the HC series was made following
protocol Mogu. Isotropic controls were added to the experimental series (BM and HC groups).
Experimental parameters per specimen type and resulting mean density, mean Young’s modulus
and mean modulus of rupture are presented in Table 10. Box plots of the results for Young’s
modulus and modulus of rupture are presented in Fig.30, and parameters interaction in Fig.31.
We can witness that besides rattan, the reinforcements did not have the same eﬀect over the
ﬂexural metrics in both series. Considering the signiﬁcant variability in specimen manufacturing
it can hardly be concluded that jacketing or inner hessian can contribute to stirring the composite
ﬂexural behaviour in the Mogu series with these experimental data. The mean of Young’s
modulus for the BM_R series, 1.38 GPa, is the closest in the reported MBC state-of-the-art to
industrially viable products for furniture and interior applications, as medium-density ﬁbreboard
(MDF) elastic modulus is typically of 4 GPa, with a modulus of rupture of 10 MPa for a density
It can be noted that the mechanical failure of the specimens was related to
of 750 kg/m3.
dewetting of the principal substrate, rattan ﬁbres did not fail nor deform plastically. In Fig.29
we can observe tested specimens crossing at mid-length: they have been softened by aggregates
debonding but stayed integral.

Statistical analysis

The result distributions are two-tailed. The mean of Fisher’s deﬁned kurtosis for Young’s mod-
ulus series is (cid:0)0:8114 (s.d. 0.5732) and (cid:0)0:5973 for modulus of rupture (s.d. 0.3400). Fisher-
Pearson’s skewness coeﬃcient mean for Young’s modulus is 0.4158 (s.d. 0.6169), and (cid:0)0:1262
for modulus of rupture (s.d. 0.7371). The distributions are considered normal [54], but did not

Page 46 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 26: One specimen of G. lucidum colonised pine wood used for compressive characterisa-
tion, after having been tested.

Deliverable D5.2

Page 47 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 27: One specimen of G. resinaceum colonised beech wood used for compressive charac-
terisation, after having been tested.

Page 48 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Mean density (s.d.) Mean Young’s
modulus (s.d.)
192.71 MPa (52.40)
197.33 MPa (45.56)
375.14 MPa (1.82)
1:38e3 MPa (630.13)
38.30 MPa (6.72)
20.41 MPa (11.48)
11.02 MPa (3.73)
318.19 MPa (68.50)

216.27 kg/m3 (17.00)
215.12 kg/m3 (8.01)
228.77 kg/m3 (11.60)
245.51 kg/m3 (9.62)
123.55 kg/m3 (2.42)
94.74 kg/m3 (10.96)
93.49 kg/m3 (4.98)
103.51 kg/m3 (6.11)

Mean modulus
of rupture (s.d.)
0.12 MPa (0.03)
0.11 MPa (0.02)
0.18 MPa (0.03)
0.62 MPa (0.14)
0.05 MPa (0.01)
0.04 MPa (0.02)
0.03 MPa (0.01)
0.31 MPa (0.02)

Type

Fibre
composition
Control
Inner hessian

BM
BM_I
BM_H Hessian jacketing
BM_R Rattan
Control
HC
HC_I
Inner hessian
HC_H Hessian jacketing
HC_R Rattan

Table 10: Summary of specimen types parameters, resulting dried densities, and ﬂexural prop-
erties.

satisfy the Shapiro-Wilk test (modulus: p=2:1097e (cid:0) 09, modulus of rupture: p=2:7600e (cid:0) 07,
(cid:11)=0.01). Equality of variances was controlled with the Levene test; Young’s modulus result vari-
ances are not equal (p=0.0005, (cid:11)=0.05), neither are modulus of rupture ones (p=9:9545e (cid:0) 05,
(cid:11)=0.05). Welch’s ANOVA was conducted for Young’s modulus and modulus of rupture regard-
ing reinforcement strategies (respectively p=0.035 and p=0.000009), and regarding "cultivation
media", between CITA and Mogu protocols (respectively p=0.0071 and p=0.043). The mean val-
ues are signiﬁcantly diﬀerent ((cid:11)=0.05). Across the two series, using the pairwise Games-Howell
test we conﬁrm the signiﬁcant diﬀerence between the rattan group and the other reinforcements
for Young’s modulus and modulus of rupture ((cid:11)=0.05), besides for inner hessian group against
the controls, and hessian jacketing against rattan (p=0.062). We conﬁrm the signiﬁcant diﬀer-
ence between cultivation media for Young’s modulus and modulus of rupture too ((cid:11)=0.02). In
the CITA cultivated series, we conﬁrm the signiﬁcant diﬀerence between all groups for Young’s
modulus and modulus of rupture ((cid:11)=0.05), but for inner hessian against control.

5.4 Tension: eﬀect of particle size

We investigated the eﬀect of particle size on the mechanical behaviour in tension of MBC, using
two granulations: small (BS), medium (BM). The species is a G. lucidum. The dog-bone specimen
design is presented in Fig.32; the smallest section of the dried dog-bones was of 20 x 36 mm,
and 51 mm in length. Experimental parameters per specimen type and resulting mean density,
mean Young’s modulus and mean ultimate strength are presented in Table 11. Box plots of the
results for Young’s modulus and ultimate strength are presented in Fig.34.

Statistical analysis

The result distributions are two-tailed. The mean of Fisher’s deﬁned kurtosis for Young’s mod-
ulus series is (cid:0)1:4313 (s.d. 0.0889) and (cid:0)1:4840 for ultimate strength (s.d. 0.0013). Fisher-
Pearson’s skewness coeﬃcient mean for Young’s modulus is (cid:0)0:2065 (s.d. 0.4445), and (cid:0)0:1546
for ultimate strength (s.d. 0.5246). The distributions are considered normal [54], which was
veriﬁed for ultimate strength and Young’s modulus results with the Shapiro-Wilk test ((cid:11)=0.05).
Equality of variances was controlled with the Levene test; Young’s modulus and ultimate strength
variances are considered equal (respectively p=0.1011, p=0.7427; (cid:11)=0.05). Welch’s ANOVA was
conducted for Young’s modulus and ultimate strength (respectively p=0.0030 and p=0.0238).
The mean values of specimen groups are signiﬁcantly diﬀerent ((cid:11)=0.05). Using the pairwise
Games-Howell test we conﬁrm the signiﬁcant diﬀerence between groups ((cid:11)=0.05), but not re-

Deliverable D5.2

Page 49 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 28: Fibre placement strategies in the ﬂexion series (left to right): control, inner hessian,
hessian jacketing, rattan ﬁbres.

Figure 29: CITA cultivated specimens used for ﬂexural characterisation, after having been tested.

Page 50 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 30: Box plots for Young’s modulus results (a) and modulus of rupture results (b).

Deliverable D5.2

Page 51 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 31: Parameters interaction graph for mean Young’s modulus (a) and mean modulus of
rupture (b).

Type Granulate size

BM
BS

0.75 - 3.0 mm
0.5 - 1.0 mm

Mean density (s.d.) Mean Young’s
modulus (s.d.)
0.84 MPa (0.15)
0.55 MPa (0.08)

217.35 kg/m3 (17.71)
185.15 kg/m3 (4.82)

Mean ultimate
strength (s.d.)
0.08 MPa (0.01)
0.07 MPa (0.01)

Table 11: Summary of specimen types parameters, resulting dried densities, and tensile proper-
ties.

garding the inner hessian group compared to the control group (both regarding Young’s modulus
and ultimate strength).

5.5 Kagome structural analysis

We investigated the eﬀect of the structural performance of the kagome Weaves for the mycelium
assembly reinforcement. The Strategy is to embed the weaves in the mycelium panels, through
the FEM analysis to simulate the structural capacity in the diﬀerent weave densities of the weaves
and the material. The simulation examines two parameters, the types of ﬁbres and the schemes
in the kagome. The ﬁbres parameter focused between artiﬁcial ﬁbres and natural ﬁbres, carbon
ﬁbres, bamboo and rattan. The size is based on the standard industrial dimensions. Experiment
the diﬀerence of types of ﬁbres is the Young’s modulus, the carbon ﬁbres are around 1:7e6 MPa
and bamboo is in the range 4e3 – 6e3 MPa. The analysis of the kagome was conducted to assess
and validate the structural load-bearing capacity of the simulation model. The strips that cross
the singularity have high bending and torsion. Due to high bending and torsion, the damage
of ﬁbres could either happen during the fabrication or in the diﬀerent load cases Fig.35. We
investigated two approaches for a diﬀerent condition, one is only to consider the behaviour of
weaves without the weight from mycelium assembly. In this setup, the strip elements connect by
the cables where the interlacing point, to give some degree of freedom in rotation. The other one
integrated the mycelium assembly in one single mesh. Each quad face has been materialised in

Page 52 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 32: ASTM D1037 informed dog-bone design for the ﬂexion series (wet specimens dimen-
sions).

Figure 33: Two of the specimens tested for tensile characterisation; notice the arrangement of
particles and mycelium in the section.

Deliverable D5.2

Page 53 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 34: Box plots for Young’s modulus results (a) and ultimate tensile strength results (b).

diﬀerent mechanical properties enable to demonstrate the integration of the kagome weaves and
mycelium assembly. The idea for two analyses is to respond to the diﬀerent stages of construction.
The ﬁrst one is to test the bending and torsion stress, and the second is to understand the
behaviour of the ﬁnal assembly.

Figure 35: The cable at the intersection constrains the movement and connects two strips.

Structural analysis

Three diﬀerent materials of the kagome weave and mycelium assemblies could be tested by meas-
uring maximum displacement. At the simulation, the material set is as follows: the bamboo is
Young’s Modulus 5e3 MPa, carbon ﬁbres is 1:7e5 MPa, and mycelium is 17 MPa. In comparison
of the kagome weaves with the 0.2 kN/m2 point load on the top, the maximum displacement:
bamboo is 53.1 mm, for the carbon ﬁbres is 9.87 mm and the hybrid is 21.0 mm. The highest von
Mises stress surrounds the singularity: bamboo 117.0 MPa, carbon ﬁbres 1:379e3 MPa, and the
hybrid 760.2 MPa (Fig.36). The result conﬁrms the considerable improvement by introducing
carbon ﬁbres reinforcement as the kagome hybrid system. It reduced the stress and increased

Page 54 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

the load capacity, and lessened the use of carbon ﬁbres.
In the second analysis, the shell model is only considering mycelium use, with a density of 200
kg/m3. The model shows a 10.5 mm maximum displacement under its own dead load (Fig.37).
Most displacement occurs near the ground support. After applying the kagome substructure, the
maximum displacement was reduced to 1.84 mm – 2.05 mm (Fig.38,39,40), with considering the
wet density of a mycelium composite (350 kg/m3). The results contribute to supporting the idea
that kagome weaves can be used as a structural framework for a mycelium composite secondary
structure (which could be used as an insulation material for instance), and the eﬀectiveness of
hybridising the weave with highly performative synthetic composite (carbon ﬁbre) along with
natural ﬁbres (such as bamboo) suggests that an eﬃcient distribution could contribute to rein-
force structurally demanded areas, while controlling the environmental impact of the architctural
system.

Figure 36: The von Mises stress map of bamboo (a), carbon (b), and hybrid structures (d).

5.6 Conclusion

The result of the experimental series are plotted in Ashby maps along with a curation of state-of-
the-art values for compression (modulus: Fig.42, ultimate strength: Fig.43), tension (modulus:
Fig.44, ultimate strength: Fig.45), and ﬂexion (modulus: Fig.46, ultimate strength: Fig.47).
These maps gather evidences produced with approximately ﬁfteen diﬀerent fungal species. The
series described at section 5.1 is labelled rigobello_myceliumbased_2021 in the maps, cita_Un-
publishedData_2021 refers to values currently only reported in this document and issued from
CITA cultivated specimens. moguCita_UnpublishedData_2021 refers to values currently only
reported in this document and issued from Mogu cultivated specimens.
Both in compression and tension, the eﬀect of particle of the smallest granulation (BS) was
signiﬁcantly lower with regards to ultimate strength and elastic modulus ((cid:11)=0.05). As seen in
section 2.3, this may be directly inﬂuenced by the presence of air, of cavities between substrate
particles for a mycelium to form. Being dust-like, the small granulation (0.5 – 1.0 mm) leave the
smallest amount of air within the substrate. Mycelium develops nonetheless, and as the number
of particles is higher when their average unit size lowers, we hypothesise that properties of a
mycelium interfacing between aggregates plays a more signiﬁcant role in the artefact mechanical
role. Furthermore, the ﬁner grinding of wood aggregates contributes to softening the individual
particles; this is due to particles resulting from the shredding process being more elongated on
average, and being structurally damaged. A picture of BS, BM, and BL particles is presented
in Fig.41. While being consistent industrial products, the granulates display considerable geo-
from small elongated ﬁbres, to ﬂaky planar medium-sized
metrical variations between scales:

Deliverable D5.2

Page 55 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 37: Mycelium shell without kagome reinforcement; compression stress concentrates near
ground conditions.

Figure 38: Top view of the bamboo weaves and mycelium simulation in the maximum vertical
loading.

Page 56 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 39: Top view of the carbon weaves and mycelium simulation in the maximum vertical
loading.

Figure 40: Top view of the hybrid weaves and mycelium simulation in the maximum vertical
loading.

Deliverable D5.2

Page 57 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

particles, to bulky and angular large granulates).
Comparing them to the compression results of BM specimens, we observed that changing fungal
species from G. lucidum to G. resinaceum for an identical experimental protocol resulted in a
signiﬁcantly lower strength and elasticity. To tie this result to section 2.4, a comparative study
of isolated mycelia from these two species could be predicted to display a greater elasticity of
G. resinaceum mycelium, thus expanding the manufacturable material properties. As expected
from the enzymatic activity review, and although G. lucidum is one of the most versatile select-
ive deligniﬁer, specimens cultivated on pine wood sawdust showed a limited colonisation density
and lower performances. Cultivation protocols on gymnospermous woods appear to necessitate
changes to ones applied on hardwoods, but the statistical signiﬁcance of this experiment being
non-satisfactory it should be conducted again to be conclusive.
Reinforcement wise, we studied three typologies in compression: rattan ﬁbres perpendicular to
load, wheat straw ﬁbres coaxial to load, and hessian jacketing coaxial to load. The addition of
ﬁbre coaxial to load had a signiﬁcant eﬀect over Young’s elastic modulus and ultimate strength
((cid:11)=0.05). Because of the wide range in particle sizes used and ﬁbre composition typologies,
the signiﬁcant diﬀerence between specimen groups supports our hypothesis that the two-phase
particulate model is suited for future MBC studies ((cid:11)=0.005). From this observation, future
studies might involve exploring a wider variety of particle shapes, natures, and distributions as
these parameters have been shown to have a signiﬁcant inﬂuence over the elastic and plastic be-
haviour of composites [115]. We demonstrated that the modifying of specimens could be attained
with contrasting examples of coaxial reinforcement and perpendicular fracture initiators, with
signiﬁcant eﬀect ((cid:11)=0.005). However, it should be noted that ﬁbre placements were subjected to
variability as ﬁbres could partially misalign with the load axis or its perpendicular during pro-
duction. This suggests that the standard deviation of the results can be reduced by improving
the accuracy in production protocol.
Beyond the performances, the various reinforcement strategies in the ﬂexion specimens resulted
in diﬀerent dimensional stability. Across CITA specimens, no shrinkage was measured in thick-
ness nor width, while controls reduced by 3.5 % in length, 2.7 % for the inner hessian group, 1.7
% for the hessian jacketing group, and 0.8 % regarding the rattan group. This is of great prac-
tical interest to ensure process control and product quality, and can be integrated and studied
further for a variety of geometries. Then both hessian jacketing and rattan ﬁbres introduction
contributed to signiﬁcantly improve the elasticity and strength of composites ((cid:11)=0.05). In the
Ashby map Fig.46 we can observe that reinforcements can increase slightly the density of the
composites depending on their nature, but nonetheless eﬃciently increase the stiﬀness of the
specimens (ultimate ﬂexural strength plotted in Fig.47). While the number of represented data
points is limited in Fig.46, we can compare the steep increase eﬀect of reinforcements compared
to a ﬂatter improvement by a densiﬁcation strategy from the state-of-the-art [4]. Compressive
Young’s modulus as a function of ultimate strength is presented in Fig.48. Compressive results
that we report in section 5.1 have us observe a steeper increase in stiﬀness by means of composi-
tion as compared to densiﬁcation strategies [50] (densiﬁcation can be observed for this data set in
Fig.42). Furthermore, we can observe that strategies of supplementation and/or species curating
can lead in a performative departure from the state-of-the-art [142]. It can also be noted that
no peer-reviewed reproducibility studies have been published yet.
Structural analysis of a mycelium shell and a hybrid kagome structural gridshell have demon-
strated the material eﬃciency interest. Such conﬁgurations have only been explored digitally,
and will be prototyped in the months to come in parallel to the development of a constructive
methodology for embedding computationally active mycelium composites.

Page 58 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 41: Small (BS), medium (BM), and large (BL) granulations of beech wood.

Deliverable D5.2

Page 59 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 42: Compressive Young’s modulus as a function of density. Data retrieved from [142, 138,
100, 50, 38, 60]. Larger points correspond to composition strategies.

Figure 43: Ultimate compressive strength as a function of density. Data retrieved from [72, 4,
60]. Larger points correspond to composition strategies.

Page 60 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 44: Tensile Young’s modulus as a function of density. Data retrieved from [142, 4]. Larger
points correspond to composition strategies.

Figure 45: Ultimate tensile strength as a function of density. Data retrieved from [142, 4, 78].
Larger points correspond to composition strategies.

Deliverable D5.2

Page 61 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 46: Flexural Young’s modulus as a function of density. Data retrieved from [72, 4, 121].
Larger points correspond to composition strategies.

Figure 47: Modulus of rupture as a function of density. Data retrieved from [37, 72, 4, 78, 121].
Larger points correspond to composition strategies.

Page 62 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 48: Compressive Young’s modulus as a function of ultimate compressive strength. Data
retrieved from [138, 142, 100, 50, 60]. Larger points correspond to composition strategies.

Deliverable D5.2

Page 63 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

6 Thermal characterisation

We investigated the thermal transmittance (U-value) of two series of MBC panels - a hemp shive
substrate series and a birch shaving substrate series. Both series were cultivated as described in
section 3.2. A composed picture of representative specimens is presented in Fig.49. The hemp
shive series consisted of panel thicknesses: 40 mm; 80 mm; 120 mm, with six replicates of each.
The birch shaving series consisted of panel thicknesses: 40 mm; 80 mm, with six replicates of each.
A custom thermal chamber was constructed to perform U-value measurements on four samples
simultaneously. The measurements were taken and analysed according to ISO 9869, as described
in section 3.9. The thermal conductivity ((cid:21)) was determined from the U-value results for each
MBC material series. The thermal conductivity values were then used to calculate predicted
U-values for various assemblies of MBC materials. Two assembly conﬁgurations were empirically
tested and results compared with their predictions to verify their use for design speciﬁcation.

6.1 Material focused U-value measurement results

For this part of the experimental plan, and in common with studies reported in the literature,
we follow a material focused thermal characterisation. However, in contrast to the literature,
we measure thermal transmittance (U-value) rather than thermal conductivity or speciﬁc heat
transfer. Our motivation for this deviation is that the experimental plan extends to consider
the composite behaviour of preliminary propositions for assemblies of MBC building envelopes.
Therefore, an ’in-situ’ measurement methodology is necessary. Conducting all measurements
with the same method allows portability of results. Figure 15 illustrates the thermal chamber
and measurement setup for diﬀerent panel thicknesses. Figure 50 presents a comparison of U-
value results for hemp shive (H) and birch shaving (BS) substrate series, across 40 mm & 80
mm panel thicknesses and all replicates. Figure 54 presents the calculated thermal conductivity
results for H & BS substrate series, across all panel thicknesses and replicates.

6.2 Assembly focused U-value predictions

A cavity wall typology was selected as the basis for making thermal transmittance (U-value)
predictions of hypothetical MBC building envelopes. A series of conﬁgurations was designed
with variations of panel thickness, but keeping consistency of substrate - i.e not mixing H &
BS series panels. Figure 51 presents the calculation tables of these preliminary MBC envelope
build-ups, together with the predicted U-value results. These calculation tables use the mean
thermal conductivity values established from the material tests and presented in Table 12. For
comparison, Part L of the UK building regulations (Conservation of fuel and power), stipulates
that new build external wall construction must achieve 0.18 W/(m²K) or lower to comply. H_-
120-20-120 provides the best U-value prediction of 0.20 W/(m²K) for a build-up of 260 mm
envelope depth. It is expected that this assembly is predicted to be the best performing as it
has the largest insulation depth. The U-value could be improved with an increase in material
depth, or an increase in the air-cavity. Very generally, conventional construction of cavity walls
will yield a depth of >300 mm. From this prediction of performance, it is plausible to suggest
that MBC could potentially result in space saving envelopes.

6.3 Assembly focused U-value measurement results

A cavity wall conﬁguration was implemented for empirical measurement and tested using either
H-series or BS-series panels. The conﬁguration was: 40 mm panel (outer face); 20 mm air cavity;

Page 64 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 49: Representative specimens tested for thermal characterisation. 40 mm birch shavings
(a), 80 mm birch shavings (b), 40 mm hemp shives (c), 80 mm hemp shives (d), 120 mm hemp
shives (e).

Deliverable D5.2

Page 65 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 50: Comparison of thermal transmittance results (U-value) across two material series with
two panel thicknesses (0.04 & 0.08m) and six replicates. On average, the hemp series provides
better insulation performance.

40 mm (inner face). Three replicates for each material series were measured. The replicates were
assembled using panels 1 & 2 for conﬁguration sample 1; panels 3 & 4 for conﬁguration sample
2; panels 5 & 6 for conﬁguration sample 3. Figure 52 presents the measured values and ﬁgure 53
presents a comparison of the predicted and measured U-value results. The measured results are
substantially higher than their equivalent predictions. Despite the scarce number of data points,
the trend is clear and requires further investigation before publication of results.

Statistical analysis

Because only three replicates could be tested for the two buildups (H_40-20-40 and BS_40-
20-40), these series results are not considered signiﬁcant and were not statistically analysed.
The result distributions are two-tailed. The mean of Fisher’s deﬁned kurtosis for thermal con-
ductivity is (cid:0)0:9254 (s.d. 0.3667). Fisher-Pearson’s skewness coeﬃcient mean is 0.3471 (s.d.
0.5406). The distributions are considered normal [54], which was veriﬁed with the Shapiro-Wilk
test (p=0.2846, (cid:11)=0.05). Equality of variances between specimen groups was controlled with
the Levene test; they are considered not equal (p=0.0348, (cid:11)=0.05). Welch’s ANOVA was con-
ducted; the mean values of specimen groups are signiﬁcantly diﬀerent with regards to thickness
(p=0.00007, (cid:11)=0.01), but not for substrate type (p=0.8054, (cid:11)=0.01). Using the pairwise Games-
Howell test we conﬁrm that substrate types do not display a signiﬁcant diﬀerence between groups
(p=0.812, (cid:11)=0.05), while the 40 mm thickness group are signiﬁcant (p(cid:20)0.05, (cid:11)=0.05), but not
80 mm specimens against 120 mm thick ones (p=0.332, (cid:11)=0.05).

Page 66 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 51: Predicted U-values of hypothetical cavity wall building envelopes constructed using
MBC

Deliverable D5.2

Page 67 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 52: Measured U-values of hypothetical cavity wall building envelope using either H or
BS-series MBC 40 mm panels for both interior and exterior wall sections and 20 mm air cavity

Figure 53: Comparison of predicted and measured U-value results for the tested assembly using
two diﬀerent MBC substrate series.

Page 68 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Type

H_40

H_80

H_120

Substrate

Hemp shives

Total
thickness
40 mm

80 mm

120 mm

BS_40

Birch shavings

40 mm

BS_80

80 mm

H_40-20-40

Hemp shives

100 mm

BS_40-20-40 Birch shavings

100 mm

n Mean density

(s.d.)
107.30 kg/m3
(4.85)
93.21 kg/m3
(2.23)
85.47 kg/m3
(3.83)
96.97 kg/m3
(1.52)
87.76 kg/m3
(8.32)
-

-

6

6

6

6

6

3

3

Mean U-value
(s.d.)
0.9583 W/m2K
(0.0604)
0.6183 W/m2K
(0.0893)
0.4867 W/m2K
(0.0665)
1.0767 W/m2K
(0.0602)
0.7000 W/m2K
(0.0400)
0.5367 W/m2K
(0.0802)
0.5633 W/m2K
(0.0629)

Mean (cid:21) (s.d.)

0.0383 W/mK
(0.0024)
0.0495 W/mK
(0.0071)
0.0584 W/mK
(0.0079)
0.0431 W/mK
(0.0024)
0.0560 W/mK
(0.0032)
0.0537 W/mK
(0.0080)
0.0563 W/mK
(0.0064)

Table 12: Summary of specimen type parameters, number of replicates and thermal properties
for each type.

Figure 54: Box plots for thermal conductivity results.

Deliverable D5.2

Page 69 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

6.4 Conclusion

According to DIN 4108 Thermal insulation and energy saving in buildings, materials must have a
thermal conductivity below 0.1 W/mK to be deﬁned as insulation materials [107]. Our material
focused results show that all panel specimens across both material series qualify as insulation
materials, with thermal conductivity values (cid:21) in the range of 0.035 - 0.072 W/mK. This is
compared to orthodox insulation materials used in construction such as glass wool (0.04 W/mK),
mineral wool (0.04 W/mK), XPS foams (0.029 - 0.032 W/mK) and EPS foams (0.030 - 0.040
W/mK). This suggests that, with further improvements in production, MBC could act as a
plausible alternative to current mineral and oil based insulation materials. Improvements might
be sought through additives to the substrate to further reduce both conductive and radiative heat
transfer. It is generally expected that lower density materials should act as better insulators.
It is therefore notable that for equivalent panel thicknesses, the H-series out-performs the BS-
series (Fig. ) despite being composed of a higher density substrate. Causality has not been
investigated further, but we hypothesise that geometric features of internal morphology (e.g
air pocket volume and geometry, wall thickness of pockets, etc.) could play a role on thermal
performance, as has been reported in the literature [93]. The results from the assembly tests
are limited, with only three data points per assembly conﬁguration. To increase this dataset we
can perform a combinatorial mixing of panels which will yield an additional 12 data points, per
conﬁguration. This is being implemented in advance of planned publication. In addition, the
inclusion of primary structural elements, connectors and surface ﬁnishes should be considered
for a more complete assessment of the composite behaviour of MBC based building envelopes.

Page 70 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

7 Conclusions and perspectives

From the enzymes review we come to understand that resulting substrate ﬁtness and mycelia
mechanical behaviour can be adjusted by means of supplementation. This can be approached
by pH and temperature regulation, and sugars, nitrogen and metal ions supplementation. The
properties of mycelia have furthermore been shown to be inﬂuenced by such. Use of compounds
such as pH buﬀer CaCO3 could be systematically investigated in the future. Lignocellulosic sub-
strates selection can also have a signiﬁcant eﬀect, unsupplemented softwoods being theoretically
not chemically ﬁt for optimal white-rot decay. Use of sycamore or ash wood could be investigated
in the future as they display higher syringyl monomer contents. The C:N content of sycamore
being 401, supplementation may be necessary for optimal fermentation.
The eﬀect of water activity upon MBC colonisation-related performances, such as mechanical
response, has not been reported yet in the state-of-the-art. Considering reports in SSF stud-
ies of several folds increases in colonisation rates, investigating its eﬀect in MBC may result in
improved lead production times and performances. Moreover, qualifying the eﬀect of hydroxyl
groups availability in hemicellulose and cellulose depending on previous wood processing could
lead to further medium optimisation. We expect this particular aspect to be speciﬁcally useful
for paste-like substrate design, such as found in mycelium-related 3D printing.
Of course, stirring the focus of future researches towards such speciﬁc aspects may result in a
higher barrier to entry of a technique that has a strong vernacular interest. The use of FTIR
spectrometry by experts within a network of amateur practitioners could be useful to adapt
cultivation strategies in a constructive manner: the qualiﬁcation of an available substrate may
lead to specifying necessary processing for it, supplementation, species selection, and/or relevant
composition strategies. While industrial and academic researches may result in deﬁning a panel
of protocols for a selection of performative materials, constructive practices informed by locally
available materials could coexist.
To the best of our knowledge, we report for the ﬁrst time on particle geometries eﬀect on mech-
anical properties of MBC. Because the cohesion between them is critical to composite stiﬀness,
we expect the state-of-the-art to beneﬁt from systematically investigating the shape, size, and
orientation of these. This may lead in the future to the emergence of heterogeneous materials
with gradients of substrate qualities. Such development may lead to material optimisation in
design, and creative explorations.
Similarly, the investigation of natural ﬁbres composition in the substrate has proved to lead
to signiﬁcant increases in stiﬀness and ultimate strength. Composites manufacturing accuracy
improvements are expected to contribute to reducing the standard deviation of results. Further-
more, reinforcements may be strategised in developing eﬃcient or multi-functional composites,
for instance, in designing the principal substrate to be thermally performative and introducing
reinforcements to perform structurally. Such strategy could be used to mitigate manufacturing
risks: if a composite is subjected to production inconsistencies, its composition may increase its
stiﬀness and user safety. Moreover, the mean of Young’s modulus for the BM_R series, 1.38
GPa, is still close to three times lower than that of MDF (4 GPa), and approximately 16 times
lower in modulus of rupture (0.62 MPa, MDF: 10 MPa). Considering the higher density of MDF,
750 kg/m3, as compared to the BM_R series (245.51 kg/m3), a future investigation could be
a hybrid strategy of composition and densiﬁcation so to improve aggregates interlocking and
the resulting strength. The arming strategy adopted with the introduction of inner hessian did
not prove signiﬁcant, as it was positioned at mid-height of the specimens it was theoretically not
subjected to tension and could only partially contribute to the bending stiﬀness. A next iteration
of the experiment may look into diﬀerent soft arming positioning strategies by reﬂecting on the
most mechanically demanded areas of the specimens. This elastic modulus, to the best of our

Deliverable D5.2

Page 71 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

knowledge, is the highest reported for MBC, hence the closest to industrially viable products for
furniture and interior applications. We report on various specimen dimensional stability after
drying linked to diverse reinforcements, this aspect can be investigated in the future both as a
means to explore its design consequences, and for production control.
We analysed the plotting of various data sets from the state-of-the-art along with data from this
report against the various tuning strategies (supplementation, densiﬁcation, and composition).
Popular densiﬁcation techniques, happening at substrate preparation or by cold or hot-pressing,
display a larger increase in ultimate strength than stiﬀness compared to composition techniques.
We have experimented and reported the eﬀect of various ﬁbre addition: the use of rattan ﬁbres
perpendicular to the load case have resulted in a tighter standard deviation under compression
and lower stiﬀness and strength; this may be used in the future to produce materials with engin-
eered failure modes. In contrast, the addition of reed ﬁbres coaxial to the load case have been
shown to increase both stiﬀness and strength in a more balanced manner than densiﬁcation. The
addition of these hollow ﬁbres contributed to a slight decrease in density, thus resulting in a more
eﬃcient performance tuning. Substrate supplementation has also been shown to increase per-
formances without inﬂuencing density. Graphical and numerical analysis of MBC performances
reports may be developed in the future so as to analyse further the role of cultivation factors
upon composite behaviour.
While MBC are still in early development and necessitate further studies to become competitive
to other interior or structural applications, the idea of structuring the composites may scale to
architectural applications with the translation of composition strategies to using kagome.
In
this spirit, the structural analysis that we reported in this study probed the design space by
looking at the structural capacity of kagome gridshells when applied a wet mycelium composite
dead load. The use of members with higher bending stiﬀness, such as carbon ﬁbre, results in a
stable structure under load. Because this material is particularly environmentally detrimental,
we developed a hybridisation strategy in the weave, by using carbon ﬁbre members along the
principal lines of stress and bamboo members in the remaining of it. The simulation of this case
resulted in minimal displacements, it shall be prototyped in the future and other load cases will
be simulated (such as snow load, punctual load, wind load).
Finally, we report on the thermal characterisation of hemp shive based and birch shaving based
MBC. In the 40 and 80 mm thickness groups, the hemp substrate was approximately 11 % less
conductive than birch. The birch substrate material density was XX % lower than that of the
hemp material. Similarly to the diﬀerence observed between extruded polystyrene (XPS, (cid:21) 2
[0.029 - 0.032] W/mK at 25 °C for a 50 mm thickness) and expanded polystyrene (EPS, (cid:21) 2
[0.030 - 0.040] W/mK at 25 °C for a 75 mm thickness), the increase in material density pairs
in our case with a densiﬁcation of the mycelium present between substrate particles (we can
notice the diﬀerence in packing density in Fig.49), leading to an increase in hyphal density and
a higher number of phase-change. Molecular excitation is then diﬀused more eﬃciently in the
hemp substrate, resulting in a lower material conductivity. Future studies may investigate the
eﬀect of substrate chemical proﬁle to mycelial expression, and for a normalised density, report
on its eﬀect upon conductivity. To the best of our knowledge, we report for the ﬁrst time on the
eﬀect of composing buildups of MBC insulation. Although the unitary series supports the idea
that mycelium can be intrumentalised for its ability to build highly redundant hyphal networks
contributing to oppose energy transmission, such buildups have not been investigated yet. They
may be explored in the future with the addition of low thermal conductivity materials such as
ceramics.

Across this report we identiﬁed and have attempted to make the principles driving MBC design
and engineering comprehensible. But a number of other wide ranging consequences of this prac-

Page 72 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

tice regard architecture, of which sustainability. Because mycelium-based composites have a
higher vernacular potential [99], they can be made of recycled substrate, or materials available
locally and seasonally. MBC beneﬁt also from a unique aesthetic as a result of the phylogenic
expression of fungi through craft and substrate design [101] (Fig.55). We hope that the on-going
uncovering of the principles for predicting MBC performance contributes to an engineering prac-
tice that may satisfy industrial needs (which actualises in patents [24]), but also fosters a rich,
open, and creative citizen craft. It is not everyday that a novel, fully sustainable, and aﬀordable
material is discovered.

Deliverable D5.2

Page 73 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 55: For a given fungal species, a range of more-than-visual aesthetics can be crafted.

Page 74 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

References

[1] J. E. Adaskaveg, R. L. Gilbertson and M. R. Dunlap. “Eﬀects of Incubation Time and
Temperature on In Vitro Selective Deligniﬁcation of Silver Leaf Oak by Ganoderma Co-
lossum”. In: Applied and Environmental Microbiology 61.1 (Jan. 1995), pp. 138–144. issn:
0099-2240. url: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1388321/ (visited
on 06/11/2021).

[2] Alexandra M. C. R. Alves et al. “Highly Eﬃcient Production of Laccase by the Basi-
diomycete Pycnoporus Cinnabarinus”. In: Applied and Environmental Microbiology 70.11
(1st Nov. 2004), pp. 6379–6384. doi: 10 . 1128 / AEM . 70 . 11 . 6379 - 6384 . 2004. url:
https://journals.asm.org/doi/full/10.1128/aem.70.11.6379-6384.2004 (visited
on 02/11/2021).

[3] Maria Elena Antinori et al. “Fine-Tuning of Physicochemical Properties and Growth Dy-
namics of Mycelium-Based Materials”. In: ACS Applied Bio Materials 3.2 (17th Feb.
2020), pp. 1044–1051. doi: 10.1021/acsabm.9b01031. url: https://doi.org/10.1021/
acsabm.9b01031 (visited on 11/11/2021).

[4] Freek V. W. Appels et al. “Fabrication Factors Inﬂuencing Mechanical, Moisture- and
Water-Related Properties of Mycelium-Based Composites”. In: Materials & Design 161
(5th Jan. 2019), pp. 64–71. issn: 0264-1275. doi: 10.1016/j.matdes.2018.11.027. url:
https://www.sciencedirect.com/science/article/pii/S0264127518308347 (visited
on 22/07/2021).

[5] Freek V. W. Appels et al. “Fungal Mycelium Classiﬁed in Diﬀerent Material Families
Based on Glycerol Treatment”. In: Communications Biology 3.1 (1 26th June 2020), pp. 1–
5. issn: 2399-3642. doi: 10.1038/s42003-020-1064-4. url: https://www.nature.com/
articles/s42003-020-1064-4 (visited on 07/07/2020).

[6] Rickard Arvidsson, Duong Nguyen and Magdalena Svanström. “Life Cycle Assessment
of Cellulose Nanoﬁbrils Production by Mechanical Treatment and Two Diﬀerent Pre-
treatment Processes”. In: Environmental Science & Technology 49.11 (2nd June 2015),
pp. 6881–6890. issn: 0013-936X. doi: 10.1021/acs.est.5b00888. url: https://doi.
org/10.1021/acs.est.5b00888 (visited on 24/11/2021).

[7] Muhammad Asgher, Yasir Sharif and H. N. Bhatti. “Enhanced Production of Ligninolytic
Enzymes by Ganoderma Lucidum IBL-06 Using Lignocellulosic Agricultural Wastes”. In:
International Journal of Chemical Reactor Engineering 8.1 (27th Mar. 2010). issn: 1542-
6580. doi: 10.2202/1542-6580.2203. url: https://www.degruyter.com/document/
doi/10.2202/1542-6580.2203/html (visited on 09/11/2021).

[8] R. H. Atalla et al. “Structures of Plant Cell Wall Celluloses.” In: Biomass recalcitrance:
deconstructing the plant cell wall for bioenergy (2009), pp. 188–212. url: https://www.
cabdirect.org/cabdirect/abstract/20123189083 (visited on 08/11/2021).

[9] Noam Attias et al. “Mycelium Bio-Composites in Industrial Design and Architecture:
Comparative Review and Experimental Analysis”. In: Journal of Cleaner Production 246
(10th Feb. 2020), p. 119037. issn: 0959-6526. doi: 10.1016/j.jclepro.2019.119037.
url: http : / / www . sciencedirect . com / science / article / pii / S0959652619339071
(visited on 08/01/2020).

Deliverable D5.2

Page 75 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[10]

Iván Ayuso-Fernández et al. “Peroxidase Evolution in White-Rot Fungi Follows Wood
Lignin Evolution in Plants”. In: Proceedings of the National Academy of Sciences 116.36
(3rd Sept. 2019), pp. 17900–17905. issn: 0027-8424, 1091-6490. doi: 10 . 1073 / pnas .
1905040116. url: https : / / www . pnas . org / content / 116 / 36 / 17900 (visited on
07/11/2021).

[11] Matthew J. Baker et al. “Using Fourier Transform IR Spectroscopy to Analyze Biological
Materials”. In: Nature Protocols 9.8 (Aug. 2014), pp. 1771–1791. issn: 1750-2799. doi:
10.1038/nprot.2014.110. url: https://www.nature.com/articles/nprot.2014.110
(visited on 23/08/2021).

[12] Petr Baldrian. “Fungal Laccases – Occurrence and Properties”. In: FEMS Microbiology
Reviews 30.2 (1st Mar. 2006), pp. 215–242. issn: 0168-6445. doi: 10 . 1111 / j . 1574 -
4976.2005.00010.x. url: https://doi.org/10.1111/j.1574- 4976.2005.00010.x
(visited on 05/11/2021).

[13] George L. Barron. “Predatory Fungi, Wood Decay, and the Carbon Cycle”. In: Biodiversity
4.1 (1st Feb. 2003), pp. 3–9. issn: 1488-8386. doi: 10.1080/14888386.2003.9712621.
url: https://doi.org/10.1080/14888386.2003.9712621 (visited on 28/10/2021).
[14] Georgios Bekiaris et al. “Pleurotus Mushrooms Content in Glucans and Ergosterol As-
sessed by ATR-FTIR Spectroscopy and Multivariate Analysis”. In: Foods 9.4 (Apr. 2020),
p. 535. doi: 10.3390/foods9040535. url: https://www.mdpi.com/2304-8158/9/4/535
(visited on 23/08/2021).

[15] Robert A. Blanchette. “Degradation of the Lignocellulose Complex in Wood”. In: Cana-
dian Journal of Botany 73.S1 (Dec. 1995), pp. 999–1010. doi: 10.1139/b95-350. url:
https://cdnsciencepub.com/doi/abs/10.1139/b95-350 (visited on 06/11/2021).
[16] L. Boddy. “Microenvironmental Aspects of Xylem Defenses to Wood Decay Fungi”. In:
Defense Mechanisms of Woody Plants Against Fungi. Ed. by Robert A. Blanchette and
Alan R. Biggs. Springer Series in Wood Science. Berlin, Heidelberg: Springer, 1992, pp. 96–
132. isbn: 978-3-662-01642-8. url: https://doi.org/10.1007/978-3-662-01642-8_6
(visited on 08/11/2021).

[17] Lynne Boddy. “Microclimate and Moisture Dynamics of Wood Decomposing in Terrestrial
Ecosystems”. In: Soil Biology and Biochemistry 15.2 (1st Jan. 1983), pp. 149–157. issn:
0038-0717. doi: 10.1016/0038-0717(83)90096-2. url: https://www.sciencedirect.
com/science/article/pii/0038071783900962 (visited on 08/11/2021).

[18] Lynne Boddy et al. “Climate Variation Eﬀects on Fungal Fruiting”. In: Fungal Ecology.
Fungi in a Changing World: The Role of Fungi in Ecosystem Response to Global Change
10 (1st Aug. 2014), pp. 20–33. issn: 1754-5048. doi: 10.1016/j.funeco.2013.10.006.
url: https://www.sciencedirect.com/science/article/pii/S1754504813001116
(visited on 10/11/2021).

[19] Christian Brischke and Gry Alfredsen. “Wood-Water Relationships and Their Role for
Wood Susceptibility to Fungal Decay”. In: Applied Microbiology and Biotechnology 104.9
(1st May 2020), pp. 3781–3795. issn: 1432-0614. doi: 10.1007/s00253- 020- 10479- 1.
url: https://doi.org/10.1007/s00253-020-10479-1 (visited on 28/09/2021).

Page 76 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[20] Christian Brischke, Arved Soetbeer and Linda Meyer-Veltrup. “The Minimum Moisture
Threshold for Wood Decay by Basidiomycetes Revisited. A Review and Modiﬁed Pile Ex-
periments with Norway Spruce and European Beech Decayed by Coniophora Puteana and
Trametes Versicolor”. In: Holzforschung 71.11 (1st Nov. 2017), pp. 893–903. issn: 1437-
434X. doi: 10.1515/hf- 2017- 0051. url: https://www.degruyter.com/document/
doi/10.1515/hf-2017-0051/html (visited on 30/09/2021).

[21] Qian Cai et al. “Dissolving Process of Bamboo Powder Analyzed by FT-IR Spectroscopy”.
In: Journal of Molecular Structure 1171 (1st June 2018). doi: 10.1016/j.molstruc.
2018.06.066.

[22] Michael D. Cameron and Steven D. Aust. “Cellobiose Dehydrogenase–an Extracellular
Fungal Flavocytochrome”. In: Enzyme and Microbial Technology 28.2 (1st Feb. 2001),
pp. 129–138. issn: 0141-0229. doi: 10 . 1016 / S0141 - 0229(00 ) 00307 - 0. url: https :
/ / www . sciencedirect . com / science / article / pii / S0141022900003070 (visited on
06/11/2021).

[23] Jesus D. Castaño et al. “Oxidative Damage Control during Decay of Wood by Brown
Rot Fungus Using Oxygen Radicals”. In: Applied and Environmental Microbiology 84.22
(2018), e01937–18. doi: 10.1128/AEM.01937- 18. url: https://journals.asm.org/
doi/10.1128/AEM.01937-18 (visited on 03/11/2021).

[24] Kustrim Cerimi et al. “Fungi as Source for New Bio-Based Materials: A Patent Review”.
In: Fungal Biology and Biotechnology 6.1 (26th Oct. 2019), p. 17. issn: 2054-3085. doi:
10.1186/s40694-019-0080-y. url: https://doi.org/10.1186/s40694-019-0080-y
(visited on 24/11/2021).

[25] Xin Ying Chan et al. “Mechanical Properties of Dense Mycelium-Bound Composites under
Accelerated Tropical Weathering Conditions”. In: Scientiﬁc Reports 11.1 (11th Nov. 2021),
p. 22112. issn: 2045-2322. doi: 10.1038/s41598- 021- 01598- 4. url: https://www.
nature.com/articles/s41598-021-01598-4 (visited on 16/11/2021).

[26] Shilin Chen et al. “Genome Sequence of the Model Medicinal Mushroom Ganoderma
Lucidum”. In: Nature Communications 3.1 (26th June 2012), p. 913. issn: 2041-1723.
doi: 10 . 1038 / ncomms1923. url: https : / / www . nature . com / articles / ncomms1923
(visited on 14/07/2021).

[27] Diana Ciolacu, Florin Ciolacu and Valentin I. Popa. “Amorphous Cellulose – Structure

and Characterization.” In: Cellulose chemistry and technology 45.1 (2011), pp. 13–21.

[28] Marie Couturier et al. “Lytic Xylan Oxidases from Wood-Decay Fungi Unlock Biomass
Degradation”. In: Nature Chemical Biology 14.3 (3 Mar. 2018), pp. 306–310. issn: 1552-
4469. doi: 10 . 1038 / nchembio . 2558. url: https : / / www . nature . com / articles /
nchembio.2558 (visited on 23/11/2021).

[29] S.F. Curling, C.A. Clausen and J.E. Winandy. “Relationships between Mechanical Prop-
erties, Weight Loss, and Chemical Composition of Wood during Incipient Brown-Rot
Decay”. In: Forest products journal 52.7/8 (July 2002), pp. 34–39. issn: 0015-7473. url:
https://handle.nal.usda.gov/10113/29430 (visited on 07/11/2021).

[30] Savitha Desai et al. “Isolation of Laccase Producing Fungi and Partial Characterization
of Laccase”. In: Biotechnology, Bioinformatics and Bioengineering 1.4 (1st Jan. 2011),
pp. 543–549. issn: 2249-9075.

Deliverable D5.2

Page 77 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[31] Patrick Pereira Dias, Laddu Bhagya Jayasinghe and Daniele Waldmann. “Investigation of
Mycelium-Miscanthus Composites as Building Insulation Material”. In: Results in Materi-
als 10 (1st June 2021), p. 100189. issn: 2590-048X. doi: 10.1016/j.rinma.2021.100189.
url: https://www.sciencedirect.com/science/article/pii/S2590048X21000224
(visited on 24/11/2021).

[32] Susan V. Diehl, M. Lynn Prewitt and Fatima Moore Shmulsky. “Use of Fatty Acid Proﬁles
To Identify White-Rot Wood Decay Fungi”. In: Wood Deterioration and Preservation.
Vol. 845. 0 vols. ACS Symposium Series 845. American Chemical Society, 31st Mar.
2003, pp. 313–324. isbn: 978-0-8412-3797-1. doi: 10.1021/bk-2003-0845.ch017. url:
https://doi.org/10.1021/bk-2003-0845.ch017 (visited on 14/07/2021).

[33] José Dorado et al. “Infrared Spectroscopy Analysis of Hemp (Cannabis Sativa) after Select-
ive Deligniﬁcation by Bjerkandera Sp. at Diﬀerent Nitrogen Levels”. In: Enzyme and mi-
crobial technology 28 (1st May 2001), pp. 550–559. doi: 10.1016/S0141-0229(00)00363-
X.

[34] Benjamin Minge Duggar. The Principles of Mushroom Growing and Mushroom Spawn

Making. U.S. Government Printing Oﬃce, 1905. 90 pp. isbn: 1-120-91803-0.

[35] Michaela Eder et al. “Wood and the Activity of Dead Tissue”. In: Advanced Materi-
als 33.28 (2021), p. 2001412. issn: 1521-4095. doi: 10 . 1002 / adma . 202001412. url:
https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.202001412 (visited on
21/11/2021).

[36] C. Eggert, U. Temp and K. E. Eriksson. “The Ligninolytic System of the White Rot
Fungus Pycnoporus Cinnabarinus: Puriﬁcation and Characterization of the Laccase”. In:
Applied and Environmental Microbiology 62.4 (Apr. 1996), pp. 1151–1158. issn: 0099-
2240. doi: 10.1128/aem.62.4.1151-1158.1996.

[37] Elise Elsacker et al. “Growing Living and Multifunctional Mycelium Composites for Large-
Scale Formwork Applications Using Robotic Abrasive Wire-Cutting”. In: Construction
and Building Materials 283 (10th May 2021), p. 122732. issn: 0950-0618. doi: 10.1016/
j . conbuildmat . 2021 . 122732. url: https : / / www . sciencedirect . com / science /
article/pii/S095006182100492X (visited on 13/11/2021).

[38] Elise Elsacker et al. “Mechanical, Physical and Chemical Characterisation of Mycelium-
Based Composites with Diﬀerent Types of Lignocellulosic Substrates”. In: PLOS ONE
14.7 (22nd July 2019), e0213954. issn: 1932-6203. doi: 10.1371/journal.pone.0213954.
url: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.
0213954 (visited on 22/07/2021).

[39] Emil Tang Engelund et al. “A Critical Discussion of the Physics of Wood–Water Interac-
tions”. In: Wood Science and Technology 47.1 (1st Jan. 2013), pp. 141–161. issn: 1432-5225.
doi: 10.1007/s00226- 012- 0514- 7. url: https://doi.org/10.1007/s00226- 012-
0514-7 (visited on 25/02/2021).

[40] Dietrich Fengel and Gerd Wegener. Wood: Chemistry, Ultrastructure, Reactions. Walter

de Gruyter, 2nd Aug. 2011. 633 pp. isbn: 978-3-11-083965-4.

[41] André Ferraz et al. “Mapping of Cell Wall Components in Ligniﬁed Biomass as a Tool to
Understand Recalcitrance”. In: Biofuels in Brazil: Fundamental Aspects, Recent Develop-
ments, and Future Perspectives. Ed. by Silvio Silvério da Silva and Anuj Kumar Chandel.
Cham: Springer International Publishing, 2014, pp. 173–202. isbn: 978-3-319-05020-1.
url: https://doi.org/10.1007/978-3-319-05020-1_9 (visited on 07/11/2021).

Page 78 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[42]

Isvett Joseﬁna Flores-Sanchez and Robert Verpoorte. “Secondary Metabolism in Can-
nabis”. In: Phytochemistry Reviews 7.3 (1st Oct. 2008), pp. 615–639. issn: 1572-980X.
doi: 10.1007/s11101- 008- 9094- 4. url: https://doi.org/10.1007/s11101- 008-
9094-4 (visited on 14/07/2021).

[43] Dimitrios Floudas et al. “The Paleozoic Origin of Enzymatic Lignin Decomposition Re-
constructed from 31 Fungal Genomes”. In: Science 336.6089 (29th June 2012), pp. 1715–
1719. doi: 10.1126/science.1221748. url: https://www.science.org/doi/10.1126/
science.1221748 (visited on 03/11/2021).

[44] Grégoire T. Freschet et al. “Interspeciﬁc Diﬀerences in Wood Decay Rates: Insights from
a New Short-Term Method to Study Long-Term Wood Decomposition”. In: Journal of
Ecology 100.1 (2012), pp. 161–170. issn: 1365-2745. doi: 10.1111/j.1365-2745.2011.
01896 . x. url: https : / / onlinelibrary . wiley . com / doi / abs / 10 . 1111 / j . 1365 -
2745.2011.01896.x (visited on 07/11/2021).

[45] M. J. Fuhr et al. Study of the Combined Eﬀect of Temperature, pH and Water Activity on
the Radial Growth Rate of the White-Rot Basidiomycete Physisporinus Vitreus by Using
a Hyphal Growth Model. 22nd June 2011. arXiv: 1106 . 4521 [physics, q-bio]. url:
http://arxiv.org/abs/1106.4521 (visited on 04/10/2021).

[46] Anton Geﬀert, Jarmila Geﬀertova and Michal Dudiak. “Direct Method of Measuring the
pH Value of Wood”. In: Forests 10.10 (Oct. 2019), p. 852. doi: 10.3390/f10100852. url:
https://www.mdpi.com/1999-4907/10/10/852 (visited on 08/11/2021).

[47] Patrick Gervais and Paul Molin. “The Role of Water in Solid-State Fermentation”. In: Bio-
chemical Engineering Journal. Solid-State Fermentation . 13.2 (1st Mar. 2003), pp. 85–
101. issn: 1369-703X. doi: 10 . 1016 / S1369 - 703X(02 ) 00122 - 5. url: https : / / www .
sciencedirect.com/science/article/pii/S1369703X02001225 (visited on 28/09/2021).
[48] Carolina Girometta et al. “Physico-Mechanical and Thermodynamic Properties of Mycelium-

Based Biocomposites: A Review”. In: Sustainability 11.1 (1 Jan. 2019), p. 281. doi:
10.3390/su11010281. url: https://www.mdpi.com/2071- 1050/11/1/281 (visited
on 08/07/2021).

[49]

I. Gomes et al. “Production of Cellulase and Xylanase by a Wild Strain of Trichoderma Vi-
ride”. In: Applied Microbiology and Biotechnology 36.5 (1st Feb. 1992), pp. 701–707. issn:
1432-0614. doi: 10.1007/BF00183253. url: https://doi.org/10.1007/BF00183253
(visited on 06/11/2021).

[50] Leyu Gou et al. “Morphological and Physico-Mechanical Properties of Mycelium Biocom-
posites with Natural Reinforcement Particles”. In: Construction and Building Materi-
als 304 (18th Oct. 2021), p. 124656. issn: 0950-0618. doi: 10.1016/j.conbuildmat.
2021 . 124656. url: https : / / www . sciencedirect . com / science / article / pii /
S0950061821024119 (visited on 13/10/2021).

[51] Joze Grdadolnik. “ATR-FTIR Spectroscopy: Its Advantages and Limitations”. In: Acta

Chimica Slovenica 49 (1st Sept. 2002), pp. 631–642.

[52] David W. Green, Jerrold E. Winandy and David E. Kretschmann. “Mechanical Properties
of Wood”. In: Wood handbook : wood as an engineering material. Madison, WI : USDA
Forest Service, Forest Products Laboratory, 1999. General technical report FPL ; GTR-
113: Pages 4.1-4.45 113 (1999). url: https://www.fs.usda.gov/treesearch/pubs/
7149 (visited on 22/07/2021).

Deliverable D5.2

Page 79 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[53] Frederick Green, Thomas A. Kuster and Thierry L. Highley. “Pectin Degradation dur-
ing Colonization of Wood by Brown-Rot Fungi”. In: Recent Research Developments in
Plant Pathology 1 (1996), pp. 83–93. url: https://www.fpl.fs.fed.us/products/
publications/specific_pub.php?posting_id=14872 (visited on 04/11/2021).
[54] Joseph F. Hair et al. Multivariate Data Analysis. Pearson Education Limited, 2013. 734 pp.

isbn: 978-1-292-02190-4.

[55] Muhammad Haneef et al. “Advanced Materials From Fungal Mycelium: Fabrication and
Tuning of Physical Properties”. In: Scientiﬁc Reports 7.1 (1 24th Jan. 2017), p. 41292.
issn: 2045-2322. doi: 10.1038/srep41292. url: https://www.nature.com/articles/
srep41292 (visited on 14/07/2021).

[56] Sudha Hariharan and Padma Nambisan. “Optimization of Lignin Peroxidase, Manganese
Peroxidase, and Lac Production from Ganoderma Lucidum Under Solid State Fermenta-
tion of Pineapple Leaf”. In: BioResources 8.1 (2013), pp. 250–271. issn: 1930-2126. url:
https://ojs.cnr.ncsu.edu/index.php/BioRes/article/view/BioRes_08_1_250_
Hariharan_Nambisan_Optimization_Ganoderma_Pineapple (visited on 09/11/2021).

[57] Paul V. Harris et al. “Stimulation of Lignocellulosic Biomass Hydrolysis by Proteins of
Glycoside Hydrolase Family 61: Structure and Function of a Large, Enigmatic Family”.
In: Biochemistry 49.15 (20th Apr. 2010), pp. 3305–3316. issn: 1520-4995. doi: 10.1021/
bi100009p. pmid: 20230050.

[58] Juan He et al. “Study on the Mechanical Properties of the Latex-Mycelium Composite”. In:
Applied Mechanics and Materials 507 (2014), pp. 415–420. issn: 1662-7482. doi: 10.4028/
www.scientific.net/AMM.507.415. url: https://www.scientific.net/AMM.507.415
(visited on 24/11/2021).

[59] Pete Heinzelman et al. “A Family of Thermostable Fungal Cellulases Created by Structure-
Guided Recombination”. In: Proceedings of the National Academy of Sciences 106.14
(7th Apr. 2009), pp. 5610–5615. issn: 0027-8424, 1091-6490. doi: 10.1073/pnas.0901417106.
url: https://www.pnas.org/content/106/14/5610 (visited on 06/11/2021).

[60] Felix Heisel et al. “Design, Cultivation and Application of Load-Bearing Mycelium Com-
ponents: The MycoTree at the 2017 Seoul Biennale of Architecture and Urbanism”. In:
International Journal of Sustainable Energy Development 6.1 (1st June 2018), pp. 296–
303. doi: 10.20533/ijsed.2046.3707.2017.0039.

[61] J. Hiscox, J. O’Leary and L. Boddy. “Fungus Wars: Basidiomycete Battles in Wood De-
cay”. In: Studies in Mycology. Leading Women in Fungal Biology 89 (1st Mar. 2018),
pp. 117–124. issn: 0166-0616. doi: 10 . 1016 / j . simyco . 2018 . 02 . 003. url: https :
/ / www . sciencedirect . com / science / article / pii / S016606161830006X (visited on
10/11/2021).

[62] Martin Hofrichter et al. “New and Classic Families of Secreted Fungal Heme Peroxidases”.
In: Applied Microbiology and Biotechnology 87.3 (July 2010), pp. 871–897. issn: 1432-0614.
doi: 10.1007/s00253-010-2633-0.

[63] G. A. Holt et al. “Fungal Mycelium and Cotton Plant Materials in the Manufacture of
Biodegradable Molded Packaging Material: Evaluation Study of Select Blends of Cot-
ton Byproducts”. In: Journal of Biobased Materials and Bioenergy 6.4 (1st Aug. 2012),
pp. 431–439. doi: 10.1166/jbmb.2012.1241.

Page 80 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[64] M. Hrmova et al. “Substrate Binding and Catalytic Mechanism of a Barley Beta-D-
Glucosidase/(1,4)-Beta-D-Glucan Exohydrolase”. In: The Journal of Biological Chemistry
273.18 (1st May 1998), pp. 11134–11143. issn: 0021-9258. doi: 10.1074/jbc.273.18.
11134.

[65] Haﬁz Muhammad Nasir Iqbal, Muhammad Asgher and Haz Nawaz Bhatti. “Optimization
of Physical Andnutritional Factors for Synthesis of Lignin Degrading Enzymes by a Novel
Strain of Trametes Versicolor”. In: BioResources 6.2 (1st Mar. 2011), pp. 1273–1287. issn:
1930-2126. url: https : / / ojs . cnr . ncsu . edu / index . php / BioRes / article / view /
BioRes_06_2_1273_Iqbal_A_Optim_Phys_Nutritional_Ligninases_Strain (visited
on 09/11/2021).

[66] M. Irshad and M. Asgher. “Production and Optimization of Ligninolytic Enzymes by
White Rot Fungus Schizophyllum Commune IBL-06 in Solid State Medium Banana
Stalks”. In: African Journal of Biotechnology 10.79 (2011), pp. 18234–18242. issn: 1684-
5315. doi: 10 . 4314 / ajb . v10i79. url: https : / / www . ajol . info / index . php / ajb /
article/view/98593 (visited on 08/11/2021).

[67] M. R. Islam et al. “Morphology and Mechanics of Fungal Mycelium”. In: Scientiﬁc Re-
ports 7.1 (1 12th Oct. 2017), p. 13070. issn: 2045-2322. doi: 10 . 1038 / s41598 - 017 -
13295-2. url: https://www.nature.com/articles/s41598-017-13295-2 (visited on
31/10/2021).

[68] M. R. Islam et al. “Stochastic Continuum Model for Mycelium-Based Bio-Foam”. In:
Materials & Design 160 (15th Dec. 2018), pp. 549–556. issn: 0264-1275. doi: 10.1016/j.
matdes.2018.09.046. url: http://www.sciencedirect.com/science/article/pii/
S0264127518307482 (visited on 28/01/2021).

[69] Joseph E. Jakes et al. “Eﬀects of Moisture on Diﬀusion in Unmodiﬁed Wood Cell Walls: A
Phenomenological Polymer Science Approach”. In: Forests 10.12 (12 Dec. 2019), p. 1084.
doi: 10 . 3390 / f10121084. url: https : / / www . mdpi . com / 1999 - 4907 / 10 / 12 / 1084
(visited on 21/11/2021).

[70] A. Jankowska and P. Kozakiewicz. “Determination of Fibre Saturation Point of Selected
Tropical Wood Species Using Diﬀerent Methods”. In: Drewno : prace naukowe, doniesi-
enia, komunikaty vol. 59, nr 197 (2016). issn: 1644-3985. doi: 10.12841/wood.1644-
3985.C07.12. url: http://yadda.icm.edu.pl/baztech/element/bwmeta1.element.
baztech-d215694b-8df1-49d0-b87c-9b2266de627c (visited on 02/10/2021).

[71] Fei Jiang et al. “Eﬀects of pH and Temperature on Recombinant Manganese Peroxi-
dase Production and Stability”. In: Applied Biochemistry and Biotechnology 146.1-3 (Mar.
2008), pp. 15–27. issn: 1559-0291. doi: 10.1007/s12010-007-8039-5.

[72] Lai Jiang et al. “Manufacturing of Biocomposite Sandwich Structures Using Mycelium-
Bound Cores and Preforms”. In: Journal of Manufacturing Processes 28 (1st Aug. 2017),
pp. 50–59. issn: 1526-6125. doi: 10 . 1016 / j . jmapro . 2017 . 04 . 029. url: https :
/ / www . sciencedirect . com / science / article / pii / S1526612517301019 (visited on
13/10/2021).

[73] Christian P. Kubicek. Fungi and Lignocellulosic Biomass. Biomass and Biofuels. Wiley-
Blackwell, 2012. isbn: 978-0-470-96009-7. url: https://www.wiley.com/en-us/Fungi+
and+Lignocellulosic+Biomass-p-9780470960097 (visited on 05/11/2021).

Deliverable D5.2

Page 81 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[74] Shin Kwang-Soo, Kim Young Hwan and Lim Jong-Soon. “Puriﬁcation and Characteriza-
tion of Manganese Peroxidase of the White-Rot Fungus Irpex Lacteus”. In: Journal of Mi-
crobiology 43.6 (2005), pp. 503–509. issn: 1225-8873. url: https://www.koreascience.
or.kr/article/JAKO200509409866134.page (visited on 06/11/2021).

[75] Kirsi Leppänen et al. “X-Ray Scattering and Microtomography Study on the Structural
Changes of Never-Dried Silver Birch, European Aspen and Hybrid Aspen during Drying”.
In: 65.6 (1st Oct. 2011), pp. 865–873. issn: 1437-434X. doi: 10.1515/HF.2011.108. url:
https://www.degruyter.com/document/doi/10.1515/HF.2011.108/html (visited on
09/11/2021).

[76] Xiaoli Li et al. “Quantitative Visualization of Lignocellulose Components in Transverse
Sections of Moso Bamboo Based on FTIR Macro- and Micro-Spectroscopy Coupled with
Chemometrics”. In: Biotechnology for Biofuels 11.1 (26th Sept. 2018), p. 263. issn: 1754-
6834. doi: 10.1186/s13068- 018- 1251- 4. url: https://doi.org/10.1186/s13068-
018-1251-4 (visited on 23/08/2021).

[77] Mae-Ling Lokko et al. “Development of Aﬀordable Building Materials Using Agricultural
Waste By-Products and Emerging Pith, Soy and Mycelium Biobinders”. In: PLEA2016
Los Angeles - Cities, Buildings, People: Towards Regenerative Environments, Los Angeles,
11th July 2016.

[78] J. A. López Nava et al. “Assessment of Edible Fungi and Films Bio-Based Material Simu-
lating Expanded Polystyrene”. In: Materials and Manufacturing Processes 31.8 (10th June
2016), pp. 1085–1090. issn: 1042-6914. doi: 10 . 1080 / 10426914 . 2015 . 1070420. url:
https://doi.org/10.1080/10426914.2015.1070420 (visited on 13/11/2021).

[79] Taina K. Lundell, Miia R. Mäkelä and Kristiina Hildén. “Lignin-Modifying Enzymes
in Filamentous Basidiomycetes – Ecological, Functional and Phylogenetic Review”. In:
Journal of Basic Microbiology 50.1 (2010), pp. 5–20. issn: 1521-4028. doi: 10 . 1002 /
jobm.200900338. url: https://onlinelibrary.wiley.com/doi/abs/10.1002/jobm.
200900338 (visited on 04/11/2021).

[80] Jacqueline MacDonald et al. “Transcriptomic Responses of the Softwood-Degrading White-
Rot Fungus Phanerochaete Carnosa during Growth on Coniferous and Deciduous Wood”.
In: Applied and Environmental Microbiology 77.10 (15th May 2011), pp. 3211–3218. doi:
10.1128/AEM.02490-10. url: https://journals.asm.org/doi/full/10.1128/AEM.
02490-10 (visited on 14/10/2021).

[81] Linda Meyer et al. “Critical Moisture Conditions for Fungal Decay of Modiﬁed Wood
by Basidiomycetes as Detected by Pile Tests”. In: Holzforschung 70.4 (1st Apr. 2016),
pp. 331–339. issn: 1437-434X. doi: 10 . 1515 / hf - 2015 - 0046. url: https : / / www .
degruyter.com/document/doi/10.1515/hf-2015-0046/html (visited on 25/02/2021).
[82] Nicholas P. Money. “Turgor Pressure and the Mechanics of Fungal Penetration”. In: Ca-
nadian Journal of Botany 73.S1 (Dec. 1995), pp. 96–102. doi: 10.1139/b95-231. url:
https://cdnsciencepub.com/doi/abs/10.1139/b95-231 (visited on 04/10/2021).
[83] L. R. S. Moreira and E. X. F. Filho. “An Overview of Mannan Structure and Mannan-
Degrading Enzyme Systems”. In: Applied Microbiology and Biotechnology 79.2 (1st May
2008), pp. 165–178. issn: 1432-0614. doi: 10.1007/s00253- 008- 1423- 4. url: https:
//doi.org/10.1007/s00253-008-1423-4 (visited on 06/11/2021).

[84] Franziska Julia Moser et al. Fungal Mycelium as a Building Material. RWTH-2017-08964.
Fachgruppe Biologie, 2017. url: https:/ /publications .rwth- aachen . de/record/
706992 (visited on 13/10/2021).

Page 82 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[85] F. L. Motta, C. C. P. Andrade and M. H. A. Santana. A Review of Xylanase Production by
the Fermentation of Xylan: Classiﬁcation, Characterization and Applications. IntechOpen,
15th May 2013. isbn: 978-953-51-1119-1. url: https://www.intechopen.com/chapters/
44332 (visited on 06/11/2021).

[86] Subash Nataraja, D.M and M. Krishnappa. “Eﬀect of Temperature on Cellulase Enzyme
Activity in Crude Extracts Isolated from Solid Wastes Microbes”. In: International Journal
of Microbiology Research 2 (31st Dec. 2010), pp. 44–47. doi: 10.9735/0975-5276.2.2.
44-47. url: http://bioinfopublication.org/files/articles/2_2_6_IJMR.pdf.
[87] Robin A. Ohm et al. “Genome Sequence of the Model Mushroom Schizophyllum Com-
mune”. In: Nature Biotechnology 28.9 (9 Sept. 2010), pp. 957–963. issn: 1546-1696. doi:
10.1038/nbt.1643. url: https://www.nature.com/articles/nbt.1643 (visited on
02/11/2021).

[88] Jason Ongpeng et al. “Using Waste in Producing Bio-Composite Mycelium Bricks”. In:

Applied Sciences 10 (31st July 2020). doi: 10.3390/app10155303.

[89] Krishna K. Pandey. “A Study of Chemical Structure of Soft and Hardwood and Wood
Polymers by FTIR Spectroscopy”. In: Journal of Applied Polymer Science 71.12 (1999),
pp. 1969–1975. issn: 1097-4628. doi: 10.1002/(SICI)1097-4628(19990321)71:12<1969::
AID-APP6>3.0.CO;2-D. url: https://onlinelibrary.wiley.com/doi/abs/10.1002/
%28SICI%291097- 4628%2819990321%2971%3A12%3C1969%3A%3AAID- APP6%3E3.0.CO%
3B2-D (visited on 15/07/2021).

[90] Krishna K. Pandey and Andrew J. Pitman. “FTIR Studies of the Changes in Wood Chem-
istry Following Decay by Brown-Rot and White-Rot Fungi”. In: International Biodeteri-
oration & Biodegradation 52.3 (Oct. 2003), pp. 151–160. issn: 09648305. doi: 10.1016/
S0964- 8305(03)00052- 0. url: https://linkinghub.elsevier.com/retrieve/pii/
S0964830503000520 (visited on 10/03/2020).

[91] Saroj Paramjeet, P. Manasa and Narasimhulu Korrapati. “Biofuels: Production of Fungal-
Mediated Ligninolytic Enzymes and the Modes of Bioprocesses Utilizing Agro-Based
Residues”. In: Biocatalysis and Agricultural Biotechnology 14 (1st Apr. 2018), pp. 57–
71. issn: 1878-8181. doi: 10 . 1016 / j . bcab . 2018 . 02 . 007. url: https : / / www .
sciencedirect.com/science/article/pii/S1878818117304644 (visited on 06/11/2021).

[92] David Parﬁtt et al. “Do All Trees Carry the Seeds of Their Own Destruction? PCR
Reveals Numerous Wood Decay Fungi Latently Present in Sapwood of a Wide Range of
Angiosperm Trees”. In: Fungal Ecology 3.4 (1st Nov. 2010), pp. 338–346. issn: 1754-5048.
doi: 10 . 1016 / j . funeco . 2010 . 02 . 001. url: https : / / www . sciencedirect . com /
science/article/pii/S1754504810000061 (visited on 07/11/2021).

[93] E Placido, MC Arduini-Schuster and J Kuhn. “Thermal properties predictive model for

insulating foams”. In: Infrared Physics & Technology 46.3 (2005), pp. 219–231.

[94] G. Rajhans et al. “Elucidation of Fungal Dye-Decolourizing Peroxidase (DyP) and Lign-
inolytic Enzyme Activities in Decolourization and Mineralization of Azo Dyes”. In: Journal
of Applied Microbiology 129.6 (2020), pp. 1633–1643. issn: 1365-2672. doi: 10.1111/jam.
14731. url: https://onlinelibrary.wiley.com/doi/abs/10.1111/jam.14731 (vis-
ited on 06/11/2021).

[95] M. V. Ramiah. “Thermogravimetric and Diﬀerential Thermal Analysis of Cellulose, Hemi-
cellulose, and Lignin”. In: Journal of Applied Polymer Science 14.5 (1970), pp. 1323–1337.
issn: 1097-4628. doi: 10.1002/app.1970.070140518. url: https://onlinelibrary.
wiley.com/doi/abs/10.1002/app.1970.070140518 (visited on 08/11/2021).

Deliverable D5.2

Page 83 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[96] R. Rashmi and K. R. Siddalingamurthy. “Microbial Xyloglucanases: A Comprehensive
Review”. In: Biocatalysis and Biotransformation 36.4 (4th July 2018), pp. 280–295. issn:
1024-2422. doi: 10.1080/10242422.2017.1417394. url: https://doi.org/10.1080/
10242422.2017.1417394 (visited on 06/11/2021).
Iuliana Răut et al. “Fungal Based Biopolymer Composites for Construction Materials”.
In: Materials 14.11 (11 Jan. 2021), p. 2906. doi: 10 . 3390 / ma14112906. url: https :
//www.mdpi.com/1996-1944/14/11/2906 (visited on 31/10/2021).

[97]

[98] Aarthi Ravichandran et al. “Augmenting Versatile Peroxidase Production from Lentinus
Squarrosulus and Its Role in Enhancing Ruminant Feed :: BioResources”. In: BioResources
16.1 (2021), pp. 1600–1615. doi: 10 . 15376 / biores . 16 . 1 . 1600 - 1615. url: https :
//bioresources.cnr.ncsu.edu/ (visited on 06/11/2021).

[99] Adrien Rigobello and Phil Ayres. “Fragile Computation: Rethinking Information Techno-
logies to Foster Situated Ecologies”. In: Proceedings of the Deep City 2021 Conference.
Deep City. Lausanne, 25th Mar. 2021.

[100] Adrien Rigobello and Phil Ayres. “Mycelium-Based Composites as Two-Phase Particulate
Composites: Compressive Behaviour of Anisotropic Designs.” In: (13th Oct. 2021). issn:
2693-5015. doi: 10.21203/rs.3.rs-943974/v1. url: https://www.researchsquare.
com/article/rs-943974/v1 (visited on 13/10/2021).

[101] Adrien Rigobello and Nadja Gaudillière-Jami. “Designing the Gross. In Search for Social
Inclusion.” In: Design Culture(s). Cumulus Conference Proceedings Roma 2021. Vol. 2.
2 vols. 19th Apr. 2021, pp. 811–827. isbn: 978-952-64-9004-5.

[102] Dana Sàez et al. Analyzing a Fungal Mycelium and Chipped Wood Composite for Use in

Construction. 1st Aug. 2021.

[103] Dana Sàez et al. Developing Sandwich Panels with a Mid-Layer of Fungal Mycelium Com-

posite for a Timber Panel Construction System. 2nd Sept. 2021.

[104] Lennart Salmén. “Wood Morphology and Properties from Molecular Perspectives”. In:
Annals of Forest Science 72.6 (1st Sept. 2015), pp. 679–684. issn: 1297-966X. doi: 10.
1007/s13595- 014- 0403- 3. url: https://doi.org/10.1007/s13595- 014- 0403- 3
(visited on 09/11/2021).

[105] Jozef Šandula et al. “Microbial (1→3)--D-Glucans, Their Preparation, Physico-Chemical

Characterization and Immunomodulatory Activity”. In: Carbohydrate Polymers 38 (1st Mar.
1999), pp. 247–253. doi: 10.1016/S0144-8617(98)00099-X.

[106] James A. Sawitzke et al. “Recombineering: Using Drug Cassettes to Knock out Genes in
Vivo”. In: Methods in Enzymology. Laboratory Methods in Enzymology: Cell, Lipid and
Carbohydrate 533 (1st Jan. 2013). Ed. by Jon Lorsch, pp. 79–102. doi: 10.1016/B978-
0-12-420067-8.00007-6. url: https://www.sciencedirect.com/science/article/
pii/B9780124200678000076 (visited on 02/11/2021).

[107] Helge Schritt, Stephan Vidi and Daniel Pleissner. “Spent Mushroom Substrate and Saw-
dust to Produce Mycelium-Based Thermal Insulation Composites”. In: Journal of Cleaner
Production 313 (1st Sept. 2021), p. 127910. issn: 0959-6526. doi: 10.1016/j.jclepro.
2021 . 127910. url: https : / / www . sciencedirect . com / science / article / pii /
S0959652621021284 (visited on 17/06/2021).

Page 84 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[108] F. W. M. R. Schwarze, S. Baum and S. Fink. “Dual Modes of Degradation by Fistulina
Hepatica in Xylem Cell Walls of Quercus Robur”. In: Mycological Research 104.7 (1st July
2000), pp. 846–852. issn: 0953-7562. doi: 10.1017/S0953756299002063. url: https:
/ / www . sciencedirect . com / science / article / pii / S0953756208617177 (visited on
07/11/2021).

[109] F. W. M. R. Schwarze, S. Baum and S. Fink. “Resistance of Fibre Regions in Wood of Acer
Pseudoplatanus Degraded by Armillaria Mellea”. In: Mycological Research 104.9 (Sept.
2000), pp. 1126–1132. issn: 1469-8102, 0953-7562. doi: 10.1017/S0953756200002525.
url: https://www.cambridge.org/core/journals/mycological-research/article/
abs/resistance-of-fibre-regions-in-wood-of-acer-pseudoplatanus-degraded-
by-armillaria-mellea/76D7C39D52960D65ED041CBB055422CB (visited on 07/11/2021).
[110] F. W.M.R. Schwarze and D. Ferner. “Ganoderma on Trees—Diﬀerentiation of Species and
Studies of Invasiveness”. In: Arboricultural Journal 27.1 (1st June 2003), pp. 59–77. issn:
0307-1375. doi: 10.1080/03071375.2003.9747362. url: https://doi.org/10.1080/
03071375.2003.9747362 (visited on 07/11/2021).

[111] Francis W. M. R. Schwarze. “Wood Decay under the Microscope”. In: Fungal Biology
Reviews 21.4 (1st Nov. 2007), pp. 133–170. issn: 1749-4613. doi: 10 . 1016 / j . fbr .
2007 . 09 . 001. url: https : / / www . sciencedirect . com / science / article / pii /
S1749461307000449 (visited on 14/10/2021).

[112] Francis W. M. R. Schwarze, Julia Engels and Claus Mattheck. Fungal Strategies of Wood
Decay in Trees. Springer Science & Business Media, 2000. 214 pp. isbn: 978-3-540-67205-0.
[113] Francis W. M. R. Schwarze and Helge Landmesser. “Preferential Degradation of Pit Mem-
branes within Tracheids by the Basidiomycete Physisporinus Vitreus”. In: Wood Research
and Technology 54.5 (6th Sept. 2000), pp. 461–462. issn: 1437-434X. doi: 10.1515/HF.
2000.077. url: https://www.degruyter.com/document/doi/10.1515/HF.2000.077/
html (visited on 07/11/2021).

[114] Ernest T. Selig and Carl J. Roner. “Eﬀects of Particle Characteristics on Behavior of
Granular Material”. In: Transportation Research Record 1131 (1987). issn: 0361-1981.
url: https://trid.trb.org/view/282802 (visited on 19/11/2021).

[115] Yu-Lin Shen et al. “Eﬀective Elastic Response of Two-Phase Composites”. In: Acta Metal-
lurgica et Materialia 42.1 (1st Jan. 1994), pp. 77–97. issn: 0956-7151. doi: 10.1016/0956-
7151(94)90050- 7. url: https://www.sciencedirect.com/science/article/pii/
0956715194900507 (visited on 19/07/2021).

[116] Jiangtao Shi, Dong Xing and Jian Lia. “FTIR Studies of the Changes in Wood Chemistry
from Wood Forming Tissue under Inclined Treatment”. In: Energy Procedia. 2012 Inter-
national Conference on Future Energy, Environment, and Materials 16 (1st Jan. 2012),
pp. 758–762. issn: 1876-6102. doi: 10 . 1016 / j . egypro . 2012 . 01 . 122. url: https :
/ / www . sciencedirect . com / science / article / pii / S1876610212001324 (visited on
22/07/2021).

[117] Ayyappa Kumar Sista Kameshwar and Wensheng Qin. “Comparative Study of Genome-
Wide Plant Biomass-Degrading CAZymes in White Rot, Brown Rot and Soft Rot Fungi”.
In: Mycology 9.2 (3rd Apr. 2018), pp. 93–105. issn: 2150-1203. doi: 10.1080/21501203.
2017.1419296. pmid: 30123665. url: https://doi.org/10.1080/21501203.2017.
1419296 (visited on 14/10/2021).

Deliverable D5.2

Page 85 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[118] Laura Sisti et al. “Valorization of Wheat Bran Agro-Industrial Byproduct as an Upgrading
Filler for Mycelium-Based Composite Materials”. In: Industrial Crops and Products 170
(15th Oct. 2021), p. 113742. issn: 0926-6690. doi: 10.1016/j.indcrop.2021.113742.
url: https://www.sciencedirect.com/science/article/pii/S0926669021005069
(visited on 11/11/2021).

[119] S Sivaprasad et al. “Development of a Novel Mycelium Bio-Composite Material to Substi-

tute for Polystyrene in Packaging Applications”. In: Materials Today: Proceedings (19th May
2021). doi: 10.1016/j.matpr.2021.04.622.

[120] Brian C. Smith. “Organic Nitrogen Compounds, VII: Amides—The Rest of the Story”. In:

Spectroscopy. Vol. 35. 1. 1st Jan. 2020, pp. 10–15. url: https://www.spectroscopyonline.
com / view / organic - nitrogen - compounds - vii - amides - rest - story (visited on
14/07/2021).

[121] Wenjing Sun et al. “Fully Bio-Based Hybrid Composites Made of Wood, Fungal Mycelium
and Cellulose Nanoﬁbrils”. In: Scientiﬁc Reports 9.1 (1 6th Mar. 2019), p. 3766. issn: 2045-
2322. doi: 10.1038/s41598-019-40442-8. url: https://www.nature.com/articles/
s41598-019-40442-8 (visited on 11/11/2021).

[122] Xueguang Sun and Ming Tang. “Comparison of Four Routinely Used Methods for Assess-
ing Root Colonization by Arbuscular Mycorrhizal Fungi”. In: Botany 90 (1st Nov. 2012).
doi: 10.1139/b2012-084.

[123] Zeynep Tacer-Caba et al. “Comparison of Novel Fungal Mycelia Strains and Sustain-
able Growth Substrates to Produce Humidity-Resistant Biocomposites”. In: Materials
& Design 192 (1st July 2020), p. 108728. issn: 0264-1275. doi: 10 . 1016 / j . matdes .
2020 . 108728. url: https : / / www . sciencedirect . com / science / article / pii /
S0264127520302628 (visited on 31/10/2021).

[124] Vanessa de Cássia Teixeira da Silva et al. “Eﬀect of pH, Temperature, and Chemicals
on the Endoglucanases and -Glucosidases from the Thermophilic Fungus Myceliophthora
Heterothallica F.2.1.4. Obtained by Solid-State and Submerged Cultivation”. In: Biochem-
istry Research International 2016 (8th May 2016), e9781216. issn: 2090-2247. doi: 10.
1155/2016/9781216. url: https://www.hindawi.com/journals/bri/2016/9781216/
(visited on 06/11/2021).

[125] Emil E. Thybring, Lisbeth G. Thygesen and Ingo Burgert. “Hydroxyl Accessibility in
Wood Cell Walls as Aﬀected by Drying and Re-Wetting Procedures”. In: Cellulose 24.6
(June 2017), pp. 2375–2384. issn: 1572-882X. doi: 10.1007/s10570-017-1278-x. url:
https://www.research-collection.ethz.ch/handle/20.500.11850/130350 (visited
on 08/11/2021).

[126] Emil Engelund Thybring, Maija Kymäläinen and Lauri Rautkari. “Moisture in Modiﬁed
Wood and Its Relevance for Fungal Decay”. In: IForest 11.3 (3 2018), pp. 418–422. issn:
19717458. doi: 10.3832/ifor2406-011. url: https://www.mendeley.com/catalogue/
ff3e8410-5e61-325f-8fbd-a8fdc622e1ee/ (visited on 25/02/2021).

[127] Raunel Tinoco, Jorge Verdin and Rafael Vazquez-Duhalt. “Role of Oxidizing Mediators
and Tryptophan 172 in the Decoloration of Industrial Dyes by the Versatile Peroxidase
from Bjerkandera Adusta”. In: Journal of Molecular Catalysis B: Enzymatic 46.1 (2nd May
2007), pp. 1–7. issn: 1381-1177. doi: 10.1016/j.molcatb.2007.01.006. url: https:
/ / www . sciencedirect . com / science / article / pii / S1381117707000197 (visited on
05/11/2021).

Page 86 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[128] George Tsoumis. Science and Technology of Wood: Structure, Properties and Utilization.
Verlag Kessel, 2009. 494 pp. isbn: 978-3-941300-22-4. url: https://www.cabdirect.
org/cabdirect/abstract/20103064578 (visited on 07/11/2021).

[129] H. M. G. van der Werf et al. “Quality of Hemp (Cannabis Sativa L.) Stems as a Raw Ma-
terial for Paper”. In: Industrial Crops and Products 2.3 (1st May 1994), pp. 219–227. issn:
0926-6690. doi: 10.1016/0926-6690(94)90039-6. url: https://www.sciencedirect.
com/science/article/pii/0926669094900396 (visited on 07/11/2021).

[130] B. R. M. Vyas, J. Volc and V. Šašek. “Eﬀects of Temperature on the Production of Man-
ganese Peroxidase and Lignin Peroxidase byPhanerochaete Chrysosporium”. In: Folia Mi-
crobiologica 39.1 (1st Feb. 1994), pp. 19–22. issn: 1874-9356. doi: 10.1007/BF02814523.
url: https://doi.org/10.1007/BF02814523 (visited on 06/11/2021).

[131] Takashi Watanabe et al. “Production and Chemiluminescent Free Radical Reactions of
Glyoxal in Lipid Peroxidation of Linoleic Acid by the Ligninolytic Enzyme, Manganese
Peroxidase”. In: European Journal of Biochemistry 268.23 (2001), pp. 6114–6122. issn:
1432-1033. doi: 10.1046/j.0014-2956.2001.02557.x. url: https://onlinelibrary.
wiley.com/doi/abs/10.1046/j.0014-2956.2001.02557.x (visited on 05/11/2021).

[132] R. Whetten and R. Sederoﬀ. “Lignin Biosynthesis”. In: The Plant Cell 7.7 (July 1995),
pp. 1001–1013. issn: 1532-298X. doi: 10.1105/tpc.7.7.1001. pmid: 12242395.
[133] Alex Wiedenhoeft. “Structure and Function of Wood”. In: Wood handbook : wood as an
engineering material: chapter 3. Centennial ed. General technical report FPL ; GTR-190.
Madison, WI : U.S. Dept. of Agriculture, Forest Service, Forest Products Laboratory, 2010:
p. 3.1-3.18. 190 (2010), pp. 3.1–3.18. url: http://www.fs.usda.gov/treesearch/pubs/
37429 (visited on 07/11/2021).

[134] Guido Wimmers et al. “Fundamental Studies for Designing Insulation Panels from Wood
Shavings and Filamentous Fungi”. In: BioResources 14.3 (2019), pp. 5506–5520. url:
https : / / bioresources . cnr . ncsu . edu / resources / fundamental - studies - for -
designing - insulation - panels - from - wood - shavings - and - filamentous - fungi/
(visited on 16/11/2021).

[135] Pan Wu et al. “Origins and Features of Pectate Lyases and Their Applications in Industry”.
In: Applied Microbiology and Biotechnology 104.17 (1st Sept. 2020), pp. 7247–7260. issn:
1432-0614. doi: 10.1007/s00253- 020- 10769- 8. url: https://doi.org/10.1007/
s00253-020-10769-8 (visited on 06/11/2021).

[136] Yangang Xing et al. “Growing and Testing Mycelium Bricks as Building Insulation Ma-
terials”. In: IOP Conference Series: Earth and Environmental Science 121 (Feb. 2018),
p. 022032. issn: 1755-1315. doi: 10 . 1088 / 1755 - 1315 / 121 / 2 / 022032. url: https :
//doi.org/10.1088/1755-1315/121/2/022032 (visited on 24/11/2021).

[137] Sangeeta Yadav et al. “Puriﬁcation and Characterization of Pectin Lyase Produced by
Aspergillus Terricola and Its Application in Retting of Natural Fibers”. In: Applied Bio-
chemistry and Biotechnology 159.1 (1st Oct. 2009), pp. 270–283. issn: 1559-0291. doi:
10.1007/s12010-008-8471-1. url: https://doi.org/10.1007/s12010-008-8471-1
(visited on 06/11/2021).

[138] Zhaohui Yang et al. “Physical and Mechanical Properties of Fungal Mycelium-Based
Biofoam”. In: Journal of Materials in Civil Engineering 29 (23rd Mar. 2017), p. 04017030.
doi: 10.1061/(ASCE)MT.1943-5533.0001866.

Deliverable D5.2

Page 87 of 89

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[139] Marcel Zamocky et al. “Cellobiose Dehydrogenase–a Flavocytochrome from Wood-Degrading,

Phytopathogenic and Saprotropic Fungi”. In: Current Protein & Peptide Science 7.3 (June
2006), pp. 255–280. issn: 1389-2037. doi: 10.2174/138920306777452367.

[140] Zhi-Min Zhang, Shan Chen and Yi-Zeng Liang. “Baseline Correction Using Adaptive Iter-
atively Reweighted Penalized Least Squares”. In: The Analyst 135.5 (May 2010), pp. 1138–
1146. issn: 1364-5528. doi: 10.1039/b922045c. pmid: 20419267.

[141] Shuai Zhou et al. “Investigation of Lignocellulolytic Enzymes during Diﬀerent Growth

Phases of Ganoderma Lucidum Strain G0119 Using Genomic, Transcriptomic and Secretomic
Analyses”. In: PLOS ONE 13.5 (31st May 2018), e0198404. issn: 1932-6203. doi: 10.
1371/journal.pone.0198404. url: https://journals.plos.org/plosone/article?
id=10.1371/journal.pone.0198404 (visited on 20/10/2021).

[142] A. R. Ziegler et al. “Evaluation of Physico-Mechanical Properties of Mycelium Reinforced
Green Biocomposites Made from Cellulosic Fibers”. In: Applied Engineering in Agriculture
32.6 (5th Dec. 2016), pp. 931–938. issn: 08838542, 19437838. doi: 10.13031/aea.32.
11830. url: http://elibrary.asabe.org/abstract.asp?aid=47570&t=3&dabs=Y&
redir=&redirType= (visited on 12/11/2021).

Page 88 of 89

Deliverable D5.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Appendices

Deliverable D5.2

Page 89 of 89

