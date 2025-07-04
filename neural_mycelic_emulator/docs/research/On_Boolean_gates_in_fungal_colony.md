See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/339471229

On Boolean gates in fungal colony

Preprint · February 2020

DOI: 10.48550/arXiv.2002.09680

CITATIONS
0

6 authors, including:

Andrew Adamatzky

University of the West of England, Bristol

922 PUBLICATIONS   14,614 CITATIONS   

SEE PROFILE

Alexander E Beasley

University of Hertfordshire

28 PUBLICATIONS   138 CITATIONS   

SEE PROFILE

READS
495

Martin Tegelaar

Utrecht University

27 PUBLICATIONS   303 CITATIONS   

SEE PROFILE

Richard Mayne

University of Oxford

102 PUBLICATIONS   1,098 CITATIONS   

SEE PROFILE

All content following this page was uploaded by Alexander E Beasley on 02 July 2021.

The user has requested enhancement of the downloaded file.

0
2
0
2

b
e
F
2
2

]
T
E
.
s
c
[

1
v
0
8
6
9
0
.
2
0
0
2
:
v
i
X
r
a

On Boolean gates in fungal colony

Andrew Adamatzkya,b, Martin Tegelaarc, Han A. B. Wostenc,
Anna L. Powella, Alexander E. Beasleya, Richard Maynea,d

aUnconventional Computing Laboratory, UWE, Bristol, UK
bCorresponding author: andrew.adamatzky@uwe.ac.uk
cMicrobiology Department, University of Utrecht, Utrecht, The Netherlands
dDepartment of Applied Sciences, University of the West of England, UWE, Bristol, UK

Abstract

A fungal colony maintains its integrity via ﬂow of cytoplasm along mycelium
network. This ﬂow, together with possible coordination of mycelium tips
propagation, is controlled by calcium waves and associated waves of elec-
trical potential changes. We propose that these excitation waves can be
employed to implement a computation in the mycelium networks. We use
FitzHugh-Nagumo model to imitate propagation of excitation in a single
colony of Aspergillus niger. Boolean values are encoded by spikes of extra-
cellular potential. We represent binary inputs by electrical impulses on a pair
of selected electrodes and we record responses of the colony from sixteen elec-
trodes. We derive sets of two-inputs-on-output logical gates implementable
the fungal colony and analyse distributions of the gates.

Keywords: mycelium network, Boolean gates, unconventional computing

1. Introduction

A vibrant ﬁeld of unconventional computing aims to employ space-time
dynamics of physical, chemical and biological media to design novel com-
putational techniques, architectures and working prototypes of embedded
computing substrates and devices. Interaction-based computing devices, is
one of the most diverse and promising families of the unconventional com-
puting structures. They are based on interactions of ﬂuid streams, sig-
[67,
nals propagating along conductors or excitation wave-fronts, see e.g.
64, 10, 10, 81, 76, 37, 29, 30, 70, 27, 73, 70, 31]. Typically, logical gates
and their cascade implemented in an excitable medium are ‘handcrafted’

Preprint submitted to Elsevier

February 25, 2020

 
 
 
 
 
 
Figure 1: Exemplar spikes of extracellular electrical potential propagating in fungal
mycelium.

to address exact timing and type of interactions between colliding wave-
fronts [67, 64, 1, 9, 74, 8, 20, 71, 82, 72, 32, 7, 69]. The artiﬁcial design of
logical circuits might be suitable when chemical media or functional materi-
als are used. However, the approach might be not feasible when embedding
computation in living systems, where the architecture of conductive path-
ways may be diﬃcult to alter or control. In such situations an opportunistic
approach to outsourcing computation can be adopted. The system is per-
turbed via two or more input loci and its dynamics if recorded at one or
more output loci. A wave-front appearing at one of the output loci is in-
terpreted as logical truth or ‘1’. Thus the system with relatively unknown
structure implements a mapping {0, 1}n → {0, 1}m, where n is a number of
input loci and m is a number of output loci, n, m > 0 [12, 6]. The approach
belong to same family of computation outsourcing techniques as in materio
computing [47, 48, 68, 49, 50] and reservoir computing [77, 43, 18, 42, 19].

Fungal colonies are characterised by rich typology of mycelium networks [36,

28, 23, 24, 38] in some cases aﬃne to fractal structures [53, 56, 16, 46, 15, 55].
Rich morphological features might imply rich computational abilities and
In
thus worse to analyse from realising Boolean functions point of view.
numerical experiments we study implementation of logical gates via inter-
action of numerous travelling excitation waves, seen as as action potentials,
on an image of a real fungal colony. Action potential-like spikes of electrical
potential have been discovered using intra-cellular recording of mycelium of
Neurospora crassa [66] and further conﬁrmed in intra-cellular recordings of
action potential in hypha of Pleurotus ostreatus and Armillaria bulbosa [54]

2

Potential, V−0.00100.0010.0020.0030.0040.0050.006Time, sec9×10510×105and in extra-cellular recordings of fruit bodies of and substrates colonized
by mycelium of Pleurotus ostreatus [4] (Fig. 1). While the exact nature of
the travelling spikes remains uncertain we can speculate, by drawing analo-
gies with oscillations of electrical potential of slime mould Physarum poly-
cephalum [39, 40, 41, 45], that the spikes in fungi are triggered by calcium
waves, reversing of cytoplasmic ﬂow, translocation of nutrients and metabo-
lites. Studies of electrical activity of higher plants can brings us even more
clues. Thus, the plants use the electrical spikes for a long-distance communi-
cation aimed to coordinate an activity of their bodies [75, 26, 83]. The spikes
of electrical potential in plants relate to a motor activity [65, 25, 63, 80],
responses to changes in temperature [51], osmotic environment [79] and me-
chanical stimulation [59, 58]. The paper is structured as follows. Colony
imaging and numerical solutions of FitzHugh-Nagumo equations are intro-
duced in Sect. 2. Section 3 studies a role of excitability on the coverage of
the network by travelling waves of excitation and exempliﬁes distributions of
Boolean computable on the given mycelium network. We discuss operation
characteristics of the mycelium computer in Sect. 4.

2. Methods

2.1. Colony imaging

Aspergillus niger strain AR9#2 [78], expressing Green Fluorescent Pro-
tein (GFP) from the glucoamylase (glaA) promoter, was grown at 30oC on
minimal medium (MM) [21] with 25 mM xylose and 1.5% agarose (MMXA).
MMXA cultures were grown for three days, after which conidia were har-
vested using saline-Tween (0.8% NaCl and 0.005% Tween-80). 250 ml liquid
cultures were inoculated with 1.25·109 freshly harvested conidia and grown at
200 rpm and 30oC in 1 L Erlenmeyer ﬂasks in complete medium (CM) (MM
containing 0.5% yeast extract and 0.2% enzymatically hydrolyzed casein)
supplemented with 25 mM xylose (repressing glaA expression). Mycelium
was harvested after 16 h and washed twice with PBS. Ten g of biomass (wet
weight) was transferred to MM supplemented with 25 mM maltose (inducing
glaA expression).

Fluorescence of GFP was localised in micro-colonies using a DMI 6000
CS AFC confocal microscope (Leica, Mannheim, Germany). Micro-colonies
were ﬁxed overnight at 4oC in 4% paraformaldehyde in PBS, washed twice
with PBS and taken up in 50 ml PBS supplemented with 150 mM glycine to

3

quench autoﬂuorescence. Micro-colonies were then transferred to a glass bot-
tom dish (Cellview™, Greiner Bio-One, Frickenhausen, Germany, PS, 35/10
MM) and embedded in 1% low melting point agarose at 45oC. Micro-colonies
were imaged at 20× magniﬁcation (HC PL FLUOTAR L 20 × 0.40 DRY).
GFP was excited by white light laser at 472 nm using 50% laser intensity
(0.1 kW/cm2) and a pixel dwell time of 72 ns. Fluorescent light emission was
detected with hybrid detectors in the range of 490–525 nm. Pinhole size was
1 Airy unit. Z-stacks of imaged micro-colonies were made using 100 slices
with a slice thickness of 8.35 µm. 3D projections were made with Fiji [62].

2.2. Numerical modelling

We used still image of the colony as a conductive template. The image of
the fungal colony (Fig. 2a) was projected onto a 1000 × 960 nodes grid. The
original image M = (mij)1≤j≤ni,1≤j≤nj , mij ∈ {rij, gij, bij}, where ni = 1000
and nj = 960, and 1 ≤ r, g, b ≤ 255 (Fig. 2a), was converted to a conductive
matrix C = (mij)1≤i,j≤n (Fig. 2b) derived from the image as follows: mij = 1
if rij < 20, (gij > 40) and bij < 20; a dilution operation was applied to C.

FitzHugh-Nagumo (FHN) equations [22, 52, 57] is a qualitative approxi-
mation of the Hodgkin-Huxley model [14] of electrical activity of living cells:

∂v
∂t
∂v
∂t

= c1u(u − a)(1 − u) − c2uv + I + Du∇2

= b(u − v),

(1)

(2)

where u is a value of a trans-membrane potential, v a variable accountable
for a total slow ionic current, or a recovery variable responsible for a slow
negative feedback, I is a value of an external stimulation current. The cur-
rent through intra-cellular spaces is approximated by Du∇2, where Du is a
conductance. Detailed explanations of the ‘mechanics’ of the model are pro-
vided in [60], here we shortly repeat some insights. The term Du∇2u governs
a passive spread of the current. The terms c2u(u − a)(1 − u) and b(u − v)
describe the ionic currents. The term u(u − a)(1 − u) has two stable ﬁxed
points u = 0 and u = 1 and one unstable point u = a, where a is a threshold
of an excitation.

We integrated the system using the Euler method with the ﬁve-node
Laplace operator, a time step ∆t = 0.015 and a grid point spacing ∆x = 2,
while other parameters were Du = 1, a = 0.13, b = 0.013, c1 = 0.26. We
controlled excitability of the medium by varying c2 from 0.05 (fully excitable)

4

(a)

(b)

(c)

(d)

Figure 2: Image of the fungal colony, 1000 × 960 pixels used as a template conductive
for FHN. (a) Original image, mycelium is seen as green pixels. (b) Conductive matrix C,
conductive pixels are black. (c) Distribution of neigbourhood sizes. (d) Conﬁguration of
electrodes.

5

Ratio ν00.050.100.150.20Neighbourhood size k12345678to 0.015 (non excitable). Boundaries are considered to be impermeable:
∂u/∂n = 0, where n is a vector normal to the boundary.

The waves of excitation propagated on conductive nodes of the grid of C,
in addition to the parameter c2, excitability of each conductive node was de-
pendent on a number k of its immediate conductive neighbours. Distribution
of neighbourhood sizes are shown in Fig. 2c.

x at an electrode location x as px = (cid:80)

To show dynamics of excitation in the network we simulated electrodes by
calculating a potential pt
y:|x−y|<2(ux −
vx). Conﬁguration of electrodes 1, · · · , 16 is shown in Fig. 2d. The numerical
integration code written in Processing was inspired by previous methods of
numerical integration of FHN and our own computational studies of the
impulse propagation in biological networks [33, 57, 60, 5, 6]. Time-lapse
snapshots provided in the paper were recorded at every 100th time step, and
we display sites with u > 0.04; videos and ﬁgures were produced by saving
a frame of the simulation every 100th step of the numerical integration and
assembling the saved frames into the video with a play rate of 30 fps. Videos
are available at [13].

3. Results

While implementing numerical experiments were selected a range the net-
work excitability (Subsect. 3.1) and then realised sets of logical gates for
excitability values selected (Subsect. 3.2).

3.1. Eﬀect of excitability on overall activity

For c2 < 0.0945 any source of excitation triggers excitation dynamics
which occupies all parts of the network accessible, via mycelial strands, from
the source. Due to the high level of excitability the network remains in the
excitable state (Fig. 3a). For values c2 from 0.094 to 0.00965 we observe
propagation of ‘classical’ excitation wave-fronts resembling circular, target
and spiral waves in a continuous medium. Examples of wave-fronts propa-
gating in networks with excitability levels c2 = 0.095 and c2 = 0.096, excited
at the same loci shown in Fig. 4a, are shown in Figs.4 and 5. In the network
with c2 there are many pathways for propagation of the excitation wave-
fronts, therefore, despite being fully deterministic, the network exhibit disor-
dered oscillations of its activity (Fig. 3c). A number of conductive pathways
decreases when c2 increases from 0.095 to 0.096. Thus many propagating
wave-fronts become, relatively, quickly conﬁned to a limited domains of the

6

(a) c2 = 0.094

(b) c2 = 0.097

(c) c2 = 0.095

(d) c2 = 0.096

Figure 3: Dynamics of the activity α for various values of excitability c2, the values are
shown in sub-captions. For every iteration t we measured the activity of the network as a
number of conductive nodes x with ut

x > 0.1.

7

Activity α, ratio00.010.020.030.04Time, iterations02×105Activity α, ratio00.00050.00100.00150.0020Time, iterations020,000Activity α, ratio00.0020.0040.0060.0080.010Time, iterations01×1052×1053×1054×105Activity α, ratio00.0010.0020.0030.004Time, iterations01×1052×1053×1054×1050.00350.00400.00452.0×1052.2×105(a) t = 200

(b) t = 1900

(c) t = 40500

(d) t = 70400

(e) t = 104800

(f) t = 115850

Figure 4: Snapshots of excitation dynamics for c2 = 0.095.

8

(a) t = 1900

(b) t = 40500

(c) t = 70400

(d) t = 104800

(e) t = 115850

(f) t = 155000

Figure 5: Snapshots of excitation dynamics for c2 = 0.096. Compare (c) and (e): the
pattern of excitation returns to the exact point of the cycle.

(a) c2 = 0.095

(b) c2 = 0.096

(c) c2 = 0.097

Figure 6: Coverage of the network for excitability c2 (a) 0.095, (b) 0.096, (c) 0.097. If the
a pixel p of the image was excited, ut
p > 0.1, it is assumed to be covered and coloured red
in the pictures (abc); the pixels which never were excited are coloured gray.

9

Figure 7: Fragment of electrical potential record on electrode 7 in response to inputs (01),
black dashed line, (10), red dotted line, (11), solid green line, entered as impulses via
electrodes Ex = 5 and Ey = 15. See locations of electrodes in Fig. 2d. To make the
individual plots visible in places of exact overlapping, we added potential −5 to recording
in response to input (01) and and potential 5 to recording in response to input (11).

network, where they continue ‘circling’ indeﬁnitely. A set of regular oscil-
lations of activity becomes evidence after a number of iterations (Fig. 3d).
The coverage of the network by excitation wave-fronts reduced with increase
of c2 from 0.095 to 0.096 (Fig. 6ab) and becomes localised when c2 reaches
0.097 (Figs. 6c and 3b). Thus we used networks with c2 = 0.095 or 0.096 for
implementation of Boolean functions.

3.2. Distribution of Boolean gates

Input Boolean values are encoded as follows. We earmark two sites of the
network as dedicated inputs, x and y, and represent logical True, or ‘1’, as
an excitation, or an impulse injected injected in the network via electrodes.
If x = 1 then the site corresponding to x is excited, if x = 0 the site is not
excited.

We here assume that each spike represents logical True and that spikes
occurring within less than 2·102 iterations happen simultaneously. By select-
ing speciﬁc intervals of recordings we can realise several gates in a single site
of recording. In this particular case we assumed that spikes are separated if
their occurrences lie more than 103 iterations apart. An example is shown in
Fig. 7.

10

SySySySxx+yxyxyx ⊕yPotential, units−50050Time, iterations30,00040,00050,00060,000Figure 8: Recording of electrical potential from all electrodes in responses to inputs in
response to inputs (01), black line, (10), red line, (11), green line, injected as spikes via
electrodes Ex = 5 and Ey = 15.

11

(a) Ex = 3, Ey = 13, c2 = 0.095

x + y
0
0
0
0
1
0
0
1
1
0
0
0
0
1
2
0
6

Sy
0
0
0
0
0
0
0
0
0
0
1
0
0
5
5
1
12

x ⊕ y
0
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
1

Sx
0
2
0
0
7
2
2
8
6
0
2
4
3
0
1
5
42

xy
0
0
0
0
1
2
0
2
1
1
0
2
2
0
0
2
13

xy
0
0
0
0
0
0
0
0
0
0
1
0
0
1
1
0
3

xy
0
0
0
0
0
0
0
0
0
0
2
0
0
0
0
0
2

(c) Ex = 7, Ey = 14, c2 = 0.094

x + y
0
0
0
0
0
0
0
0
0
0
0
0
1
0
1
0
2

Sy
0
0
0
0
0
0
0
0
0
0
0
0
0
2
3
2
7

x ⊕ y
0
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
1

Sx
0
2
0
0
1
1
1
0
3
0
1
0
2
2
0
0
13

xy
0
1
0
0
3
1
1
2
4
1
1
6
3
0
3
0
26

xy
0
0
0
0
0
0
0
0
0
2
0
0
1
1
0
1
5

xy
0
0
0
0
0
0
0
0
0
0
0
0
0
0
2
0
2

(e) Ex = 5, Ey = 15, c2 = 0.095

x + y
0
0
0
0
1
3
0
1
0
0
1
2
1
3
1
1
14

Sy
0
0
0
0
4
0
0
3
5
0
0
4
7
1
5
3
32

x ⊕ y
0
0
0
0
0
0
0
1
0
0
2
0
0
0
0
1
4

Sx
0
8
0
0
0
4
0
1
1
3
4
2
0
2
0
2
27

xy
0
0
0
0
0
0
1
0
0
0
2
2
0
0
0
2
7

xy
0
0
0
0
2
0
0
1
2
0
0
0
3
1
6
1
16

xy
0
0
0
0
0
0
0
1
0
0
2
1
0
0
0
1
5

E
0
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
12
13
14
15
Total

E
0
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
12
13
14
15
Total

E
0
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
12
13
14
15
Total

Total
0
2
0
0
9
4
2
11
8
1
7
6
5
7
9
8
79

Total
0
3
0
0
4
2
2
2
7
3
2
6
8
5
9
3
56

Total
0
8
0
0
7
7
1
8
8
3
11
11
11
7
12
11
105

E
0
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
12
13
14
15
Total

E
0
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
12
13
14
15
Total

E
0
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
12
13
14
15
Total

(b) Ex = 7, Ey = 14, c2 = 0.095
x + y
0
0
0
0
0
0
0
1
0
0
1
1
2
1
0
0
6

x ⊕ y
0
0
0
0
0
0
0
0
0
0
1
0
0
0
0
0
1

Sx
0
5
0
0
6
6
2
4
7
0
5
4
3
1
2
8
53

Sy
0
0
0
0
0
0
0
0
0
0
0
0
0
5
4
0
9

xy
0
0
0
0
1
0
0
2
0
0
0
3
0
0
1
2
9

xy
0
0
0
0
0
0
0
0
0
0
0
0
0
2
2
0
4

xy
0
0
0
0
0
0
0
0
0
0
0
0
0
0
1
0
1

(d) Ex = 5, Ey = 15, c2 = 0.094
x + y
0
0
0
0
0
0
0
1
0
0
0
1
0
2
0
0
4

x ⊕ y
0
0
0
0
0
0
0
0
0
0
0
0
0
2
0
0
2

Sx
0
1
0
0
0
0
0
0
1
1
2
0
0
1
1
0
7

Sy
0
0
0
0
1
0
0
2
1
0
0
5
6
0
1
0
16

xy
0
1
0
0
2
1
1
0
2
2
0
1
1
1
0
0
12

xy
0
0
0
0
1
0
0
4
1
0
0
1
2
1
5
1
16

xy
0
0
0
0
1
0
0
0
0
0
1
1
0
1
0
0
4

(f) Ex = 5, Ey = 15, c2 = 0.096

x + y
0
0
0
0
1
0
0
1
1
0
0
2
0
0
1
0
6

Sy
0
0
0
0
3
0
0
4
5
0
1
3
4
2
3
3
28

x ⊕ y
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0

Sx
0
2
0
0
0
4
2
1
0
7
5
1
0
2
2
2
28

xy
0
1
0
0
0
2
1
0
0
0
1
1
0
0
0
0
6

xy
0
0
0
0
1
0
0
1
1
0
1
1
0
2
3
2
12

xy
0
0
0
0
0
0
0
0
0
0
0
1
0
0
0
0
1

Total
0
5
0
0
7
6
2
7
7
0
7
8
5
9
10
10
83

Total
0
2
0
0
5
1
1
7
5
3
3
9
9
8
7
1
61

Total
0
3
0
0
5
6
3
7
7
7
8
9
4
6
9
7
81

Table 1: Numbers of Boolean gates detected for selected pairs of input electrodes Ex and
Ey.

12

Figure 9: Comparative ratios of Boolean gates discovered in mycelium network in
present analysed in present paper, black disc and solid line; slime mould Physarum poly-
cephalum [34], black circle and dotted line; succulent plant [11], red snowﬂake and dashed
line; single molecule of protein verotoxin [2], light blue ‘+’ and dash-dot line; actin bundles
network [12], green triangle pointing right and dash-dot-dot line; actin monomer [3], ma-
genta triangle pointing left and dashed line. Area of xor gate is magniﬁed in the insert.
Lines are to guide eye only.

13

x+ySxSyx⊕yxyxyxyNumbers of Boolean gates detected on the electrodes for selected pairs
of input electrodes are shown in Tab. 1. We see that select x and select y
gates, Sx and Sy are most frequent. They usually are detected with the same
frequency (Tab. 1d–f), however there are examples of input electrode pairs,
where one of the select gates is found much more often than another. This
is most visible for the pair (Ex, Ey) = (3, 13) where Sx dominates (Tab. 1a),
and the pair (7,14) and (Tab. 1bc) where Sy dominates. Next common gates
in the hierarchy are xy and xy. The gates xy and x + y are detected with
nearly the same frequency with gate x + y being slightly more common. The
x ⊕ y is the most rare gate.

The sub-tables Tab. 1d, Tab. 1e and Tab. 1f show how excitability of the
network aﬀects numbers of gates detected. The networks with high excitabil-
ity, c2 = 0.094, and low excitability, c2 = 0.0096 realise smaller number of
gates then that realised by sub-excitable network, c2 = 0.095.

Overall distribution (average of outputs of input electrode pairs (3,13),
(5,15), (7,14), (4,13), (13,7)) of a ratio of gates discovered is shown in Fig. 9.
This is accompanied by distributions of gates discovered in experimental lab-
oratory reservoir computing with slime mould Physarum polycephalum [34],
succulent plant [11] and numerical modelling experiments on computing with
protein verotoxin [2], actin bundles network [12], and actin monomer [3]. All
the listed distributions have very similar structure with gates selecting one
of the inputs in majority, followed by or gate, not-and an and-not gates.
The gate and is usually underrepresented in experimental and modelling
experiments. The gate xor is a rare ﬁnd.

4. Discussion

We have demonstrated how sets of logical gates can be implemented in
single colony mycelium networks via initiation of electrical impulses. The
impulses travel in the network, interact with each other (annihilate, reﬂect,
change their phase). Thus for diﬀerent combinations of input impulses and
record diﬀerent combinations of output impulses, which in some cases can be
interpreted as representing two-inputs-one-output functions.

To estimate a speed of computation we refer to Olsson and Hansson’s [54]
original study, in which they proposed that electrical activity in fungi could
be used for communication with message propagation speed 0.5 mm/sec.
Diameter of the colony (Fig. 2a), which experimental laboratory images has
been used to run FHN model, is c. 1.7 mm. Thus, it takes the excitation

14

waves initiated at a boundary of the colony up to 3-4 sec to span the whole
mycelium network (this time is equivalent to c. 70K iterations of the numer-
ical integration model). In 3-4 sec the mycelium network can compute up to
a hundred logical gates. This gives us the rate of a gate per 0.03 sec, or, in
terms of frequency this will be c. 30 Hz. The mycelium network computing
can not compete with existing silicon architecture however its application
domain can be a unique of living biosensors (a distribution of gates realised
might be aﬀected by environmental conditions) [44] and computation em-
bedded into structural elements where fungal materials are used [61, 35, 17].

5. Acknowledgement

This project has received funding from the European Union’s Horizon
2020 research and innovation programme FET OPEN “Challenging current
thinking” under grant agreement No 858132.

References

References

[1] Andrew Adamatzky.

Collision-based computing in Belousov–
Zhabotinsky medium. Chaos, Solitons & Fractals, 21(5):1259–1264,
2004.

[2] Andrew Adamatzky.

Computing in verotoxin.

ChemPhysChem,

18(13):1822–1830, 2017.

[3] Andrew Adamatzky. Logical gates in actin monomer. Scientiﬁc reports,

7(1):1–14, 2017.

[4] Andrew Adamatzky. On spiking behaviour of oyster fungi pleurotus

djamor. Scientiﬁc reports, 8(1):1–7, 2018.

[5] Andrew Adamatzky. On interplay between excitability and geometry.

arXiv preprint arXiv:1904.06526, 2019.

[6] Andrew Adamatzky. Plant leaf computing. Biosystems, 2019.

[7] Andrew Adamatzky, Ben de Lacy Costello, and Larry Bull. On poly-
morphic logical gates in subexcitable chemical medium. International
Journal of Bifurcation and Chaos, 21(07):1977–1986, 2011.

15

[8] Andrew Adamatzky, Ben De Lacy Costello, Larry Bull, and Julian Hol-
ley. Towards arithmetic circuits in sub-excitable chemical media. Israel
Journal of Chemistry, 51(1):56–66, 2011.

[9] Andrew Adamatzky and Benjamin de Lacy Costello. Binary colli-
sions between wave-fragments in a sub-excitable Belousov–Zhabotinsky
medium. Chaos, Solitons & Fractals, 34(2):307–315, 2007.

[10] Andrew Adamatzky, Benjamin de Lacy Costello, Chris Melhuish, and
Norman Ratcliﬀe. Experimental implementation of mobile robot taxis
with onboard Belousov–Zhabotinsky chemical medium. Materials Sci-
ence and Engineering: C, 24(4):541–548, 2004.

[11] Andrew Adamatzky, Simon Harding, Victor Erokhin, Richard Mayne,
Nina Gizzie, Frantisek Baluˇska, Stefano Mancuso, and Georgios Ch Sir-
akoulis. Computers from plants we never made: Speculations. In In-
spired by nature, pages 357–387. Springer, 2018.

[12] Andrew Adamatzky, Florian Huber, and J¨org Schnauß. Computing on

actin bundles network. Scientiﬁc reports, 9(1):1–10, 2019.

[13] Andrew Adamatzky, Martin Tegelaar, Han Wosten, Anna Powell, and
Alexander Beasley. Supplementary materials. on boolean gates in fungal
colony.

[14] Go W Beeler and H Reuter. Reconstruction of the action potential of
ventricular myocardial ﬁbres. The Journal of physiology, 268(1):177–
210, 1977.

[15] Lynne Boddy, John M Wells, Claire Culshaw, and Damian P Donnelly.
Fractal analysis in studies of mycelium in soil. Geoderma, 88(3):301–328,
1999.

[16] Rory G Bolton and Lynne Boddy. Characterization of the spatial aspects
of foraging mycelial cord systems using fractal geometry. Mycological
research, 97(6):762–768, 1993.

[17] Joseph Dahmen. Soft matter: Responsive architectural operations.

Technoetic Arts, 14(1-2):113–125, 2016.

16

[18] Matthew Dale, Julian F Miller, and Susan Stepney. Reservoir computing
as a model for in-materio computing. In Advances in Unconventional
Computing, pages 533–571. Springer, 2017.

[19] Matthew Dale, Julian F Miller, Susan Stepney, and Martin A Trefzer.
A substrate-independent framework to characterize reservoir computers.
Proceedings of the Royal Society A, 475(2226):20180723, 2019.

[20] Ben de Lacy Costello, Rita Toth, Christopher Stone, Andrew
Implementation of glider guns in the
Physical Review E,

Adamatzky, and Larry Bull.
light-sensitive Belousov-Zhabotinsky medium.
79(2):026114, 2009.

[21] Ronald P De Vries, Kim Burgers, Peter JI van de Vondervoort, Jens C
Frisvad, Robert A Samson, and Jaap Visser. A new black aspergillus
species, a. vadensis, is a promising host for homologous and heterologous
protein production. Appl. Environ. Microbiol., 70(7):3954–3959, 2004.

[22] Richard FitzHugh. Impulses and physiological states in theoretical mod-

els of nerve membrane. Biophysical journal, 1(6):445–466, 1961.

[23] M Fricker, L Boddy, and D Bebber. Network organisation of mycelial
fungi. In Biology of the fungal cell, pages 309–330. Springer, 2007.

[24] Mark D Fricker, Luke LM Heaton, Nick S Jones, and Lynne Boddy. The
mycelium as a network. The Fungal Kingdom, pages 335–367, 2017.

[25] J¨org Fromm. Control of phloem unloading by action potentials in mi-

mosa. Physiologia Plantarum, 83(3):529–533, 1991.

[26] J¨org Fromm and Silke Lautner. Electrical signals and their physiological
signiﬁcance in plants. Plant, cell & environment, 30(3):249–257, 2007.

[27] Pier Luigi Gentili, Viktor Horvath, Vladimir K Vanag, and Irving R
Epstein. Belousov-Zhabotinsky “chemical neuron” as a binary and fuzzy
logic processor. IJUC, 8(2):177–192, 2012.

[28] Manuela Giovannetti, Cristiana Sbrana, Luciano Avio, and Patrizia
Strani. Patterns of below-ground plant interconnections established by
means of arbuscular mycorrhizal networks. New Phytologist, 164(1):175–
181, 2004.

17

[29] Jerzy Gorecki and Joanna Natalia Gorecka.

Information processing
with chemical excitations–from instant machines to an artiﬁcial chem-
ical brain. International Journal of Unconventional Computing, 2(4),
2006.

[30] Jerzy Gorecki, Joanna Natalia Gorecka, and Yasuhiro Igarashi. Informa-
tion processing with structured excitable medium. Natural Computing,
8(3):473–492, 2009.

[31] Gerd Gruenert, Konrad Gizynski, Gabi Escuela, Bashar Ibrahim, Jerzy
Gorecki, and Peter Dittrich. Understanding networks of computing
chemical droplet neurons based on information ﬂow. International jour-
nal of neural systems, 25(07):1450032, 2015.

[32] Shan Guo, Ming-Zhu Sun, and Xin Han. Digital comparator in ex-
citable chemical media. International Journal Unconventional Comput-
ing, 2015.

[33] Peter Hammer. Spiral waves in monodomain reaction-diﬀusion model,

2009.

[34] Simon Harding, Jan Koutn´ık, J´urgen Schmidhuber, and Andrew
Adamatzky. Discovering boolean gates in slime mould. In Inspired by
Nature, pages 323–337. Springer, 2018.

[35] Felix Heisel, Juney Lee, Karsten Schlesier, Matthias Rippmann, Nazanin
Saeidi, Alireza Javadian, Adi Reza Nugroho, Tom Van Mele, Philippe
Block, and Dirk E Hebel. Design, cultivation and application of load-
bearing mycelium components. International Journal of Sustainable En-
ergy Development, 6(2), 2018.

[36] D Hitchcock, CA Glasbey, and K Ritz. Image analysis of space-ﬁlling by
networks: Application to a fungal mycelium. Biotechnology Techniques,
10(3):205–210, 1996.

[37] Yasuhiro Igarashi and Jerzy Gorecki. Chemical diodes built with con-

trolled excitable media. IJUC, 7(3):141–158, 2011.

[38] MR Islam, G Tudryn, R Bucinell, L Schadler, and RC Picu. Morphology
and mechanics of fungal mycelium. Scientiﬁc reports, 7(1):1–12, 2017.

18

[39] Tatsuichi Iwamura. Correlations between protoplasmic streaming and
bioelectric potential of a slime mold, Physarum polycephalum. Shokubut-
sugaku Zasshi, 62(735-736):126–131, 1949.

[40] Noburo Kamiya and Shigemi Abe. Bioelectric phenomena in the myx-
omycete plasmodium and their relation to protoplasmic ﬂow. Journal
of Colloid Science, 5(2):149–163, 1950.

[41] U Kishimoto. Rhythmicity in the protoplasmic streaming of a slime
mold, Physarum polycephalum. I. a statistical analysis of the electric
potential rhythm. The Journal of general physiology, 41(6):1205–1222,
1958.

[42] Zoran Konkoli, Stefano Nichele, Matthew Dale, and Susan Stepney.
In Computational

Reservoir computing with computational matter.
Matter, pages 269–293. Springer, 2018.

[43] Mantas Lukoˇseviˇcius and Herbert Jaeger. Reservoir computing ap-
proaches to recurrent neural network training. Computer Science Re-
view, 3(3):127–149, 2009.

[44] Veronica Manzella, Claudio Gaz, Andrea Vitaletti, Elisa Masi, Luisa
Santopolo, Stefano Mancuso, D Salazar, and JJ De Las Heras. Plants
as sensing devices: the PLEASED experience.
In Proceedings of the
11th ACM conference on embedded networked sensor systems, pages 1–
2, 2013.

[45] R Meyer and W Stockem. Studies on microplasmodia of Physarum
polycephalum V: electrical activity of diﬀerent types of microplasmodia
and macroplasmodia. Cell biology international reports, 3(4):321–330,
1979.

[46] JD Mihail, M Obert, JN Bruhn, and SJ Taylor. Fractal geometry of
diﬀuse mycelia and rhizomorphs of armillaria species. Mycological Re-
search, 99(1):81–88, 1995.

[47] Julian F Miller and Keith Downing. Evolution in materio: Looking
beyond the silicon box. In Proceedings 2002 NASA/DoD Conference on
Evolvable Hardware, pages 167–176. IEEE, 2002.

19

[48] Julian F Miller, Simon L Harding, and Gunnar Tufte. Evolution-in-
materio: evolving computation in materials. Evolutionary Intelligence,
7(1):49–67, 2014.

[49] Julian F Miller, Simon J Hickinbotham, and Martyn Amos. In materio
computation using carbon nanotubes. In Computational Matter, pages
33–43. Springer, 2018.

[50] Julian Francis Miller. The alchemy of computation: designing with the

unknown. Natural Computing, 18(3):515–526, 2019.

[51] PV Minorsky. Temperature sensing by plants: a review and hypothesis.

Plant, Cell & Environment, 12(2):119–135, 1989.

[52] Jinichi Nagumo, Suguru Arimoto, and Shuji Yoshizawa. An active
pulse transmission line simulating nerve axon. Proceedings of the IRE,
50(10):2061–2070, 1962.

[53] M Obert, P Pfeifer, and M Sernetz. Microbial growth patterns described

by fractal geometry. Journal of Bacteriology, 172(3):1180–1185, 1990.

[54] S Olsson and BS Hansson. Action potential-like activity found in fungal
mycelia is sensitive to stimulation. Naturwissenschaften, 82(1):30–31,
1995.

[55] Maria Papagianni. Quantiﬁcation of the fractal nature of mycelial aggre-
gation in aspergillus niger submerged cultures. Microbial Cell Factories,
5(1):5, 2006.

[56] Dhananjay B Patankar, Tuan-Chi Liu, and Timothy Oolman. A fractal
model for the characterization of mycelial morphology. Biotechnology
and bioengineering, 42(5):571–578, 1993.

[57] Arkady M Pertsov, Jorge M Davidenko, Remy Salomonsz, William T
Baxter, and Jose Jalife. Spiral waves of excitation underlie reentrant
activity in isolated cardiac muscle. Circulation research, 72(3):631–650,
1993.

[58] Barbara G Pickard. Action potentials in higher plants. The Botanical

Review, 39(2):172–201, 1973.

20

[59] G Roblin. Analysis of the variation potential induced by wounding in

plants. Plant and cell physiology, 26(3):455–461, 1985.

[60] Jack M Rogers and Andrew D McCulloch. A collocation-Galerkin ﬁnite
element model of cardiac action potential propagation. IEEE Transac-
tions on Biomedical Engineering, 41(8):743–757, 1994.

[61] Philip Ross. Your rotten future will be great. The Routledge Companion

to Biology in Art and Architecture, page 252, 2016.

[62] Johannes Schindelin, Ignacio Arganda-Carreras, Erwin Frise, Verena
Kaynig, Mark Longair, Tobias Pietzsch, Stephan Preibisch, Curtis Rue-
den, Stephan Saalfeld, Benjamin Schmid, et al. Fiji: an open-source
platform for biological-image analysis. Nature methods, 9(7):676–682,
2012.

[63] Takao Sibaoka. Rapid plant movements triggered by action potentials.

The botanical magazine= Shokubutsu-gaku-zasshi, 104(1):73–95, 1991.

[64] Jakub Sielewiesiuk and Jerzy G´orecki. Logical functions of a cross junc-
tion of excitable chemical media. The Journal of Physical Chemistry A,
105(35):8189–8195, 2001.

[65] PJ Simons. The role of electricity in plant movements. New Phytologist,

87(1):11–37, 1981.

[66] Cliﬀord L Slayman, W Scott Long, and Dietrich Gradmann. “Action
potentials” in Neurospora crassa, a mycelial fungus. Biochimica et Bio-
physica Acta (BBA) — Biomembranes, 426(4):732–744, 1976.

[67] Oliver Steinbock, Petteri Kettunen, and Kenneth Showalter. Chemical
wave logic gates. The Journal of Physical Chemistry, 100(49):18970–
18975, 1996.

[68] Susan Stepney. Co-designing the computational model and the comput-
ing substrate. In International Conference on Unconventional Compu-
tation and Natural Computation, pages 5–14. Springer, 2019.

[69] William M Stevens, Andrew Adamatzky,

Ishrat Jahan, and Ben
de Lacy Costello. Time-dependent wave selection for information pro-
cessing in excitable media. Physical Review E, 85(6):066129, 2012.

21

[70] James Stovold and Simon O’Keefe. Simulating neurons in reaction-
In International Conference on Information Pro-

diﬀusion chemistry.
cessing in Cells and Tissues, pages 143–149. Springer, 2012.

[71] Ming-Zhu Sun and Xin Zhao. Multi-bit binary decoder based on
The Journal of chemical physics,

Belousov-Zhabotinsky reaction.
138(11):114106, 2013.

[72] Ming-Zhu Sun and Xin Zhao. Crossover structures for logical compu-
tations in excitable chemical medium. International Journal Unconven-
tional Computing, 2015.

[73] Hisako Takigawa-Imamura and Ikuko N Motoike. Dendritic gates for
signal integration with excitability-dependent responsiveness. Neural
Networks, 24(10):1143–1152, 2011.

[74] Rita Toth, Christopher Stone, Ben de Lacy Costello, Andrew
Adamatzky, and Larry Bull. Simple collision-based chemical logic gates
with adaptive computing. Theoretical and Technological Advancements
in Nanotechnology and Molecular Computation: Interdisciplinary Gains:
Interdisciplinary Gains, page 162, 2010.

[75] Kazimierz Trebacz, Halina Dziubinska, and Elzbieta Krol. Electrical
signals in long-distance communication in plants. In Communication in
plants, pages 277–290. Springer, 2006.

[76] Alejandro Vazquez-Otero, Jan Faigl, Natividad Duro, and Raquel
Dormido.
for au-
tonomous mobile robot exploration of unknown environments. IJUC,
10(4):295–316, 2014.

Reaction-diﬀusion based computational model

[77] David Verstraeten, Benjamin Schrauwen, Michiel d’Haene, and Dirk
Stroobandt. An experimental uniﬁcation of reservoir computing meth-
ods. Neural networks, 20(3):391–403, 2007.

[78] Arman Vinck, Charissa de Bekker, Adam Ossin, Robin A Ohm,
Ronald P de Vries, and Han AB W¨osten. Heterogenic expression of
genes encoding secreted proteins at the periphery of Aspergillus niger
colonies. Environmental microbiology, 13(1):216–225, 2011.

22

[79] Alexander G Volkov. Green plants: electrochemical interfaces. Journal

of Electroanalytical Chemistry, 483(1-2):150–156, 2000.

[80] Alexander G Volkov, Justin C Foster, Talitha A Ashby, Ronald K
Walker, Jon A Johnson, and Vladislav S Markin. Mimosa pudica: elec-
trical and mechanical stimulation of plant movements. Plant, cell &
environment, 33(2):163–173, 2010.

[81] Hiroshi Yokoi, Andy Adamatzky, Ben de Lacy Costello, and Chris Mel-
huish. Excitable chemical medium controller for a robotic hand: Closed-
loop experiments.
International Journal of Bifurcation and Chaos,
14(09):3347–3354, 2004.

[82] Guo-Mao Zhang, Ieong Wong, Meng-Ta Chou, and Xin Zhao. Towards
constructing multi-bit binary adder based on Belousov-Zhabotinsky re-
action. The Journal of chemical physics, 136(16):164108, 2012.

[83] Matthias R Zimmermann and Axel Mith¨ofer. Electrical long-distance
signaling in plants. In Long-Distance Systemic Signaling and Commu-
nication in Plants, pages 291–308. Springer, 2013.

23

View publication stats

