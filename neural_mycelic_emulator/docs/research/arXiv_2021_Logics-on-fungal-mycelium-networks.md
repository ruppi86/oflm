See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/357046923

Logics in fungal mycelium networks

Preprint · December 2021

DOI: 10.48550/arXiv.2112.07236

CITATIONS
0

7 authors, including:

Andrew Adamatzky

University of the West of England, Bristol

922 PUBLICATIONS   14,614 CITATIONS   

SEE PROFILE

Alexander E Beasley

University of Hertfordshire

28 PUBLICATIONS   138 CITATIONS   

SEE PROFILE

READS
534

Phil Ayres

Royal Danish Academy

107 PUBLICATIONS   628 CITATIONS   

SEE PROFILE

Martin Tegelaar

Utrecht University

27 PUBLICATIONS   303 CITATIONS   

SEE PROFILE

All content following this page was uploaded by Phil Ayres on 14 March 2022.

The user has requested enhancement of the downloaded file.

1
2
0
2

c
e
D
4
1

]
T
E
.
s
c
[

1
v
6
3
2
7
0
.
2
1
1
2
:
v
i
X
r
a

Logics in fungal mycelium networks

Andrew Adamatzky, Phil Ayres, Alexander E. Beasley, Nic
Roberts, Martin Tegelaar, Michail-Antisthenis
Tsompanas and Han A. B. W¨osten

Abstract. The living mycelium networks are capable of eﬃcient sen-
sorial fusion over very large areas and distributed decision making. The
information processing in the mycelium networks is implemented via
propagation of electrical and chemical signals en pair with morpholog-
ical changes in the mycelium structure. These information processing
mechanisms are manifested in experimental laboratory ﬁndings that
show that the mycelium networks exhibit rich dynamics of neuron-like
spiking behaviour and a wide range of non-linear electrical properties.
On an example of a single real colony of Aspergillus niger, we demon-
strate that the non-linear transformation of electrical signals and trains
of extracellular voltage spikes can be used to implement logical gates
and circuits. The approaches adopted include numerical modelling of
excitation propagation on the mycelium network, representation of the
mycelium network as a resistive and capacitive (RC) network and an
experimental laboratory study on mining logical circuits in mycelium
bound composites.

Mathematics Subject Classiﬁcation (2010). Primary 68Q07; Sec-
ondary 92B25.
Keywords. Fungi, Boolean circuits, Unconventional computing.

1. Introduction

The fungi is among the largest, most widely distributed group of living organ-
isms [15]. Fungi can grow as individual cells or in a interconnected network
of hyphae. These hyphae that grow at their tips and branch sub-apically can
be compartmentalized by porous septa that can be either in a closed or open
state. In the open state, cytoplasm can stream from one compartment to the
other or even from hypha to hypha. In the closed state, the compartments can

To be published in special issue of Logica Universalis — “Logic, Spatial Algorithms and
Visual Reasoning”, edited by Andrew Schumann and Jerzy Kr´ol, 2022

 
 
 
 
 
 
2 Adamatzky, Ayres, Beasley, Roberts, Tegelaar, Tsompanas and W¨osten

Figure 1. Exemplar spikes of extracellular electrical poten-
tial propagating in fungal mycelium.

act as individual entities although there is still interaction with neighboring
cells [34]. Mycelia can be thousands of years old and can cover large surface
areas. The largest known mycelium, belonging to the Armillaria genus covers
an area of 965 hectares [30]. The fungi show a high degree of adaptability
to environmental conditions. They are demonstrated to eﬃciently explore
conﬁned spaces with their hyphae [17, 19, 20, 22, 21]. In fact, they even
form diﬀerent types of hyphae within the mycelium [31]. Optimisation of the
mycelial network [14] is quite similar to that of slime mould P. polycephalum,
e.g. in terms of proximity graphs [1] and transport networks [2]. Taking into
account the ubiquity, range of length scales and spatio-temporal dynamics
exhibited by the fungi, they represent a promising research target within the
context of unconventional computing.

The motivation of this paper is to contribute to uncovering basic mech-
anisms of decision making in the fungal network in terms of Boolean gates
and circuits. Mechanisms of computation discovered in mycelium networks
could be utilised in future designs of electrical analog computing circuits [32]
and to design and program computing schemes embedded into living fungal
architectures [6].

A ﬁrst step toward discovering the computing potential of fungal net-
works would be to estimate frequencies of logical gates and simple circuits
realisable in a single fungal colony. We implement this idea using three tech-
niques: numerical modelling of spiking events on fungal colony, modelling the
colony as a resistive and capacitive (RC) network and mining logical circuits.
In the ﬁrst technique, logical gates are calculated based on the temporal
co-occurrence of spikes emerging as responses to diﬀerent input strings [10].
Why do we consider spikes of electrical potential? Because these spikes are
manifestations of the calcium waves that travels along mycelium networks and
implement information between distant parts of the mycelium network and,

Potential, mV−6.0−5.5−5.0−4.5t, sec75,00080,00085,000Logics in fungal mycelium networks

3

possibly, participate in the information processing. First discovery of the elec-
trical potential spikes has been done via intra-cellular recording of mycelium
of Neurospora crassa [29]. Further conﬁrmed in intra-cellular recordings of
action potential in hypha of Pleurotus ostreatus and Armillaria bulbosa [25],
and observed in the extra-cellular recordings of fruit bodies resulting from
substrates colonized by the mycelium of Pleurotus ostreatus [5] (Fig. 1).

Spikes of fungal electrical potential are notoriously slow, with a min-
imum spike duration of 2 mins and maximum up to an hour. Thus the
techniques of spikes based logical circuits might not be suitable for practi-
cal applications. Two other techniques exploit principles of electrical analog
computing [11, 27]. True and False values are represented by above thresh-
old and below threshold voltages. Due to the non-linearity of the conductive
substrate along electrical current pathways between input and output elec-
trodes, the input voltages are transformed and thus logical mappings are
implemented.

Detailed descriptions of these techniques can be found in [10, 11, 27].
Here we provide an updated overview of the approaches and provide an in-
tegrative analysis of the results.

2. Methods

2.1. Colony imaging
We have grown Aspergillus niger fungus strain AR9#2 [33]. This strain ex-
presses Green Fluorescent Protein (GFP) from the glucoamylase (glaA) pro-
moter. A ﬂuorescence of GFP was localised in micro-colonies using a DMI
6000 CS AFC confocal microscope (Leica, Mannheim, Germany). Micro-
colonies were imaged at 20× magniﬁcation (HC PL FLUOTAR L 20 × 0.40
DRY). Z-stacks of imaged micro-colonies were made using 100 slices with a
slice thickness of 8.35 µm. 3D projections were made with Fiji [28].

2.2. Numerical modelling
We used a selected image of the colony, from the middle of the z-stack, as a
conductive template. The image of the fungal colony (Fig. 2) was projected
onto a 1000 × 960 nodes grid C.

We simulated electrical activity of the colony with FitzHugh-Nagumo

(FHN) equations [16, 24, 26]:

∂v
∂t
∂v
∂t

= c1u(u − a)(1 − u) − c2uv + I + Du∇2

= b(u − v),

(2.1)

(2.2)

where u is a value of a trans-membrane potential, v a variable accountable for
a total slow ionic current, or a recovery variable responsible for a slow negative
feedback, I is a value of an external stimulation current. The current through
intra-cellular spaces is approximated by Du∇2, where Du is a conductance.
We integrated the system using the Euler method with the ﬁve-node Laplace

4 Adamatzky, Ayres, Beasley, Roberts, Tegelaar, Tsompanas and W¨osten

Figure 2. Image of the fungal colony, 1000 × 960 pixels
used as a template conductive for FHN. A conﬁguration of
electrodes is superimposed on the image.

operator, a time step ∆t = 0.015 and a grid point spacing ∆x = 2, while other
parameters were Du = 1, a = 0.13, b = 0.013, c1 = 0.26, c2 = 0.095. To show
dynamics of excitation in the network, we simulated electrodes by calculating
a potential pt

x at an electrode location x as px = (cid:80)

y:|x−y|<2(ux − vx).

2.3. Resistive and capacitive network modelling

The z-stacks of the colony were converted to a 3D graph (Fig. 3). The 3D
graph was converted to a resistive and capacitive (RC) network, by assigning
to each edge a function of a resistor or a capacitor at random. The magni-
tudes of the resistance and capacitance were functions of the length of the
edges/connections. We have chosen resistances in the order of kOhms and val-
ues of capacitance in the order of pF. We selected ground nodes and sources
(positive voltage nodes) at random. The trials were run on 1000 networks
(with the same architecture but diﬀerent values of resistance and capaci-
tance). During SPICE modelling we used two voltage pulses of 60 mV on
randomly chosen positive nodes.

We modelled the fungal colony in serial RC networks (resistors and
capacitors are connected in series) and parallel RC networks (resistors and
capacitors are connected in parallel), see basics in [23]. The output voltages
have been binarised with the threshold θ: V > θ symbolises logical True
otherwise False.

Logics in fungal mycelium networks

5

Figure 3. A graph representation of a single fungal colony.
Each frame shows the graph after a 36◦ rotation around the
z-axis.

2.4. Experimental laboratory mining of circuits

A hemp shavings substrate was colonised by the mycelium of the grey oyster
fungi, P. ostreatus (Ann Miller’s Speciality Mushrooms Ltd, UK). Hardware
was developed that was capable of sending sequences of 4 bit strings to a
mycelium substrate. The strings were encoded as step voltage inputs where
-5 V denoted a logical 0 and 5 V a logical 1. The hardware was based around
an Arduino Mega 2560 (Elegoo, China) and a series of programmable sig-
nal generators, AD9833 (Analog, USA). The 4 input electrodes were 1 mm
diameter platinum rods inserted to a depth of 50 mm in the substrate in
a straight line with a separation of 20 mm. Data acquisition (DAQ) probes
were placed in a parallel line 50 mm away separated by 10 mm. The electron
sink and source was placed 50 mm on from DAQ probes. There were 7 DAQ
diﬀerential inputs from the mycelium substrate to a Pico 24 (Pico Technol-
ogy, UK) analogue-to-digital converter (ADC), the 8th channel was used to

6 Adamatzky, Ayres, Beasley, Roberts, Tegelaar, Tsompanas and W¨osten

Figure 4. Examples of electrical potential spikes recorded
on the electrode 7. The data represent responses to input
impulse strings, entered via electrodes 5 and 15, inputs (01),
black dashed line, (10), red dotted line, (11), solid green line.
The locations of electrodes are shown in Fig. 2.

pass a pulse to the ADC on every input state change. There were a total
of 14 repeats. A sequence of 4 bit strings counting up from binary 0000 to
1111, with a state change every hour, were passed into the substrate. Boolean
strings were extracted from the data, where a logic ‘1’ was noted for a channel
if it had a peak outside the threshold band for a particular state, else a value
of ‘0’ was recorded. The polarity of the peak was not considered. The sum of
products (SOP) Boolean functions were calculated for each output channel.
For each repeat there were 7 channels and 32 thresholds giving total of 3136
individual truth tables.

3. Results

3.1. Spikes derived logical gates
We adopt the encoding procedure developed by us in [9]. We select two elec-
trodes as inputs x and y. We represent logical True, or ‘1’ as an impulse
injected in the network via input electrode. For example, if x = 1 then the
site corresponding to x is excited, if x = 0 the site is not excited.

Each spike represents logical True. The spikes occurring within less
than 2 · 102 iterations are seen as occuring simultaneously.We assume that
spikes are separated if their occurrences lie more than 103 iterations apart.
An example is shown in Fig. 4.

Numbers of Boolean gates detected on the electrodes for selected pair
of input electrodes are shown in Tab. 1. The most frequent gates are select x
and select y gates and occur similar frequencies. The and-not gates xy and
xy less common than select gates. The gates xy and x + y are detected with

SySySySxx+yxyxyx ⊕yPotential, units−50050Time, iterations30,00040,00050,00060,000Logics in fungal mycelium networks

7

Figure 5. Recording of electrical potential from all elec-
trodes in responses to inputs in response to inputs (01),
black line, (10), red line, (11), green line, injected as spikes
via electrodes Ex = 5 and Ey = 15.

8 Adamatzky, Ayres, Beasley, Roberts, Tegelaar, Tsompanas and W¨osten

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

x + y Sy x ⊕ y Sx xy xy xy Total
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

Table 1. Numbers of Boolean gates detected for selected
pairs of input electrodes 3 and 13.

nearly the same frequency with gate x + y being slightly more common. The
most rare gate is a logical exclusion x ⊕ y.

The overall distribution of the ratio of gates discovered is shown in
Fig. 6. The distribution demonstrates frequencies of discoveries of the four-
input-one-output logical gates and could be used for characterisation of a
computational power of the fungal substrates. This is accompanied by distri-
butions of gates discovered in experimental laboratory reservoir computing
with slime mould Physarum polycephalum [18], succulent plant [8] and nu-
merical modelling experiments on computing with protein verotoxin [3], actin
bundles network [9], and actin monomer [4]. The distributions of gates dis-
covered in natural systems are alike to each other in the hierarchies of the
gates frequencies. Namely, gates selecting one of the inputs are most com-
mon, they are followed by or gate, then by not-and an and-not gates. The
gate and is typically less frequent. The gate xor is a totally rare.

3.2. Resistive and capacitive (RC) networks
There are sixteen types of two-input-one-output Boolean gates. The ‘active’
gates, i.e. those where zero inputs evoke a non-zero response could not be
realised in the passive electrical model of a fungal colony. They realisable
gates are and, or, and-not (x and not y and not x and y), select
(select x and select y) and xor. The exclusion gates xor have not been
detected in any of the RC models of the fungal colony.

Logics in fungal mycelium networks

9

Figure 6. Comparative ratios of Boolean gates discovered
in mycelium network analysed in present paper, black disc
and solid line; slime mould Physarum polycephalum [18],
black circle and dotted line; succulent plant [8], red snowﬂake
and dashed line; single molecule of protein verotoxin [3],
light blue ‘+’ and dash-dot line; actin bundles network [9],
green triangle pointing right and dash-dot-dot line; actin
monomer [4], magenta triangle pointing left and dashed line.
Area of xor gate is magniﬁed in the insert. Lines are to guide
eye only.

In the model of serial RC networks, we found gates and, select and
and-not; no or gates have been found. The number n of the gates discovered
decreases by a power law with increase of θ: nand-not = 72 · x−0.98, nselect =
2203 · x−0.48, nand = 0.02 · x−1.6. Frequency of and gate oscillates, as shown
in zoom insert in Fig. 7a, more likely due to its insigniﬁcant presence in the
samples. The oscillations reach near zero base when θ exceeds 0.001.

In the model of parallel RC networks we found only gates and, select
and or. The number of or gates decreases quadratically and becomes nil
when θ > 0.03. The number of and gates increases near linearly, nand =
−1.72 · 106 + 2.25 · 108 · x, with increase of θ. The number of select gates
reaches its maximum at θ = 0.023, and then starts to decreases with the
further increase of θ: nselect = 9.61 · 106 + 1.21 · 109 · x − 2.7 · x2.

3.3. Experimental laboratory mining
We have discovered total of 3136 4-inputs-1-output Boolean functions. 470
unique functions are detailed in [27]. Figure 8 shows the Boolean function
distribution. The two peak values were logical False, n = 238, and logical
True, n = 237. The highest occurring non-trivial gate was A + B + C + D,
n = 145. The top nine occurring non-trivial Boolean functions are listed in
table 3.3. The only single gate functions found were for nand (A+B +C +D),
n = 145, or (A + B + C + D), n = 46, and and (ABCD), n = 8.

x+ySxSyx⊕yxyxyxy10Adamatzky, Ayres, Beasley, Roberts, Tegelaar, Tsompanas and W¨osten

(a)

(b)

Figure 7. Occurrences of the gates from the groups and,
black, or, green, and-not, red, and select, blue, for θ ∈
[0.0001, 0.05], with θ increment 0.0001, in (a) fungal colony
modelled with serial RC networks, (b) fungal colony mod-
elled with parallel RC networks.

4. Discussion

In numerical modelling and experimental laboratory studies we demonstrated
that a wide range of Boolean circuits are implemented in a single fungal colony
and a substrate colonised by mycelium. In the models where logical functions
are implemented with spikes (travelling excitation waves), the xor gate is
the rarest, or and and are more common and and-not are most common
(select is a rather trivial gate). The frequency distribution of the gate is
generally in line with the distributions of gates discovered in other living
substrates. In the resistive and capacitive (RC) network model of a single

ANDSELECTAND-NOTNumber of gates05×10410×10415×104θ00.010.020.030.040.05010002000300000.0020.0040.0060.00845,00050,00055,00000.001ANDORSELECTNumber of gates05×10610×10615×10620×10625×106θ00.010.020.030.040.05Logics in fungal mycelium networks

11

Figure 8. Counts of realised Boolean functions discovered
in laboratory experiments. Horizontal axis is a decimal repre-
sentation of functions. Vertical axis is a number of functions
discovered in experiments.

Count
145
83

81
59
55
53
47
46
46

F1
F2

F3
F4
F5
F6
F7
F8
F9

Boolean function
A + B + C + D (nand)
AB + AC + AD + AB + BC + BD + AC + BC +
CD + AD + BD + CD
ACD + ABC + ABC + ABD
AC + AD + AC + CD + AD + BD + CD
AB + CD + AD
ABCD
BD + CD + AD + BCD
ABCD
A + B + C + D (or)

Table 2. Top nine highest occurring Boolean functions dis-
covered in experimental laboratory mining with a substrate
colonised by living mycelium.

fungal colony, we discovered and-not gate in serial networks, and or and
and in parallel networks. This relatively poor representation of logical func-
tions might be due to the absence of capacitive elements. In contrast to the
RC model, sets of logical circuits discovered in laboratory experiments with
living mycelium are impressively large [27]. This is because living mycelium
networks are active, i.e. they generate spikes of electrical potential [5] and
spikes of resistance [7], capacitive [12] and memfractive properties [13].

12Adamatzky, Ayres, Beasley, Roberts, Tegelaar, Tsompanas and W¨osten

Acknowledgement
This research has received funding from the European Union’s Horizon 2020
research and innovation programme FET OPEN “Challenging current think-
ing” under grant agreement No 858132 / project Fungal Architectures.
(www.fungar.eu).

References

[1] Andrew Adamatzky. Developing proximity graphs by Physarum polycephalum:
Does the plasmodium follow the toussaint hierarchy? Parallel Processing Let-
ters, 19(01):105–127, 2009.

[2] Andrew Adamatzky, editor. Bioevaluation of World Transport Networks. World

Scientiﬁc, 2012.

[3] Andrew Adamatzky. Computing in verotoxin. ChemPhysChem, 18(13):1822–

1830, 2017.

[4] Andrew Adamatzky. Logical gates in actin monomer. Scientiﬁc reports, 7(1):1–

14, 2017.

[5] Andrew Adamatzky. On spiking behaviour of oyster fungi pleurotus djamor.

Scientiﬁc reports, 8(1):1–7, 2018.

[6] Andrew Adamatzky, Phil Ayres, Gianluca Belotti, and Han W¨osten. Fungal
architecture position paper. International Journal of Unconventional Comput-
ing, 14, 2019.

[7] Andrew Adamatzky and Antoni Gandia. On electrical spiking of ganoderma

resinaceum. bioRxiv, 2021.

[8] Andrew Adamatzky, Simon Harding, Victor Erokhin, Richard Mayne, Nina
Gizzie, Frantisek Baluˇska, Stefano Mancuso, and Georgios Ch Sirakoulis. Com-
puters from plants we never made: Speculations. In Inspired by nature, pages
357–387. Springer, 2018.

[9] Andrew Adamatzky, Florian Huber, and J¨org Schnauß. Computing on actin

bundles network. Scientiﬁc reports, 9(1):1–10, 2019.

[10] Andrew Adamatzky, Martin Tegelaar, Han AB Wosten, Anna L Powell,
Alexander E Beasley, and Richard Mayne. On boolean gates in fungal colony.
Biosystems, 193:104138, 2020.

[11] Alexander E Beasley, Phil Ayres, Martin Tegelaar, Michail-Antisthenis Tsom-
panas, and Andrew Adamatzky. On electrical gates on fungal colony. Biosys-
tems, page 104507, 2021.

[12] Alexander E Beasley, Anna L Powell, and Andrew Adamatzky. Capacitive

storage in mycelium substrate. arXiv preprint arXiv:2003.07816, 2020.

[13] Alexander E Beasley, Anna L Powell, and Andrew Adamatzky. Fungal photo-

sensors. arXiv preprint arXiv:2003.07825, 2020.

[14] Lynne Boddy, Juliet Hynes, Daniel P Bebber, and Mark D Fricker. Sapro-
trophic cord systems: dispersal mechanisms in space and time. Mycoscience,
50(1):9–19, 2009.

[15] Michael John Carlile, Sarah C Watkinson, and Graham W Gooday. The fungi.

Gulf Professional Publishing, 2001.

Logics in fungal mycelium networks

13

[16] Richard FitzHugh. Impulses and physiological states in theoretical models of

nerve membrane. Biophysical journal, 1(6):445–466, 1961.

[17] Kristi L Hanson, Dan V Nicolau Jr, Luisa Filipponi, Lisen Wang, Abraham P
Lee, and Dan V Nicolau. Fungi use eﬃcient algorithms for the exploration of
microﬂuidic networks. small, 2(10):1212–1220, 2006.

[18] Simon Harding, Jan Koutn´ık, J´urgen Schmidhuber, and Andrew Adamatzky.
Discovering boolean gates in slime mould. In Inspired by Nature, pages 323–
337. Springer, 2018.

[19] Marie Held, Clive Edwards, and Dan V Nicolau. Examining the behaviour
of fungal cells in microconﬁned mazelike structures. In Imaging, Manipula-
tion, and Analysis of Biomolecules, Cells, and Tissues VI, volume 6859, page
68590U. International Society for Optics and Photonics, 2008.

[20] Marie Held, Clive Edwards, and Dan V Nicolau. Fungal intelligence; or on the
behaviour of microorganisms in conﬁned micro-environments. In Journal of
Physics: Conference Series, volume 178, page 012005. IOP Publishing, 2009.

[21] Marie Held, Clive Edwards, and Dan V Nicolau. Probing the growth dynamics
of neurospora crassa with microﬂuidic structures. Fungal biology, 115(6):493–
505, 2011.

[22] Marie Held, Abraham P Lee, Clive Edwards, and Dan V Nicolau. Microﬂuidics
structures for probing the dynamic behaviour of ﬁlamentous fungi. Microelec-
tronic Engineering, 87(5-8):786–789, 2010.

[23] Paul Horowitz and Winﬁeld Hill. The art of electronics. Cambridge Univ. Press,

1989.

[24] Jinichi Nagumo, Suguru Arimoto, and Shuji Yoshizawa. An active pulse trans-
mission line simulating nerve axon. Proceedings of the IRE, 50(10):2061–2070,
1962.

[25] S Olsson and BS Hansson. Action potential-like activity found in fungal mycelia

is sensitive to stimulation. Naturwissenschaften, 82(1):30–31, 1995.

[26] Arkady M Pertsov, Jorge M Davidenko, Remy Salomonsz, William T Baxter,
and Jose Jalife. Spiral waves of excitation underlie reentrant activity in isolated
cardiac muscle. Circulation research, 72(3):631–650, 1993.

[27] Nic Roberts and Andrew Adamatzky. Mining logical circuits in fungi. arXiv

preprint arXiv:2108.05336, 2021.

[28] Johannes Schindelin, Ignacio Arganda-Carreras, Erwin Frise, Verena Kaynig,
Mark Longair, Tobias Pietzsch, Stephan Preibisch, Curtis Rueden, Stephan
Saalfeld, Benjamin Schmid, et al. Fiji: an open-source platform for biological-
image analysis. Nature methods, 9(7):676–682, 2012.

[29] Cliﬀord L Slayman, W Scott Long, and Dietrich Gradmann. “Action poten-
tials” in Neurospora crassa, a mycelial fungus. Biochimica et Biophysica Acta
(BBA) — Biomembranes, 426(4):732–744, 1976.

[30] Myron L Smith, Johann N Bruhn, and James B Anderson. The fungus
Armillaria bulbosa is among the largest and oldest living organisms. Nature,
356(6368):428, 1992.

[31] Martin Tegelaar, Robert-Jan Bleichrodt, Benjamin Nitsche, Arthur FJ Ram,
and Han AB W¨osten. Subpopulations of hyphae secrete proteins or resist heat
stress in aspergillus oryzae colonies. Environmental microbiology, 22(1):447–
455, 2020.

14Adamatzky, Ayres, Beasley, Roberts, Tegelaar, Tsompanas and W¨osten

[32] Bernd Ulmann. Analog and hybrid computer programming. De Gruyter Olden-

bourg, 2020.

[33] Arman Vinck, Charissa de Bekker, Adam Ossin, Robin A Ohm, Ronald P
de Vries, and Han AB W¨osten. Heterogenic expression of genes encoding se-
creted proteins at the periphery of Aspergillus niger colonies. Environmental
microbiology, 13(1):216–225, 2011.

[34] Han AB W¨osten, G Jerre van Veluw, C de Bekker, and Pauline Krijgsheld.
Heterogeneity in the mycelium: implications for the use of fungi as cell factories.
Biotechnology letters, 35(8):1155–1164, 2013.

Andrew Adamatzky
Unconventional Computing Laboratory, UWE, Bristol, UK
e-mail: andrew.adamatzky@uwe.ac.uk

Phil Ayres
Centre for Information Technology and Architecture (CITA), Royal Danish Acad-
emy, Copenhagen, Denmark
e-mail: phil.ayres@kglakademi.dk

Alexander E. Beasley
Centre for Engineering Research, University of Hertfordshire, UK
e-mail: andrew.adamatzky@uwe.ac.uk

Nic Roberts
Unconventional Computing Laboratory, UWE, Bristol, UK
e-mail: andrew.adamatzky@uwe.ac.uk

Martin Tegelaar
Microbiology, Department of Biology, University of Utrecht, Utrecht, The Nether-
lands
e-mail: m.tegelaar@uu.nl

Michail-Antisthenis Tsompanas
Unconventional Computing Laboratory, UWE, Bristol, UK
e-mail: andrew.adamatzky@uwe.ac.uk

Han A. B. W¨osten
Microbiology, Department of Biology, University of Utrecht, Utrecht, The Nether-
lands
e-mail: h.a.b.wosten@uu.nl

View publication stats

