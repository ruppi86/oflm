1
2
0
2

g
u
A
1
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
3
5
0
.
8
0
1
2
:
v
i
X
r
a

Mining logical circuits in fungi

Nic Roberts, Andrew Adamatzky

Unconventional Computing Laboratory, UWE, Bristol, UK

Abstract

Living substrates are capable for nontrivial mappings of electrical signals
due to the substrate nonlinear electrical characteristics. This property can
be used to realise Boolean functions.
Input logical values are represented
by amplitude or frequency of electrical stimuli. Output logical values are
decoded from electrical responses of living substrates. We demonstrate how
logical circuits can be implemented in mycelium bound composites. The
mycelium bound composites (fungal materials) are getting growing recog-
nition as building, packaging, decoration and clothing materials. Presently
the fungal materials are passive. To make the fungal materials adaptive,
i.e.
sensing and computing, we should embed logical circuits into them.
We demonstrate experimental laboratory prototypes of many-input Boolean
functions implemented in fungal materials from oyster fungi P. ostreatus.
We characterise complexity of the functions discovered via complexity of the
space-time conﬁgurations of one-dimensional cellular automata governed by
the functions. We show that the mycelium bound composites can implement
representative functions from all classes of cellular automata complexity in-
cluding the computationally universal. The results presented will make an
impact in the ﬁeld of unconventional computing, experimental demonstration
of purposeful computing with fungi, and in the ﬁeld of intelligent materials,
as the prototypes of computing mycelium bound composites.

Keywords: mycelium network, Boolean gates, unconventional computing

1. Introduction

The fungi are one of the largest, the oldest, most adaptive and widely dis-
tributed group of organisms [9]. Smallest fungi are single cells. The largest
mycelium spreads in hectares [46]. When growing in a bulk medium of wood

Preprint submitted to BioSystems

August 12, 2021

 
 
 
 
 
 
or plant shavings fungi bind the medium in a solid monolith with outstanding
mechanical properties. The mycelium bound composites are seen as future
environmentally sustainable growing biomaterials [26, 25, 10, 1]. They are
already used in acoustic [40, 15, 41] and thermal [52, 51, 19, 14, 49, 8] in-
sulation panels and cladding, materials for packaging [21, 45, 36] and wear-
ables [2, 44, 26, 4, 24]. The currently used fungal materials are passive and
inert because the fungi in the composites are dead and treated to prevent
decay. To make the fungal materials adaptive and intelligent we must either
(1) leave part of the fungal materials alive, or (2) dope the materials with
functional nanoparticles and polymers. In the present paper we explore the
ﬁrst option of sensing and computing with living mycelium.

Fungal colonies are characterised by rich typology of mycelium networks [20,

18, 16, 17, 23] in some cases similar to fractal structures [37, 39, 7, 31, 6, 38].
Rich morphological features might imply rich computational abilities and
thus worth to analyse from a realising Boolean functions point of view. To
implement logical functions we adopted a theoretical approach developed in
[3, 43]. The technique is based on selecting a pair of input sites, apply-
ing all possible combinations of inputs, where logical values are represented
by electrical characteristics of input signals, to the sites and recording out-
puts, represented by electrical responses of the substrate, on a set of the
selected output sites. The approach belong to the family of reservoir com-
puting [48, 28, 11, 27, 12] and in materio computing [32, 33, 47, 34, 35]
techniques of analysing computational properties of physical and biological
substrates.

The paper is structured as follows. First, the experimental setup will
be described, then the procedure for data gathering and analysis will be
outlined.

2. Methods

A hemp shavings substrate was colonised by the mycelium of the grey
oyster fungi, P. ostreatus (Ann Miller’s Speciality Mushrooms Ltd, UK).
Recordings were carried out in a stable indoor environment with the temper-
ature remaining stable at 22 ± 0.5°and relative humidity of air 40 ± 5%. The
humidity of the substrate colonised by fungi was kept at c. 70-80%.

Hardware was developed that was capable of sending sequences of 4 bit
strings to a mycelium substrate. The strings were encoded as step voltage
inputs where -5 V denoted a logical 0 and 5 V a logical 1. The hardware

2

Figure 1: Left: Schematic of the mycelium communications system; PC — laptop for
generating sequences; CU – control unit, dashed section is a breakdown of a single channel;
ADC — analogue to digital converter. Right: experimental set up.

was based around an Arduino Mega 2560 (Elegoo, China) and a series of
programmable signal generators, AD9833 (Analog, USA). The 4 input elec-
trodes were 1 mm diameter platinum rods inserted to a depth of 50 mm in
the substrate in a straight line with a separation of 20 mm. Data acquisition
(DAQ) probes were placed in a parallel line 50 mm away separated by 10 mm.
The electron sink and source was placed 50 mm on from DAQ probes. There
were 7 DAQ diﬀerential inputs from the mycelium substrate to a Pico 24
(Pico Technology, UK) analogue-to-digital converter (ADC), the 8th channel
was used to pass a pulse to the ADC on every input state change, see Fig. 1
for a schematic of the apparatus. The substrate and probes were placed in
a semi-sealed container. After each experimental repeat the substrate was
sprayed with water, left for an hour and then the next repeat was conducted.
There were a total of 14 repeats.

A sequence of 4 bit strings counting up from binary 0000 to 1111, with a
state change every hour, were passed into the substrate, see Fig. 2 for timing
details. In all 14 repeats of the experiment were done on the same substrate
to capture changes in structure of the growing mycelium. Samples from 7
channels were taken at 1 Hz over the whole duration of a given experimental

3

Figure 2: Timing diagram and associated Boolean strings for four inputs into the mycelium
substrate, time step is one hour.

run. Peaks for each channel were located for a set of 32 thresholds, from
20 mV to 175 mV with step 5 mV, for each input state, 0000 to 1111.

Boolean strings were extracted from the data, where a logic ‘1’ was noted
for a channel if it had a peak outside the threshold band for a particular state
else, a value of ‘0’ was recorded, the polarity of the peak was not considered.
The strings for each experimental repeat were stored in their respective
Boolean table. To extract state graphs, a state/node was deﬁned as the string
of output values from each channel at each input state, transitions/edges were
deﬁned as a change in input state. This led to a total of 448 state graphs. The
sum of products (SOP) Boolean functions were calculated for each output
channel. For each repeat there were 7 channels and 32 thresholds giving total
of 3136 individual truth tables.

See Fig. 3 for SOP extraction. If a peak is found in Fig. 3a during an
input state then this is considered a logical 1, highlighted in yellow in table
Fig. 3b are the thresholded values for channel 5, the resulting truth table is
then reduced to a sum products shown below the table.

3. Results

We have discovered total of 3136 4-inputs-1-output Boolean functions.
470 unique functions are presented in Supplementary Materials. Figure 4
shows the Boolean function distribution. The two peak values were logical
False, n = 238, and logical True, n = 237. The highest occurring non-
trivial gate was A + B + C + D, n = 145. The top 16 occurring non-trivial

4

(a)

(b)

Figure 3: Workﬂow example. (a) The measurements taken by channel 5 of the DAQ in
blue, the synchronisation signal is shown red which marks the state change, threshold
band shown in green, peaks outside this band are highlighted with ‘x’ marker. (b) The
truth and the function extracted.

5

Figure 4: Counts of realised Boolean functions discovered in laboratory experiments. Hor-
izontal axis is a decimal representation of functions. Vertical axis is a number of functions
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
40
37
37
37
32
29
28

F1
F2
F3
F4
F5
F6
F7
F8
F9
F10
F11
F12
F13
F14
F15
F16

Table 1: Top 16 highest occurring Boolean functions.

Boolean function
A + B + C + D (nand)
AB +AC +AD+AB +BC +BD+AC +BC +CD+AD+BD+CD
ACD + ABC + ABC + ABD
AC + AD + AC + CD + AD + BD + CD
AB + CD + AD
ABCD
BD + CD + AD + BCD
ABCD
A + B + C + D (or)
AB + AD + AB + BD + AD + BD + CD
ABCD
AD + AB + BC + AD + BCD
AB + AC + AD + AD + BD + CDABC + BCD
AD + AB + BD + AC + CD + AD + ABC + BCD
C + AB + AD + AB + BDAD + BD
AB + AC + BD + BCD + ABC

6

(a) F2

(b) F3

(c) F4

(d) F5

(e) F7

(f) F10

(g) F11

(h) F12

(i) F13

(j) F14

(k) F15

(l) F16

Figure 5: Space-time conﬁgurations of one-dimensional cellular automata governed by
functions from Tab. 1. An automaton has 500 cells and evolves for 500 iterations. Initial
conﬁgurations has a random uniform distribution of cells in state ‘1’ where each cell takes
the state ‘1’ with a probability 1
2 .

Boolean functions are listed in Tab. 1. The only single gate functions found
were for nand (A + B + C + D), n = 145, or (A + B + C + D), n = 46, and
and (ABCD), n = 8.

Let us discuss complexity of the functions discovered (Tab. 1) via com-
plexity of the space-time conﬁgurations of one-dimensional cellular automata
governed by the functions. We consider an array Z of ﬁnite state machines,
called cells, where every cell takes states ‘0’ or ‘1’ and updates its state de-

7

Figure 6: Frequency of functions from Tab. 1 versus LZ complexity, measured via com-
pressibility of the space-time conﬁgurations of cellular automata governed by the functions.
Functions F1, F6, F8 and F9 are not displayed because their LZ is near zero.

pending on the states of its four immediate neighbours. All cells update
their states by the same rule and in discrete time. For example, a cell with
index i, xi ∈ Z, updates its state at time t as a function of states of its four
i+1, xt
neighbours: xt+1 = f (xt
i+2). To map functions from Tab. 1
to the rules governing the cellular automata we assume that A corresponds
to xt
i+1. For example, a cell xi of cel-
lular automaton governed by the function F5 (Tab. 1) updates its state as
xt+1 = xi−2xi−1 + xi+1xi+2 + xi−2xi+2.

i+1 and D to xt

i−2, B to xt

i−1, C to xt

i−2, xt

i−1, xt

Automaton governed by F1, F6, F8 fall into absorbing state where all cells
are in state ‘0’. The automaton governed by rule F9 falls into the state where
all cells are in state ‘1’. Space-time conﬁgurations, random initial conditions
and absorbing boundaries, of automata governed by other rules are shown in
Fig. 5. We characterise a complexity of the space-time patterns via Lempel-
Ziv complexity (compressibility) LZ. The LZ complexity is evaluated by a
size of concentration proﬁles saved as PNG ﬁles of the conﬁgurations. This
is suﬃcient because the ’deﬂation’ algorithm used in PNG lossless compres-
sion [42, 22, 13] is a variation of the classical Lempel–Ziv 1977 algorithm [53].
The frequency of the functions occurrence in the experimental circuit mining
versus LZ complexity of the functions is shown in Fig. 6. We can see that
there is no correlation between how often a function can be found and how
complexity the function is. Thus, e.g. the function F13 (Tab. 1) generates

8

F3F4F5F7F13F10F16F15F14F12F11Compressibility, kB010203040Count304050607080most complex space-time conﬁguration (Fig. 5i) yet it is in the mid-range
of the frequency of experimental occurrence. The less complex functions F5,
F7, F12, F15 span the interval [29,55] counts of occurrences in experimental
laboratory mining.

Let us consider positions of the functions Tab. 1 in the Wolfram classi-
ﬁcation [50] of cellular automaton behaviour. Functions F1, F6, F8, F9 and
F11 belong to the class I, the class of automata exhibiting a dull dynamics
and evolving to a stable state where all cells are in the same state. Functions
F2, F7, F12, F14, F15 belong to the class II: the automata fall into global cells
do not update their state or update them cyclically from ‘0’ to ‘1’. Functions
F4, F10 and F13 belong to class III: the space-time dynamics is characterised
by quasi-random behaviour and diﬃcult predictability of the successions of
the global states. These functions generate the most complex, as evaluated
by LZ measure, space-time conﬁgurations. Function F2 shows an interesting
example of the function belonging to classes II and III. Two functions F3 and
F16 belong to class IV: the space-time dynamics of automata show gliders
(compact patterns translating in space) with non-trivial interactions between
the gliders. The automata governed by rules F3 and F16 are computationally
universal, because it is possible to implement an arbitrary logical circuit via
collisions between the gliders, see e.g.

[30, 29].

4. Discussion

Mycelium bound composites transform electrical signals in a non-linear
manner due to mem-fractive and capacitive properties of the fungal tissue [5].
Whilst exact biophysical mechanisms of the signal transformation by the
mycelium remain unknown we can explore the non-linear properties of this
In experimental laboratory
living substrate to implement logical circuits.
studies we demonstrated that mycelium bound composites implement a wide
range of Boolean circuits. Analyses of the functions extracted in terms of
space-time dynamics of cellular automata helped us to order the functions in
several classes of complexity and pinpoint the functions supporting a univer-
sal computation. The ﬁrst ever prototype of the fungal reservoir computer,
presented in the paper, demonstrates that a computation can be embedded
into living materials. The research presented also pinpointed a high degree
of variability in the logical circuits implemented by the fungi. This is be-
cause the live mycelium remain in the continuous process of growth and
reconﬁguration. To decrease the variability of the results we could consider

9

to functionalise the mycelium networks with semi-conductive particles and
polymers and allow the mycelium to dry. The resulting networks will have a
permanent structure which will guarantee repeatability of the experimental
circuits discovered. This will be a topic of our future studies.

Acknowledgement

This project has received funding from the European Union’s Horizon
2020 research and innovation programme FET OPEN “Challenging current
thinking” under grant agreement No 858132.

Supplementary materials

4-inputs-1-output logical functions discovered in experiments with mycelium

bound composites.

(AD) + (CB) + (DA) + (AB) + (BDC)
(AD) + (DA) + (BDC)
(ADC) + (BDA) + (ACBD)
(BD) + (ABC) + (ACB)
(AD) + (BDA)
(BC) + (BD) + (CD) + (AD) + (DA)
(ABC) + (ACDB)
(ADB) + (DBC) + (BCDA) + (ABCD)
(BD) + (DA) + (ACD)
A + B + D
(BC) + (BD) + (ACD) + (ACD)
(AD) + (CDA)
(ADB) + (ACD) + (DBC) + (BCDA)
(BCA) + (ABDC) + (ACDB) + (ABCD)
ADC
(BC) + (CD) + (AD) + (DA)
C + (AB) + (BA) + (DA)
(ABC) + (ADB) + (BDA) + (CDA)
(ABC) + (ACD)
(ABCD) + (BCDA) + (ABCD)
(ACB) + (BDA) + (ABD)
(BA) + (CA) + (DB) + (BCD) + (ABC)

10

(ABD) + (ACD)
(ABC) + (ACB) + (BCD) + (BDC) + (CDB)
ABDC
(AB) + (BA) + (DA) + (BCD)
(AD) + (ABC) + (ACB) + (BCA) + (BCD) + (BDA) + (BDC) + (DAC) +
(ABC) + (BCD)
(ABC) +(ABD) +(ACB) +(ACD) +(ADB) +(ADC) + (BDA) +(BDC) +
(CDA) + (CDB)
(ABD) + (ACD) + (ACDB)
(BA) + (CA) + (DA) + (BCD) + (CDB)
(DA)+(DB)+(DC)+(ABC)+(ABD)+(ACB)+(ACD)+(BCA)+(BCD)
(BA) + (BC) + (BD) + (ACD) + (CAD) + (DAC) + (ACDB)
(ABCD) + (DABC)
(AD) + (DA) + (DB) + (DC)
(DA) + (DC) + (BCA)
(AD) + (DA) + (DBC)
(AD) + (DA) + (ABC) + (ACB) + (BCA) + (BCD) + (BDC) + (CDB) +
(ABC) + (BCD)
(ABD) + (ACB) + (CDA) + (DAB)
(AD) + (BCA) + (DABC)
(BCA) + (BDC) + (ACDB) + (BCD)
(AD) + (CB) + (DA) + (DC)
(ADC) + (BDA) + (CDA)
(AC) + (BCD) + (BDA) + (CDB)
(AB) + (AC) + (DA)
(AB) + (CA) + (DC)
(DA) + (DB) + (ABC) + (BCA)
(ABD) + (ACD)
(ABCD) + (ACDB) + (BCDA)
(BC) + (AD) + (DA) + (DB)
(ACB) + (BCA) + (BDC) + (ACD)
(ABC) + (ABD) + (BCDA)
(AD) + (BCD) + (DAB) + (DAC)
A + B + (CD) + (DC)
(BAC) + (BAD) + (BCD)
(ACD) + (ACDB) + (BCAD)
(ABC) + (ACDB) + (BCAD)
(ABC) + (ACB) + (ADB) + (BCA) + (ABCD)

11

(BA) + (CAD) + (ACDB)
(ACD) + (ACDB)
(ABC) + (ACDB) + (BCDA)
(ADC) + (BDA)
A + D + (BC) + (CB)
(DA) + (DB) + (DC) + (ACB) + (ACD) + (BCA) + (BCD)
(AD) + (DA) + (BCA) + (BDC) + (CDB)
C + (BD) + (AB) + (BA)
(BC) + (BD) + (CD) + (DA) + (ACD)
(AD) + (DA) + (DC)
(AD) + (BC) + (CA) + (CB) + (DA)
(AC) + (DC) + (ADB) + (BCD)
B + A + D
(ACD) + (BDA)
A + B + D
(AD) + (BA) + (BC) + (CDB)
(DA) + (DB) + (ABC)
(AD) + (BA) + (BC) + (BD) + (DA) + (AC) + (CD) + (ACB) + (CDB)
(DA) + (DC) + (ACB) + (BCA)
(ACDB) + (ABCD)
(ABDC) + (ABCD) + (DABC)
(AD) + (DA) + (DB) + (DC) + (BC)
(AB) + (AC) + (BCA) + (BDA) + (BCD)
(ABD) + (ACD)
(AB) + (AC) + (AD) + (BCA) + (BCD) + (BDA) + (BDC) + (CDA) +
(CDB) + (BCD)
(BD) + (ABC) + (BCA) + (ACD) + (ACDB)
(BCD) + (BCD) + (ACBD) + (ADBC)
(BCA) + (BDC) + (ACDB) + (ABCD)
(AD) + (BA) + (BD) + (CA) + (CD) + (DA) + (ABC) + (DBC)
(AB) + (ADC) + (BCA) + (BCD)
(AD) + (ACB) + (BCA) + (BDA)
(ACB) + (ACD) + (ABD) + (BCDA)
(BD) + (CD) + (DA) + (DBC)
(BC) + (BD) + (CB) + (CD) + (DA) + (DB) + (DC)
(AB) + (AD) + (BA) + (BD) + (CA) + (DA) + (DB)
(AD) + (ABC) + (ACB) + (BCA) + (BCD)
A + B + D + C

12

(ACB) + (BCDA)
(ABC) + (ACB) + (ABD)
(AB) + (CDB) + (ACD) + (DAC)
(AD) + (BCA) + (BDA) + (CDB)
(ABD) + (ACB) + (BCDA)
(BCA) + (CDA) + (ABD) + (ACD)
(BD) + (ABC) + (BCA) + (ACD) + (ACDB) + (DABC)
D + (AB) + (AC) + (BA) + (BC) + (CA) + (CB)
(ACB) + (ADC) + (BCDA)
(ABC) + (ABD) + (BCA) + (BCD) + (BDA) + (BDC) + (CDB)
(ACD) + (BCDA)
(ADB) + (BCDA)
(ABC) + (ACD) + (BCAD)
(BAC) + (CBD) + (DAB)
(AD) + (DA) + (BCA) + (CDB) + (ABC)
(AD) + (DA) + (DC) + (BC) + (BCA)
(BA) + (BD) + (ABC) + (CAD) + (DBC)
(ABC) + (ABD) + (ACD) + (ACDB) + (BCDA)
(CD) + (DB) + (DC) + (ABC)
(CB) + (DA) + (BD) + (ABC)
(ACB) + (ACD) + (BCDA)
(BD) + (CA) + (CB) + (CD) + (AD) + (ABC) + (ADB) + (ADC)
(ACB) + (ACD) + (BCAD)
(ABDC) + (ACDB) + (ABCD) + (DABC)
(AD) + (BCD) + (BDA) + (CBD)
(AD) + (ABC) + (BCDA)
(BDC) + (ABD)
(AB) + (AC) + (BCA) + (BCD)
(BA) + (BC) + (CD) + (ACB) + (DAC)
(ABDC) + (ACDB) + (BCDA)
(ABD) + (CBD)
(DA) + (DB) + (DC) + (ABC) + (ACB) + (BCA)
(CBD) + (ABCD)
(BC) + (DA) + (DB) + (ABD) + (CAB)
(BC) + (ABD) + (ACB) + (ACD) + (ADB) + (ADC) + (BDA) + (CDA) +
(CDB) + (ACD)
(DA) + (DB) + (DC) + (ACD)
(ABC) + (ABD) + (CDB) + (ACD) + (DAB)

13

(BD) + (ABC) + (ACB) + (BCA)
(AB) + (AC) + (AD) + (BCD)
(CD) + (DC) + (ABC) + (ADB)
(CB) + (DA) + (AB) + (ACD)
(ABC) + (ACD) + (ADC) + (BDA)
(AB) + (AD) + (BA) + (BD) + (DA) + (DB)
(DA) + (DC) + (ACB)
(AD) + (BA) + (BC) + (CD) + (ACB)
(BCD) + (ABC) + (ACB)
(BCA) + (ABD) + (ACD)
(ABC) + (ABD) + (CDA) + (CDB)
(ACB) + (CBD) + (ABCD)
(BA) + (CA) + (BCD) + (CBD) + (ADBC)
(AB) + (AC) + (BA) + (BC) + (CA) + (CB) + (DA)
(AD) + (DA) + (DB) + (BC) + (BCA)
(ABD) + (BCD) + (ABCD)
(ABD) + (ACD) + (BCD) + (BCD)
(ABCD) + (ACDB) + (BCDA) + (ABCD) + (DABC)
(ABC) + (ACB) + (ABD) + (BCDA)
(ABC) + (ABD) + (CDB) + (ACD)
ABCD
(ACB) + (BCA) + (ACD) + (BCD) + (CAD)
(ABDC) + (ACDB)
(AB) + (AC) + (BC) + (BDA)
(AC) + (BD) + (DB) + (DC)
(AB) + (BD)
(ABC) + (ACD) + (ADC) + (BDA) + (CDA)
(ACD) + (ACDB) + (BCDA)
ACDB
(BA) + (CA) + (DB) + (ABC)
(ACB) + (ACD) + (BCDA)
A + D + (BC) + (BC)
(AB) + (AC) + (CD) + (BAD)
(DA) + (DB) + (DC) + (ABC) + (ABD) + (ACB) + (ACD)
A + D + (BC)
(AD) + (BA) + (DA)
(ACD) + (BDAC)
(BCD) + (BDA) + (BDC) + (CDB) + (ACD)

14

(AB) + (AC) + (AD) + (BCA)
(AD) + (CA) + (DA) + (DB)
ABD
B + (AD) + (AC)
ACBD
(ABC) + (ADB) + (BCAD)
BCDA
(AD) + (ACB) + (BCA) + (BDC) + (DAC)
(BAC) + (BCD) + (CAD)
(AD) + (DAB) + (DAC)
(ACD) + (ACDB) + (BDAC)
(AD) + (ACB) + (BCDA)
(DA) + (DC) + (ACB) + (BAC)
AD
(ABD) + (BCDA)
A + (BC) + (BD) + (BCD)
(BDA) + (ADBC)
(ACD) + (BDC) + (ABD)
(ABCD) + (ABCD)
(ACB) + (ADC) + (BDA) + (CDA)
(ABC) + (ABD) + (CDA) + (CDB) + (ACD)
(ABCD) + (BCAD)
(ACB) + (BCA) + (BDA) + (ACD)
(AC) + (CD) + (AD) + (BA) + (DA) + (AC) + (CD)
(ABD) + (ACD)
(AD) + (DA) + (CDB)
(BAC) + (CBD) + (DAC)
(AD) + (CDA) + (CDB) + (DAB)
(AB) + (AC) + (AD) + (BD) + (DAB)
(AB) + (AC) + (BD) + (BC) + (BCA)
(AD) + (BCA) + (DAB)
(ABD) + (ACD) + (ACDB) + (BCDA) + (DABC)
(ABC) + (ABD) + (ACB) + (BDA)
C + D + (AB) + (BA)
(ABD) + (ACDB) + (BCDA)
(AB) + (BA) + (BC) + (DA)
(ABC) + (ACB) + (ABD) + (BCAD)
(BA) + (BC) + (BD) + (DA) + (DB) + (DC) + (ACB) + (ACD)

15

(BA) + (CD) + (DA)
(AD) + (ABC) + (ACB) + (BCD)
(AB) + (AD) + (BCD)
(BC) + (ACD) + (CDA) + (CDB)
(BCDA) + (ADBC)
(ACBD) + (BCAD)
(AD) + (BD) + (BCA) + (BCD)
(AB)+(AC)+(AD)+(BCA)+(BCD)+(BDA)+(BDC)+(CDA)+(CDB)
(CDB) + (ACD) + (DAC)
(AD) + (DA) + (DB) + (BCA)
(AB) + (ACD) + (BCD) + (BCD)
(AD) + (BDAC) + (CDAB)
A + (BC) + (CB) + (BD)
(ABC) + (ACB) + (ABCD)
(BCD) + (ABD) + (ACD) + (BAD) + (BCD) + (CAD) + (CBD) + (DBC)
(AB) + (AC) + (BD) + (BCA)
(ACB) + (CDA) + (ACD) + (DAB)
(AC) + (ADB) + (BCD) + (BDC)
(DC) + (BC) + (ADB) + (BDA)
(AB) + (ADC) + (BCDA) + (BCD)
(AD) + (BCA) + (BDC) + (CDB)
(AB) + (AC) + (BD) + (CD) + (BCA)
(AD) + (CBD) + (BCDA)
C + (AB) + (AD) + (BA) + (BD) + (DA) + (DB)
(AD) + (BA) + (BC) + (DA)
(BC) + (DC) + (ABD) + (ADB) + (BDA) + (CABD)
B + C + A + D
(ABC) + (BCA) + (BCD) + (ACDB) + (BCD)
(DA) + (ABC) + (ABD)
A + B + C + D
(AB) + (ACD) + (BCAD)
(ACD) + (BDA) + (CDA)
(DB) + (CDA) + (ACD)
(AB) + (BA) + (BCD) + (CDA)
A + D + (BC)
(ABC) + (ACB) + (ADB) + (BCAD)
A + C + D + B
(DA) + (DB) + (ACD) + (CAB)

16

(CD) + (AD) + (CB) + (DA)
(AD) + (BDC) + (CDB) + (BCD)
(DA) + (DC) + (ABC) + (ACD) + (BCA)
(AC) + (BD) + (CA) + (DA) + (DB)
(ABDC) + (BCDA) + (ACBD)
(ABC) + (BCA) + (BCD) + (ACD) + (ACDB)
(ABDC) + (BCDA)
(BA) + (BC) + (ACB) + (ABD)
(AC) + (BD) + (DA) + (DB)
(AD) + (BCDA)
(AD) + (BA) + (DA) + (CDB)
(BA) + (BC) + (DC) + (ACB) + (CBD)
(ACDB) + (ABCD) + (BCAD)
(BA) + (CA) + (CDB)
(BC) + (DA) + (DB) + (DC) + (AC) + (ABD) + (ACB) + (ACD)
(ABD) + (ACD) + (BCDA)
(ABCD) + (ABDC) + (ACDB) + (BCDA)
(BA) + (BD) + (CA) + (CD) + (ADBC)
ABCD
A + B + (CD)
(ABC) + (ABD) + (ACB) + (BCA) + (BCD) + (BDA) + (BDC)
(BCD) + (BCD) + (CAD) + (CBD) + (DBC)
(ABC) + (ACB) + (ADB) + (BCD)
(CBD) + (DAB) + (BACD)
(DA) + (DB) + (ABC) + (ACB)
(ADC) + (BCDA) + (ACBD)
(DB) + (DC) + (BCA) + (ABC)
(ACB) + (CBD) + (BCDA) + (ABCD)
(BA) + (ADB) + (BCD) + (CAD)
A + B + C + D
(BD) + (AB) + (BCA) + (BCD)
(ABDC) + (ABCD)
(AB) + (AC) + (AD) + (DA) + (DB) + (DC) + (BCA) + (BCD)
(AD) + (BDC) + (CDB) + (ABC)
(ACB) + (ACD) + (DABC)
(ABD) + (ADB) + (ADC) + (BCDA)
(BC) + (CB) + (DA) + (BD)
(ABCD) + (ACBD) + (BCAD)

17

(AD) + (BD) + (ABC) + (ACB) + (BCA) + (ABCD)
(ABC) + (ACB) + (BCA) + (ABD)
C + A + D
(AD) + (ACB) + (BCA) + (BDC)
(AC) + (AD) + (CA) + (CD) + (DA) + (DC) + (AB)
(ABC) + (ABD) + (ACD)
(ADB) + (ACD) + (BDAC)
(AD) + (BC) + (BD) + (AC)
(AB) + (BD) + (BAC)
(DA) + (ABC) + (ABD) + (ABC)
(ABC) + (ACD) + (ADC) + (BDC)
(ABC) + (ACB) + (BCA) + (BDC) + (BCD)
(BA) + (BC) + (BD) + (DA) + (DB) + (DC) + (AC) + (ACB) + (ACD)
(ABDC) + (ACBD)
(AD) + (ABC) + (CDA)
(AD) + (DA) + (BCA)
(ABD) + (ACB) + (BDA) + (CDB)
(BD) + (AB) + (AC) + (BCA) + (BCD)
(AD) + (ABC) + (ACB) + (BCA) + (ABC)
B + (AC) + (AD) + (CA) + (CD) + (DA) + (DC)
(BCDA) + (ACBD) + (ADBC)
(ABC) + (ACB) + (BCDA)
(BD) + (ACD) + (BCA) + (ACD)
(AD) + (DB) + (BCD) + (DAC)
(ACB) + (CAD) + (ABCD)
(ABC) + (ACB) + (BCA)
(AB) + (AC) + (AD) + (BA) + (BC) + (BD) + (CA) + (CB) + (CD) +
(DA) + (DB) + (DC)
(AD) + (CDB) + (DAC)
(ABC) + (ABD) + (ACD) + (ACDB)
(CA) + (DA) + (BCD) + (DBC)
(AB) + (AC) + (AD) + (BDA) + (BDC) + (CDA) + (CDB) + (BCD)
(BCD) + (BCD) + (ACBD)
ACD
(BDA) + (BDC) + (ACDB) + (ABC) + (BCD)
(AD) + (BCD)
(ACB) + (BCA) + (CAD) + (ABCD)
(DA) + (ABD) + (ACD) + (BCA)

18

(AD) + (BCD) + (CBD)
(ACDB) + (BCDA)
ADBC
(DA) + (BCA) + (ABD) + (ACD)
(ABC) + (ABD) + (ACD) + (BCD) + (ACDB) + (BCDA)
A + (BD) + (CD) + (DBC)
(BA) + (BCD) + (CBD) + (ADBC)
A + B + C + D
(AD) + (ABC) + (ACB)
(AB) + (BCD) + (BDA) + (CDA)
(AB) + (ACD) + (BCD) + (BDC) + (CDB) + (ACD) + (DAB) + (DAC)
(AD) + (BC) + (CB) + (DA) + (AB)
(AD) + (BC) + (DA) + (CDB)
(ACB) + (CDB) + (ACD) + (DAC)
(AB) + (BCD) + (BDA)
(AB) + (CA) + (BCD)
(AD) + (BA) + (BC) + (AC) + (ACB)
(AC) + (AD) + (BA) + (CD)
(AB) + (CD) + (AD) + (BD) + (DAB)
(ACB) + (ABDC) + (BCDA) + (ABCD)
(AD) + (BDAC)
(BA) + (BC) + (CD) + (DC) + (ADB)
A + (BCD) + (BCD)
(AD) + (BD) + (DAB) + (DAC)
(ACD) + (BDC) + (ABD) + (DAC)
(CDAB) + (ABCD)
(AC) + (DA) + (DB) + (BCD)
(BD) + (CA) + (DA) + (DBC)
(BC) + (BD) + (DA) + (ACB)
(ABC) + (ACB) + (ADB) + (BCA) + (ABD)
(BD) + (CA) + (DA) + (ABC)
(AB) + (BA) + (BC) + (CD)
(AB) + (AC) + (AD) + (BCD) + (DABC)
(AD) + (DA) + (BCA) + (DBC)
(ACB) + (ADC) + (BDA)
(AB) + (AC) + (DA) + (BC) + (BCA)
(CA) + (DA) + (DB) + (ACD)
(ABDC) + (BCDA) + (ABCD)

19

BCD
(BA) + (ADB) + (BCD) + (CBD)
(AD) + (BDA) + (CDB)
(BA) + (BC) + (DB) + (ACB)
(AD) + (DA)
ADB
(AD) + (DA) + (BC) + (BCA)
(ABD) + (ACB) + (CDB) + (BDAC)
(AD) + (DA) + (BCA) + (BDC) + (ABC)
(BCA) + (ABDC) + (ACDB)
A + C + D
(ABD) + (ACD) + (ADBC)
(ABC) + (BDC) + (CDA) + (ABD)
(ADB) + (ADC)
ABCD
(ABCD) + (ABDC) + (ACDB) + (BCDA) + (BCD)
(DA) + (ACD) + (BDC) + (CDB) + (ABD)
(ACB) + (ABDC)
(ABC) + (ACD) + (ACDB)
(ACD) + (BCA) + (BDA) + (CDA)
(AC) + (AD) + (CA) + (CD) + (DA) + (DB) + (DC)
(BDA) + (CDA) + (ACBD) + (ADBC)
(AD) + (BA) + (BC) + (DA) + (AC)
A + B + D
(AD) + (CD) + (ACB) + (BCA)
(ABC) + (ABD) + (ACD) + (BCD) + (BCD)
(ABD) + (ACD) + (ACDB) + (BCDA)
(ABCD) + (BDAC)
(DA) + (BCD) + (CDB) + (ABD)
(CDB) + (ACD)
(AB) + (AC) + (BCAD)
D + (AB) + (BA) + (BC)
(ABC) + (ACB)
(ADC) + (BCDA)
(ACDB) + (ABCD)
(ACB) + (BCDA) + (ABCD)
(CDB) + (ACD) + (BDAC)
(ABD) + (ACB) + (BDA)

20

C + A + B + D
(ABC) + (ADC) + (BDA) + (CDA)
A + D + (BC)
(ABD) + (ACB) + (BDC)
(AB) + (AC) + (DC) + (BC) + (BCA)
(ABC) + (ACB) + (ACD) + (CDA) + (BCD)
ABCD
(BDA) + (CDA) + (ADBC)
(CBD) + (BACD) + (DABC)
(ABD) + (ADBC)
B + (CD) + (AD) + (DA)
(ACD) + (ADB) + (ADC) + (BCDA)
B + (AD) + (CD)
(BCD) + (ACBD)
(ABD) + (CBD) + (BCDA)
(ACB) + (ABDC) + (BCDA)
(ABD) + (ABDC)
(ABD) + (ABDC) + (BCDA)
(ABCD) + (ACDB) + (BCDA) + (ABCD)
(AD) + (BCD) + (BDC) + (CDB)
ABD
(AD) + (BA) + (CB) + (DA)
(ABD) + (ACB) + (CBD) + (BCDA)
(BCA) + (ACD)
B + A + C + D
(BA) + (CBD) + (DAC)
(AD) + (BA) + (BC) + (DA) + (CDB)
(DA) + (DC) + (ACBD)
(ABD) + (ACB)
(ACB) + (ABD) + (ABDC)
(ABD) + (ACD) + (BDC) + (CDB)
(AC) + (CA) + (DA) + (DB) + (AB)
B + (AC) + (CD) + (AD) + (DA) + (AC) + (CD)
(BC) + (ACB) + (BDA) + (ABD)
(AB) + (BD) + (DB) + (DC)
(BCA) + (BDC) + (ACDB) + (ABCD)
(ADB) + (BCD) + (ACD) + (BDAC)
(ACB) + (ACD) + (ABD)

21

(CA) + (CB) + (CD) + (DA) + (DB) + (DC) + (ABC) + (ABD)
(AC) + (AD) + (BDA) + (BAC) + (CBD)
(ABDC) + (ACDB) + (BCDA) + (ABCD)
(AB) + (AC) + (DA) + (BCA)
(ABC) + (ACB) + (BDA)
(ABD) + (ADB) + (ADC) + (BDA) + (BDC) + (CDA) + (CDB)
ACB
CBD
(CD) + (DA) + (ABD) + (ACD)
(AD) + (DA) + (CDB) + (ABC)
(AD) + (DA) + (BCA) + (CDB)
(AB) + (AD) + (BA) + (BD) + (DA) + (DB) + (DC)
(AD) + (BC) + (CA) + (DA)
(ABC) + (ACB) + (BCA) + (ABCD)
(AB) + (AC) + (BAD)
(ABC) + (ACB) + (ADB)
(AD) + (ACB) + (CBD) + (BCDA)
(AB) + (CDB) + (ACD) + (DAC)
(BA) + (CA) + (BCD) + (ADBC)
(ABC) + (ACB) + (ABD) + (BCD) + (BCDA)
(ABC) + (ABD) + (ACDB) + (BCDA)
(DA) + (ACD) + (BDC) + (CDB) + (ABD) + (BAC)
C + (AD) + (DA) + (AB)
(AD) + (BA) + (BD) + (DA) + (AC) + (CD) + (ACB) + (CDB)
(ACB) + (ADC)
(CD) + (DC) + (ABC) + (ADB) + (BDA)
(AB) + (AD) + (BA) + (BD) + (DA) + (DB) + (AC)
(ABC) + (ABD) + (ACD) + (BCAD)
(ACD) + (BAC) + (CAB) + (DAB)
A + B + C + D
(ADC) + (BDA) + (CDA) + (ACBD)
(AB) + (AC) + (BC) + (BD) + (CD) + (AD) + (DA) + (ABC) + (BCD)
(BA) + (DA) + (CBD)
(CDA) + (ACBD) + (ADBC)

22

References

References

[1] Andrew Adamatzky, Antoni Gandia, Phil Ayres, Han W¨osten, and Mar-
tin Tegelaar. Adaptive fungal architectures. LINKs-series, 5:66–77.

[2] Andrew Adamatzky, Anna Nikolaidou, Antoni Gandia, Alessandro Chi-
olerio, and Mohammad Mahdi Dehshibi. Reactive fungal wearable.
Biosystems, 199:104304, 2021.

[3] Andrew Adamatzky, Martin Tegelaar, Han AB Wosten, Anna L Powell,
Alexander E Beasley, and Richard Mayne. On boolean gates in fungal
colony. Biosystems, 193:104138, 2020.

[4] Freek Vincentius Wilhelmus Appels. The use of fungal mycelium for
the production of bio-based materials. PhD thesis, Universiteit Utrecht,
2020.

[5] Alexander E Beasley, Mohammed-Salah Abdelouahab, Ren´e Lozi,
Anna L Powell, and Andrew Adamatzky. Mem-fractive properties of
mushrooms. arXiv preprint arXiv:2002.06413, 2020.

[6] Lynne Boddy, John M Wells, Claire Culshaw, and Damian P Donnelly.
Fractal analysis in studies of mycelium in soil. Geoderma, 88(3):301–328,
1999.

[7] Rory G Bolton and Lynne Boddy. Characterization of the spatial aspects
of foraging mycelial cord systems using fractal geometry. Mycological
research, 97(6):762–768, 1993.

[8] Juan Pablo C´ardenas-R. Thermal insulation biomaterial based on hy-
drangea macrophylla. In Bio-Based Materials and Biotechnologies for
Eco-Eﬃcient Construction, pages 187–201. Elsevier, 2020.

[9] Michael John Carlile, Sarah C Watkinson, and Graham W Gooday. The

fungi. Gulf Professional Publishing, 2001.

[10] Kustrim Cerimi, Kerem Can Akkaya, Carsten Pohl, Bertram Schmidt,
and Peter Neubauer. Fungi as source for new bio-based materials: a
patent review. Fungal biology and biotechnology, 6(1):1–10, 2019.

23

[11] Matthew Dale, Julian F Miller, and Susan Stepney. Reservoir computing
as a model for in-materio computing. In Advances in Unconventional
Computing, pages 533–571. Springer, 2017.

[12] Matthew Dale, Julian F Miller, Susan Stepney, and Martin A Trefzer.
A substrate-independent framework to characterize reservoir computers.
Proceedings of the Royal Society A, 475(2226):20180723, 2019.

[13] Peter Deutsch and Jean-Loup Gailly. Zlib compressed data format spec-

iﬁcation version 3.3. Technical report, 1996.

[14] Patrick Pereira Dias, Laddu Bhagya Jayasinghe, and Daniele Wald-
Investigation of mycelium-miscanthus composites as building

mann.
insulation material. Results in Materials, 10:100189, 2021.

[15] Elise Elsacker, Simon Vandelook, Aur´elie Van Wylick, Joske Ruytinx,
Lars De Laet, and Eveline Peeters. A comprehensive framework for the
production of mycelium-based lignocellulosic composites. Science of The
Total Environment, 725:138431, 2020.

[16] M Fricker, L Boddy, and D Bebber. Network organisation of mycelial
fungi. In Biology of the fungal cell, pages 309–330. Springer, 2007.

[17] Mark D Fricker, Luke LM Heaton, Nick S Jones, and Lynne Boddy. The
mycelium as a network. The Fungal Kingdom, pages 335–367, 2017.

[18] Manuela Giovannetti, Cristiana Sbrana, Luciano Avio, and Patrizia
Strani. Patterns of below-ground plant interconnections established by
means of arbuscular mycorrhizal networks. New Phytologist, 164(1):175–
181, 2004.

[19] Carolina Girometta, Anna Maria Picco, Rebecca Michela Baiguera,
Daniele Dondi, Stefano Babbini, Marco Cartabia, Mirko Pellegrini, and
Elena Savino. Physico-mechanical and thermodynamic properties of
mycelium-based biocomposites: a review. Sustainability, 11(1):281,
2019.

[20] D Hitchcock, CA Glasbey, and K Ritz. Image analysis of space-ﬁlling by
networks: Application to a fungal mycelium. Biotechnology Techniques,
10(3):205–210, 1996.

24

[21] GA Holt, Gavin Mcintyre, Dan Flagg, Eben Bayer, JD Wanjura, and
MG Pelletier. Fungal mycelium and cotton plant materials in the man-
ufacture of biodegradable molded packaging material: Evaluation study
of select blends of cotton byproducts. Journal of Biobased Materials and
Bioenergy, 6(4):431–439, 2012.

[22] Paul Glor Howard. The Design and Analysis of Eﬃcient Lossless Data

Compression Systems. PhD thesis, Citeseer, 1993.

[23] MR Islam, G Tudryn, R Bucinell, L Schadler, and RC Picu. Morphology
and mechanics of fungal mycelium. Scientiﬁc reports, 7(1):1–12, 2017.

[24] Mitchell Jones, Antoni Gandia, Sabu John, and Alexander Bismarck.
Leather-like material biofabrication using fungi. Nature Sustainability,
pages 1–8, 2020.

[25] Mitchell Jones, Andreas Mautner, Stefano Luenco, Alexander Bismarck,
and Sabu John. Engineered mycelium composite construction materi-
als from fungal bioreﬁneries: A critical review. Materials & Design,
187:108397, 2020.

[26] Elvin Karana, Davine Blauwhoﬀ, Erik-Jan Hultink, and Serena Camere.
When the material grows: A case study on designing (with) mycelium-
based materials. International Journal of Design, 12(2), 2018.

[27] Zoran Konkoli, Stefano Nichele, Matthew Dale, and Susan Stepney.
In Computational

Reservoir computing with computational matter.
Matter, pages 269–293. Springer, 2018.

[28] Mantas Lukoˇseviˇcius and Herbert Jaeger. Reservoir computing ap-
proaches to recurrent neural network training. Computer Science Re-
view, 3(3):127–149, 2009.

[29] Genaro J Mart´ınez, Andrew Adamatzky, Christopher R Stephens, and
Alejandro F Hoeﬂich. Cellular automaton supercolliders. International
Journal of Modern Physics C, 22(04):419–439, 2011.

[30] Genaro Ju´arez Mart´ınez, Andrew Adamatzky, and Harold V McIntosh.
Phenomenology of glider collisions in cellular automaton rule 54 and as-
sociated logical gates. Chaos, Solitons & Fractals, 28(1):100–111, 2006.

25

[31] JD Mihail, M Obert, JN Bruhn, and SJ Taylor. Fractal geometry of
diﬀuse mycelia and rhizomorphs of armillaria species. Mycological Re-
search, 99(1):81–88, 1995.

[32] Julian F Miller and Keith Downing. Evolution in materio: Looking
beyond the silicon box. In Proceedings 2002 NASA/DoD Conference on
Evolvable Hardware, pages 167–176. IEEE, 2002.

[33] Julian F Miller, Simon L Harding, and Gunnar Tufte. Evolution-in-
materio: evolving computation in materials. Evolutionary Intelligence,
7(1):49–67, 2014.

[34] Julian F Miller, Simon J Hickinbotham, and Martyn Amos. In materio
computation using carbon nanotubes. In Computational Matter, pages
33–43. Springer, 2018.

[35] Julian Francis Miller. The alchemy of computation: designing with the

unknown. Natural Computing, 18(3):515–526, 2019.

[36] Abhik Mojumdar, Himadri Tanaya Behera, and Lopamudra Ray. Mush-
room mycelia-based material: An environmental friendly alternative to
synthetic packaging. Microbial Polymers, pages 131–141, 2021.

[37] M Obert, P Pfeifer, and M Sernetz. Microbial growth patterns described

by fractal geometry. Journal of Bacteriology, 172(3):1180–1185, 1990.

[38] Maria Papagianni. Quantiﬁcation of the fractal nature of mycelial aggre-
gation in aspergillus niger submerged cultures. Microbial Cell Factories,
5(1):5, 2006.

[39] Dhananjay B Patankar, Tuan-Chi Liu, and Timothy Oolman. A fractal
model for the characterization of mycelial morphology. Biotechnology
and bioengineering, 42(5):571–578, 1993.

[40] MG Pelletier, GA Holt, JD Wanjura, Eben Bayer, and Gavin McIn-
tyre. An evaluation study of mycelium based acoustic absorbers grown
on agricultural by-product substrates. Industrial Crops and Products,
51:480–485, 2013.

[41] Owen Robertson et al. Fungal future: A review of mycelium biocompos-
ites as an ecological alternative insulation material. DS 101: Proceedings

26

of NordDesign 2020, Lyngby, Denmark, 12th-14th August 2020, pages
1–13, 2020.

[42] Greg Roelofs and Richard Koman. PNG: the deﬁnitive guide. O’Reilly

& Associates, Inc., 1999.

[43] Stefano Siccardi and Andrew Adamatzky. Actin quantum automata:
Communication and computation in molecular networks. Nano Com-
munication Networks, 6(1):15–27, 2015.

[44] Jillian Silverman, Huantian Cao, and Kelly Cobb. Development of mush-
room mycelium composites for footwear products. Clothing and Textiles
Research Journal, 38(2):119–133, 2020.

[45] S Sivaprasad, Sidharth K Byju, C Prajith, Jithin Shaju, and CR Re-
jeesh. Development of a novel mycelium bio-composite material to sub-
stitute for polystyrene in packaging applications. Materials Today: Pro-
ceedings, 2021.

[46] Myron L Smith, Johann N Bruhn, and James B Anderson. The fungus
Armillaria bulbosa is among the largest and oldest living organisms.
Nature, 356(6368):428, 1992.

[47] Susan Stepney. Co-designing the computational model and the comput-
ing substrate. In International Conference on Unconventional Compu-
tation and Natural Computation, pages 5–14. Springer, 2019.

[48] David Verstraeten, Benjamin Schrauwen, Michiel d’Haene, and Dirk
Stroobandt. An experimental uniﬁcation of reservoir computing meth-
ods. Neural networks, 20(3):391–403, 2007.

[49] Fei WANG, Hong-qiang LI, Shu-shuo KANG, Ye-fei BAI, Guo-
zhen CHENG, and Guo-qiang ZHANG. The experimental study of
mycelium/expanded perlite thermal insulation composite material for
buildings. Science Technology and Engineering, 2016:20, 2016.

[50] Stephen Wolfram. Statistical mechanics of cellular automata. Reviews

of modern physics, 55(3):601, 1983.

27

[51] Yangang Xing, Matthew Brewer, Hoda El-Gharabawy, Gareth Griﬃth,
and Phil Jones. Growing and testing mycelium bricks as building insu-
lation materials. In IOP Conference Series: Earth and Environmental
Science, volume 121, page 022032. IOP Publishing, 2018.

[52] Zhaohui Yang, Feng Zhang, Benjamin Still, Maria White, and
Philippe Amstislavski. Physical and mechanical properties of fungal
mycelium-based biofoam. Journal of Materials in Civil Engineering,
29(7):04017030, 2017.

[53] Jacob Ziv and Abraham Lempel. A universal algorithm for sequential
data compression. IEEE Transactions on information theory, 23(3):337–
343, 1977.

28

