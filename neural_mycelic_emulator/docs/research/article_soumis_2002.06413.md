See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/339079868

Mem-fractive Properties of Mushrooms

Preprint · February 2020

DOI: 10.13140/RG.2.2.31741.10720

CITATIONS
0

5 authors, including:

Mohammed-Salah Abdelouahab

Centre Universitaire de Mila

79 PUBLICATIONS   821 CITATIONS   

SEE PROFILE

Alexander E Beasley

University of Hertfordshire

28 PUBLICATIONS   138 CITATIONS   

SEE PROFILE

READS
1,162

René Pierre Lozi

Université Côte d'Azur

211 PUBLICATIONS   1,942 CITATIONS   

SEE PROFILE

Andrew Adamatzky

University of the West of England, Bristol

922 PUBLICATIONS   14,614 CITATIONS   

SEE PROFILE

All content following this page was uploaded by René Pierre Lozi on 10 April 2020.

The user has requested enhancement of the downloaded file.

0
2
0
2

r
p
A
9

]
E
N
.
s
c
[

2
v
3
1
4
6
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

Mem-fractive Properties of Mushrooms

Alexander E. Beasley∗1,*, Mohammed-Salah Abdelouahab3, Ren´e Lozi2, Anna L. Powell1,
and Andrew Adamatzky1

1Unconventional Computing Laboratory, UWE, Bristol, UK
2Universit´e Cˆote d’Azur, CNRS, LJAD, Nice, France
3Laboratory of Mathematics and their interactions, University Centre Abdelhaﬁd Boussouf,
Mila 43000, Algeria

Abstract

Memristors close the loop for I-V characteristics of the traditional, passive, semi-conductor devices.
Originally proposed in 1971, the hunt for the memristor has been going ever since. The key feature of
a memristor is that its current resistance is a function of its previous resistance and the current passed
through it. As such, the behaviour of the device is inﬂuenced by changing the way in which potential
is applied across it. Ultimately, information can be encoded on memristors. Biological substrates have
already been shown to exhibit some memristive properties. By extension, the mem-capacitor and mem-
inductor have been proposed. Such devices change either their capacitive or inductive properties a function
of the previous voltage, similar to memristors. A device that exhibits combinations of memristors, mem-
capacitors and mem-inductors is termed a mem-fractive device. However, many passive memory devices
are yet to be found. Here we show that the fruit bodies of grey oyster fungi Pleurotus ostreatus exhibit
encouraging behaviour in the ﬁeld of organic memory devices. This paper presents the I-V characteristics
of the mushrooms. By examination of the conducted current for a given voltage applied as a function of
the previous voltage, it is shown that the mushroom exhibits the properties of a mem-fractor. Our results
demonstrate that nature continues to provide specimens that hold these unique and valuable electrical
characteristics and which have the potential to advance the ﬁeld of hybrid electronic systems.

K eywords: memristor, fungi, memfractance

1

Introduction

Originally proposed by Chua in 1971 [10], the memristor poses a fourth basic circuit element, whose char-
acteristics diﬀer from that of R, L and C elements. Memristance has been seen in nano-scale devices where
electronic and ionic transport are coupled under an external bias voltage [35]. Strukov et al. posit that
the hysteric I-V characteristics observed in thin-ﬁlm, two-terminal devices can be understood as memristive.
However, this is observed behaviour of devices that already have other, large signal behaviours.

Finding a true memristor is by no means an easy task. Nevertheless, a number of studies have turned
to nature to provide the answer, with varying success. Memristive properties of organic polymers were
discovered well before the ‘oﬃcial’ discovery of the memristor was announced in [35]. The ﬁrst examples
of memristors could go back to the singing arc, invented by Duddell in 1900, and originally used in wireless
telegraphy before the invention of the triode [18]. Memristive properties of organic polymers have been
studied since 2005 [13] in experiments with hybrid electronic devices based on polyaniline-polyethylenoxide
junction [13]. Memristive properties of living creatures and their organs and ﬂuids have been demonstrated
in skin [29], blood [24], plants [37] (including fruits [36]), slime mould [15], tubulin microtubules [12, 9].

∗Corresponding author: Alexander Beasley, alex.beasley@uwe.ac.uk

1

 
 
 
 
 
 
(a)

(b)

Figure 1: Positions of electrodes in fruit bodies. (a) Electrodes inserted 10 mm apart in the fruit body cap.
(b) One electrode is inserted in the cap with the other in the stem.

This paper presents a study of the I-V characteristics of the fruit bodies of the grey oyster fungi Pleurotus
ostreatus. Why fungi? Previously we recorded extracellular electrical potential of oyster’s fruit bodies,
basidiocarps [2] and found that the fungi generate action potential like impulses of electrical potential. The
impulses can propagate as isolated events, or in trains of similar impulses. Further, we demonstrated, albeit
in numerical modelling, that fungi can be used as computing devices, where information is represented
by spikes of electrical activity, a computation is implemented in a mycelium network and an interface is
realised via fruit bodies [3]. A computation with fungi might not be useful per se, because the speed of
spike propagation is substantially lower than the clock speed in conventional computers. However, the fungal
computation becomes practically feasible when embedded in a slow developing spatial process, e.g. growing
architecture structures. Thus, in [4] we discussed how to: produce adaptive building constructions by
developing structural substrate using live fungal mycelium, functionalising the substrate with nanoparticles
and polymers to make mycelium-based electronics, implementing sensorial fusion and decision making in the
fungal electronics.

Why we are looking for mem-fractive properties of fungi? Mem-fractors [1] have combinations of properties
exhibited by memristors, mem-capacitors and mem-inductors. A memristor is a material implication [7, 26]
and can, therefore, can be used for constructing other logical circuits, statefull logic operations [7], logic
operations in passive crossbar arrays of memristors [28], memory aided logic circuits [25], self-programmable
logic circuits [6], and, indeed, memory devices [19].
If strands of fungal mycelium in a culture substrate
and the fruit bodies show some mem-fractive properties then we can implement a large variety of mem-
ory and computing devices embedded directly into architectural building materials made from the fungal
substrates [4].

The rest of this paper is organised as follows. Section 2 details the experimental set up used to examine
the I-V characteristics of fruit bodies. Section 3 presents the results from the experimentation, with further
discussion of voltage spiking provided in section 3.2. Mathematical modelling of the mem-fractive behaviour
of the Grey Oyster mushrooms is given in section 4. A discussion of the results is given in section 5 and
ﬁnally conclusions are given in section 3.

2

Figure 2: I-V Characteristics from a model of an ideal memristor [20].

2 Experimental Set Up

We used grey oyster fungi Pleurotus ostreatus (Ann Miller’s Speciality Mushrooms Ltd, UK) cultivated on
wood shavings. The iridium-coated stainless steel sub-dermal needles with twisted cables (Spes Medica SRL,
Italy) were inserted in fruit bodies (Fig. 1) of grey oyster fungi using two diﬀerent arrangements: 10 mm apart
in the cap of the fungi (cap-to-cap), Fig. 1(a), and translocation zones (cap-to-stem), Fig. 1(b). I-V sweeps
were performed on the fungi samples with Keithley Source Measure Unit (SMU) 2450 (Keithley Instruments,
USA) under the following conditions: [-500 mV to 500 mV, -1 V to 1 V] with the samples in ambient lab light
(965 Lux). Varying the step size of the voltage sweep allowed testing the I-V characteristics of the subject
at diﬀerent frequencies. Electrodes were arranged in two diﬀerent methods: both electrodes approximately
10 mm apart in the cap of the fruit body (Fig. 1(a)); and one electrode placed in the cap with the other
electrode placed in the stem (Fig. 1(b)). The voltage ranges are limited so as not to cause the electrolysis of
water. Each condition was repeated at least six times over the samples. Voltage sweeps were performed in
both directions (cyclic voltammetry) and plots of the I-V characteristics were produced.

MATLAB was used to analyse the frequency and distribution of spiking behaviour observed in the I-V
sweeps of the fruiting bodies under test (Sect. 3.2). All histogram plots are binned according to the voltage
interval set for the Kiethley SMU.

3 Results

3.1 I-V characterisation

Fruit body samples are shown to exhibit memristive properties when subject to a voltage sweep. The ideal
memristor model (Fig. 2) is shown to display ‘lobes’ on the I-V characterisation sweeps, indicating that the
current resistance is a function of the previous resistance — hence a memristor has memory. For the purposes
of analysis, graphs are referred to by their quadrants, starting with quadrant one as the top right and being
number anti-clockwise.

The ideal memristor model has a crossing point at 0V, where theoretically no current ﬂows. From Figs. 3
and 4, it can be seen that when 0 V is applied by the source meter, a reading of a nominally small voltage
and current is performed. The living membrane is capable of generating potential across the electrodes, and
hence a small current is observed. Mem-capacitors produce similar curves to that in Fig. 2, when plotting
charge (q) against voltage (v) [38]. Additionally, mem-inductors produce similar plots for current (i) against
ﬂux (ϕ).

While the sample under test is subjected to a positive voltage (quadrant 1), it can be seen there is
nominally a positive current ﬂow. Higher voltages result in a larger current ﬂow. For an increasing voltage

3

-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage (V)-6-4-20246Current (A)10-5I-V Characteristics of an Ideal Memristor(a)

(b)

Figure 3: Raw data from cyclic voltammetry performed over -0.5 V to 0.5 V. (a) Cap-to-cap electrode
placement. (b) Stem-to-cap electrode placement.

sweep there is a larger current ﬂow for the corresponding voltage during a negative sweep.

Similarly, in quadrant 3 where there is a negative potential across the electrodes, the increasing voltage
sweep yields a current with smaller magnitude than the magnitude of the current on a negative voltage sweep.

Put simply, the fruit body has a resistance that is a function of the previous voltage conditions.
By applying averaging to the performed tests, a clear picture is produced that demonstrates for a given
set of conditions, a typical response shape can be expected (Figs. 5 and 6). The stem-to-cap placement of
the electrodes in the fruit body yields a tighter range for the response (ﬁgures 5(b) and 6(b)). This can be
expected due to the arrangement of the transportation pathways, so-called translocation zone distinct from
any vascular hyphae [33, 23], in the fruit body which run from the edge of the cap and down back through the
stem to the root structure (mycelium). Cap-to-cap placement of the electrodes applies the potential across
a number of the solutes translocation pathways and hence yields a wider range of results. However, for all
results, it is observed that the positive phase of the cyclic voltammetry produces a diﬀerent conduced current
than the negative phase. The opening of the hysteresis curve around the zero, zero point suggests the fungus
is not strictly a mem-ristor, instead it is also exhibiting mem-capacitor and mem-inductor eﬀects. The build
of charge in the device prevents the curve from closing completely to produce the classic mem-ristor pinching
shape.

Reducing the step voltage step size (by ten fold) for the I-V characterisation is synonymous to reducing

4

-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage [V]-1-0.500.51Current [I]10-7Oyster mushroom fruit bodies with cap to cap electrodessample1 run1sample1 run2sample1 run3sample1 run4sample1 run5sample1 run6sample1 run7sample1 run8sample1 run9sample1 run10-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage [V]-1-0.500.51Current [I]10-7Oyster mushroom fruit bodies with stem to cap electrodessample1 run1sample1 run2sample1 run3sample1 run4sample1 run5sample1 run6sample1 run7sample1 run8sample1 run9sample1 run10sample1 run11sample1 run12sample1 run13sample1 run14sample1 run15sample1 run16sample1 run17sample1 run18sample1 run19sample1 run20(a)

(b)

Figure 4: Raw data from cyclic voltammetry performed over -1 V to 1 V. (a) Cap-to-cap electrode placement.
(b) Stem-to-cap electrode placement.

5

-1-0.8-0.6-0.4-0.200.20.40.60.81Voltage [V]-1-0.500.51Current [I]10-6Oyster mushroom fruit bodies with cap to cap electrodessample1 run1sample1 run2sample1 run3sample1 run4sample1 run5sample1 run6sample1 run7sample1 run8sample1 run9sample1 run10-1-0.8-0.6-0.4-0.200.20.40.60.81Voltage [V]-1-0.500.51Current [I]10-6Oyster mushroom fruit bodies with stem to cap electrodessample1 run1sample1 run2sample1 run3sample1 run4sample1 run5sample1 run6sample1 run7sample1 run8sample1 run9sample1 run10sample1 run11sample1 run12sample1 run13sample1 run14sample1 run15sample1 run16sample1 run17sample1 run18sample1 run19sample1 run20(a)

(b)

Figure 5: Average grey oyster fungi fruit bodies I-V characteristics for cyclic voltammetry of -0.5 V to 0.5 V.
(a) Cap-to-cap electrode placement. (b) Stem-to-cap electrode placement.

the frequency of the voltage sweep. Decreasing the sweep frequency of the voltage causes the chances of
“pinching” in the I-V sweep to increase, as seen in quadrant 1 of ﬁgure 7. This further reinforces the presence
of some mem-capacitor behaviour. Since the charging frequency of the fungus has now been reduced there
is a greater amount of time for capacitively stored energy to dissipate, thus producing a more ‘resistive’ plot
with a pinch in the hysteresis.

3.2 Spiking

It is observed from Figs. 3 to 4 that portions of the cyclic voltammetry result in oscillations in the conduced
current, or spiking activity. Oscillations occur most prominently on the positive phase of the cyclic voltam-
metry as the applied voltage approaches 0V and similarly on the negative phase, again as the applied voltage
approaches 0 V. Current oscillations are typically in the order of nano-amps and persist for a greater number
of cycles when the electrodes are arranged as a pair on the fruit body cap (between ﬁve and ten cycles)
compared to the stem-to-cap arrangement (fewer than ﬁve repeats).

Figure 8 demonstrates the spiking frequency of a single repeat of the cyclic voltammetry performed
between -0.5 V and 0.5 V with the electrodes in a cap-to-cap arrangement. It is shown in the ﬁgure that

6

-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage [V]-1-0.500.51Current [I]10-7Average Oyster mushroom fruit bodies with cap to cap electrodes-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage [V]-1-0.500.51Current [I]10-7Average Oyster mushroom fruit bodies with stem to cap electrodes(a)

(b)

Figure 6: Average fruit bodies I-V characteristics for cyclic voltammetry of -1 V to 1 V. (a) Cap-to-cap
electrode placement. (b) Stem-to-cap electrode placement.

the voltage interval between spikes in an oscillation period are less than 0.06 V. Figure 9 concatenates the
data for all repeats of the cyclic voltammetry performed under four diﬀerent conditions. It is clearly shown
that in cap-to-cap arrangements the voltage interval between spikes is less than when the electrodes are in
a translocation arrangement. Any spikes that occur when the voltage interval becomes large can be taken
as not occurring during a period of oscillation in the sweep, instead they occur infrequently and randomly
during the sweep.

Reducing the frequency of the voltage sweep (Fig. 7) also has the eﬀect of removing the current oscillations.

7

-1-0.8-0.6-0.4-0.200.20.40.60.81Voltage [V]-1-0.500.51Current [I]10-6Average Oyster mushroom fruit bodies with cap to cap electrodes-1-0.8-0.6-0.4-0.200.20.40.60.81Voltage [V]-1-0.500.51Current [I]10-6Average Oyster mushroom fruit bodies with stem to cap electrodesFigure 7: I-V Characteristics of fungi fruit bodies with the voltage step size set to 0.001 V.

Figure 8: The voltage interval of spikes in the I-V characteristics of the fruit body for a single run.

8

-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage [V]-2-1012Current [I]10-7Oyster mushroom fruit bodies with cap to cap electrodessample1 run1sample1 run2Frequency of spiking in fruit body sample0.020.040.060.080.10.120.14Voltage interval [V]00.20.40.60.811.21.41.61.82Peak frequencyPositive cycleNegative cycle(a)

(b)

(c)

(d)

Figure 9: Concatenations of all spiking data from all data runs for four diﬀerent test conditions. (a) voltam-
metry over -0.5 V to 0.5 V, cap-to-cap electrode arrangement. (b) voltammetry over -1 V to 1 V, cap-to-cap
electrode arrangement. (c) voltammetry over -0.5 V to 0.5 V, stem-to-cap electrode arrangement. (d) voltam-
metry over -1 V to 1 V, stem-to-cap electrode arrangement. Legends ommited on (c) and (d) for clarity.

9

00.10.20.30.40.50.60.7Voltage interval [V]00.511.522.53Peak frequencyFrequency of spiking in fruit body sample over a number of runsPositive cycle 1Negative cycle 1Positive cycle 2Negative cycle 2Positive cycle 3Negative cycle 3Positive cycle 4Negative cycle 4Positive cycle 5Negative cycle 5Positive cycle 6Negative cycle 6Positive cycle 7Negative cycle 7Positive cycle 8Negative cycle 8Positive cycle 9Negative cycle 9Positive cycle 10Negative cycle 1000.20.40.60.811.2Voltage interval [V]00.511.522.53Peak frequencyFrequency of spiking in fruit body sample over a number of runsPositive cycle 1Negative cycle 1Positive cycle 2Negative cycle 2Positive cycle 3Negative cycle 3Positive cycle 4Negative cycle 4Positive cycle 5Negative cycle 5Positive cycle 6Negative cycle 6Positive cycle 7Negative cycle 7Positive cycle 8Negative cycle 8Positive cycle 9Negative cycle 9Positive cycle 10Negative cycle 1000.10.20.30.40.50.6Voltage interval [V]00.511.522.53Peak frequencyFrequency of spiking in fruit body sample over a number of runsFigure 10: Non-binary solution space showing mem-fractive properties of a memory element

4 Mathematical Model of Mushroom Memfractance

Here we report the I-V characteristics of grey oyster fungi Pleurotus ostreatus fruit bodies. It is evident from
the results that grey oyster fungi display memristive behaviour.

Although the fruit bodies typically do not demonstrate the “pinching” property of an ideal memristor [11],
it can be clearly seen that the biological matter exhibits memory properties when the electrical potential
across the substrate is swept. A positive sweep yields a higher magnitude current when the applied voltage
is positive; and a smaller magnitude current when the applied voltage is negative.

Fractional Order Memory Elements (FOME) are proposed as a combination of Fractional Order Mem-
Capacitors (FOMC) and Fractional Order Mem-Inductors (FOMI) [1]. The FOME (1) is based on the
generalised Ohm’s law and parameterised as follows: α1, α2 are arbitrary real numbers — it is proposed that
0 ≥ α1, α2 ≤ 1 models the solution space by [5], F α1,α2
is the memfractance, q(t) is the time dependent
charge, ϕ(t) is the time dependent ﬂux. Therefore, the memfractance (F α1,α2
M ) is an interpolation between
four points: M C — mem-capacitance, RM — memristor, M I — mem-inductance, and R2 M — the second
order memristor. Full derivations for the generalised FOME model are given by [1, 5]. The deﬁnition of
memfractance can be straightforward generalised to any value of α1, α2 (see [1, Fig. 27]).

M

Dα1

t ϕ(t) = F α1,α2

M (t)Dα2

t q(t)

(1)

The appearance of characteristics from various memory elements in the fungal I-V curves supports the

assertion that the fungal is a mem-fractor where α1 and α2 are both greater than 0 and less than 2.

There is no biological reason for memfractance of Ooyster fungi fruit bodies with stem to cap electrodes,
be a usual closed formula. Therefore, one can get only a mathematical approximation of this function. In
this section, we propose two alternatives to obtain the best approximation for memfractance in the case of
average fruit bodies I-V characteristics for cyclic voltammetry of Fig. 6(b) (Fig. 11).

4.1 Approximation by polynomial on the whole interval of voltage

Raw data include the time, voltage and intensity of each reading. There are 171 readings for each run.
The process of these data, in order to obtain a mathematical approximation of memfractance, in the ﬁrst
alternative, takes 4 steps as follows. First step: approximate v(t) by a twenty-four-degree polynomial (Fig. 12)
whose coeﬃcients are given in Tab, 1.

v(t) ≈ P (t) =

j=24
(cid:88)

j=0

ajtj

10

(2)

Mem-ristorMem-InductorMem-Capacitor2nd order mem-ristorα1 α2 (1,1)(0,1)(1,0)(0,0)Mem-FractorResistorCapacitorInductor(2,2)(1,2)(2,1)α1 α2 Figure 11: Raw data from average fruit bodies I-V cyclic voltammetry performed over -1 V to 1 V. Stem-to-
cap electrode placement.

Table 1: Coeﬃcient of P(t)

a0
a1
a2
a3
a4
a5
a6
a7
a8
a9
a10
a11
a12

2.36810109946699e-43
-4.78342788514078e-40
4.45025298649318e-37
-2.52211206380669e-34
9.68672841708898e-32
-2.64464369703488e-29
5.19611819410190e-27
-7.12198496974121e-25
5.80230108481181e-23
1.59013702626457e-22
-8.60157726907686e-19
1.48292987584698e-16
-1.56317950862153e-14

a13
a14
a15
a16
a17
a18
a19
a20
a21
a22
a23
a24

1.18302125464207e-12
-6.72265349925510e-11
2.90458838155410e-09
-9.51752589043893e-08
2.33484036114612e-06
-4.19159536470121e-05
0.000531866967507868
-0.00453232038841485
0.0240895989682110
-0.0726485498614107
0.135299293073760
-1.04736115240006

Sum of squared estimate of errors SSE = (cid:80)j=n
SSR = (cid:80)j=n

j=1 (vj − ˆvj)2
j=1 (ˆvj − v)2

Table 2: Goodness of ﬁt

Sum of squared residuals
Sum of square total
Coeﬃcient of determination

SST = SSE + SSR
R − square = SSR
SST

0.0680517563652170
133.688517134422
133.756568890787
0.999491226809049

The polynomial ﬁts very well the experimental voltage curve, as the statistical indexes show in Tab, 2.
Step 2: in the same way approximate the current i(t) using a twenty-four-degree polynomial (Fig. 13) whose
coeﬃcients are given in Table 3.

i(t) ≈ Q(t) =

j=24
(cid:88)

j=0

bjtj

(3)

Again, the polynomial ﬁts well the experimental intensity curve, as displayed in Table 4. Step 3: From

(1) used under the following form Dα2

t q(t) (cid:54)= 0.

11

-1-0.8-0.6-0.4-0.200.20.40.60.81v(t)-1-0.8-0.6-0.4-0.200.20.40.60.81i(t)#10-6Figure 12: Voltage versus time and its approximation by a 24-degree polynomial

Table 3: Coeﬃcient of Q(t)

b0
b1
b2
b3
b4
b5
b6
b7
b8
b9
b10
b11
b12

8.73846352218898e-49
-1.71535341852628e-45
1.55262364796050e-42
-8.56384614589988e-40
3.19831491989559e-37
-8.46294727047340e-35
1.59708666114599e-32
-2.04593370725626e-30
1.36036443304302e-28
8.14484000064489e-27
-3.66183256804588e-24
5.61870303550308e-22
-5.69947824465678e-20

b13
b14
b15
b16
b17
b18
b19
b20
b21
b22
b23
b24

4.25821973203331e-18
-2.42471413376463e-16
1.06306849079070e-14
-3.58289124918788e-13
9.19419585703268e-12
-1.76692818009608e-10
2.48346182702953e-09
-2.47326661661364e-08
1.67584032221916e-07
-7.34738169887512e-07
1.95479195837707e-06
-2.69478636561017e-06

Table 4: Goodness of ﬁt

Sum of squared estimate of errors
Sum of squared residuals
Sum of square total
Coeﬃcient of determination

5.84247524503151e-13
4.07366051979587e-11
4.13208527224619e-11
0.985860709883522

Figure 13: Current versus time and its approximation by a 24 degree polynomial

12

020406080100120140160180t-1-0.8-0.6-0.4-0.200.20.40.60.81v(t)020406080100120140160180t-6-4-20246i(t)#10-7Figure 14: Zeros t∗(α2) of the denominator of F α1,α2

M (t).

Dα1
t ϕt
Dα2
t q(t)
and the Rieman-Liouville fractional derivative deﬁned by [31]

M (t) =

F α1,α2

0 Dα
RL

t f (t) =

1
Γ(m − α)

dm
dtm

(cid:90) t

0

(t − s)m−α−1f (s)ds, m - 1 < α < m

together with the formula for the power function

0 Dα
RL
t

(cid:0)atβ(cid:1) =

aΓ(β + 1)
Γ(β − α + 1)

tβ−α, β > −1, α > 0,

(4)

(5)

(6)

we obtain the closed formula of F α1,α2

M (t), approximation of the true biological memfractance of the Oyster

mushroom

F α1,α2

M (t) =

Dα1
Dα2

t ϕ(t)
t ϕ(t)

=

0 Dα1
RL
t
0 Dα2
RL

t

(cid:80)j=24
j=0
(cid:80)j=24
j=0

aj
j+1 tj+1
bj
j+1 tj+1

=

(cid:80)j=24
j=0
(cid:80)j=24
j=0

aj Γ(j+1)
Γ(j+2−α1) tj+1−α1
bj Γ(j+1)
Γ(j+2−α2) tj+1−α2

(7)

Step 4 choice of parameter α1 and α2: We are looking for the best value of these parameters in the range
(α1, α2) ∈ [0, 2]2. In this goal, we are considering ﬁrst the singularities of F α1,α2
M (t) in order to avoid their
existence, using suitable values of the parameters. Secondly, we will choose the most regular approximation.
We compute numerically, the values t∗(α2) which vanish the denominator of F α1,α2

M (t) (Fig. 14).

We observe one, two or three coexisting solutions depending on the value of α2. Moreover, there is no
value of α2 without zero of the denominator. Therefore, in order to eliminate the singularities, we need to
determine the couples (α1, α2) ∈ [0, 2]2, vanishing simultaneously denominator and numerator of F α1,α2
M (t)
(Figs. 15,16).

In the second part of step 4, we choose the most regular approximation. We consider that the most

regular approximation is the one for which the function range (F α1,α2

M (t)) is minimal (Figs. 17,18)

range (F α1,α2

M (t)) = max

(F α1,α2

M (t)) − min

(F α1,α2

M (t))

t∈[0,171]
From the numerical results, the best couple (α1, α2) and the minimum range of F α1,α2

t∈[0,171]

Table 5, and the corresponding Memfractance is displayed in Fig. 19.

(8)

M (t) are given in

The value of (α1, α2) given in Table 5 belongs to the triangle of Fig. 10, whose vertices are Memristor,
Memcapacitor and Capacitor. Which means that Oyster mushroom fruit bodies with stem to cap electrodes,
is like a mix of such basic electronic devices.

As a counter-example of our method for choosing the best possible Memfractance, Fig. 20 displays, the

Memfractance for a non-optimal couple (α1, α2) = (1, 1.78348389322388) which presents two singularities.

13

00.20.40.60.811.21.41.61.82,2*020406080100120140160180t*3 solutionsFigure 15: Zeros t∗(α2) of F α1,α2

M (t) denominator (red dots), and zeros t∗(α1) of the numerator (blue dots).

Figure 16: Values of (α1, α2) ∈ [0, 2]2 for which the zeros t∗(α2) of denominator of F α1,α2
the zeros t∗(α1) of denominator.

M (t) correspond to

Figure 17: Values of range (F α1,α2

M (t)) for (α1, α2) ∈ [0, 2]2

Table 5: Minimum F α1,α2

M (t)
Minimum range F α1,α2

M (t)

α1
1.08642731

α2
0.25709492

825770.46017259

14

020406080100120140t*00.20.40.60.811.21.41.61.82,1, ,2(t,,1)(t,,2)00.20.40.60.811.21.41.61.82,100.511.52,200.20.40.60.811.21.41.61.82,1, ,200.511.52Range(Fm,1,,2(t))#109Minimum Range of smooth Mushroom memfractance Rm= 825770.46  for ( ,1, ,2)= (1.09, 0.26)(,1,Range(Fm,1,,2(t))(,2,Range(Fm,1,,2(t))X: 0.2571Y: 8.258e+05X: 1.086Y: 8.258e+05Figure 18: Magniﬁcation of Fig. 17.

Figure 19: Memfractance for (α1, α2) given in Table 5.

Figure 20: Memfractance with two singularities for (α1, α2) = (1, 1.78348389322388).

15

00.20.40.60.811.21.41.61.82,1, ,200.511.52Range(Fm,1,,2(t))#106Minimum Range of smooth Mushroom memfractance Rm= 825770.46  for ( ,1, ,2)= (1.09, 0.26)(,1,Range(Fm,1,,2(t))(,2,Range(Fm,1,,2(t))X: 0.2571Y: 8.258e+05X: 1.086Y: 8.258e+05020406080100120140160180t-10123456Fm,1,,2(t)#105Mushroom memfractance with ( ,1, ,2)= (1.09, 0.26)020406080100120140160180t-1-0.8-0.6-0.4-0.200.20.40.60.81Fm,1,,2(t)#109Mushroom memfractance with ( ,1, ,2)= (1.00, 1.78)Figure 21: Comparison between average experimental data of cyclic voltammetry performed over -1 V to 1 V,
Stem-to-cap electrode placement, and approximate values of v(t) and i(t).

4.2 Approximate cycling voltammetry
From the closed formula of F α∗
(1).

1 ,α∗

M

2

(t) it is possible to retrieve the formula of the current function i(t) using

i(t) = D1−α2

t

(cid:21)

(cid:20) Dα1
F α1,α2

t ϕ(t)
M (t)

= D1−α2
t

= D1−α2
t





j=24
(cid:88)

j=0

bjΓ(j + 1)
Γ(j + 2 − α2)








(cid:80)j=24
j=0

(cid:80)j=24
j=0
(cid:80)j=24
j=0









aj Γ(j+1)
Γ(j+2−α1) tj+1−α1
aj Γ(j+1)
Γ(j+2−α1) tj+1−α1
bj Γ(j+1)
Γ(j+2−α2) tj+1−α2

tj+1−α2



(9)

=

=

j=24
(cid:88)

j=0

j=24
(cid:88)

j=0

Γ(j + 2 − α2)bjΓ(j + 1)
Γ(j + 2 − α2)Γ(j + 1)

tj+1−α2−(1−α2)

bjtj

The comparison of average experimental data of cyclic voltammetry performed over -1 V to 1 V, Stem-to-
cap electrode placement, and closed approximative formula is displayed in Fig. 21, showing a good agreement
between both curves except near the maximum value of v(t) and i(t).

Figure 22 shows that the curve computed from closed approximative formula belongs to the histogram of

data of all runs.

The discrepancy between both curves is due to the method of approximation chosen in (2) and (3).
It is possible, as we show in the next subsection to improve the ﬁtting of the approximated curve near

the right hand-side vertex, using piecewise polynomial approximation of both v(t) and i(t).

4.3 Alternative approximation of the cycling voltammetry

Due to the way of conducting the experiments, the voltage curve presents a vertex, that means that the
function v(t) is non-diﬀerentiable for T = 87.23747459. In fact, the value of T is the average value of the
non-diﬀerentiable points for the 20 runs.

In this alternative approximation, we follow the same 4 steps as in 4.1, changing the approximation by
a twenty-four-degree polynomial to an approximation by a 2-piecewise ﬁfth-degree-polynomial, for both v(t)
and i(t).

16

-1-0.8-0.6-0.4-0.200.20.40.60.81v(t)-1-0.8-0.6-0.4-0.200.20.40.60.81i(t)#10-6Figure 22: Both average experimental data curve and the curve computed from closed approximative formula
are nested into the histogram of data of all runs.

Table 6: Coeﬃcient for i(t)
Coeﬃcient Value for 0 ≤ t ≤ T Coeﬃcient Value for T ≤ t < T

a0
a1
a2
a3
a4
a5

-0.98299
0.02665
-5.91565 E -4
1.12211 E -5
-6.28483 E -8
6.9675 E -11

a(cid:48)
0
a(cid:48)
1
a(cid:48)
2
a(cid:48)
3
a(cid:48)
4
a(cid:48)
5

37.16955
-1.2986
0.01826
-1.25146 E -4
4.12302 E -7
-5.25359 E-19

Approximation
Coeﬃcient of determination

Table 7: Goodness of ﬁt
t < T
0.99983

t > T
0.9999

First step: approximation of v(t) by a 2-piecewise ﬁfth-degree-polynomial (Fig. 23) whose coeﬃcients are

given in Table 6.

v(t) =

(cid:40)

P1(t) = a0 + a1t + a2t2 + a3t3 + a4t4 + a5t5, for 0 ≤ t ≤ T
P2(t) = a(cid:48)

5t5, for T ≤ t < 171

4t4 + a(cid:48)

3t3 + a(cid:48)

2t2 + a(cid:48)

1t + a(cid:48)

0 + a(cid:48)

The ﬂux is obtained integrating v(t) versus time.

(cid:40)

ϕ(t) =

IP1(t) = a0t + a1
0t + a(cid:48)
IP2(t) = a(cid:48)

2 t2 + a2
2 t2 + a(cid:48)

3 t3 + a3
3 t3 + a(cid:48)

4 t4 + a4
4 t4 + a(cid:48)

5 t5 + a5
5 t5 + a(cid:48)

6 t6, for 0 ≤ t ≤ T
6 t6, for T ≤ t < 171

1

3

2

5

4

(10)

(11)

The polynomial ﬁts very well the experimental voltage curve, as the statistical indexes show in Table 7.
Step 2: in the same way, one approximates the current i(t) using a 2-piecewise ﬁfth degree polynomial

(Fig. 24) whose coeﬃcients are given in Table 8.

17

 Figure 23: Voltage versus time and its approximation by 2-piecewise ﬁfth degree polynomial

Table 8: Coeﬃcient for i(t)
Coeﬃcient Value for 0 ≤ t ≤ T Coeﬃcient Value for T ≤ t < 171

b0
b1
b2
b3
b4
b5

-7.21418 E -7
1.11765 E -7
-6.3792 E -9
1.57327 E -10
-1.7745 E -12
7.52304 E -15

b(cid:48)
0
b(cid:48)
1
b(cid:48)
2
b(cid:48)
3
b(cid:48)
4
b(cid:48)
5

2.69466 E -4
-1.05461 E -5
1.63678 E -7
-1.25915 E -9
4.80107 E -12
-7.26253 E-15

Approximation
Coeﬃcient of determination

Table 9: Goodness of ﬁt
t < T
0.99171

t > T
0.98613

(cid:40)

i(t) =

P3(t) = b0 + b1t + b2t2 + b3t3 + b4t4 + b5t5, for 0 ≤ t ≤ T
3t3 + b(cid:48)
P4(t) = b(cid:48)

5t5, for T ≤ t < 171

4t4 + b(cid:48)

2t2 + b(cid:48)

1t + b(cid:48)

0 + b(cid:48)

(12)

Again, the polynomial ﬁts very well the experimental voltage curve, as the statistical indexes show in

Table 9.

Therefore, the charge is given by

(cid:40)

5 t5 + b5
IP3(t) = b0t + b1
0t + b(cid:48)
5 t5 + b(cid:48)
IP4(t) = b(cid:48)
Step 3: Following the same calculus as before with (4), one obtains

4 t4 + b4
4 t4 + b(cid:48)

3 t3 + b3
3 t3 + b(cid:48)

2 t2 + b2
2 t2 + b(cid:48)

q(t) =

1

5

4

2

3

6 t6, for 0 ≤ t ≤ T
6 t6, for T ≤ t < 171

for 0 ≤ t ≤ T , F α1,α2

M (t) =

0 Dα1
RL
0 Dα2
RL

t ϕ(t)
t q(t)

=

0 Dα1
RL
t
0 Dα2
RL
t

[IP1(t)]
[IP3(t)]

=

(cid:80)j=5
j=1
(cid:80)j=5
j=0

aj Γ(j+1)
Γ(j+2−α1) tj+1−α1
bj Γ(j+1)
Γ(j+2−α2) tj+1−α2

(13)

(14)

However, because fractional derivative has memory eﬀect, for T < t < 171, the formula is slightly more

complicated

18

020406080100120140160180t-1-0.8-0.6-0.4-0.200.20.40.60.81v(t)Experimental dataDegree 5 polynomial fittingFigure 24: Current versus time and its approximation by 2-piecewise ﬁfth degree polynomial

F α1,α2

M (t) =

=

=

1
Γ(m1−α1)
1
Γ(m2−α2)
(cid:104)(cid:82) T

0 Dα1
RL
0 Dα2
RL

t ϕ(t)
t q(t)

=

1
Γ(m1−α1)

dm1
dtm1

1
Γ(m2−α2)

dm2
dtm2

1
Γ(m1−α1)

dm1
dtm1
(cid:80)

1
Γ(m2−α2)

dm2
dtm2

j=0 j = 5

dm1
dtm1
dm2
dtm2

(cid:82) t
0 (t − s)m1−α1−1ϕ(s)ds
(cid:82) t
0 (t − s)m2−α2−1q(s)ds

, m1 − 1 < α1 < m1 and m2 − 1 < α2 < m2

(cid:104)(cid:82) T

0 (t − s)m1−α1−1IP1(s)ds + (cid:82) t
0 (t − s)m2−α2−1IP3(s)ds + (cid:82) t
(cid:80)j=5
j=0

(cid:104) aj
j+1

(cid:105)
T (t − s)m1−α1−1IP2(s)ds
(cid:105)
T (t − s)m2−α2−1IP4(s)ds
a(cid:48)
j
j+1

(cid:82) T
0 (t − s)m1−α1−1sj+1ds +

(cid:105)
(cid:82) t
T (t − s)m1−α1−1sj+1ds

(cid:104) bj
j+1

(cid:82) T
0 (t − s)m2−α2−1sj+1ds +

b(cid:48)
j
j+1

(cid:105)
(cid:82) t
T (t − s)m2−α2−1sj+1ds

(15)

Using integration by part repeatedly six times we obtain

19

020406080100120140160180t-6-4-20246i(t)#10-7Experimental dataCalculated by fractional model)
6
1
(

(cid:105)
(cid:105)
k
−
1
+
j
T
2
α
−
k
+
2
m
)
T
−

t
(
)
2
α
−
2

m
(
Γ
!
)
1
+
j
(

)
1
α
−
1
+
k
+
1

m
(
Γ
!
)
k
−
1
+
j
(

(cid:105)
(cid:105)
k
−
1
+
j
T
1
α
−
k
+
1
m
)
T
−

t
(
)
1
α
−
1

m
(
Γ
!
)
1
+
j
(

)
2
α
−
1
+
k
+
2

m
(
Γ
!
)
k
−
1
+
j
(

1
+
j
=
k
(cid:80)

0
=
k

(cid:104)

(cid:48)j
a

1
+
j

+

(cid:105)
1
α
−
k
+
1
m
t
)
1
α
−
1

m
(
Γ
!
)
1
+
j
(

)
1
α
−
1
+
j
+
1

m
(
Γ

+

1
+
j
=
k
(cid:80)

0
=
k

(cid:104)

(cid:48)j
b

1
+
j

+

(cid:105)
2
α
−
k
+
2
m
t
)
2
α
−
2

m
(
Γ
!
)
1
+
j
(

)
2
α
−
1
+
j
+
2

m
(
Γ

+

)
2
α
−
1
+
k
+
2

m
(
Γ
!
)
k
−
1
+
j
(

(cid:105)
k
−
1
+
j
T
2
α
−
k
+
2
m
)
T
−

t
(
)
2
α
−
2

m
(
Γ
!
)
1
+
j
(
−
(cid:104)
1
+
j
=
k
(cid:80)

(cid:104)

j
b

1
+
j

(cid:104)
5
=
j
(cid:80)

0
=
j

1
α
−
1
+
k
+
1

m
(
Γ
!
)
k
−
1
+
j
(

(cid:105)
k
−
1
+
j
T
1
α
−
k
+
1
m
)
T
t
(
)
1
α
−
1

m
(
Γ
!
)
1
+
j
(
−
(cid:104)
1
+
j
=
k
(cid:80)

(cid:104)

j
a

1
+
j

(cid:104)
5
=
j
(cid:80)

0
=
j

1
m
d

1
m
t
d

1

)
1
α
−
1

m
(
Γ

0
=
k

0
=
k

(cid:105)
1
α
−
1
+
j
+
1
m
t
)
1
α
−
1

m
(
Γ

!
j

(cid:105)
2
α
−
1
+
j
+
2
m
t
)
2
α
−
2

m
(
Γ

!
j

1
α
−
2
+
j
+
1

m
(
Γ

)
2
α
−
2
+
j
+
2

m
(
Γ

j
a
+

j
b
+

(cid:105)
k
−
1
+
j
T
1
α
−
k
+
1
m
)
T
−

t
(
)
1
α
−
1

m
(
Γ

)
1
α
−
1
+
k
+
1

m
(
Γ
!
)
k
−
1
+
j
(

!
j
(cid:104)
1
+
j
=
k
(cid:80)

0
=
k

)
j
a
−

(cid:48)j
a
(

(cid:104)
5
=
j
(cid:80)

0
=
j

(cid:105)
k
−
1
+
j
T
2
α
−
k
+
2
m
)
T
−

t
(
)
2
α
−
2

m
(
Γ

)
2
α
−
1
+
k
+
2

m
(
Γ
!
)
k
−
1
+
j
(

!
j
(cid:104)
1
+
j
=
k
(cid:80)

0
=
k

)
j
b
−

(cid:48)j
b
(

(cid:104)
5
=
j
(cid:80)

0
=
j

2
m
d

2
m
t
d

1
m
d

1
m
t
d

2
m
d

2
m
t
d

1

)
2
α
−
2

m
(
Γ

1

)
1
α
−
1

m
(
Γ

1

)
2
α
−
2

m
(
Γ

)
t
(
2
α

,
1
M
α
F

(cid:105)

1
α
−
1
+
j
t
!
j

)
1
α
−
2
+
j
(
Γ
j
a
+

(cid:105)

2
α
−
1
+
j
t
!
j

)
2
α
−
2
+
j
(
Γ
j
b
+

(cid:105)

(cid:105)

)
1
α
−
1
+
k
(
Γ
!
)
k
−
1
+
j
(

0
=
k

k
−
1
+
j
T
1
α
−
k
)
T
−

t
(
!
j
(cid:104)
1
+
j
=
k
(cid:80)

)
2
α
−
1
+
k
(
Γ
!
)
k
−
1
+
j
(

0
=
k

k
−
1
+
j
T
2
α
−
k
)
T
−

t
(
!
j
(cid:104)
1
+
j
=
k
(cid:80)

)
j
a
−

)
j
b
−

(cid:48)j
a
(

(cid:104)
5
=
j
(cid:80)

0
=
j

(cid:48)j
b
(

(cid:104)
5
=
j
(cid:80)

0
=
j

=

=

=

20

Figure 25: The ﬁrst zero t∗(α2) ≥ T , of the denominator of F α1,α2

M (t), as function of α2.

In this 2-piece wise approximation, the vertex is non-diﬀerentiable, this implies that (16) expression has

a singularity at T (because (t − T )−α1,2 → ∞).

It could be possible to avoid this singularity, using a 3-piece wise approximation, smoothing the vertex.

However, the calculus are very tedious. We will explain, below, what our simpler choice implies.

Then F α1,α2

M (t) =

(t − T )−α1

(t − T )−α2

(cid:104)(cid:80)j=5
j=0
(cid:104)(cid:80)j=5
j=0

(cid:104)
(a(cid:48)
(cid:104)
(b(cid:48)

j − aj) (cid:80)k=j+1
j − bj) (cid:80)k=j+1

k=0

(cid:104) j!(t−T )kT j+1−k

(j+1−k)!Γ(k+1−α1
(cid:104) j!(t−T )kT j+1−k

(cid:105)

)
(cid:105)

+ aj

j!tj+1−α1 (t−T )α1
Γ(j+2−α1)

+ bj

j!tj+1−α2 (t−T )α2
Γ(j+2−α2)

(cid:105)(cid:105)

(cid:105)(cid:105)

(cid:104)

(a(cid:48)

(cid:80)j=5
j=0
(t − T )α1−α2 (cid:80)j=5
j=0

=

j − aj) (cid:80)k=j+1

k=0

(j+1−k)!Γ(k+1−α2)

k=0
(cid:104) j!(t−T )kT j+1−k

(j+1−k)!Γ(k+1−α1)

(cid:105)

+ aj

j!tj+1−α1 (t−T )α1
Γ(j+2−α1)

(cid:105)

(cid:104)
(b(cid:48)

j − bj) (cid:80)k=j+1

k=0

(cid:104) j!(t−T )kT j+1−k

(cid:105)

(j+1−k)!Γ(k+1−α2)

+ bj

j!tj+1−α2 (t−T )α2
Γ(j+2−α2)

(cid:105)

(17)

Finally

F α1,α2

M (t) =






(cid:80)j=5
j=0
(cid:80)j=5
j=0

aj Γ(j+1)
Γ(j+2−α1) tj+1−α1
bj Γ(j+1)
Γ(j+2−α2) tj+1−α2
(cid:80)j=5
(a(cid:48)
j=0

(cid:20)

j −aj ) (cid:80)k=j+1

k=0

,

(cid:20)

j!(t−T )k T j+1−k
(j+1−k)!Γ(k+1−α1 )

(cid:21)

+aj

j!tj+1−α1 (t−T )α1
Γ(j+2−α1)

(cid:21)

(t−T )α1−α2 (cid:80)j=5
j=0

(cid:20)

(b(cid:48)

j −bj ) (cid:80)k=j+1

k=0

(cid:104) j!(t−T )k T j+1−k

(cid:105)

(j+1−k)!Γ(k+1−α2 )

+bj

j!tj+1−α2 (t−T )α2
Γ(j+2−α2)

for0 ≤ t ≤ T

(cid:21) ,

forT < t < 171

(18)

Step 4 choice of parameter α1 and α2: Following the same idea as for the ﬁrst alternative, we try to
avoid singularity for F α1,α2
M (t), except of course the singularity near T , which is of mathematical nature
(non-diﬀerentiability of voltage and intensity at t = T ). Figure 25 display the ﬁrst zero t∗(α2) ≥ T , of the
denominator of F α1,α2

M (t). One can see that t∗(1) ∼= T .

Figure 26 displays the curves of couples (α1, α2) for which the denominator and numerator of F α1,α2
M (t)
are null simultaneously for t < T and t > T . On this ﬁgure, the value of α1, that corresponds to α2 = 1 is
α1 ≈ 1.78348389322388. The corresponding Memfractance is displayed in Fig. 27.

The singularity observed in Figs. 27-28 is due to the non-diﬀerentiability of both voltage and intensity
It is only a mathematical problem of approximation which can be solved using a
functions at point T .
3-piecewise polynomial instead of the 2-piecewise polynomial (P 1(t), P 2(t)) and (P 3(t), P 4(t)). The third
added piecewise polynomials for v(t) and i(t) being deﬁned on the tiny interval [87.24, 87.90]. However due
to more tedious calculus, we do not consider this option in the present article. It is only a math problem, and
one can consider that Fig. 28 represents the value of the memfractance in the interval [0, 87.24] ∪ [87.90, 171].
The value of (α1 = 1.78, α2 = 1.00) belongs to the line segment of Fig. 10, whose extremities are
Memristor, and Capacitor. Which means that Oyster mushroom fruit bodies with stem to cap electrodes, is
like a mix of such basic electronic devices. The comparison of average experimental data of cyclic voltammetry

21

00.10.20.30.40.50.60.70.80.91,2*8090100110120130140150160170t*Figure 26: Couples (α1, α2) for which the denominator and numerator of F α1,α2
for t < T (blue dot) and t > T (red dot).

M (t) are null simultaneously

Figure 27: Memfractance for (α1 = 1.78, α2 = 1.00) given in Table 5

Figure 28: Magniﬁcation of Fig. 27

22

0.60.811.21.41.61.82,1-0.500.511.52,2t>Tt<T020406080100120140160180t00.511.522.533.544.5Fm,1,,2(t)#106Mushroom memfractance with ( ,1, ,2)= (1.78, 1.00)X: 87.54Y: 4.665e+06X: 87.24Y: 1.139e+05X: 87.85Y: 1.489e+06020406080100120140160180t-20246810Fm,1,,2(t)#105Mushroom memfractance with ( ,1, ,2)= (1.78, 1.00)X: 87.24Y: 1.139e+05Figure 29: Comparison between average experimental data of cyclic voltammetry performed over -1 V to 1 V,
Stem-to-cap electrode placement, and closed approximative formula.

performed over -1 V to 1 V, Stem-to-cap electrode placement, and closed approximative formula is displayed
in Fig. 29, showing a very good agreement between both curves.

5 Discussion

First at all, we have two remarks:

1. Both approximations used in Section 4 converge to Memfractance with parameter value (α1, α2) -
belonging inside or on edge of the triangle of Fig. 10, whose vertices are Memristor, Memcapacitor and
Capacitor.

Of course, the value for these approximations are not exactly the same. This is due, in part to the fact
that we consider that the most regular approximation is the one for which the function range (F α1,α2
M (t)) is
minimal. Other choices based on physiology of Mushroom could be invoked. Moreover, the Memfractance is
computed on the averaged curve of 20 runs which do not present exactly the same characteristic voltammetry.
Oyster mushroom fruit bodies are biological material, which prevent exact reproduction of electrical property.
2. The use of fractional derivatives to analyze the memfractance, is obvious if one considers that fractional
derivatives have memory, which allow a perfect modeling of memristive elements. Their handling is however
delicate if one wants to avoid any ﬂaw.

Similar I-V characteristics have been experienced for slime mould [15] and apples [36]. The cyclic voltam-
metry experiments demonstrate that the I-V curve produced from these living substrates is a closed loop
where the negative path does not match the positive path. Hence the fungi display the characteristics of a
memristor . A similar conclusion is drawn for the microtubule experiments [8]. The microtubule exhibits
diﬀerent resistive properties for the same applied voltage depending on the history of applied voltages.

Additionally, the fruit bodies produce current oscillations during the cyclic voltammetry. This oscillatory
eﬀect is only observed on one phase of the voltammetry for a given voltage range which is, again, a behaviour
that can be associated to a device whose resistance is a function of its previous resistance. This spiking
activity is typical of a device that exhibits memristive behaviours. Firstly, it was reported in experiments
with electrochemical devices using graphite reference electrodes, that a temporal dependence of the current of
the device - at constant applied voltage - causes charge accumulation and discharge [14]. The spiking is also
apparent in some plots, for a large electrode size, in experiments with electrode metal on solution-processed
ﬂexible titanium dioxide memristors [17]. A detailed analysis of types of spiking emerging in simulated
memristive networks was undertaken in [16]. Repeatable observations of the spiking behaviour in I-V of the
fungi is very important because this opens new pathways for the implementation of neuromorphic computing
with fungi. A fruitful theoretical foundation of this ﬁeld is already well developed [34, 21, 32, 30, 27, 22].

23

-1-0.8-0.6-0.4-0.200.20.40.60.81v(t)-1-0.8-0.6-0.4-0.200.20.40.60.81i(t)#10-6Experimental dataCalculated by fractional model6 Conclusion

The fruit bodies of grey oyster fungi Pleurotus ostreatus were subjected to I-V characterisation a number of
times, from which it was clearly shown that they exhibit mem-fractor properties. Under cyclic voltammetry,
the fruit body will conduct diﬀerently depending on the phase (positive or negative) of the voltammetry.
This behaviour produces the classic “lobes” in the I-V characteristics of a memristor.

However, a biological medium, such as the fruit body of the grey oyster fungi presented here, will diﬀer
from that of the ideal memristor model since the “pinching” behaviour and size of the hysteresis lobes are
functions of the frequency of the voltage sweep as well as the previous resistance. Typically, the biological
medium generates its own potential across the electrodes, therefore, even when no additional potential is
supplied, there is still current ﬂow between the probes. This property of the fungi produces an opening in the
I-V curve that is a classic property of the mem-capacitor. Since the fungi are exhibiting properties of both
memristors and mem-capacitors, their electrical memory behaviour puts them somewhere in the mem-fractor
solution space where 0 < α1, α2 < 1. Hence, it has been shown that fungi act as mem-fractors.

Acknowledgement

This project has received funding from the European Union’s Horizon 2020 research and innovation pro-
gramme FET OPEN “Challenging current thinking” under grant agreement No 858132.

Author contributions

A.A. conceived the idea of experiments. A.A. and A.P. prepared the substrate colonised by mycelium. A.B.
performed experiments, collected data and produced all plots in the manuscript. R.L. and MS.A. derived the
mathematical model of mem-fractance as seen in grey oyster fungi. All authors prepared manuscript (wrote
and reviewed all contents).

References

[1] M.-S. Abdelouahab, R. Lozi, and L. Chua. Memfractance: a mathematical paradigm for circuit elements

with memory. International Journal of Bifurcation and Chaos, 24(9):1430023, 2014.

[2] Andrew Adamatzky. On spiking behaviour of oyster fungi pleurotus djamor. Scientiﬁc reports, 8(1):1–7,

2018.

[3] Andrew Adamatzky. Towards fungal computer. Interface focus, 8(6):20180029, 2018.

[4] Andrew Adamatzky, Phil Ayres, Gianluca Belotti, and Han Wosten. Fungal architecture. arXiv preprint

arXiv:1912.13262, 2019.

[5] Nariman A.Khalil, Lobna A.Said, Ahmed G.Radwan, and Ahmed M.Solimane. General fractional order

mem-elements mutators. International Journal of Bifurcation and Chaos, 90:211–221, 2019.

[6] Julien Borghetti, Zhiyong Li, Joseph Straznicky, Xuema Li, Douglas AA Ohlberg, Wei Wu, Duncan R
Stewart, and R Stanley Williams. A hybrid nanomemristor/transistor logic circuit capable of self-
programming. Proceedings of the National Academy of Sciences, 106(6):1699–1703, 2009.

[7] Julien Borghetti, Gregory S Snider, Philip J Kuekes, J Joshua Yang, Duncan R Stewart, and R Stan-
‘memristive’switches enable ‘stateful’ logic operations via material implication. Nature,

ley Williams.
464(7290):873–876, 2010.

[8] Alessandro Chiolerio, Thomas C. Draper, Richard Mayne, and Andrew Adamatzky. On resistance
switching and oscillations in tubulin microtuble droplets. Journal of Colloid and Interface Science,
560:589–595, Feb 2020.

24

[9] Alessandro Chiolerio, Thomas C Draper, Richard Mayne, and Andrew Adamatzky. On resistance switch-
ing and oscillations in tubulin microtubule droplets. Journal of colloid and interface science, 560:589–595,
2020.

[10] L. Chua. Memristor-the missing circuit element. IEEE Transactions on Circuit Theory, 18(5):507–519,

Sep. 1971.

[11] Leon Chua.
2014.

If it’s pinched it’s a memristor. Semiconductor Science and Technology, 29(10):104001,

[12] Mar´ıa del Roc´ıo Cantero, Paula L Perez, Noelia Scarinci, and Horacio F Cantiello. Two-dimensional

brain microtubule structures behave as memristive devices. Scientiﬁc reports, 9(1):1–10, 2019.

[13] Victor Erokhin, Tatiana Berzina, and Marco P Fontana. Hybrid electronic device based on polyaniline-

polyethyleneoxide junction. Journal of applied physics, 97(6):064501, 2005.

[14] Victor Erokhin and Marco P Fontana. Electrochemically controlled polymeric device: a memristor (and

more) found two years ago. arXiv preprint arXiv:0807.0333, 2008.

[15] Ella Gale, Andrew Adamatzky, and Ben de Lacy Costello. Slime mould memristors. BioNanoScience,

5(1):1–8, 2015.

[16] Ella Gale, Ben de Lacy Costello, and Andrew Adamatzky. Emergent spiking in non-ideal memristor

networks. Microelectronics Journal, 45(11):1401–1415, 2014.

[17] Ella Gale, David Pearson, Steve Kitson, Andrew Adamatzky, and Ben de Lacy Costello. The eﬀect of
changing electrode metal on solution-processed ﬂexible titanium dioxide memristors. Materials Chem-
istry and Physics, 162:20–30, 2015.

[18] JEAN-MARC GINOUX and BRUNO ROSSETTO. The singing arc: The oldest memristor? In Andrew
Adamatzky and Guanrong Chen, editors, Chaos, CNNs, memristors and beyond. World Scientiﬁc, 2013.

[19] Yenpo Ho, Garng M Huang, and Peng Li. Nonvolatile memristor memory: device characteristics and
design implications. In Proceedings of the 2009 International Conference on Computer-Aided Design,
pages 485–490, 2009.

[20] Thang Hoang. Memristor model.

https://www.mathworks.com/matlabcentral/fileexchange/

25082-memristor-model, 2020. MATLAB Central File Exchange. Retrieved January 13, 2020.

[21] Giacomo Indiveri, Bernab´e Linares-Barranco, Robert Legenstein, George Deligeorgis, and Themistoklis
Prodromakis. Integration of nanoscale memristor synapses in neuromorphic computing architectures.
Nanotechnology, 24(38):384010, 2013.

[22] Giacomo Indiveri and Shih-Chii Liu. Memory and information processing in neuromorphic systems.

Proceedings of the IEEE, 103(8):1379–1397, 2015.

[23] DH Jennings. Translocation of solutes in fungi. Biological Reviews, 62(3):215–243, 1987.

[24] Shiv Prasad Kosta, Yogesh P Kosta, Mukta Bhatele, YM Dubey, Avinash Gaur, Shakti Kosta, Jyoti
Gupta, Amit Patel, and Bhavin Patel. Human blood liquid memristor. International Journal of Medical
Engineering and Informatics, 3(1):16–29, 2011.

[25] Shahar Kvatinsky, Dmitry Belousov, Slavik Liman, Guy Satat, Nimrod Wald, Eby G Friedman, Avinoam
Kolodny, and Uri C Weiser. Magic—memristor-aided logic. IEEE Transactions on Circuits and Systems
II: Express Briefs, 61(11):895–899, 2014.

[26] Shahar Kvatinsky, Guy Satat, Nimrod Wald, Eby G Friedman, Avinoam Kolodny, and Uri C Weiser.
Memristor-based material implication (imply) logic: Design principles and methodologies. IEEE Trans-
actions on Very Large Scale Integration (VLSI) Systems, 22(10):2054–2066, 2013.

25

[27] Bernabe Linares-Barranco, Teresa Serrano-Gotarredona, Luis A Camu˜nas-Mesa, Jose A Perez-Carrasco,
Carlos Zamarre˜no-Ramos, and Timothee Masquelier. On spike-timing-dependent-plasticity, memristive
devices, and building a self-learning visual cortex. Frontiers in neuroscience, 5:26, 2011.

[28] Eike Linn, R Rosezin, Stefan Tappertzhofen, U B¨ottger, and Rainer Waser. Beyond von neumann—logic
operations in passive crossbar arrays alongside memory operations. Nanotechnology, 23(30):305205, 2012.

[29] Ø G Martinsen, S Grimnes, CA L¨utken, and GK Johnsen. Memristance in human skin. Journal of

Physics: Conference Series, 224(1):012071, 2010.

[30] Matthew D Pickett, Gilberto Medeiros-Ribeiro, and R Stanley Williams. A scalable neuristor built with

mott memristors. Nature materials, 12(2):114–117, 2013.

[31] I. Podlubny. Fractional Diﬀerential Equations. Academic Press, San Diego, 1999.

[32] Mirko Prezioso, Y Zhong, D Gavrilov, Farnood Merrikh-Bayat, Brian Hoskins, G Adam, K Likharev, and
D Strukov. Spiking neuromorphic networks with metal-oxide memristors. In 2016 IEEE International
Symposium on Circuits and Systems (ISCAS), pages 177–180. IEEE, 2016.

[33] Karl H Sch¨utte. Translocation in the fungi. The New Phytologist, 55(2):164–182, 1956.

[34] Teresa Serrano-Gotarredona, Themistoklis Prodromakis, and Bernabe Linares-Barranco. A proposal for
hybrid memristor-CMOS spiking neuromorphic learning systems. IEEE cIrcuIts and systEms magazInE,
13(2):74–88, 2013.

[35] Dmitri B. Strukov, Gregory S. Snider, Duncan R. Stewart, and R. Stanley Williams. The missing

memristor found. Nature, 453(7191):80–83, May 2008.

[36] A G. Volkov and V S. Markin. Electrochemistry of gala apples: Memristors in vivo. Russian Journal of

Electrochemistry, 53(9):1011–1018, Sept. 2017.

[37] Alexander G Volkov, Clayton Tucket, Jada Reedus, Maya I Volkova, Vladislav S Markin, and Leon

Chua. Memristors in plants. Plant signaling & behavior, 9(3):e28152, 2014.

[38] Z. Yin, H. Tian, G. Chen, and L. O. Chua. What are memristor, memcapacitor, and meminductor?

IEEE Transactions on Circuits and Systems II: Express Briefs, 62(4):402–406, April 2015.

View publication stats

26

