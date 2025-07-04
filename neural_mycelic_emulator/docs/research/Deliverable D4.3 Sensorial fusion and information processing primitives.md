Ref. Ares(2023)3121721 - 04/05/2023

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Horizon 2020

Deliverable  D4.3
Sensorial  fusion  and  information  processing  primitives

Date of preparation: 30/05/2023

Revision: 1

Start date of project: 2019/12/01

Duration:  48  months 

Project coordinator: UWE

Classification: public

Partners:
lead: UWE

contribution: CITA, UU, Mogu

Project website:

http://fungar.eu/

H2020-FETopen-2019

Deliverable D3.3

Page 1 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

DELIVERABLE SUMMARY SHEET

Grant agreement number:

858132

Project acronym:

FUNGAR

Deliverable No:

Deliverable D3.3

Due date:

M48

Delivery date:

30/05/2023

Name:

Sensorial fusion and information processing primitives

Description:

In this deliverable we report about implementation of massive-
parallel computing circuits with simulated and living mycelium
networks. The report includes results on implementation of
action-potential like spiking gates in a single fungal colony, simula-
tion of electrical Boolean circuits, experimental laboratory mining
of Boolean circuits in substrates colonised by fungi and experi-
mental laboratory implementation of fungal circuits in responsive
insoles.

Partners owning:

UWE

Partners contributed:

CITA, UU, Mogu

Made available to:

public

Page 2 of 14

Deliverable D3.3

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Table of contents

1 Background

2 Spikes based Boolean gates in a model of a single fungal colony

3 Electrical analog Boolean gates

4 Experimental mining of Boolean circuits

5 Exemplar application in fungal insoles

6 Conclusion

References

4

5

6

6

12

14

14

Deliverable D3.3

Page 3 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

1 Background

A vibrant field of unconventional computing aims to employ space-time dynamics of physical,
chemical and biological media to design novel computational techniques, architectures and work-
ing prototypes of embedded computing substrates and devices.
Interaction-based computing
devices, is one of the most diverse and promising families of the unconventional computing struc-
tures. They are based on interactions of fluid streams, signals propagating along conductors or
excitation wave-fronts. Typically, logical gates and their cascade implemented in an excitable
medium are ‘handcrafted’ to address exact timing and type of interactions between colliding
wave-fronts. The artificial design of logical circuits might be suitable when chemical media or
functional materials are used. However, the approach might be not feasible when embedding
computation in living systems, where the architecture of conductive pathways may be difficult
to alter or control. In such situations an opportunistic approach to outsourcing computation can
be adopted. The system is perturbed via two or more input loci and its dynamics if recorded
at one or more output loci. A wave-front appearing at one of the output loci is interpreted as
logical truth or ‘1’. Thus the system with relatively unknown structure implements a mapping
{0, 1}n → {0, 1}m, where n is a number of input loci and m is a number of output loci, n, m > 0.
The approach belong to same family of computation outsourcing techniques as in materio com-
puting. Fungal colonies are characterised by rich typology of mycelium networks in some cases
affin to fractal structures. Rich morphological features might imply rich computational abilities
and thus worse to analyse from realising Boolean functions point of view. In numerical experi-
ments we study implementation of logical gates via interaction of numerous travelling excitation
waves, seen as as action potentials, on an image of a real fungal colony. Detailed results related
to the Deliverable are presented in [roberts_mining_2021, 1, 2, 3].

(a)

(b)

(c)

Figure 1: Image of the fungal colony, 1000 × 960 pixels used as a template conductive for FHN.
(a) Original image, mycelium is seen as green pixels. (b) Conductive matrix C, conductive pixels
are black. (c) Configuration of electrodes.

Page 4 of 14

Deliverable D3.3

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 2: Fragment of electrical potential record on electrode 7 in response to inputs (01),
black dashed line, (10), red dotted line, (11), solid green line, entered as impulses via electrodes
Ex = 5 and Ey = 15. See locations of electrodes in Fig. 1d. To make the individual plots visible
in places of exact overlapping, we added potential −5 to recording in response to input (01) and
and potential 5 to recording in response to input (11).

2 Spikes based Boolean gates in a model of a single

fungal colony

A fungal colony maintains its integrity via flow of cytoplasm along mycelium network. This flow,
together with possible coordination of mycelium tips propagation, is controlled by calcium waves
and associated waves of electrical potential changes. We propose that these excitation waves can
be employed to implement a computation in the mycelium networks. We use FitzHugh-Nagumo
model to imitate propagation of excitation in a single colony of Aspergillus niger (Fig. 1ab). The
waves of excitation are recorded by an array of simulated electrodes (Fig. 1c).

Boolean values are encoded by spikes of extracellular potential (Fig. 2). We represent binary
inputs by electrical impulses on a pair of selected electrodes and we record responses of the colony
from sixteen electrodes. We derive sets of two-inputs-on-output logical gates implementable the
fungal colony and analyse distributions of the gates.

In paper [1] we have demonstrated how sets of logical gates can be implemented in single
colony mycelium networks via initiation of electrical impulses. The impulses travel in the network,
interact with each other (annihilate, reflect, change their phase). Thus for different combinations
of input impulses and record different combinations of output impulses, which in some cases can
be interpreted as representing two-inputs-one-output functions.

To estimate a speed of computation we refer to Olsson and Hansson’s [4] original study,
in which they proposed that electrical activity in fungi could be used for communication with
message propagation speed 0.5 mm/sec. Diameter of the colony (Fig. 1a), which experimental
laboratory images has been used to run FHN model, is c. 1.7 mm. Thus, it takes the excitation
waves initiated at a boundary of the colony up to 3-4 sec to span the whole mycelium network
(this time is equivalent to c. 70K iterations of the numerical integration model). In 3-4 sec the
mycelium network can compute up to a hundred logical gates. This gives us the rate of a gate per
0.03 sec, or, in terms of frequency this will be c. 30 Hz. The mycelium network computing can

Deliverable D3.3

Page 5 of 14

SySySySxx+yxyxyx ⊕yPotential, units−50050Time, iterations30,00040,00050,00060,000EU-H2020 FET grant agreement no. 858132 — fungal architectures

not compete with existing silicon architecture however its application domain can be a unique of
living biosensors (a distribution of gates realised might be affected by environmental conditions)
and computation embedded into structural elements where fungal materials are used.

3 Electrical analog Boolean gates

In numerical modelling and experimental laboratory setup we exploited principles of electrical
analog computing [2]. True and False values are represented by above threshold and below
threshold voltages. Due to the non-linearity of the conductive substrate along electrical current
pathways between input and output electrodes, the input voltages are transformed and thus
logical mappings are implemented. Detailed descriptions of these techniques can be found in [2].
The z-stacks of a single colony of Aspergillus niger fungus strain AR9#2 were converted
to a 3D graph (Fig. 3). We modelled the colony as a resistive and capacitive (RC) network.
RC networks are circuits consisting of resistances and capacitors, the most fundamental passive
circuit elements needed to design from a low-pass filter up to an equivalent network of a nerve
cell. The 3D graph was converted to the RC network, whose magnitudes are a function of the
length of the connections.

Resistances were in the order of kOhms and capacitance in the order of pF. The positive
voltage and ground nodes were randomly assigned from the sample and 1000 networks were
created in each arrangement for analysis. SPICE analysis consisted of transient analysis using a
two voltage pulses of 60 mV on the randomly assigned positive nodes. We modelled the fungal
colony in serial RC networks and parallel RC networks. The output voltages were binarised with
the threshold θ: V > θ symbolises logical True otherwise False.

There are 16 possible logical gates realisable for two inputs and one output. The gates
implying input 0 and evoking a response 1, i.e. f (0, 0) = 1, are not realisable because the
simulated fungal circuit is passive. The remaining 8 gates are and, or, and-not (x and not
y and not x and y), select (select x and select y) and xor. In the model of serial RC
networks, we found gates and, select and and-not; no or gates have been found. The number
n of the gates discovered decreases by a power law with increase of θ. The frequency of and gates
oscillates, as shown in the zoom insert in Fig. 4a, most likely due to its insignificant presence in
the samples. The oscillations reach near zero base when θ exceeds 0.001. In the model of parallel
RC networks we only found the gates and, select and or. The number of or gates decreases
quadratically and becomes nil when θ > 0.03. The number of and gates increases near linearly
with increase of θ. The number of select gates reaches its maximum at θ = 0.023, and then
starts to decreases with the further increase of θ.To conclude, mycelium bound composites can
act as computing media and implement a wide range of Boolean circuits, thus opening a new
perspective in biological analog and hybrid computing.

4 Experimental mining of Boolean circuits

Living substrates are capable for nontrivial mappings of electrical signals due to the substrate
nonlinear electrical characteristics. This property can be used to realise Boolean functions. In-
put logical values are represented by amplitude or frequency of electrical stimuli. Output logical
values are decoded from electrical responses of living substrates. We demonstrate how logical
circuits can be implemented in mycelium bound composites. The mycelium bound composites
(fungal materials) are getting growing recognition as building, packaging, decoration and cloth-
ing materials. Presently the fungal materials are passive. To make the fungal materials adaptive,

Page 6 of 14

Deliverable D3.3

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 3: Perspective views of the 3D Graph. Each frame shows the graph after a 36◦ rotation
around the z-axis with origin located approximately in the centre of the colony, on the x − −y
plane indicated with registration marks.

Deliverable D3.3

Page 7 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 4: Occurrences of the gates from the groups and, black, or, green, and-not, red, and
select, blue, for θ ∈ [0.0001, 0.05], with θ increment 0.0001, in (a) fungal colony modelled with
serial RC networks, (b) fungal colony modelled with parallel RC networks.

Figure 5: Left: Schematic of the mycelium communications system; PC — laptop for generating
sequences; CU – control unit, dashed section is a breakdown of a single channel; ADC — analogue
to digital converter. Right: experimental set up.

i.e. sensing and computing, we should embed logical circuits into them. We demonstrate experi-
mental laboratory prototypes of many-input Boolean functions implemented in fungal materials
from oyster fungi P. ostreatus. We characterise complexity of the functions discovered via com-
plexity of the space-time configurations of one-dimensional cellular automata governed by the
functions. We show that the mycelium bound composites can implement representative functions
from all classes of cellular automata complexity including the computationally universal. The
results presented will make an impact in the field of unconventional computing, experimental
demonstration of purposeful computing with fungi, and in the field of intelligent materials, as
the prototypes of computing mycelium bound composites.

A hemp shavings substrate was colonised by the mycelium of the grey oyster fungi, P. ostreatus
(Ann Miller’s Speciality Mushrooms Ltd, UK). Recordings were carried out in a stable indoor

Page 8 of 14

Deliverable D3.3

ANDSELECTAND-NOTNumber of gates05×10410×10415×104θ00.010.020.030.040.05010002000300000.0020.0040.0060.00845,00050,00055,00000.001ANDORSELECTNumber of gates05×10610×10615×10620×10625×106θ00.010.020.030.040.05EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 6: Timing diagram and associated Boolean strings for four inputs into the mycelium
substrate, time step is one hour.

environment with the temperature remaining stable at 22 ± 0.5°and relative humidity of air
40 ± 5%. The humidity of the substrate colonised by fungi was kept at c. 70-80%.

Hardware was developed that was capable of sending sequences of 4 bit strings to a mycelium
substrate. The strings were encoded as step voltage inputs where -5 V denoted a logical 0 and
5 V a logical 1. The hardware was based around an Arduino Mega 2560 (Elegoo, China) and a
series of programmable signal generators, AD9833 (Analog, USA). The 4 input electrodes were
1 mm diameter platinum rods inserted to a depth of 50 mm in the substrate in a straight line
with a separation of 20 mm. Data acquisition (DAQ) probes were placed in a parallel line 50 mm
away separated by 10 mm. The electron sink and source was placed 50 mm on from DAQ probes.
There were 7 DAQ differential inputs from the mycelium substrate to a Pico 24 (Pico Technology,
UK) analogue-to-digital converter (ADC), the 8th channel was used to pass a pulse to the ADC
on every input state change, see Fig. 5 for a schematic of the apparatus. The substrate and
probes were placed in a semi-sealed container. After each experimental repeat the substrate was
sprayed with water, left for an hour and then the next repeat was conducted. There were a total
of 14 repeats.

A sequence of 4 bit strings counting up from binary 0000 to 1111, with a state change every
hour, were passed into the substrate, see Fig. 6 for timing details.
In all 14 repeats of the
experiment were done on the same substrate to capture changes in structure of the growing
mycelium. Samples from 7 channels were taken at 1 Hz over the whole duration of a given
experimental run. Peaks for each channel were located for a set of 32 thresholds, from 20 mV to
175 mV with step 5 mV, for each input state, 0000 to 1111.

Boolean strings were extracted from the data, where a logic ‘1’ was noted for a channel if it
had a peak outside the threshold band for a particular state else, a value of ‘0’ was recorded, the
polarity of the peak was not considered.

The strings for each experimental repeat were stored in their respective Boolean table. To
extract state graphs, a state/node was defined as the string of output values from each channel
at each input state, transitions/edges were defined as a change in input state. This led to a total
of 448 state graphs. The sum of products (SOP) Boolean functions were calculated for each
output channel. For each repeat there were 7 channels and 32 thresholds giving total of 3136
individual truth tables.

See Fig. 7 for SOP extraction. If a peak is found in Fig. 7a during an input state then this
is considered a logical 1, highlighted in yellow in table Fig. 7b are the thresholded values for

Deliverable D3.3

Page 9 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 7: Workflow example. (a) The measurements taken by channel 5 of the DAQ in blue,
the synchronisation signal is shown red which marks the state change, threshold band shown in
green, peaks outside this band are highlighted with ‘x’ marker. (b) The truth and the function
extracted.

Page 10 of 14

Deliverable D3.3

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 8: Counts of realised Boolean functions discovered in laboratory experiments. Horizontal
axis is a decimal representation of functions. Vertical axis is a number of functions discovered in
experiments.

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
AB + AC + AD + AB + BC + BD + AC + BC + CD + AD + BD + CD
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

channel 5, the resulting truth table is then reduced to a sum products shown below the table.

We have discovered total of 3136 4-inputs-1-output Boolean functions. 470 unique functions
are presented in Supplementary Materials. Figure 8 shows the Boolean function distribution.
The two peak values were logical False, n = 238, and logical True, n = 237. The highest
occurring non-trivial gate was A + B + C + D, n = 145. The top 16 occurring non-trivial

Deliverable D3.3

Page 11 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Boolean functions are listed in table 1. The only single gate functions found were for nand
(A + B + C + D), n = 145, or (A + B + C + D), n = 46, and and (ABCD), n = 8. More details
are provided in [5].

5 Exemplar application in fungal insoles

Whilst living fungal Boolean circuits could be used in computing devices in future, it would be
useful to test have some application domains at present. Here we represent results of experiments
on living fungal insoles, further detail can be found here [3].

Figure 9: (a) capillary matting cut into insole pattern (b) insole on bed of spawn (c) well colonised
insole

Figure 10: Bespoke insole test rig (a) setup inside growth tent (b) weight uniformly distributed
via pivot joint on prosthetic foot (c) heel bias (d) toes bias.

Mycelium bound composites are promising materials for a diverse range of applications in-
cluding wearables and building elements. Their functionality surpasses some of the capabilities

Page 12 of 14

Deliverable D3.3

EU-H2020 FET grant agreement no. 858132 — fungal architectures

of traditionally passive materials, such as synthetic fibres, reconstituted cellulose fibres and nat-
ural fibres. Thereby, creating novel propositions including augmented functionality (sensory)
and aesthetic (personal fashion). Biomaterials can offer multiple modal sensing capability such
as mechanical loading (compressive and tensile) and moisture content. To assess the sensing po-
tential of fungal insoles we undertook laboratory experiments on electrical response of bespoke
insoles made from capillary matting colonised with oyster fungi Pleurotus ostreatus (Fig. 9) to
compressive stress which mimics human loading when standing and walking (Fig. 10). We have
shown changes in electrical activity with compressive loading. The results advance the develop-
ment of intelligent sensing insoles which are a building block towards more generic reactive fungal
wearables. Using FitzHugh-Nagumo model we numerically illustrated how excitation wave-fronts
behave in a mycelium network colonising an insole and shown that it may be possible to discern
pressure points from the mycelium electrical activity.

Electrical activity (spiking) was recorded in mycelium bound composites fabricated into in-
soles. The number and periodicity of electrical spikes change when the mycelium is subjected to
compressive loading. We have shown that it might be possible to discern the loading from the
electrical response of the fungi to stimuli [3]. The results advance the development of intelligent
sensing insoles which are a building block towards more generic reactive fungal wearables. Elec-
trical activity changes in both spatial and temporal domains. Using FitzHugh-Nagumo model we
numerically illustrated how excitation wave-fronts behave in a mycelium network colonising an
insole and shown that it might be possible to discern pressure points from the mycelium electrical
activity. Fungal based insoles offer augmented functionality (sensory) and aesthetic (personal
fashion). We presented results of scoping experiments on living biowearables. The results open
new horizons in exploring feasibility of living fungal materials in everyday life. Directions of
future research will involve bench marking of the prototypes and testing them in real life.

Deliverable D3.3

Page 13 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

6 Conclusion

In a series of computer models and laboratory experiments we have demonstrated that mycelium
networks are capable for sensorial fusion, information processing and realisation of logical circuits.
The findings open new exciting pathways for integration of sensing and computing fungal devices
into fungal materials, buildings and architectures.

References

[1] Andrew Adamatzky et al. “On boolean gates in fungal colony”. In: Biosystems 193 (2020),

p. 104138.

[2] Alexander E Beasley et al. “On electrical gates on fungal colony”. In: Biosystems 209 (2021),

p. 104507.

[3] Anna Nikolaidou et al. “Responsive fungal insoles for pressure detection”. In: Scientific

Reports 13.1 (2023), p. 4595.

[4] Stefan Olsson and BS Hansson. “Action potential-like activity found in fungal mycelia is

sensitive to stimulation”. In: Naturwissenschaften 82 (1995), pp. 30–31.

[5] Nic Roberts and Andrew Adamatzky. “Mining logical circuits in fungi”. In: Scientific Reports

12.1 (2022), p. 15930.

Page 14 of 14

Deliverable D3.3

