EU-H2020 FET grant agreement no. 858132 — fungal architectures

Ref. Ares(2022)79090 - 05/01/2022

Horizon 2020

Deliverable D4.2
A catalogue of electrical activity patterns related to
chemical and physical stimulation

Date of preparation: 2021/11/30

Revision: 1

Start date of project: 2019/12/01 Duration: 36 months

Project coordinator: UWE

Classification: public

Partners:
lead: UWE

contribution: CITA, UU, Mogu

Project website:

http://fungar.eu/

Deliverable D4.2

Page 1 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

H2020-FETopen-2019

Page 2 of 14

Deliverable D4.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

DELIVERABLE SUMMARY SHEET

Grant agreement number:

858132

Project acronym:

FUNGAR

Deliverable No:

Deliverable D4.2

Due date:

M24

Delivery date:

2021/11/30

Name:

Description:

A catalogue of electrical activity patterns related to
chemical and physical stimulation

In this deliverable, we report on activities and results related to
WP4 as defined in the scope of works. This includes uncovering
patterns of electrical activity related to chemical and mechanical
stimulation. These results provide the basis for design and pro-
totyping of chemical, optical and mechanical sensors and imple-
mentation of inputs into future reservoir computing devices made
with living mycelium bound composites.

Partners owning:

UWE

Partners contributed:

CITA, UU, Mogu

Made available to:

public

Deliverable D4.2

Page 3 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Table of contents

1 Background

2 Endogeneous spiking: Fungal oscillators

3 Response to electrical stimulation: fungal memristors

4 Response to mechanical stimulation: Fungal pressure sensor

5 Response to optical stimulation: Fungal photosensor

6 Response to chemical stimulation: Fungal chemical sensor

7 Conclusion

5

6

7

9

10

10

12

Page 4 of 14

Deliverable D4.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 1: A scheme off the electronic interface with fungi.

1 Background

Spikes of electrical potential are typically considered to be key attributes of neurons and neuronal
spiking activity is interpreted as a language of a nervous system. However, almost all creatures
without nervous system produce spikes of electrical potential — Protozoa [15], Hyrdoroza [?],
slime moulds [26, 27] and plants [31, 22, 35]. Fungi also exhibit trains of action-potential like
spikes, detectable by intra- and extra-cellular recordings [30, 29, 2]. In experiments with recording
of electrical potential of oyster fungi Pleurotus djamor we discovered two types of spiking activity:
high-frequency (period 2.6 min) and low-freq (period 14 min) [2]. While studying other species
of fungus, Ganoderma resinaceum, we found that most common width of an electrical potential
spike is 5-8 min [5]. In both species of fungi we observed bursts of spiking in the trains of the
spike similar to that observed in central nervous system. Whilst the similarly could be just
phenomenological this indicates a possibility that mycelium networks transform information via
interaction of spikes and trains of spikes in manner homologous to neurons. First evidence has
been obtained that indeed fungi respond to mechanical, chemical and optical stimulation by
changing pattern of its electrically activity and, in many cases, modifying characteristics of their
spike trains [7, 9]. There is also evidence of electrical current participation in the interactions
between mycelium and plant roots during formation of mycorrhiza [13]. In [20] we compared
complexity measures of the fungal spiking train and sample text in European languages and
found that the ’fungal language’ exceeds the European languages in morphological complexity.
In our venture to decode the language of fungi a first step would be to uncover if all species of
fungi exhibit similar characteristics of electrical spiking activity.

In this article we catalogue electrical activity of living mycelium bound composites based

Deliverable D4.2

Page 5 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 2: (a) Examples of high amplitude and high frequency spikes. (b) Oscillation of electrical
potential under 10 V DC applied, where spikes analysed are marked by ‘*’. From [8].

on their responses to electrical, mechanical, chemical and optical stimulation (Fig. 1).. The
characterisation provided aims to produce practical devices from mycelium bound composites.
The device include oscillators, memristors, photosensors, pressure sensors and chemical sensors.

2 Endogeneous spiking: Fungal oscillators

An electronic oscillator is a device which converts direct current to an alternating current signal.
A fungal oscillator is based on endogenous oscillations of an electrical resistance of mycelium
bound composites. A nearly homogeneous sheet of mycelium of P. ostreatus, grown on the surface
of a growth substrate, exhibits trains of resistance spikes (Fig. 2(a)) [8]. The average width of
spikes is c. 23 min and the average amplitude is c. 1 kΩ. The distance between neighbouring
spikes in a train of spikes is c. 30 min. Typically there are 4-6 spikes in a train of spikes. Two
low frequency and high
types of electrical resistance spikes trains are found in fruit bodies:
amplitude (28 min spike width, 1.6 kΩ amplitude, 57 min distance between spikes) and high
frequency and low amplitude (10 min width, 0.6 kΩ amplitude, 44 min distance between spikes).

Page 6 of 14

Deliverable D4.2

Resistance, Ohm1.6×1051.7×1051.8×1051.9×1052.0×105Time, ×10 sec8400845085008550860086508700***s***Potential, V7.8807.8857.890Time, ×10 sec20002500300035004000EU-H2020 FET grant agreement no. 858132 — fungal architectures

To assess feasibility of the living fungal oscillator, we conducted a series of scoping experiments
by applying direct voltage to the fungal substrate and measuring output voltage. An example of
the electrical potential of a substrate colonised by fungi under 10 V applied is shown in Fig. 2(b).
Voltage spikes are clearly observed. Spikes with amplitude above 1 mV, marked by ‘*’, except
the spike marked by ‘s’ have been analysed. We can see two trains of three spikes each. Average
width of the spikes is 103 sec, average amplitude 2.5 mV, while average distance between spikes
is c. 2 · 103K sec. To conclude, fungi can be used as a very low frequency electronic oscillators
in designs of biological circuits.

One of the feasible explanations of the resistance oscillations could be in the translocation
of water and metabolites taking place in the mycelium. This translocation is periodic, and
more likely guided by calcium waves. Increase in a liquid in the mycelium loci leads to reduced
resistance. When the translocated mass of metabolites leaves the area, the resistance increases.

3 Response to electrical stimulation: fungal memris-

tors

A memristor, also known as Resistive Switching Device (RSD), is a two or three-terminal device
whose resistance depends on one or more internal state variables of the device [4]. A memris-
tor is defined by a state-dependent Ohm’s law. Its resistance depends on the entire past signal
waveform of the applied voltage, or current, across the memristor. Using memristors, one can
achieve circuit functionalities that it is not possible to establish with resistors, capacitors and
inductors, therefore the memristor is of great pragmatic usefulness. Potential unique applica-
tions of memristors have been enabled by their physical implementation and are expected to
occur in spintronic devices, ultra-dense information storage, neuromorphic circuits, human brain
interfaces and programmable electronics [19, 18].

Memristive properties of living creatures, their organs and fluids have been demonstrated
in skin, blood, plants, slime mould, tubulin microtubules, see details in [12]. A mechanism of
the memristance is likely in the relocation of ions and temporary physical changes of the cell
membranes.

In experimental laboratory studies (see the setup in Figs. 3(a) and 3(b)), we demonstrated
that P. ostreatus fruit bodies exhibit memristive properties when subject to a voltage sweep [12].
The ideal memristor model has a crossing point at 0V, where theoretically no current flows.
Figures 3(c) and 3(d) show the results of cyclic voltammetry of grey oyster mushrooms with
electrodes positioned in the caps and/or stems. When 0 V is applied by the source meter, a
reading of a nominally small voltage and current is performed.

When the sample under test is subjected to a positive voltage (quadrant 1), it can be seen
there is nominally a positive current flow. Higher voltages result in a larger current flow. For
an increasing voltage sweep there is a larger current flow for the corresponding voltage during
a negative sweep. In quadrant 3 where there is a negative potential across the electrodes, the
increasing voltage sweep yields a current with smaller magnitude than the magnitude of the
current on a negative voltage sweep. Put simply, the fruit body has a resistance that is a
function of the previous voltage conditions.

The living membrane is capable of generating potential across the electrodes, and hence a
small current is observed. To conclude, living fungi can be used as memristors (resistors with
memory) in biocomputing circuits.

Deliverable D4.2

Page 7 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

(d)

Figure 3: Fungal memristors. (ab) Positions of electrodes in fruit bodies. (a) Electrodes inserted
10 mm apart in the fruit body cap. (b) One electrode is inserted in the cap with the other in the
stem. (cd) Raw data from cyclic voltammetry performed over -0.5 V to 0.5 V. (c) Cap-to-cap
electrode placement. (d) Stem-to-cap electrode placement. From [12].

Page 8 of 14

Deliverable D4.2

-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage [V]-1-0.500.51Current [A]10-7Oyster mushroom fruit bodies with cap to cap electrodes-0.5-0.4-0.3-0.2-0.100.10.20.30.40.5Voltage [V]-1-0.500.51Current [A]10-7Oyster mushroom fruit bodies with stem to cap electrodesEU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 4: (a) Experimental setup. Pairs of differential electrodes inserted in a fungal block and
16 kg kettle bell placed on top of the fungal block. Channels are from the top right clockwise
(1-2), (3-5), . . . , (15-16). (b) The activity of the block stimulated with 16 kg load. Moments of
the loads applications are labelled by ‘ON’ and lifting the loads by ‘OFF’. Channels are colour
coded as (1-2) – black, (3-4) – red, (5-6) – blue, (7-8) – green, (9-10) – magenta, (11-12)– orange,
(13-14) – yellow. From [6].

4 Response to mechanical stimulation: Fungal pressure

sensor

We stimulated blocks of G. resinaceum mycelium colonised substrate by placing a 16 kg cast iron
weight on their top face (Fig. 4(a)). Electrical activity of the fungal composite block was recorded

Deliverable D4.2

Page 9 of 14

ONOFFONOFFONOFFONOFFONOFFONOFFONOFFPotential, mV−4−20246810121416Time, sec01×1052×1053×1054×1055×105−1012.4×1052.6×1052.8×1053.0×1053.2×105EU-H2020 FET grant agreement no. 858132 — fungal architectures

using 8 pairs of differential electrodes, as specified in Fig. 4(a). An example of electrical activity
recorded on 8 channels, during the stimulation with 16 kg weight, is shown in Fig. 4(b) [6]. In
response to application of 16 kg weight the fungal blocks produced spikes with median amplitude
1.4 mV and median duration 456 sec; average amplitude of ON spikes was 2.9 mV and average
duration 880 sec. OFF spikes were characterised by median amplitude 1 mV and median duration
216 sec; average amplitude 2.1 mV and average duration 453 sec. ON spikes are 1.4 higher
than and twice as longer as OFF spikes. Based on this comparison of the response spikes we
can claim that fungal blocks recognise when a weight was applied or removed [6]. The results
complement our studies on tactile stimulation of fungal skin (mycelium sheet with no substrate)
[7]: the fungal skin responds to application and removal of pressure with spikes of electrical
potential. The fungal blocks can discern whether a weight was applied or removed because
the blocks react to the application of weights with higher amplitude and longer duration spikes
than the spikes responding to the removal of the weights. The fungal responses to stimulation
show habituation. This is in accordance with previous studies on stimulation of plants, fungi,
bacteria, and protists [10, 23, 25, 16, 34]. To conclude, mycelium bound composites are capable
of detecting pressure, therefore fungal pressure sensors can be incorporated into living loci of
fungal building materials.

Electrical response of mycelium bound composites to mechanical pressure could be caused by
polarisation of the cell membranes caused by mechanical deformation and blockade of calcium
waves pathways due to mechanical constriction of mycelium strands.

5 Response to optical stimulation: Fungal photosensor

Fungal response to illumination was analysed using a fungal skin — a 1.5 mm thick sheet of pure
mycelium of G. resinaceum fungi (Fig. 5(a)) [7]. The response of the fungal skin to illumination
is manifested in the raising of the baseline potential, as illustrated in the exemplar recordings in
Fig. 5(b). The response-to-illumination spike does not subside but the electrical potential stays
raised until illumination is switched off. An average amplitude of the response is 0.6 mV. The rise
in potential starts immediately after the illumination is switched on. The potential saturation
time is c. 3 · 103 sec on average; the potential relaxation time is c. 3 · 103 sec. Typically, we did
not observe any spike trains after the illumination was switched off, however, in a couple of trials
we witnessed spike trains on top of the raised potential, as shown in Fig. 5(c). To conclude,
living fungal materials respond to illumination by changing their electrical activity, therefore
fungal materials can be incorporated in logical circuits and actuators with optical inputs.

Electrical responses of fungi to illumination are due photosensitive nature of the fungi, with
research showing fungi can be more photosensitive than green plants [17, 24]. Fungi are most
receptive towards the UV end of the spectrum but exhibit photoresponses across the entire light
spectrum. Briefly exposing fungi to light can interrupt their current growth cycle, triggering
other responses.

6 Response to chemical stimulation: Fungal chemical

sensor

We demonstrated that hemp pads colonised by the fungus P. ostreatus (Fig. 6(a)) show distinctive
sets of responses to chemical stimulation [9, 21]. We stimulated colonised hemp pads with 96%
ethanol, malt extract powder (Sigma Aldrich, UK) dissolved in distilled water, dextrose (Ritchie
Products Ltd, UK) and hydrocortisone (Solu-Cortef trademark, 4 mL Act-O-Vial, Pfizer, Athens,

Page 10 of 14

Deliverable D4.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

Figure 5: Fungal response to optical stimulation. (a) A photograph of electrodes inserted into
the fungal skin. (b) Exemplar response of fungal skin to illumination, recorded on three pairs
of differential electrodes. ‘L*’ indicates illumination is applied, ‘Lo’ illumination is switched off.
() A train of spikes on the raised potential as a response to illumination. From [7].

(a)

(b)

Figure 6: (a) Experimental setup. Exemplar locations of electrodes. (b) Response to application
of dextrose. The moment of application is shown by asterisk.

Deliverable D4.2

Page 11 of 14

L*LoPotential, mV−1012Time, sec20,00025,00030,000L*LoPotential, mV1.21.41.61.82.02.2Time, sec35,00040,000*Potential, mV0123456Time, sec02×1044×1046×1048×10410×10412×10414×104EU-H2020 FET grant agreement no. 858132 — fungal architectures

Greece). An example of the response to chemical stimulation is shown in Fig. 6(b). A response
to stimulation with ethanol is characterised by a drop of electrical potential, up to 8 mV, followed
by repolarisation phase, lasting for up to 15 sec. Fungi respond to the application of nutrients
by increasing the frequency of electrical potential spiking [9]. Exposure to hydrocortisone leads
to a series of electrical disturbance events propagating along the mycelium networks with further
indications of suppressed electrical activity [21]. Fungal chemical sensors show a great potential
for future applications, however substantial research should be invested in their calibration.

7 Conclusion

Practical implementations of the electronic properties of fungi would be in sensorial and comput-
ing circuits embedded into mycelium bound composites. For example, an approach of exploiting
reservoir computing for sensing [11], where the information about the environment is encoded in
the state of the reservoir memristive computing medium, can be employed to prototype sensing-
memristive devices from living fungi. A very low frequency of fungal electronic oscillators does
not preclude us from considering inclusion of the oscillators in fully living or hybrid analog cir-
cuits embedded into fungal architectures [3] and future specialised circuits and processors made
from living fungi functionalised with nanoparticles, as have been illustrated in prototypes of
hybrid electronic devices with slime mould [33, 32, 28, 1, 14]. Potential devices made of living
fungi might include environmental sensors integrated in building structures [3] and wearables [9],
patches monitoring chemical parameters of human body [21].

References

[1] Andrew Adamatzky. Twenty five uses of slime mould in electronics and computing: Survey.

International Journal of Unconventional Computing, 11, 2015.

[2] Andrew Adamatzky. On spiking behaviour of oyster fungi pleurotus djamor. Scientific

reports, 8(1):1–7, 2018.

[3] Andrew Adamatzky, Phil Ayres, Gianluca Belotti, and Han Wösten. Fungal architecture

position paper. International Journal of Unconventional Computing, 14, 2019.

[4] Andrew Adamatzky and Leon Chua, editors. Memristor networks. Springer Science &

Business Media, 2013.

[5] Andrew Adamatzky and Antoni Gandia. On electrical spiking of ganoderma resinaceum.

Biophysical Reviews and Letters, 0(0):1–9, 0.

[6] Andrew Adamatzky and Antoni Gandia. Living mycelium composites discern weights. arXiv

preprint arXiv:2106.00063, 2021.

[7] Andrew Adamatzky, Antoni Gandia, and Alessandro Chiolerio. Fungal sensing skin. Fungal

biology and biotechnology, 8(1):1–6, 2021.

[8] Andrew Adamatzky and Jeff Jones. On electrical correlates of physarum polycephalum
spatial activity: Can we see physarum machine in the dark? Biophysical Reviews and
Letters, 6(01n02):29–57, 2011.

[9] Andrew Adamatzky, Anna Nikolaidou, Antoni Gandia, Alessandro Chiolerio, and Mo-

hammad Mahdi Dehshibi. Reactive fungal wearable. Biosystems, 199:104304, 2021.

Page 12 of 14

Deliverable D4.2

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[10] Philip B Applewhite. Learning in bacteria, fungi, and plants. Invertebrate learning, 3:179–

186, 1975.

[11] Vasileios Athanasiou and Zoran Konkoli. On using reservoir computing for sensing ap-
plications: exploring environment-sensitive memristor networks. International Journal of
Parallel, Emergent and Distributed Systems, 33(4):367–386, 2018.

[12] Alexander E Beasley, Mohammed-Salah Abdelouahab, René Pierre Lozi, Michail Antisthenis
Tsompanas, Anna Powell, and Andrew Adamatzky. Mem-fractive properties of mushrooms.
Bioinspiration & biomimetics, 2021.

[13] RLL Berbara, BM Morris, HMAC Fonseca, B Reid, NAR Gow, and MJ Daft. Electrical
currents associated with arbuscular mycorrhizal interactions. New phytologist, 129(3):433–
438, 1995.

[14] Tatiana Berzina, Alice Dimonte, Angelica Cifarelli, and Victor Erokhin. Hybrid slime mould-
International Journal of General Systems,

based system for unconventional computing.
44(3):341–353, 2015.

[15] MS Bingley. Membrane potentials in amoeba proteus. Journal of Experimental Biology,

45(2):251–267, 1966.

[16] Aurèle Boussard, Julie Delescluse, Alfonso Pérez-Escudero, and Audrey Dussutour. Memory
inception and preservation in slime moulds: the quest for a common mechanism. Philosoph-
ical Transactions of the Royal Society B, 374(1774):20180368, 2019.

[17] M J Carlile. The Photobiology of Fungi, volume 16. Annual Reviews, 1965.

[18] Alessandro Chiolerio, Michela Chiappalone, Paolo Ariano, and Sergio Bocchini. Coupling
resistive switching devices with neurons: State of the art and perspectives. Frontiers in
Neuroscience, 11:70, 2017.

[19] Leon Chua, Georgios Ch Sirakoulis, and Andrew Adamatzky. Handbook of Memristor Net-

works. Springer Nature, 2019.

[20] Mohammad Mahdi Dehshibi and Andrew Adamatzky. Electrical activity of fungi: Spikes

detection and complexity analysis. Biosystems, 203:104373, 2021.

[21] Mohammad Mahdi Dehshibi, Alessandro Chiolerio, Anna Nikolaidou, Richard Mayne, Ant-
oni Gandia, Mona Ashtari-Majlan, and Andrew Adamatzky. Stimulating fungi pleurotus
ostreatus with hydrocortisone. ACS Biomaterials Science & Engineering, 7(8):3718–3726,
2021.

[22] Jörg Fromm and Silke Lautner. Electrical signals and their physiological significance in

plants. Plant, cell & environment, 30(3):249–257, 2007.

[23] Yu Fukasawa, Melanie Savoury, and Lynne Boddy. Ecological memory and relocation de-
cisions in fungal mycelial networks: responses to quantity and location of new resources.
The ISME journal, 14(2):380–388, 2020.

[24] Masaki Furuya. Photobiology of fungi. In: Kendrick R.E., Kronenberg G.H.M. (eds) Pho-

tomorphogenesis in plants. Springer, Dordrecht, 1986.

[25] Simona Ginsburg and Eva Jablonka. Evolutionary transitions in learning and cognition.

Philosophical Transactions of the Royal Society B, 376(1821):20190766, 2021.

Deliverable D4.2

Page 13 of 14

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[26] Tatsuichi Iwamura. Correlations between protoplasmic streaming and bioelectric potential
of a slime mold, Physarum polycephalum. Shokubutsugaku Zasshi, 62(735-736):126–131,
1949.

[27] Noburo Kamiya and Shigemi Abe. Bioelectric phenomena in the myxomycete plasmodium
and their relation to protoplasmic flow. Journal of Colloid Science, 5(2):149–163, 1950.

[28] Vasileios Ntinas, Ioannis Vourkas, Georgios Ch Sirakoulis, and Andrew I Adamatzky.
Oscillation-based slime mould electronic circuit model for maze-solving computations. IEEE
Transactions on Circuits and Systems I: Regular Papers, 64(6):1552–1563, 2017.

[29] S Olsson and BS Hansson. Action potential-like activity found in fungal mycelia is sensitive

to stimulation. Naturwissenschaften, 82(1):30–31, 1995.

[30] Clifford L Slayman, W Scott Long, and Dietrich Gradmann.

in
Neurospora crassa, a mycelial fungus. Biochimica et Biophysica Acta (BBA)-Biomembranes,
426(4):732–744, 1976.

“Action potentials”

[31] Kazimierz Trebacz, Halina Dziubinska, and Elzbieta Krol. Electrical signals in long-distance
communication in plants. In Communication in plants, pages 277–290. Springer, 2006.

[32] Xavier Alexis Walter, Ian Horsfield, Richard Mayne, Ioannis A Ieropoulos, and Andrew
Adamatzky. On hybrid circuits exploiting thermistive properties of slime mould. Scientific
reports, 6:23924, 2016.

[33] James GH Whiting, Richard Mayne, Nadine Moody, Ben de Lacy Costello, and Andrew
Adamatzky. Practical circuits with physarum wires. Biomedical Engineering Letters,
6(2):57–65, 2016.

[34] Kiichi Yokochi et al. An investigation on the habituation of amoeba. Aichi Igakkwai Zasshi=

Jl. Aichi Med. Soc., 33(3), 1926.

[35] Matthias R Zimmermann and Axel Mithöfer. Electrical long-distance signaling in plants. In
Long-Distance Systemic Signaling and Communication in Plants, pages 291–308. Springer,
2013.

Page 14 of 14

Deliverable D4.2

