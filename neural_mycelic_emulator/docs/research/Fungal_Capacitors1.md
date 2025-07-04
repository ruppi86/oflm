See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/339997867

Capacitive storage in mycelium substrate

Preprint · March 2020

DOI: 10.48550/arXiv.2003.07816

CITATIONS
0

3 authors, including:

Alexander E Beasley

University of Hertfordshire

28 PUBLICATIONS   138 CITATIONS   

SEE PROFILE

READS
1,291

Andrew Adamatzky

University of the West of England, Bristol

922 PUBLICATIONS   14,614 CITATIONS   

SEE PROFILE

All content following this page was uploaded by Alexander E Beasley on 18 March 2020.

The user has requested enhancement of the downloaded file.

Capacitive storage in mycelium substrate

Alexander E. Beasley∗1, Anna L. Powell1, and Andrew Adamatzky1

1Unconventional Computing Laboratory, UWE, Bristol, UK

March 18, 2020

Abstract

The emerging ﬁeld of living technologies aims to create new functional hybrid materials in which living
systems interface with artiﬁcial ones. Combining research into living technologies with emerging devel-
opments in computing architecture has enabled the generation of organic electronics from plants and
slime mould. Here, we expand on this work by studying capacitive properties of a substrate colonised
by mycelium of grey oyster fungi, Pleurotus ostreatus. Capacitors play a fundamental role in traditional
analogue and digital electronic systems and have a range of uses including sensing, energy storage and
ﬁlter circuits. Mycelium has the potential to be used as an organic replacement for traditional capacitor
technology. Here, were show that the capacitance of mycelium is in the order of hundreds of pico-Farads.
We also demonstrate that the charge density of the mycelium ‘dielectric’ decays rapidly with increasing
distance from the source probes. This is important as it indicates that small cells of mycelium could be
used as a charge carrier or storage medium, when employed as part of an array with reasonable density.

Keywords: fungi, capacitance, mycelium, storage, biocomputing

1

Introduction

The study of novel substrates for sensing, storing and processing information draws on work from the ﬁelds of
unconventional computing, living technology and organic electronics. The ﬁeld of unconventional computing
aims to deﬁne the principles of information processing in living, physical and chemical systems and applies
this knowledge to the development of future computing devices and architectures [1]. Research into living
technologies is focused on the co-functional integration of animate and non-organic systems [9]. Finally, or-
ganic electronics [54, 31] looks to use naturally occurring materials as analogues to traditional semi-conductor
circuits [57], which often requires functionalisation using polymers or metallic compounds to exploit ionic
movement [34]. The development of organic electronics promises a technology that is low-cost and has low
production temperature requirements [37]. However, diﬃculties such as relatively low gain of organic tran-
sistors (approx. 5), and the behavioural variability inherently means that there is a large amount of research
eﬀort being placed into organic thin ﬁlm transistors [66, 59, 24, 58], organic LEDs [52, 40], and organic
capacitors [35, 51, 41]. The capacitive properties of a device allows it to either store energy or react to
AC/DC signals diﬀerently. There are a number of applications in which this property may be utilised, such
as energy harvesting [7], memory [60], or ﬁlter circuits [16]. Hybrid electronic circuits are a concept that
looks to combine traditional silicon, semi-conductor devices with elements found in nature [36, 8].

The capacitive properties of living tissues [38] have a wide range of potential applications, e.g. the estima-
tion of a plan root system size [18, 45], quantifying the DNA content of eukaryotic cells [56], analysing water
transport pathways in plants [13], measuring heat injury in plants [65], measuring contents of minerals in
bones [62], gauging ﬁrmness of apples [11], sugar contents of citrus fruits [12], maturity of avocados [64], esti-
mating depth of epidermal barriers [15], studies of endo- and exocytosis of single cells [48], and approximating
mass and morphology of microbial colonies [25, 42, 53].

∗Corresponding author: Alexander Beasley, alex.beasley@uwe.ac.uk

1

In the present study we focus on the capacitive properties of the mycelium of the grey oyster fungi

Pleurotus ostreatus for several reasons.

Firstly, research into the capacitive properties of fungi is lacking, despite their huge potential for bioelec-
tronic applications. Fungi are the largest, most widely distributed and oldest group of living organisms on
the planet [17]. The smallest fungi are microscopic single cells, the largest, Armillaria bulbosa, occupies 15
hectares and weighs 10 tons [55]. Fungi sense light, chemicals, gases, gravity and electric ﬁelds [5] as well
as demonstrating mechanosensing behaviour [32]. Thus, their electrical properties can be tuned via various
inputs.

Secondly, fungi have the potential to be used as distributed living computing devices, i.e.

large-scale
networks of mycelium, which collect and analyse information about environment and execute some decision
making processes [2].

Finally, there is a growing interest in developing buildings from pre-fabricated blocks of substrates
colonised by fungi [50, 4, 20]. A recent initiative aims to grow monolithic constructions in which living
mycelium coexists with dried mycelium, functionalised with nanoparticles and polymers [3]. In such a case,
fungi could be act as optical, tactile and chemical sensors, fuse and process information and perform decision
making computations [2].

Providing local charge to areas of mycelium allows the storage of information inside the substrate. Iden-
tifying the area around which the induced charge can be detected allows the construction of an array in the
substrate where each cell can contain individual bits of information. Determining the capacitive properties
of fungi takes a step towards the realisation of fungal analogue circuits — circuits that use fungi to replace
traditional semiconductors.

The rest of this paper is organised as follows. Section 2 describes the experimental methods used for the
analysis of the substrate. Section 3 presents the results with discussions. Finally, conclusions are drawn in
Sect. 4.

2 Experimental method

Mycelium of the grey oyster fungi Pleurotus ostreatus (Ann Miller’s Speciality Mushrooms Ltd, UK) was
cultivated on damp wood shavings (Fig. 1(a)). Control samples of the growth medium, woodshavings, were
Iridium-coated stainless steel sub-dermal needles with twisted cables (Spes
not colonised by mycelium.
Medica SRL, Italy) were inserted in the colonised substrate. An LCR meter (BK Precision, model number )
was used to provide a nominal reading of the capacitance of the sample with probes at 10 mm, 20 mm, 40 mm
and 50 mm separation.

The samples were charged using a bench top DC power supply (BK precision 9206) to 50 V. The power
supply output was de-activated and the discharge curve was measured using a bench top digital multi-meter
(DMM) Fluke 8846A (Fig. 1(b)). To fully characterise the capacitance of the samples, both the charge and
discharge curves were monitored [23] with a number of probe separations (e.g. 10 mm, 20 mm, 40 mm, and
50 mm). Measurements from the DMM were automated through a serial terminal from a host PC. All bench-
top equipment was high impedance to limit power lost through leakage in the test equipment. All plots were
generated using MATLAB. The sample interval was approximately 0.33 s.

3 Results

Capacitance measurement

Samples of the growth medium and mycelium were measured for their capacitance using a standard bench
top LCR meter (Tab. 1). The measured capacitance value of the mycelium substrate was two to four fold
greater than that of the growth medium alone. The values of capacitance were also eﬀected by the moisture
content such that, if the capacitance of the mycelium was measured straight after it is sprayed with water,
the capacitance typically increased compared to that of dry substrate.

2

(a)

(b)

Figure 1: Experimental setup. (a) A sample of the mycelium under test. (b) Voltage discharge measuring
set up for mycelium samples

3

Mycelium Sample+-V~10mmIridium-coated stainless steel sub-dermal needlesTable 1: Capacitance of mycelium and growth mediums.

Sample

Electrode spacing Capacitance (pF)

Dry wood shavings

Damp wood shavings

Mycelium (drying)

Mycelium (freshly watered)

10 mm
20 mm
10 mm
20 mm
10 mm
20 mm
30 mm
40 mm
50 mm
10 mm
20 mm
30 mm
40 mm
50 mm

37.6
37.4
57
58.6
184
144
146
120
118
193
186
134
125
139

(a)

(b)

Figure 2: Discharge of a substrate after being charged to 50 V with probes separation of 10 mm. (a) Dry
wood shavings. (b) Damp wood shavings — shavings are immersed in water for half and hour after which
excess water is drained. Data are discrete. Line is for eye guidance only.

Figure 3: Mycelium sample is charged to 50 V then allowed to discharge. Electrode spacing is varied (10 mm,
20 mm, 40 mm and 50 mm). Data are discrete. Line is for eye guidance only.

4

05101520253035Seconds [s]01020304050Voltage [V]05101520253035Seconds [s]01020304050Voltage [V]05101520253035Seconds [s]01020304050Voltage [V]10mm20mm40mm50mmTable 2: Discharge curve ﬁtness approximation coeﬃcients (with 95% conﬁdence bounds)

Sample
Dry wood shavings
Damp wood shavings
Mycelium w/10 mm probe separation
Mycelium w/20 mm probe separation
Mycelium w/40 mm probe separation
Mycelium w/50 mm probe separation

a
104.9 (102.6, 107.2)
6205 (5164, 7245)
2276 (2198, 2354)
2688 (2544, 2833)
1569 (1458, 1680)
1413 (1311, 1516)

b
-0.4079 (-0.4151, -0.4007)
-0.6261 (-0.6464, -0.6058)
-0.4446 (-0.4481, -0.441)
-0.4639 (-0.4695, -0.4583)
-0.3948 (-0.4021, -0.3875)
-0.3817 (-0.3891, -0.3743)

(a)

(b)

Figure 4: Charging mycelium to 50 V with source electrodes 10 mm apart. Substance was charged from
approx. 10 minutes, readings are taken for approx. 22 minuets in total. (a) Sense electrodes were placed
in series, with the source terminals (10 mm away from either positive or negative electrodes).
(b) Sense
electrodes were in parallel from the source electrodes (10 mm clearance). Data are discrete. Line is for eye
guidance only.

Discharge characteristics

Discharge characteristics of the growth medium and mycelium samples are shown in Figs. 2–3. Discharge
curves were produced by setting up the DC power supply and DMM in parallel with each other. The
substrate being tested is then charged to 50 V and the power supply output is disabled. The DMM continued
to periodically measure the remaining charge in the substrate for a period of time. The sample interval was
approximately 0.33 s.

The discharge curves for both the growth medium and the mycelium are very steep - approximated by

an exponential (1) .

f (x) = a · eb·x

(1)

Where the parameters for 95% ﬁtness for diﬀerent mediums can be found in Table 2.

The discharge time is governed by equation τ = RC, where τ is the time constant, R is a resistance, and

C is a capacitance.

With capacitance in the order of pico-Farads, and input impedance of the source and measurement

equipment in the order of mega-Ohms, it is expected that the discharge will be in the order of seconds.

Comparing the discharge curves of the growth medium to that of the mycelium samples, it is evident that
the discharge was not as steep in the mycelium due to the increase in capacitance over the growth medium.
However, it was still in the pico-Farad range and, therefore, the majority of charge was lost after just over
5 s. Increasing the separation distance of the probes (Fig. 3) had only a minimal eﬀect on the capacitance of
the substrate (shown in Tab. 1), and therefore minimal eﬀect on the discharge curve.

Observing the charging and discharging behaviour of the sample around the source electrodes helps to
build a better picture of the current density of the mycelium substrate. Figure 4 shows how we placed the
measurement equipment electrodes 10 mm away from the source electrodes in the mycelium, in three diﬀerent

5

0200400600800100012001400Seconds [s]-1-0.500.511.522.5Voltage [V]PositiveNegative0200400600800100012001400Seconds [s]051015Voltage [V]Figure 5: Measurement probes are arranged around the source probes to examine the charge ﬁeld in the
substrate.

locations shown in Fig. 5. The sample was then charged for approximately 10 mins before the supply was
turned oﬀ. The electrodes placed in ‘series’ (locations (a) and (b) on Fig. 5) with the charge electrodes
(Fig. 4(a)) show minimum voltage detected beyond the supply electrodes in the horizontal plane, when the
supply is active. Placing the measurement probes in ‘parallel’ (location (c) on Fig. 5) with the supply probes
(Fig. 4(b)) demonstrates the fact that considerably more current is conducted between the supply probes
in the vertical plane. When the supply is de-activated, the voltage around the supplies collapses almost
instantly.

Charge characteristics

The charge curves in the specimens around the supply electrodes provide an insight into the ability of the
substrate to conduct current. The sense electrodes were distanced from the source electrodes by varying
amounts. Initially, the growth medium was studied on its own (Fig. 6). The dry wood shavings (Fig. 6(a))
showed very low voltage across the sense electrodes placed 10 mm away from the source electrodes (parallel).
The dry shavings essentially acted as an open circuit and the measurement electrodes picked up noise. Damp
wood shavings form a more contiguous mass and the introduction of the water helped to conduct current
(Fig. 6(b)). From a 50 V source a maximum voltage of approx. 15 V was reached across the measurement
electrodes. Although a diﬀerent charge curve was generated for the two repeated runs with electrodes at
diﬀerent distances, we were primarily interested in the maximum observed voltage over the period Fig. 6(c)
superimposes the charge curves from both dry and damp growth medium on to the same axis and Fig. 8
shows all charge curves for growth medium and mycelium.

Preforming similar experiments with the mycelium substrate (Fig. 7), it was observed that moving the
measurement electrodes further from the supply reduced the measured Vmax over the ˜22 min measurement
window. Figure 9 shows that the maximum measured voltage dropped rapidly as the distance from the
source electrodes increased. At 10 mm away from the source, less than 1/5th of the supply was measured over
22 mins, decreasing to less than 1/10th at 15 mm separation. Figure 10 shows that, for the shortest paths
between the two source electrodes, there was a higher current density, indicated by the ﬁeld lines being closer
together. As we move further away from the centre of the two probes, the current density decreased (arrows
are shown further apart). Beyond the two probes in the ‘y–direction’, we would expect to experience very
little current ﬂow, however there are still fringing ﬁeld eﬀects which give rise to the small voltages shown in
Fig. 4(a).

Additionally, the moisture content of the mycelium had an impact on its ability to conduct current.
Figure. 11 shows that, if the sample of mycelium is continually charged over a period where it is also drying
out, the conducted charge can decrease rapidly as the water content vanishes. The measurement electrodes
for this sample are 5 mm away from the source, however we see a decrease in measured voltage of almost
15 V over the measurement window.

6

++++Increasing distanceSourceMeasureMeasureMeasure(a)(b)(c)(a)

(b)

Figure 6: Charge dynamics of growth substrate with measurement equipment set in parallel with charge
electrodes. Electrode pairs were 10 mm apart. (a) Dry wood shavings. (b) Damp wood shavings. (c) Dry
and damp wood shavings. Data are discrete. Line is for eye guidance only.

(c)

Figure 7: Charging of mycelium sample with measurement equipment set to measure at diﬀerent distances
from supply (5 mm, 10 mm, 15 mm, 20 mm, and 50 mm). Supply electrodes and measurement electrodes were
arranged in parallel with each other.

7

020406080100120140Seconds [s]-202468Voltage [V]10-30200400600800100012001400Seconds [s]05101520Voltage [V]10mm20mm020406080100120140Seconds [s]05101520Voltage [V]Dry shavings 10mm probesDamp shavings 10mm probesDamp shavings 20mm probes0200400600800100012001400Seconds [s]010203040Voltage [V]5mm10mm15mm20mm50mmFigure 8: Charge characteristics of wet and dry growth medium and and mycelium with measurement probes
at diﬀerent distances from source probes.

Figure 9: Maximum measured voltage at increasing distance from supply probes in mycelium. Data are
discrete. Line is for eye guidance only.

8

020406080100120140Seconds [s]-5051015202530Voltage [V]Dry shavings 10mm probesDamp shavings 10mm probesDamp shavings 20mm probesMycelium 5mmMycelium 10mmMycelium 15mmMycelium 20mmMycelium 50mm5101520253035404550Distance [mm]010203040Voltage [V]Figure 10: Current density between the two source electrodes. Current ﬂow is shown in the physical direction
rather than convention.

Figure 11: Mycelium charge measured 5 mm away from 50 V source while mycelium sample starts to dry out.
Data are discrete. Line is for eye guidance only.

9

+Fringing fields02004006008001000120014000510152025Voltage [V]4 Conclusions

Mycelium exhibits traditional capacitive characteristics. The capacitance of the substance is in the order of
100’s of pico-Farads. Whilst this does not represent vast amounts of storage, it is up to four fold more than
that of the growth medium alone. Most noticeably, the charge carrying capability of the substance drops oﬀ
rapidly when measurements are taken away from the source electrodes. This shows great potential for the
use of mycelium networks to conduct or store charge in local ‘hot spots’ that are isolated from other areas
in the immediate vicinity. However, it is crucial that the moisture content of the mycelium is kept constant
since the ability to carry charge is strongly inﬂuenced by moisture content.

Any potential analogue circuits implemented with live mycelium will be vulnerable to environmental
conditions, especially humidity, availability of nutrients and removal of metabolites. Ideally, the mycelium
networks should be stabilized so they continue functioning whilst drying. This stabilization can be achieved
either by coating or priming the mycelium with polyaniline (PANI) or poly(3,4-ethylenedioxythiophene) and
polystyrene sulfonate (PEDOT-PSS). This approach has been proven to be successful in experiments with
slime mould P. polycephaum [6, 6, 19], and thus it is likely that a similar technique may be applied to
fungi. Moreover, PABI and PEDOT-PSS incorporated in, or interfaced with, mycelium can bring additional
functionality in terms of conductive pathways [63], memory switches [30, 21] and synaptic-like learning [10, 33].
An optional route toward the functional ﬁxation of mycelium would be doping the networks with substances
that aﬀect the electrical properties of mycelium, such as carbon nanotubes, graphene oxide, aluminium oxide,
calcium phosphate. Similar studies conducted in our laboratory using slime mould and plants have shown
that such an approach is feasible [28, 27]. Moreover using a combination of PANI and carbon nanotubes
in the mycelium network aﬀord it supercapacitive properties [22, 26]. Another potential direction of future
studies would be to increase the capacity of the mycelium as a result of modifying the network geometry
by varying nutritional conditions and temperature [14, 29, 46, 47], concentration of nutrients [49] or with
chemical and physical stimuli [5]. With regards to the impact of our ﬁnding for the ﬁeld of unconventional
computing, we believe further research on experimental laboratory implementation of capacitive threshold
logic [43, 39], adiabatic capacitive logic [44] and capacitive neuromorphic architectures [61] will yield fruitful
insights.

Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation pro-
gramme FET OPEN “Challenging current thinking” under grant agreement No 858132.

Author contributions

A.B. conceived the idea of experiments. A.A. and A.P. prepared the substrate colonised by mycelium. A.B.
performed experiments, collected data and produced all plots in the manuscript. A.B. and A.A. prepared
manuscript (wrote and reviewed all contents). A.P. reviewed manuscript.

References

[1] Andrew Adamatzky. Advances in Unconventional Computing. Springer, 2016.

[2] Andrew Adamatzky. Towards fungal computer. Interface focus, 8(6):20180029, 2018.

[3] Andrew Adamatzky, Phil Ayres, Gianluca Belotti, and Han Wosten. Fungal architecture. arXiv preprint

arXiv:1912.13262, 2019.

[4] Freek VW Appels, Serena Camere, Maurizio Montalti, Elvin Karana, Kaspar MB Jansen, Jan Dijkster-
huis, Pauline Krijgsheld, and Han AB W¨osten. Fabrication factors inﬂuencing mechanical, moisture-and
water-related properties of mycelium-based composites. Materials & Design, 161:64–71, 2019.

10

[5] Yong-Sun Bahn, Chaoyang Xue, Alexander Idnurm, Julian C Rutherford, Joseph Heitman, and Maria E
Cardenas. Sensing the environment: lessons from fungi. Nature Reviews Microbiology, 5(1):57, 2007.

[6] Silvia Battistoni, Alice Dimonte, and Victor Erokhin. Organic memristor based elements for bio-inspired

computing. In Advances in Unconventional Computing, pages 469–496. Springer, 2017.

[7] A. E. Beasley, C. R Bowen, D. A Zabek, and C. T. Clarke. Use it or lose it: The inﬂuence of second
order eﬀects of practical components on storing energy harvested by pyroelectric eﬀects. Technisches
Messen, 85(9):522–540, 2017.

[8] Raymond G Beausoleil, Philip J Kuekes, Gregory S Snider, Shih-Yuan Wang, and R Stanley Williams.

Nanoelectronic and nanophotonic interconnect. Proceedings of the IEEE, 96(2):230–247, 2008.

[9] Mark A Bedau, John S McCaskill, Norman H Packard, and Steen Rasmussen. Living technology:

Exploiting life’s principles in technology. Artiﬁcial Life, 16(1):89–97, 2010.

[10] Tatiana Berzina, Anteo Smerieri, Marco Bernab`o, Andrea Pucci, Giacomo Ruggeri, Victor Erokhin, and
MP Fontana. Optimization of an organic memristor as an adaptive memory element. Journal of Applied
Physics, 105(12):124515, 2009.

[11] A. A. Bhosale and K. K. Sundaram. Firmness prediction of the apple using capacitance measurement.

Procedia Technology, 12:163–167, 2014.

[12] Ajit. A. Bhosale. Detection of sugar contents in citrus fruits by capacitance method. In 10th international

conference interdisciplinarity in engineering, INTER-ENG 2016, pages 466–471, 2016.

[13] Chris J Blackman and Tim J Brodribb. Two measures of leaf capacitance:

insights into the water
transport pathway and hydraulic conductance in leaves. Functional Plant Biology, 38(2):118–126, 2011.

[14] Lynne Boddy, John M Wells, Claire Culshaw, and Damian P Donnelly. Fractal analysis in studies of

mycelium in soil. Geoderma, 88(3):301–328, 1999.

[15] Steven T Boyce, Andrew P Supp, M Dana Harriger, William L Pickens, R Randall Wickett, and Steven B
Hoath. Surface electrical capacitance as a noninvasive index of epidermal barrier in cultured skin sub-
stitutes in athymic mice. Journal of investigative dermatology, 107(1):82–87, 1996.

[16] Robert W Brodersen, Paul R Gray, and David A Hodges. Mos switched-capacitor ﬁlters. Proceedings

of the IEEE, 67(1):61–75, 1979.

[17] Michael John Carlile, Sarah C Watkinson, and Graham W Gooday. The fungi. Gulf Professional

Publishing, 2001.

[18] Oldrich Chloupek. Evaluation of the size of a plant’s root system using its electrical capacitance. Plant

and Soil, 48(2):525–532, 1977.

[19] Angelica Cifarelli, Tatiana Berzina, and Victor Erokhin. Bio-organic memristive device: polyaniline–

physarum polycephalum interface. physica status solidi (c), 12(1-2):218–221, 2015.

[20] Joseph Dahmen. Soft matter: Responsive architectural operations. Technoetic Arts, 14(1-2):113–125,

2016.

[21] VA Demin, VV Erokhin, PK Kashkarov, and MV Kovalchuk. Electrochemical model of polyaniline-
based memristor with mass transfer step. In AIP Conference Proceedings, volume 1648, page 280005.
AIP Publishing LLC, 2015.

[22] Bin Dong, Ben-Lin He, Cai-Ling Xu, and Hu-Lin Li. Preparation and electrochemical characterization
of polyaniline/multi-walled carbon nanotubes composites for supercapacitor. Materials Science and
Engineering: B, 143(1-3):7–13, 2007.

[23] M. Dulik and S. Jurecka. Measuring capacitance of various types of structures. In 2014 ELEKTRO,

pages 640–644, May 2014.

11

[24] H. Endoh, S. Toguchi, and K. Kudo. High performance vertical-type organic transistors and organic
light emitting transistors. In Polytronic 2007 - 6th International Conference on Polymers and Adhesives
in Microelectronics and Photonics, pages 139–142, Jan 2007.

[25] R Fehrenbach, M Comberbach, and JO Petre. On-line biomass monitoring by capacitance measurement.

Journal of biotechnology, 23(3):303–314, 1992.

[26] Elzbieta Frackowiak, Volodymyd Khomenko, Krzysztof Jurewicz, Katarzyna Lota, and Francois B´eguin.
Journal of Power Sources,

Supercapacitors based on conducting polymers/nanotubes composites.
153(2):413–418, 2006.

[27] N Gizzie, R Mayne, S Yitzchaik, M Ikbal, and A Adamatzky. Living wires — eﬀects of size and coating of
gold nanoparticles in altering the electrical properties of Physarum polycephalum and lettuce seedlings.
Nano LIFE, 1(6):1650001, 2015.

[28] Nina Gizzie, Richard Mayne, David Patton, Paul Kendrick, and Andrew Adamatzky. On hybridising
lettuce seedlings with nanoparticles and the resultant eﬀects on the organisms’ electrical characteristics.
Biosystems, 147:28–34, 2016.

[29] Ha Thi Hoa and Chun-Li Wang. The eﬀects of temperature and nutritional conditions on mycelium
growth of two oyster mushrooms (Pleurotus ostreatus and Pleurotus cystidiosus). Mycobiology, 43(1):14–
23, 2015.

[30] Gerard David Howard, Larry Bull, Ben de Lacy Costello, Andrew Adamatzky, and Victor Erokhin. A
spice model of the peo-pani memristor. International Journal of Bifurcation and Chaos, 23(06):1350112,
2013.

[31] Hagen Klauk. Organic electronics: materials, manufacturing, and applications. John Wiley & Sons,

2006.

[32] Ching Kung. A possible unifying principle for mechanosensation. Nature, 436(7051):647, 2005.

[33] DA Lapkin, AV Emelyanov, VA Demin, TS Berzina, and VV Erokhin. Spike-timing-dependent plasticity

of polyaniline-based memristive element. Microelectronic Engineering, 185:43–47, 2018.

[34] J. M. Leger. Organic electronics: The ions have it. Advanced Materials, 20(4):837–841, 2008.

[35] Wen-Hua Li, Kui Ding, Han-Rui Tian, Ming-Shui Yao, Bhaskar Nath, Wei-Hua Deng, Yaobing Wang,
and Gang Xu. Conductive metal–organic framework nanowire array electrodes for high-performance
solid-state supercapacitors. Adv. Funct. Mater, 27:1702067, 2017.

[36] Wei Lu and Charles M Lieber. Nanoelectronics from the bottom up. In Nanoscience And Technology:

A Collection of Reviews from Nature Journals, pages 137–146. World Scientiﬁc, 2010.

[37] Hagen Marien, Michiel Steyaert, and Paul Heremans. Analog Organic Electronics. Springer, 2013.

[38] ET McAdams and J Jossinet. Tissue impedance: a historical overview. Physiological measurement,

16(3A):A1, 1995.

[39] Alejandro Medina-Santiago, Mario Alfredo Reyes-Barranca, Ignacio Algredo-Badillo, Alfonso Martinez
Cruz, Kelsey Alejandra Ram´ırez Guti´errez, and Adri´an Eleazar Cort´es-Barr´on. Reconﬁgurable arith-
metic logic unit designed with threshold logic gates. IET Circuits, Devices & Systems, 13(1):21–30,
2018.

[40] M. Mizukami, S. Cho, K. Watanabe, M. Abiko, Y. Suzuri, S. Tokito, and J. Kido. Flexible organic
light-emitting diode displays driven by inkjet-printed high-mobility organic thin-ﬁlm transistors. IEEE
Electron Device Letters, 39(1):39–42, Jan 2018.

[41] T. Morimoto, M. Tsushima, M. Suhara, K. Hiratsuka, Y. Sanada, and T. Kawasato. Electric double-

layer capacitor using organic electrolyte. MRS Proceedings, 496:627, 1997.

12

[42] Andr´e A Neves, Dora A Pereira, Luıs M Vieira, and Jos´e C Menezes. Real time monitoring biomass
concentration in streptomyces clavuligerus cultivations with industrial media using a capacitance probe.
Journal of biotechnology, 84(1):45–52, 2000.

[43] Hakan Ozdemir, Asim Kepkep, Banu Pamir, Yusuf Leblebici, and Ugur Cilingiroglu. A capacitive

threshold-logic gate. IEEE Journal of Solid-State Circuits, 31(8):1141–1150, 1996.

[44] Ga¨el Pillonnet, Herv´e Fanet, and Samer Houri. Adiabatic capacitive logic: a paradigm for low-power
logic. In 2017 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–4. IEEE, 2017.

[45] K´alm´an Rajkai, KR V´egh, and T Nacsa. Electrical capacitance as the indicator of root size and activity.

Agrok´emia ´es Talajtan, 51(1-2):89–98, 2002.

[46] Alan DM Rayner. The challenge of the individualistic mycelium. Mycologia, pages 48–71, 1991.

[47] CM Regalado, JW Crawford, K Ritz, and BD Sleeman. The origins of spatial heterogeneity in vegetative

mycelia: a reaction-diﬀusion model. Mycological Research, 100(12):1473–1480, 1996.

[48] Boˇstjan Rituper, Alenka Guˇcek, Jernej Jorgaˇcevski, Ajda Flaˇsker, Marko Kreft, and Robert Zorec. High-
resolution membrane capacitance measurements for the study of exocytosis and endocytosis. Nature
protocols, 8(6):1169, 2013.

[49] Karl Ritz. Growth responses of some soil fungi to spatially heterogeneous nutrients. FEMS Microbiology

Ecology, 16(4):269–279, 1995.

[50] Philip Ross. Your rotten future will be great. The Routledge Companion to Biology in Art and Archi-

tecture, page 252, 2016.

[51] Marco Sangermano, Alessandra Vitale, Nicolo Razza, Alain Favetto, Marco Paleari, and Paolo Ariano.

Multilayer UV-cured organic capacitors. Polymer, 56:131–134, 2015.

[52] T. Sano, Y. Suzuri, M. Koden, T. Yuki, H. Nakada, and J. Kido. Organic light emitting diodes for
In 2019 26th International Workshop on Active-Matrix Flatpanel Displays and

lighting applications.
Devices (AM-FPD), volume 26th, pages 1–4, July 2019.

[53] M Sarra, AP Ison, and MD Lilly. The relationships between biomass concentration, determined by a
capacitance-based probe, rheology and morphology of saccharopolyspora erythraea cultures. Journal of
Biotechnology, 51(2):157–165, 1996.

[54] Jane M Shaw and Paul F Seidler. Organic electronics:

introduction. IBM Journal of Research and

Development, 45(1):3–9, 2001.

[55] Myron L Smith, Johann N Bruhn, and James B Anderson. The fungus Armillaria bulbosa is among the

largest and oldest living organisms. Nature, 356(6368):428, 1992.

[56] LL Sohn, OA Saleh, GR Facer, AJ Beavis, RS Allan, and Daniel A Notterman. Capacitance cytometry:
Measuring biological cells one by one. Proceedings of the National Academy of Sciences, 97(20):10687–
10690, 2000.

[57] Eleni Stavrinidou, Roger Gabrielsson, Eliot Gomez, Xavier Crispin, Ove Nilson, Daniel T. Simon, and

Magnus Berggren. Electronic plants. Scientiﬁc Advances, 1(10):1–8, 2015.

[58] W. Tang, J. Zhao, Q. Li, and X. Guo. Highly sensitive low power ion-sensitive organic thin-ﬁlm tran-
sistors. In 2018 9th Inthernational Conference on Computer Aided Design for Thin-Film Transistors
(CAD-TFT), pages 1–1, Nov 2018.

[59] S. Tokito. Flexible printed organic thin-ﬁlm transistor devices and integrated circuit applications. In

2018 International Flexible Electronics Technology Conference (IFETC), pages 1–2, Aug 2018.

[60] L. L. Vadasz, H. T. Chua, and A. S. Grove. Semiconductor random-access memories. IEEE Spectrum,

8(5):40–48, May 1971.

13

[61] Zhongrui Wang, Mingyi Rao, Jin-Woo Han, Jiaming Zhang, Peng Lin, Yunning Li, Can Li, Wenhao
Song, Shiva Asapu, Rivu Midya, et al. Capacitive neural network with neuro-transistors. Nature
communications, 9(1):1–10, 2018.

[62] Paul Allen Williams and Subrata Saha. The electrical and dielectric properties of human bone tissue and
their relationship with density and bone mineral content. Annals of biomedical engineering, 24(2):222–
233, 1996.

[63] Joung Eun Yoo, Tracy L Bucholz, Suyong Jung, and Yueh-Lin Loo. Narrowing the size distribution of
the polymer acid improves pani conductivity. Journal of Materials Chemistry, 18(26):3129–3135, 2008.

[64] Gerald Zachariah and Louis C. Erickson. Evaluation of some physical methods for determining avocado

maturity, volume 49. California Avocado Society, 1965.

[65] MIN Zhang, JHM Willison, MA Cox, and SA Hall. Measurement of heat injury in plant tissue by using

electrical impedance analysis. Canadian journal of botany, 71(12):1605–1611, 1993.

[66] Ute Zschieschang and Hagen Klauk. Organic transistors on paper: a brief review. J. Mater. Chem. C,

7:5522–5533, 2019.

View publication stats

14

