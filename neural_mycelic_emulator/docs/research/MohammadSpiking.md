See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/349193934

Electrical activity of fungi: Spikes detection and complexity analysis

Article  in  Biosystems · February 2021

DOI: 10.1016/j.biosystems.2021.104373

CITATIONS
45

2 authors:

Mohammad Mahdi Dehshibi

University Carlos III de Madrid

105 PUBLICATIONS   974 CITATIONS   

SEE PROFILE

READS
407

Andrew Adamatzky

University of the West of England, Bristol

922 PUBLICATIONS   14,614 CITATIONS   

SEE PROFILE

All content following this page was uploaded by Mohammad Mahdi Dehshibi on 22 February 2021.

The user has requested enhancement of the downloaded file.

Electrical activity of fungi: Spikes detection and complexity
analysis

Mohammad Mahdi Dehshibia,b,∗, Andrew Adamatzkyb

aDepartment of Computer Science, Multimedia and Telecommunications, Universitat Oberta de Catalunya, Barcelona, Spain
bUnconventional Computing Laboratory, University of the West England, Bristol, UK

A R T I C L E I N F O

A B S T R A C T

Keywords:
Pleurotus djamor
electrical activity
spikes
complexity

Oyster fungi Pleurotus djamor generate actin potential like spikes of electrical potential. The
trains of spikes might manifest propagation of growing mycelium in a substrate, transportation of
nutrients and metabolites and communication processes in the mycelium network. The spiking
activity of the mycelium networks is highly variable compared to neural activity and therefore
can not be analysed by standard tools from neuroscience. We propose original techniques for
detecting and classifying the spiking activity of fungi. Using these techniques, we analyse the
information-theoretic complexity of the fungal electrical activity. The results can pave ways for
future research on sensorial fusion and decision making of fungi.

1. Introduction

Excitation is an essential property of all living organisms, bacteria Masi et al. (2015), Protists Eckert and Brehm
(1979); Hansma (1979); Bingley (1966), fungi McGillviray and Gow (1987) and plants Trebacz et al. (2006); Fromm
and Lautner (2007); Zimmermann and Mithöfer (2013) to vertebrates Hodgkin and Huxley (1952); Aidley and Ashley
(1998); Nelson and Lieberman (2012); Davidenko et al. (1992). Waves of excitation could be also found in various
physical Kittel (1958); Tsoi et al. (1998); Slonczewski (1999); Gorbunov and Kirsanov (1987), chemical Belousov
(1959); Zhabotinsky (1964); Zhabotinsky (2007) and social systems Farkas et al. (2002, 2003). Extracellular (EC)
action potential recordings have been widely used to record and measure neural activity in organisms with excitation.
When recorded with diﬀerential electrodes, the spike manifests a propagating wave of excitation.

In our recent studies Adamatzky (2018b, 2019); Adamatzky et al. (2020), we have shown that the Pleurotus djamor
oyster fungi generate action potentials like electrical potential impulses. We observed spontaneous spike1 trains with
two types of activity, i.e. high-frequency (2.6 min period) and low-frequency (14 min period). However, the proper use
of this information is subject to the accurate extraction of the EC spike waveform, separating it from the background
activity of the neighbouring cells and sorting the characteristics.

The lack of an algorithmic framework for the exhaustive characterisation of the electrical activity of the substrate
colonised by mycelium of oyster fungi Pleurotus djamor has inspired us to develop a framework to extract spike
patterns, quantify the diversity of spike events and measure the complexity of fungal electrical communication. We
evidenced the spiking activity of the mycelium (see an example in Figure 1), which will enable us to build an experi-
mental prototype of fungi-based information processing devices.

We evaluated the proposed framework in comparison with existing spike detection techniques in neuroscience Ne-
nadic and Burdick (2004); Shimazaki and Shinomoto (2010) and observed a signiﬁcant improvement in the spike
activity extraction. The evaluation of the proposed method for detecting spike events compared to the speciﬁed spike
arrival time by the expert shows true-positive and false-positive rates of 76% and 16%, respectively. We found that the
average dominant duration of an action-potential-like spike is 402 sec. The amplitude of the spikes ranges from 0.5 mV
to 6 mV and depends on the location of the source of electrical activity (the position of electrodes). We have found
that the complexity of the Kolmogorov fungal spike ranges from 11 × 10−4 to 57 × 10−4. In Vicnesh and Hagiwara
(2019), the human brain’s Kolmogorov complexity is measured in normal, pre-ictal and ictal states resulting in 6.01,
5.59 and 7.12 values, respectively. Although the fungi’ complexity is considerably smaller than that of the human

∗Corresponding author
ORCID(s): 0000-0001-8112-5419 (M.M. Dehshibi); 0000-0003-1073-2662 (A. Adamatzky)
1Calling the spikes spontaneous means that the intentional external stimulus does not invoke them. Otherwise, the spikes actually reﬂect the

ongoing physiological and morphological processes in the mycelial networks.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 1 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

(c)

(d)

Figure 1: The electrical behaviour of the mycelium of the grey oyster fungi. (a) Example of electrical potential dynamics
recorded in seven channels of the same cluster during 409 hours. (b) Two channels are zoomed in the inserts to show the
rich combination of slow (hours) drift of base electrical potential combined with relatively fast (minutes) oscillations of the
potential. (c) DC levelling for two channels is plotted. The mismatch of DC levels indicates the resistance and diﬀerent
levels of intra-communication in the substrate. (d) All ’classical’ parts of the spike, i.e. depolarisation, depolarisation
and refractory period, can be found in this sample spike. This spike has a length of 220 s, from the base-level potential
to the refractory-like phase, and a refractory period of 840 s. The depolarisation and depolarisation rates are 0.03 and
0.009 mV/s, respectively.

brain, its changes suggest a degree of intra-communication in the mycelium sub-network. In fact, diﬀerent parts of the
substrate transmit diﬀerent information to other parts of the mycelium network, where the more prolonged propagation
of excitation waves leads to higher levels of complexity.

The rest of this paper is structured as follows: Sect. 2 presents the experimental setup. Details of the proposed spike
detection methods are explained in Sect. 3. Experimental results and complexity analyses are discussed in Sect. 4.
Finally, the discussion is given in Sect. 5.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 2 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

2. Experimental set-up

A wood shavings substrate was colonised by the mycelium of the grey oyster fungi, Pleurotus ostreatus (Ann
Miller’s Speciality Mushrooms Ltd, UK). The substrate was placed in a hydroponic growing tent with a silver Mylar
lightproof inner lining (Green Box Tents, UK). Recordings were carried out in a stable indoor environment with the
temperature remaining stable at 22 ± 0.5°and relative humidity of air 40 ± 5%. The humidity of the substrate colonised
by fungi was kept at c. 70-80%. Figure 2 shows examples of the experimental setups.

(a)

(b)

(c)

Figure 2: (a) In-line placement of electrodes (1 cm distance), (b) random electrode placement, (c) the experimental setup.

We inserted pairs of iridium-coated stainless steel sub-dermal needle electrodes (Spes Medica SRL, Italy) with
twisted cables into the colonised substrate for recording electrical activity. Using a high-resolution ADC-24 (Pico
Technology, UK) data logger with a 24-bit A/D converter, galvanic insulation and software-selectable sample rates
all lead to superior noise-free resolution. We recorded electrical activity one sample per second, where the minimum
and maximum logging times were 60.04 and 93.45 hours, respectively. During recording, the logger makes as many
measurements as possible (basically up to 600 per second) and saves the average value. We set the acquisition voltage
range to 156 mV with an oﬀset accuracy of 9 𝜇V at 1 Hz to preserve a gain error of 0.1%. Each electrode pair
was considered independently with a 17-bit noise-free resolution and a 60 ms conversion time. In our experiments,
electrode pairs were placed in one of two conﬁgurations: random placement or in-line placement. The distance between
the electrodes was between 1-2 cm. In each cluster, we recorded 5–16 pairs of electrodes (channels) simultaneously.
In six trials, we also undertook recordings of the fruit body’s resistance, where electrodes were inserted in stalks of
the bodies. We measured and logged a range of resistances 1-1.6kΩ using Fluke 8846A precision multimeter, where
the test current being 1±0.0013𝜇A, once per 10 seconds, 5×104 samples per trial Adamatzky et al. (2021). It should be
noted that the placement of the electrodes in two experiments was in-lines with a distance of 1 cm, in two experiments
it was in-lines with a distance of 2 cm, and in two experiments it was random with a distance of approximately 2 cm.

3. Proposed method

A spike event can be formally deﬁned as an extracellular signal that exceeds a simple amplitude threshold and
passes through a corresponding pair of user-speciﬁc time-voltage boxes. The spike, which involves depolarisation,

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 3 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

depolarisation and refractory cycles, represents physiological and morphological processes in mycelial networks. To
extract spike events, we proposed an unsupervised approach consisting of three major steps. The pipeline of the
proposed approach is shown in Figure 3

Figure 3: The pipeline for the identiﬁcation of spike events.

– Step 1: We split the entire recording duration (𝐹 (𝑡)) into 𝑘 chunks (𝑓𝑘(𝑡)) with respect to the signal transitions. In
order to evaluate the transitions, we determined the state level of the signal by its histogram and identiﬁed all regions
that cross the upper boundary of the low state and the lower boundary of the high state. Then, we measure scale-to-
frequency conversions of the analytic signal in each chunk using Morse wavelet basis Lilly and Olhede (2012). To
assess the existence of spike-like events, we scaled the wavelet coeﬃcients at each frequency and obtained the sum
of the scales below the threshold speciﬁed in Algorithm 1. Finally, we selected regions of interest (ROI) enclosed
between a consecutive local minimum and a maximum of more than 30 sec.
– Step 2: We used spline interpolation to measure the analytic signal envelopes around local maximum values. To
determine the analytical signal, we ﬁrst applied the discrete approximation of Laplace’s diﬀerential operator to 𝑓𝑘(𝑡)
to obtain a ﬁnite sequence of equally-spaced samples. Then, we applied discrete-time Fourier transform to this ﬁnite
sequence. From the average signal envelope, we extracted regions spanning between a consecutive local minimum and
a maximum. These regions created constraints that contributed to the identiﬁcation of spike events.
– Step 3: We retained the ROIs extracted in the ﬁrst step, which met the constraints of the second step. The signal
envelope could direct wavelet decomposition in an unsupervised manner in order to cluster the signal into the spike,
pseudo-spike, and background activity of the adjacent cells. In the following sub-sections, we detailed the proposed
process.

3.1. Slicing fungi electrical activity

To split the electrical activity of fungi (𝐹 (𝑡)) with a duration of (𝑡) second into (𝑘) chunks (𝑓 𝑘(𝑡), 1 ≤ 𝑘 ≤ 𝑡 − 1),
we used the signal transitions that constitute each pulse. To determine the transitions, we estimated the state level of
𝐹 (𝑡) using the histogram method IEEE (2011). Then, we identiﬁed all regions that cross the upper boundary of the
low state and the lower boundary of the high state. We followed the following steps to estimate the signal states:

1. Determine the minimum, maximum and range of amplitudes.
2. Sort the amplitude values in the histogram bins and determine the width of the bin by dividing the amplitude

range by the number of bins.

3. Identify the lowest- and highest-indexed histogram bins, ℎ𝑏𝑙𝑜𝑤
4. Divide the histogram into two sub-histograms, where the indices of the lower and upper histogram bins are

, with non-zero counts.

, ℎ𝑏ℎ𝑖𝑔ℎ

ℎ𝑏𝑙𝑜𝑤

(ℎ𝑏ℎ𝑖𝑔ℎ − ℎ𝑏𝑙𝑜𝑤) ≤ ℎ𝑏 ≤ ℎ𝑏ℎ𝑖𝑔ℎ
5. Calculate the mean of the lower and upper histogram to compute the state levels.

(ℎ𝑏ℎ𝑖𝑔ℎ − ℎ𝑏𝑙𝑜𝑤) and ℎ𝑏𝑙𝑜𝑤 + 1
2

≤ ℎ𝑏 ≤ 1
2

, respectively.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 4 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

Each chunk is then enclosed between the last negative-going transitions of each positive-polarity pulse and the next
positive-going transition. Figure 4 shows the slicing results of two channels.

(a)

(b)

Figure 4: Slicing electrical potential recordings for two channels.

3.2. Detecting time-localised events by Morse-based wavelets

The electrical activity of mycelium shows modulated behaviour with changes in amplitude and frequency over
time. This feature suggests that the signal can be analysed with analytic wavelets, which are naturally grouped into
pairs of even or cosine-like and odd or sine-like pairs, allowing them to capture phase variability. A wavelet (𝜓(𝑡))
is a ﬁnite energy function that projects 𝑓 (𝑡) to a family of time-scale waveforms through translation and scaling. The
Morse wavelet (𝜓𝛽,𝛾 (𝑡)) is an analytical wavelet whose Fourier transform is supported only on a positive real axis Lilly
and Olhede (2012); Lilly (2017). This wavelet is deﬁned in the frequency domain using Eq. 1.

𝜓𝛽,𝛾 (𝑡) =

Ψ𝛽,𝛾 (𝜔) ≡

∞

Ψ𝛽,𝛾 (𝜔) 𝑒𝑖𝜔𝑡 d𝜔,

1
2𝜋 ∫
−∞
𝑎𝛽,𝛾 𝜔𝛽𝑒−𝜔𝛾
𝜔 > 0
⎧
(𝑎𝛽,𝛾 𝜔𝛽𝑒−𝜔𝛾 ) 𝜔 = 0
⎪
1
⎨
2
⎪
𝜔 < 0
0
⎩

.

(1)

𝛾 is the amplitude coeﬃcient used as a real-
where 𝛽 ≥ 0 and 𝛾 > 0, 𝜔 is the angular frequency and 𝑎𝛽,𝛾
valued normalised constant. Here, 𝑒 is Euler’s number, 𝛽 characterises the low-frequency behaviour, and 𝛾 deﬁnes the
high-frequency decay. We can rewrite Eq. 1 in the Fourier domain, parameterised by 𝛽 and 𝛾 as in Eq. 2.

≡ 2

) 1

( 𝑒𝛾
𝛽

𝜙𝛽,𝛾 (𝜏, 𝑠) ≡

∞

∫

−∞

1
𝑠

𝜓 ∗

𝛽,𝛾 (

𝑡 − 𝜏
𝑠

)𝑓 (𝑡) dt =

1
2𝜋 ∫

∞

−∞

ei𝜔𝜏 Ψ∗

𝛽,𝛾 (s𝜔)F(𝜔) d𝜔.

(2)

𝛽,𝛾 (𝜔) is real-valued, the
where 𝐹 (𝜔) is the Fourier transform of 𝑓 (𝑡), and ∗ denotes the complex conjugate. When Ψ∗
conjugation may be omitted. The scale variable 𝑠 allows the wavelet to stretch or compress in time. In order to reﬂect
the energy of 𝑓 (𝑡) and to normalise time-domain wavelets to preserve constant energy, 1
is typically used. However,
√
𝑠
instead, we used 1
𝑠
we can use the inverse Fourier transform by 𝑓 (𝑡) = 1
2𝜋
𝛿(𝜔) is the Dirac delta function.

since we deﬁne the amplitude of the time-located signals. To recover time-domain representation,
−∞ 𝑒𝑖𝜔𝑡 dt = 2𝜋𝛿(𝜔), where

−∞ 𝑒𝑖𝜔𝑡 𝐹 (𝜔) d𝜔 and 𝜓𝛽,𝛾 (𝑡) = ∫ ∞
∫ ∞

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 5 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

The representation of Morse wavelets would be more oscillatory when both 𝛽 and 𝛾 increase, and more localised
with impulses when these parameters decrease. On the other hand, increasing 𝛽 and holding 𝛾 ﬁxed expand the central
portion of the wavelet and increase the long-term rate of decay. Whereas, increasing 𝛾 by keeping 𝛽 constant extends
the wavelet envelope without aﬀecting the long-term decay rate. Inspired by Lilly and Olhede (2008), we set the
symmetry parameter 𝛾 to 3 and the time-bandwidth product 𝑃 2 = 𝛽𝛾 to 60. We have used 𝐿1
normalisation to provide
the same magnitude in wavelets when we have the same amplitude oscillatory components at diﬀerent scales.

(a)

(b)

Figure 5: Annotated spikes by the expert (black arrows) on the Morse wavelet scalogram for (a) 𝑆𝑙𝑖𝑐𝑒1 and (b) 𝑆𝑙𝑖𝑐𝑒2.
The scalogram is plotted as a function of time and frequency, where the maximum absolute value at each frequency is
used for the normalisation of the coeﬃcient. The frequency axis is shown on a linear scale.

Figure 5 displays two randomly chosen 3000-second chunks of fungi electrical activity (namely 𝑆𝑙𝑖𝑐𝑒1

)
and 𝑆𝑙𝑖𝑐𝑒2
with their Morse wavelet scalograms. We observed that the use of the maximum absolute value at each frequency
(level) to normalise coeﬃcients may help to identify events that may involve spikes. We then used Eq. 3 to normalise
coeﬃcients and set zero entries to 1.

(

𝑔𝛽,𝛾 (𝜏, 𝑠) =

𝜂 ×

⊺,
𝜅𝛽,𝛾 (𝜏, 𝑠) = |𝜙𝛽,𝛾 (𝜏, 𝑠)|
)⊺

𝜅𝛽,𝛾 (𝜏, 𝑠) − min𝑠(𝜅𝛽,𝛾 (𝜏, 𝑠))
max𝑠(𝜅𝛽,𝛾 (𝜏, 𝑠))

.

(3)

| ∙ |

where
and (∙)⊺ return the absolute value and the matrix transpose, respectively. Here, 𝜂 is a scaling factor that we
empirically set to 240. We used 𝑔𝛽,𝛾 (𝜏, 𝑠) in Algorithm 1 to extract the candidate ROIs shown in Figure 6. As shown
in Figure 6(c,d), some of the detected regions are either too short2 or lack repolarisation and depolarisation periods
that should be removed from .

Algorithm 2 is proposed to eliminate so-called pseudo-spike and inﬂection regions which do not reach the spike
(see Figure 7(a))
(see Figure 7(b)). We found that the

characteristics as shown in Figure 7. Applying Algorithm 2 resulted in the loss of two spikes in 𝑆𝑙𝑖𝑐𝑒1
and the failure to eliminate two pseudo-spike and two inﬂection regions in 𝑆𝑙𝑖𝑐𝑒2
analysis of the analytic signal by its envelope could improve the accuracy of the spike detection.

3.3. Analytical signal envelope for locating spike pattern

We calculated the magnitude of the analytic signal to obtain the signal envelope (𝜉). The analytic signal is detected
using the discrete Fourier Transform as implemented in the Hilbert Transform. In order to highlight eﬀective signal
peaks and neutralise inﬂection regions, the second numerical signal derivation (𝐿 = 𝜕2𝑓 ∕4𝜕𝑡2) was calculated. A
frequency-domain approach is proposed in Marple (1999) to approximately generate a discrete-time analytic signal.

2We observed in our previous studies Adamatzky (2018b, 2019) that minimum spike length was 5 mins.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 6 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(c)

(b)

(d)

Figure 6: (a,b) Identiﬁed local maxima and minima over 𝑔𝛽,𝛾 (𝜏, 𝑠) in (a) 𝑆𝑙𝑖𝑐𝑒1 and (b) 𝑆𝑙𝑖𝑐𝑒2. The second-row of the
plot is the inverse of the ﬁrst row; therefore, the marked maximums are identical to the local minima. (c,d) Candidate
regions of interest which are alternately coloured purple and green to ease visual tracking.

In this approach, the negative frequency in half of each spectral period is set to 0, resulting in a periodic one-sided
spectrum. The procedure for generating a complex-valued 𝑁-point (𝑁 is even) discrete-time analytic signal (𝐹 (𝜔))
from a real-valued 𝑁-point discrete time signal (𝐿[𝑛]) is as follows:

1. Calculate the 𝑁-point discrete-time Fourier transform using 𝐹 (𝜔) = 𝑇 ∑𝑁−1

≤
1∕2𝑇 Hz. 𝐿[𝑛] , 0 ≤ 𝑛 ≤ 𝑁 − 1 is obtained by sampling the band-limited real-valued continuous-time sig-
nal 𝐿(𝑛𝑇 ) = 𝐿[𝑛] at periodic time intervals of 𝑇 seconds to prevent aliasing.

𝑛=0 𝐿[𝑛]𝑒−𝑖2𝜋𝜔𝑇 𝑛, where

|𝜔|

2. Calculate the 𝑁-point one-sided discrete-time analytic signal transform:

𝑍[𝑚] =

⎧
𝐹 [0],
⎪
2𝐹 [𝑚],
⎪
𝐹 [ 𝑁
⎨
],
⎪
2
0,
⎪
⎩

for 𝑚 = 0
for 1 ≤ 𝑚 ≤ 𝑁
2
for 𝑚 = 𝑁
2
for 𝑁
2

+ 1 ≤ 𝑚 ≤ 𝑁 − 1.

− 1

(4)

3. Calculate the 𝑁-point inverse discrete-time Fourier transform to obtain the complex discrete-time analytic signal

of same sample rate as the original 𝐿[𝑛]

𝑧[𝑛] =

1
𝑁𝑇

𝑁−1
∑

𝑚=0

𝑍[𝑚]𝑒

𝑖2𝜋𝑚𝑛
𝑁

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

(5)

Page 7 of 22

3

4

5

6

7

8

9

10

11

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

Electrical activity of fungi: Spikes detection and complexity analysis

Algorithm 1: Detecting candidate regions for time-localised events.

: 𝑔𝛽,𝛾 (𝜏, 𝑠) – Scaled wavelets coeﬃcients.

Input
Output:  – set of candidate regions.

1 begin
2

𝜖 = 0.05 × (max(𝑔𝛽,𝛾 (𝜏, 𝑠)) − min(𝑔𝛽,𝛾 (𝜏, 𝑠)));
𝑚𝑎𝑥𝑔 ← set of all LocalMaximum(𝑔𝛽,𝛾 (𝜏, 𝑠), 𝜖);
// LocalMaximum() returns 𝜏∗ if ∀𝜏 ∈ (𝜏∗ ± 𝜖), 𝑔𝛽,𝛾 (𝜏∗, 𝑠) ≥ 𝑔𝛽,𝛾 (𝜏, 𝑠).
𝑚𝑖𝑛𝑔 ← set of all LocalMinimum(𝑔𝛽,𝛾 (𝜏, 𝑠), 𝜖);
 ← 𝐬𝐨𝐫𝐭(𝑚𝑖𝑛𝑔
𝑛 = 𝐜𝐚𝐫𝐝( );
// 𝐜𝐚𝐫𝐝(𝐴) returns number of entries in 𝐴.
if 𝑛 ≡ 1 (mod 2) then

⋃ 𝑚𝑎𝑥𝑔);

slack ← 𝐦𝐞𝐚𝐧(diﬀerence of two consecutive entries);
Add 𝐦𝐢𝐧(
𝑛 = 𝑛 + 1;

𝑛 + slack, 𝜏) to  ;

end
 ← (

12
13 end
14 return 

𝑖, 

𝑖+1), ∀𝑖 ∈ {1, 3, ⋯ , 𝑛 − 1}

Algorithm 2: Excluding pseudo-spike and inﬂation regions form candidate ROI.

Input

:  — set of ROI, i.e., Algorithm 1 output,
𝑓 — Electrical potential.
Output:  — set of wavelet-based ROIs,

 — set of pseudo-spike and inﬂection regions.

1 begin
2

for 𝑖 = 1 to 𝐜𝐚𝐫𝐝() do

𝑙𝑏 ← (𝑖, 1);
𝑢𝑏 ← (𝑖, 2);
if (𝑢𝑏 − 𝑙𝑏) > 30 then

𝑐ℎ𝑢𝑛𝑘 = 𝑓 [𝑙𝑏 ⋯ 𝑢𝑏];
𝑚𝑖𝑛𝑖𝑚𝑎 = 𝐦𝐢𝐧(isLocalMinimum(chunk));
// isLocalMinimum() and isLocalMaximum() use spline interpolation in

locating local extreme Hall and Meyer (1976).

𝑚𝑎𝑥𝑖𝑚𝑎 = 𝐦𝐚𝐱(isLocalMaximum(chunk));
if 𝑓 (𝑚𝑖𝑛𝑖𝑚𝑎) < 𝐦𝐢𝐧(𝑓 (𝑙𝑏), 𝑓 (𝑢𝑏)) 𝐨𝐫 𝑓 (𝑚𝑎𝑥𝑖𝑚𝑎) > 𝐦𝐚𝐱(𝑓 (𝑙𝑏), 𝑓 (𝑢𝑏)) then

 ← [𝑙𝑏, 𝑢𝑏];

else

 ← [𝑙𝑏, 𝑢𝑏];

end

end

end

15
16 end
17 return , 

Obtaining an analytic signal in this way satisﬁes two properties: (1) The real part is equivalent to the original discrete-
time sequence; (2) the real and imaginary components are orthogonal. Calculating the magnitude of the analytic signal

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 8 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

Figure 7: Results of applying Algorithm 2 to (a) 𝑆𝑙𝑖𝑐𝑒1 and (b) 𝑆𝑙𝑖𝑐𝑒2. Two spike events are missed in 𝑆𝑙𝑖𝑐𝑒1. Two
pseudo-spike and two inﬂection regions still remain in 𝑆𝑙𝑖𝑐𝑒2.

using Eq. 6 results in the signal envelope (𝜉[𝑛]) containing the upper (𝜉𝐻 [𝑛]) and lower (𝜉𝐿[𝑛]) envelopes of 𝐿[𝑛].

𝜉[𝑛] = |𝑧[𝑛]|

(6)

Envelopes are calculated using spline interpolation over a local maximum separated by at least 𝑛𝑝 = 60 samples.
We considered 𝑛𝑝 = 60 because, in our previous studies Adamatzky (2018b, 2019), we did not observe any electrical
potential of spikes shorter than 60 seconds. Algorithm 3 was proposed to locate candidate regions using a signal
envelope.

Figure 8 (a,d) shows the candidate regions in  before applying Step 13. At this step, while  includes regions
that do not align with the spike deﬁnition (pointed by arrow in the plot), the correctly identiﬁed spikes are consistent
with our ﬁndings in Adamatzky (2018a,b). Steps 13 and 14 were used to remove non-spike regions marked in red
in Figure 8 (b,e). However, the output of Algorithm 3 (see Figure 8 (c,f)) still includes regions belonging to either
pseudo-spike/inﬂection regions or refractory periods attached to pseudo-spike regions.

To ﬁx mis-identiﬁed ROIs in Algorithms 2 and 3, we proposed Algorithm 4 in which regions in ( ∪ ) are used
) and update the spike
to update . Indeed, if the ROI in  is a subset of ( ∪ ), we add it to the spike event set (
length. If the ROI in ( ∪ ) is a subset of , we add it to the pseudo-spike set (
). In the case of an intersection that
does not meet the subset requirement, we concatenate ROIs and divide the new region from the intersection point into
two segments. Then, we add the segment with the minimum length to 
. Finally, regions with a length of less than
60 seconds are excluded from both 
𝑠

. The results are shown in Figure 9.

and 

𝑝

𝑝

𝑝

𝑠

4. Experimental results

This section consists of objective and complexity analyses. In the objective analysis, we demonstrated the eﬀective-
ness of the spike event detection method compared to conventional spike detection techniques in neuroscience Nenadic
and Burdick (2004); Shimazaki and Shinomoto (2010). We also compared the proposed method with the expert opinion
on the location of spikes. In the complexity analysis, we selected the complexity measures used in previous studies Mi-
noofam et al. (2012, 2014); Parsa et al. (2017); Taghipour et al. (2016); Dehshibi et al. (2015, 2020); Gholami et al.
(2020) to quantify spatio-temporal activity patterns.

4.1. Objective analysis

Various methods have been proposed for detecting and sorting spike events in EC recordings Quiroga et al. (2004);
Nenadic and Burdick (2004); Obeid and Wolf (2004); Wilson and Emerson (2002); Gotman and Wang (1991); Wilson

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 9 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

Algorithm 3: Detecting candidate spike region from signal envelope.

Input

: 𝜉[𝑛] — Envelope of signal 𝐿[𝑡],
𝑛𝑝 = 60 — Minimum distance between two consecutive local extreme.

Output:  — set of envelope-based ROIs.

1 begin
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

∕2;

(𝜉𝐻 [𝑛] + 𝜉𝐿[𝑛]

)
𝜉𝑀 [𝑛] =
[𝑣𝑎𝑙𝑚𝑖𝑛, 𝑖𝑛𝑑𝑚𝑖𝑛] = isLocalMinimum(𝜉𝑀 [𝑛], 𝑛𝑝);
[𝑣𝑎𝑙𝑚𝑎𝑥, 𝑖𝑛𝑑𝑚𝑎𝑥] = isLocalMaximum(𝜉𝑀 [𝑛], 𝑛𝑝);
// isLocalMinimum() and isLocalMaximum() locate local minimum and maximum,

respectively.

𝑗 ← index of the ﬁrst local maximum whose value is greater than the value of the ﬁrst local minimum;
for 𝑖 = 1 to 𝐜𝐚𝐫𝐝(𝑖𝑛𝑑𝑚𝑖𝑛) do
if 𝑗 ≤ 𝐜𝐚𝐫𝐝(𝑖𝑛𝑑𝑚𝑎𝑥) then

Δ ← 𝑣𝑎𝑙𝑚𝑎𝑥(𝑗) − 𝑣𝑎𝑙𝑚𝑖𝑛(𝑖);
Add (𝑖𝑛𝑑𝑚𝑖𝑛(𝑖), 𝑖𝑛𝑑𝑚𝑎𝑥(𝑗), Δ
𝑗 ← 𝑗 + 1;

) to ;

end

end
//  has 𝑗 rows and 3 columns, as 
𝜌 = 𝐦𝐞𝐚𝐧(
// 𝐦𝐞𝐚𝐧() and 𝐬𝐭𝐝() calculate the mean and standard deviation, respectively.
Remove the 𝑘𝑡ℎ entry from  where 

2, and 

3) − 𝐬𝐭𝐝(

3(𝑘) < 𝜌 – see Figure 8(b);

1, 

3);

3.

14
15 end
16 return 

et al. (1999); Franke et al. (2010); Rácz et al. (2020); Wang et al. (2020); Sablok et al. (2020); Liu et al. (2020). However,
only a few of these methods do not require additional details, such as template construction and the supervised setting
of thresholds for detecting and sorting spike events Nenadic and Burdick (2004); Shimazaki and Shinomoto (2010).
Nenadic and Burdick Nenadic and Burdick (2004) have developed an unsupervised method for detecting and locating
spikes in noisy neural recordings. This approach beneﬁts from the continuous transformation of the wavelet. They
applied multi-scale signal decomposition using the ‘bior1.3,’ ‘bior1.5,’ ‘Haar,’ or ‘db2’ wavelet basis. To determine
the presence of spikes, they separated the signal and noise at each scale and performed Bayesian hypothesis testing.
Finally, they combined decisions on diﬀerent scales to estimate the arrival times of individual spikes.

Shimazaki and Shinomoto Shimazaki and Shinomoto (2010) proposed an optimisation technique for the timing-
histogram bin width selection. This optimisation minimised the mean integrated square in the kernel density estima-
tion. This method beneﬁts from variable kernel width, which allowed grasping non-stationary phenomena. Also, this
method used stiﬀness constant to avoid possible overﬁtting due to excessive freedom in the bandwidth variability. The
calculated bandwidth was then used as a proxy for ﬁltering spike event regions. Figure 10 shows the results of applying
these methods to two chunks with a length of 3000 seconds.

Both compete methods could not correctly detect all spike events that were located by the expert. The wavelet-based
method could locate three spikes in Figure 10(b) without detecting any spike in Figure 10(a). The adaptive bandwidth
kernel-based method could detect one spike in Figure 10(b) and one pseudo-spike in Figures 10(a, b). While our
proposed method misidentiﬁed one spike event in Figure 9(a) and three spikes in Figure 9(b).

We also compared the proposed method with the expert opinion on a randomly selected 36,000-second chunk,
i.e., 10 hours of electrical activity recordings. In this quantitative comparison, the proposed approach could correctly
locate 21 spikes and four pseudo-spike events. Our method also overestimated two refractory periods, resulting in the
true-positive and false-positive rates of 76% and 16%, respectively. Figure 11(a) shows located spikes by the expert,
and Figure 11(b) demonstrates the results of the proposed spike detection method.

We applied the proposed method to six experiments where the statistical results are shown in Figures 12 and 13 and

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 10 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

(c)

(d)

(e)

(f)

Figure 8: Results of Algorithm 3 applied to 𝑆𝑙𝑖𝑐𝑒1 (ﬁrst row) and 𝑆𝑙𝑖𝑐𝑒2 (second row). (a,d) Candidate regions by ﬁnding
local minima and maxima in the analytic signal envelope. The pointed regions are also highlighted in the bar chart in
red. (b,e) The absolute diﬀerence in prominence between the successive local minima and maxima. Regions that do
3(𝑘) < 𝜌 are coloured in red. (c,f) Regions of Interest in . The grey dashed rectangle shows the correct
not satisfy 
spike, including repolarisation, depolarisation, and refractory periods. The purple dashed rectangle shows the region whose
refractory period attached to the pseudo-spike region.

(a)

(b)

Figure 9: Results of applying Algorithm 4 to (a) 𝑆𝑙𝑖𝑐𝑒1 and (b) 𝑆𝑙𝑖𝑐𝑒2. We alternatively used red/purple for colouring
spike events and blue/cyan for colouring pseudo-spike events.

summarised in Table 1. It should be noted that the placement of the electrodes in two experiments was in lines with

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 11 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

Algorithm 4: Extracting fungi spike and pseudo-spike events.

: , ,  — Regions of interest.

Input
Output: 

𝑠, 
𝑝

— Fungi spike and pseudo-spike events, respectively.

1 begin
2

foreach 𝑟𝑒 ∈  do
𝑐ℎ𝑢𝑛𝑘𝑒 ← [𝑟1
foreach 𝑟𝑤 ∈ ( ∪ ) do
𝑤];
𝑤 ⋯ 𝑟2

𝑒];
𝑒 ⋯ 𝑟2

𝑐ℎ𝑢𝑛𝑘𝑤 ← [𝑟1
switch 𝑐ℎ𝑢𝑛𝑘𝑤, 𝑐ℎ𝑢𝑛𝑘𝑒 do

case 𝑐ℎ𝑢𝑛𝑘𝑒 ⊂ 𝑐ℎ𝑢𝑛𝑘𝑤 do

𝑐ℎ𝑢𝑛𝑘𝑤(𝑒𝑛𝑑) = 𝑐ℎ𝑢𝑛𝑘𝑒(𝑒𝑛𝑑);
;

𝑠 ← 𝑐ℎ𝑢𝑛𝑘𝑤

end
case 𝑐ℎ𝑢𝑛𝑘𝑤 ⊂ 𝑐ℎ𝑢𝑛𝑘𝑒 do



𝑝 ← 𝑐ℎ𝑢𝑛𝑘𝑤

;

end
case intersect(𝑐ℎ𝑢𝑛𝑘𝑤, 𝑐ℎ𝑢𝑛𝑘𝑒) do

// intersect() checks if two chunks have an intersection point.
Split the concatenation of 𝑐ℎ𝑢𝑛𝑘𝑤
and 𝑐ℎ𝑢𝑛𝑘𝑒
sub-Chunks;
𝑝 ← sub-Chunks;


from intersection point into two

end

end

end

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

16

17

18

19

20

21

22

end
foreach 𝑟 ∈ (
Remove 𝑟 if

𝑠 ∪ 
𝑝) do
|𝑟| < 60;

end

23
24 end
25 return 

𝑠, 
𝑝

a distance of 1 cm, in two experiments it was in lines with a distance of 2 cm, and in two experiments it was random
with a distance of approximately 2 cm. The proposed method is implemented in MATLAB R2020a, where the code
and details of the experiments can be found in Dehshibi and Adamatzky (2020).

These results are consistent with the experiments carried out on the electrical activity of Physarum polycephalum
Adamatzky (2013, 2018a) where it has been reported that the length of the Physarum spike is between 60 and 120
seconds. Physarum is faster than fungi in terms of growth. Now, with further observations, we can hypothesise that
the length of the fungal spikes cannot be less than 60 seconds.

4.2. Complexity Analysis

To quantify the complexity of the electrical signalling recorded, we used the following measurements:

1. The Shannon entropy (𝐻) is calculated as 𝐻 = − ∑

𝑤∈𝑊 (𝜈(𝑤)∕𝜂 ⋅ 𝑙𝑛(𝜈(𝑤)∕𝜂)), where 𝜈(𝑤) is a number of
times the neighbourhood conﬁguration 𝑤 is found in conﬁguration 𝑊 , and 𝜂 is the total number of spike events
found in all channels of the experiment.

2. Simpson’s diversity (𝑆) is calculated as 𝑆 = ∑

𝑤∈𝑊 (𝜈(𝑤)∕𝜂)2. It linearly correlates with Shannon entropy for
𝐻 < 3 and the relationship becomes logarithmic for higher values of 𝐻. The value of 𝑆 ranges between 0 and
1, where 1 represents inﬁnite diversity and 0, no diversity.

3. Space ﬁlling (𝐷) is the ratio of non-zero entries in 𝑊 to the total length of string.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 12 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

Figure 10: Results of applying proposed algorithms in Nenadic and Burdick (2004); Shimazaki and Shinomoto (2010) to
(a) 𝑆𝑙𝑖𝑐𝑒1 and (b) 𝑆𝑙𝑖𝑐𝑒2. Note that the wavelet-based method can only locate spike arrival time. The kernel bandwidth
optimisation can, however, extract the spike region.

(a)

(b)

Figure 11: (a) Spike arrival time located by the expert. Here we used augmented pink arrow to point to these spikes.
(b) Spike regions extracted by the proposed method. Spike regions are alternatively coloured in orange and violet. The
green areas point to pseudo-spike regions that are mistaken for spikes. Blue rectangles with dash edge show overestimated
refractory periods. We used black arrows to point to the missed spikes.

4. Expressiveness (𝐸) is calculated as the Shannon entropy 𝐻 divided by space-ﬁlling ratio 𝐷, where it reﬂects

the ‘economy of diversity’.

5. Lempel–Ziv complexity (𝐿𝑍) is used to assess temporal signal diversity, i.e., compressibility. Here, we repre-
sented the spiking activity of mycelium with a binary string where ‘1s’ indicates the presence of a spike and ‘0s’
otherwise. Formally, as both the barcode and the channels’ electrical activity are stored as PNG images, LZ is
the ratio of the barcode image size to the size of the electrical activity image (see Figure 14 and 15).

6. Perturbation complexity index 𝑃 𝐶𝐼 = 𝐿𝑍∕𝐻.

To calculate Lempel–Ziv complexity, we saved each signal as a PNG image (see two examples in Figure 15), where
the ‘deﬂation’ algorithm used in PNG lossless compression Deutsch and Gailly (1996); Howard (1993); Roelofs and
Koman (1999) is a variation of the classical LZ77 algorithm Ziv and Lempel (1977). We employed this approach as
the recorded signal is a non-binary string. We take the largest PNG ﬁle size to normalise this measurement.

In order to assess signal diversity across all channels and observations, each experiment was represented by a binary
matrix with a row for each channel and a column for each observation. This binary matrix is then concatenated by
observation to form a single binary string. We used Kolmogorov complexity algorithm Kaspar and Schuster (1987)
to measure the Lempel—-Ziv complexity (𝐿𝑍𝑐) across channels. 𝐿𝑍𝑐 captures both temporal diversity on a single

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 13 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

(c)

(d)

(e)

(f)

Figure 12: Distribution of spike event lengths with superimposed Gaussian and Adaptive bandwidth kernels Shimazaki and
Shinomoto (2010). (a,b) In-line electrode arrangements with a distance of 1 cm. (c,d) In-line electrode arrangements with
a distance of 2 cm. (e,f) Random electrode arrangements with an approximate distance of 2 cm.

Table 1
The dominant value and bandwidth for the length and amplitude of the spike in each experiment across all recording
channels. Duration and amplitude of spikes are estimated by the probability density function (PDF) and the adaptive
bandwidth kernel (ABK) Shimazaki and Shinomoto (2010). The bold-faced blue and red entries show the absolute
minimum and maximum values, respectively. As we have bi-directional potential changes, we have considered absolute
value.

#Channels #Spikes

PDF

ABK

PDF

ABK

Length (sec)

Amplitude (V)

#1
#2
#3
#4
#5
#6

8
5
4
5
5
15

565
447
124
951
573
862

Dominant Bandwidth Dominant Bandwidth Dominant Bandwidth Dominant Bandwidth

84.00
366.80
84.00
534.12
334.25
1014.72

75.61
154.31
75.61
80.09
74.52
99.53

84.00
625.60
84.00
534.12
334.25
1014.72

60.22
126.47
60.22
84.80
80.9
92.67

0.00003
0.00642
0.00003
-0.00239
-0.01536
-0.00172

0.00048
0.00544
0.00048
0.00301
0.00218
0.00381

-0.00117
0.00642
-0.00117
-0.00239
-0.01462
-0.01277

0.00576
0.00667
0.00576
0.00508
0.00357
0.00591

channel and spatial diversity across channels. We also normalised 𝐿𝑍𝑐 by dividing the raw value by the randomly
shuﬄed value obtained for the same binary input sequence. Since the value of 𝐿𝑍 for a ﬁxed-length binary sequence
is maximum if the sequence is absolutely random, the normalised values represent the degree of signal diversity on a
scale from 0 to 1. The results of the calculation of these complexity measurements for all six conﬁgurations are shown
in Figure 16 and summarised in Table 2.

We calculated the aforementioned complexity criteria for three forms of writing to illustrate the communication
complexity of the mycelium substrate, including (1) news items3, (2) random alphanumeric sequences4 and (3) periodic

3https://www.sciencemag.org/news/2020/07/meet-lizard-man-reptile-loving-biologist-tackling-some-biggest-questions-evolution
4We used available service at https://www.random.org/

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 14 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

(c)

(d)

(e)

(f)

Figure 13: Distribution of spike maximum amplitudes with superimposed Gaussian and Adaptive bandwidth kernels Shi-
mazaki and Shinomoto (2010) for (a,b) in lines electrode placement with a distance of 1 cm. (c,d) in lines electrode
placement with a distance of 2 cm. (e,f) random electrode placement with an approximate distance of 2 cm.

Table 2
The mean of complexity measurements for six experiments.

#Channel #Spike

#1
#2
#3
#4
#5
#6

8
5
4
5
5
15

565
447
124
951
573
862

Lempel–Ziv
complexity
0.79
0.91
0.75
0.93
0.88
0.69

Shannon
entropy
45.81
63.27
22.57
123.11
75.75
39.96

Simpson’s
diversity
0.76
0.98
0.61
0.89
0.79
0.71

Space
ﬁlling
30.68×10−5
35.20×10−5
48.10×10−5
57.30×10−5
53.02×10−5
24.20×10−5

Kolmogorov

PCI

Expressiveness

30.36×10−4
35.78×10−4
10.94×10−4
56.05×10−4
52.80×10−4
25.06×10−4

0.365
0.021
0.333
0.072
0.077
0.207

20.8×104
18.6×104
29.71
23.8×104
16.4×104
20.4×104

alphanumeric sequences encoded with Huﬀman code Huﬀman (1952), see barcode in Figure 17. Table 3 presents the
results of the comparison. We also considered two podcasts in English (387 seconds) and Chinese (385 seconds)
to compare the complexity of fungal spiking with human speech. Both podcasts were in MP3 format at a sampling
rate of 44100 Hz. We randomly selected two chunks from two electrical activity channels with a duration of 388
and 342 seconds to compare with the English and Chinese podcasts, respectively. We observed that, in both cases, the
Kolmogorov complexity of the fungal is lower than the human speech, implying the fact that the amount of information
transmitted by the fungi is less than the human voice. From a technical point of view, we computed the DC level of
each signal and binarised the signal with respect to that level Huang and Lin (2009). To binaries the signal, we set the
inputs with values less than or equal to the DC level to 0 and the rest of the inputs to 1. The ﬁndings are shown in
Figure 18.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 15 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

(c)

(d)

(e)

(f)

Figure 14: Barcode-like representation of spike events in various channels for (a,b) in-line electrode arrangements at a
distance of 1 cm, (c,d) in-line electrode arrangements at a distance of 2 cm, and (e,f) random electrode arrangements at
an approximate distance of 2 cm.

5. Discussion

We developed algorithmic framework for exhaustive characterisation of electrical activity of a substrate colonised
by mycelium of oyster fungi Pleurotus djamor. We evidenced spiking activity of the mycelium. We found that average

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 16 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

Figure 15: Two samples from input channels, which are saved in black and white PNG format without axes and annotations.

Table 3
The complexity measurements for pieces of news, a random sequence of alphanumeric, a periodic sequence of alphanumeric
along with three chunks randomly selected from our experiments.

News
Random sequence
Periodic sequence
Chunk 1
Chunk 2
Chunk 3

Length

36187
36002
36006
36000
36000
36000

Lempel–Ziv
complexity
0.127919
0.125465
0.127090
0.067611
0.007250
0.068417

Shannon
entropy
4.421728
5.770331
3.882058
16.194914
15.478087
31.680374

Simpson’s
diversity
0.999941
0.999941
0.999937
0.947368
0.944444
0.976190

Space
ﬁlling
0.465996
0.469835
0.442426
0.000556
0.000528
0.001194

Kolmogorov

PCI

Expressiveness

0.765382
1.001850
0.076508
0.006307
0.006727
0.012613

0.173096
0.173621
0.019708
0.000389
0.000435
0.000398

9.49
12.28
8.77
29150.84
29326.90
26523.10

dominant duration of an action-potential like spike is 402 sec. The spikes amplitudes’ depends on the location of the
source of electrical activity related to the position of electrodes, thus the amplitudes provide less useful information.
The amplitudes vary from 0.5 mV to 6 mV. This is indeed low compared to 50-60 mV of intracellular recording,
nevertheless understandable due to the fact the electrodes are inserted not even in mycelium strands but in the substrate
colonised by mycelium. The shift of the distribution to higher values of spike amplitude in experiments with a distance
of 2 cm between the electrodes might indicate that the width of the propagation of the excitation wave front exceeds
1 cm and might even be close to 2 cm.

The spiking events have been characterised with several complexity measures. Most measures, apart of Kol-
mogorov complexity shown a low degree of variability between channels (diﬀerent sites of the recordings). The Kol-
mogorov complexity of fungal spiking varies from 11×10−4 to 57×10−4. This might indicated mycelium sub-networks
in diﬀerent parts of the substrate have been transmitting diﬀerent information to other parts of the mycelium network.
This is somehow echoes experimental results on communication between ants analysed with Kolmogorov complexity:
longer paths communicated ants corresponds to higher values of complexity Ryabko and Reznikova (1996).

LZ complexity of fungal language (Tab. 2) is much higher than of news, random or periodic sequences (Tab. 3).
The same can be observed for Shannon entropy. Kolmogorov complexity of the fungal language is much lower than
that of news sampler or random or periodic sequences. Complexity of European languages based on their compress-
ibility Sadeniemi et al. (2008) is shown in Figure 19, French having lowest LZ complexity 0.66 and Finnish highest
LZ complexity 0.79. Fungal language of electrical activity has minimum LZ complexity 0.61 and maximum 0.91
(media 0.85, average 0.83). Thus, we can speculate that a complexity of fungal language is higher than that of human
languages (at least for European languages).

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 17 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(c)

(e)

(b)

(d)

(f)

Figure 16: (a) Shannon entropy, (b) Simpson’s diversity, (c) Space ﬁlling, (d) Expressiveness, (e) Lempel–Ziv complexity,
and (f) Perturbation complexity index. All measurements are scaled to the range of [0, 1].

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 18 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

(a)

(b)

(c)

Figure 17: Binary representation of (a) pieces of news,(b) random sequence of alphanumeric, and (c) periodic sequence
of alphanumeric after applying Huﬀman coding.

(a) DC level = −6.49×10−5, sampling rate = 44 kHz, samples
= 17100912, Kolmogorov = 0.739866

(b) DC level = −27.23 × 10−5, sampling rate = 1 Hz, samples
= 388, Kolmogorov = 0.598645

(c) DC level = −20.05 × 10−5, sampling rate = 44 kHz, sam-
ples = 15057264, Kolmogorov = 0.753261

(d) DC level = −9.31 × 10−5, sampling rate = 1 Hz, samples
= 342, Kolmogorov = 0.574892

Figure 18: Comparison of the human voice in (a-c) English/Chinese with the electrical activity of fungi with the duration
of (b-d) 388/342 seconds.

Acknowledgement

This project has received funding from the European Union’s Horizon 2020 research and innovation programme

FET OPEN “Challenging current thinking” under grant agreement No 858132.

Declaration of competing interest

There is no conﬂict of interest with this paper.

Credit Author Statement

All authors are responsible for the experiments and writing of the paper.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 19 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

Figure 19: Lempel-Ziv complexity of European languages (data from Sadeniemi et al. (2008)) with average complexity of
fungal (‘fu’) electrical activity language added.

References
Adamatzky, A., 2013. Tactile bristle sensors made with slime mold. IEEE Sensors journal 14, 324–332.
Adamatzky, A., 2018a. On spiking behaviour of oyster fungi Pleurotus djamor. Scientiﬁc reports 8, 1–7.
Adamatzky, A., 2018b. Towards fungal computer. Interface focus 8, 20180029.
Adamatzky, A., 2019. Plant leaf computing. Biosystems 182, 59–64.
Adamatzky, A., Chiolerio, A., Sirakoulis, G., 2021. Electrical resistive spiking of fungi. Biophysical Reviews and Letters 0, 1–7. doi:10.1142/

S1793048021500016.

Adamatzky, A., Nikolaidou, A., Gandia, A., Chiolerio, A., Dehshibi, M.M., 2020. Reactive fungal wearable. Biosystems 199, 104304.
Aidley, D.J., Ashley, D., 1998. The physiology of excitable cells. volume 4. Cambridge University Press Cambridge.
Belousov, B.P., 1959. A periodic reaction and its mechanism. Compilation of Abstracts on Radiation Medicine 147, 1.
Bingley, M., 1966. Membrane potentials in amoeba proteus. Journal of Experimental Biology 45, 251–267.
Davidenko, J.M., Pertsov, A.V., Salomonsz, R., Baxter, W., Jalife, J., 1992. Stationary and drifting spiral waves of excitation in isolated cardiac

muscle. Nature 355, 349.

Dehshibi, M.M., Adamatzky, A., 2020. Supplementary material for “Electrical activity of fungi: Spikes detection and complexity analysis". https:

//doi.org/10.5281/zenodo.3997031. (Accessed on 24/08/2020).

Dehshibi, M.M., Shanbehzadeh, J., Pedram, M.M., 2020. A robust image-based cryptology scheme based on cellular nonlinear network and local

image descriptors. International Journal of Parallel, Emergent and Distributed Systems 35, 514–534.

Dehshibi, M.M., Shirmohammadi, A., Adamatzky, A., 2015. On growing persian words with l-systems: visual modeling of neyname. International

Journal of Image and Graphics 15, 1550011.

Deutsch, P., Gailly, J., 1996. Zlib compressed data format speciﬁcation version 3.3. Technical Report. RFC 1950, May.
Eckert, R., Brehm, P., 1979. Ionic mechanisms of excitation in paramecium. Annual review of biophysics and bioengineering 8, 353–383.
Farkas, I., Helbing, D., Vicsek, T., 2002. Social behaviour: Mexican waves in an excitable medium. Nature 419, 131.
Farkas, I., Helbing, D., Vicsek, T., 2003. Human waves in stadiums. Physica A: statistical mechanics and its applications 330, 18–24.
Franke, F., Natora, M., Boucsein, C., Munk, M.H., Obermayer, K., 2010. An online spike detection and spike classiﬁcation algorithm capable of

instantaneous resolution of overlapping spikes. Journal of computational neuroscience 29, 127–148.

Fromm, J., Lautner, S., 2007. Electrical signals and their physiological signiﬁcance in plants. Plant, cell & environment 30, 249–257.
Gholami, N., Dehshibi, M.M., Adamatzky, A., Rueda-Toicen, A., Zenil, H., Fazlali, M., Masip, D., 2020. A novel method for reconstructing ct
images in gate/geant4 with application in medical imaging: A complexity analysis approach. Journal of Information Processing 28, 161–168.

Gorbunov, L., Kirsanov, V., 1987. Excitation of plasma waves by an electromagnetic wave packet. Sov. Phys. JETP 66, 40.
Gotman, J., Wang, L., 1991. State-dependent spike detection: concepts and preliminary results. Electroencephalography and clinical Neurophysi-

ology 79, 11–19.

Hall, C.A., Meyer, W.W., 1976. Optimal error bounds for cubic spline interpolation. Journal of Approximation Theory 16, 105–122.
Hansma, H.G., 1979. Sodium uptake and membrane excitation in paramecium. The Journal of cell biology 81, 374–381.
Hodgkin, A.L., Huxley, A.F., 1952. A quantitative description of membrane current and its application to conduction and excitation in nerve. The

Journal of physiology 117, 500–544.

Howard, P.G., 1993. The Design and Analysis of Eﬃcient Lossless Data Compression Systems. Ph.D. thesis. Citeseer.
Huang, H., Lin, F., 2009. A speech feature extraction method using complexity measure for voice activity detection in wgn. Speech Communication

51, 714–723.

Huﬀman, D.A., 1952. A method for the construction of minimum-redundancy codes. Proceedings of the IRE 40, 1098–1101.
IEEE, 2011. Ieee standard for transitions, pulses, and related waveforms. IEEE Std 181-2011 (Revision of IEEE Std 181-2003) , 1–71doi:10.1109/

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 20 of 22

LZ complexity0.600.650.700.750.800.85LanguagefresptgaitenslnlmtdaelsvlvdeplltsketﬁfuElectrical activity of fungi: Spikes detection and complexity analysis

IEEESTD.2011.6016198.

Kaspar, F., Schuster, H., 1987. Easily calculable measure for the complexity of spatiotemporal patterns. Physical Review A 36, 842.
Kittel, C., 1958. Excitation of spin waves in a ferromagnet by a uniform rf ﬁeld. Physical Review 110, 1295.
Lilly, J.M., 2017. Element analysis: a wavelet-based method for analysing time-localized events in noisy time series. Proceedings of the Royal

Society A: Mathematical, Physical and Engineering Sciences 473, 20160776.

Lilly, J.M., Olhede, S.C., 2008. Higher-order properties of analytic wavelets. IEEE Transactions on Signal Processing 57, 146–160.
Lilly, J.M., Olhede, S.C., 2012. Generalized morse wavelets as a superfamily of analytic wavelets. IEEE Transactions on Signal Processing 60,

6036–6041.

Liu, Z., Wang, X., Yuan, Q., 2020. Robust detection of neural spikes using sparse coding based features. Mathematical Biosciences and Engineering

17, 4257.

Marple, L., 1999. Computing the discrete-time" analytic" signal via ﬀt. IEEE Transactions on signal processing 47, 2600–2603.
Masi, E., Ciszak, M., Santopolo, L., Frascella, A., Giovannetti, L., Marchi, E., Viti, C., Mancuso, S., 2015. Electrical spiking in bacterial bioﬁlms.

Journal of The Royal Society Interface 12, 20141036.

McGillviray, A.M., Gow, N.A., 1987. The transhyphal electrical current of Neuruspua crassa is carried principally by protons. Microbiology 133,

2875–2881.

Minoofam, S.A.H., Dehshibi, M.M., Bastanfard, A., Eftekhari, P., 2012. Ad-hoc ma’qeli script generation using block cellular automata. J. Cell.

Autom. 7, 321–334.

Minoofam, S.A.H., Dehshibi, M.M., Bastanfard, A., Shanbehzadeh, J., 2014. Pattern formation using cellular automata and l-systems: A case study

in producing islamic patterns, in: Cellular Automata in Image Processing and Geometry. Springer, pp. 233–252.

Nelson, P.G., Lieberman, M., 2012. Excitable cells in tissue culture. Springer Science & Business Media.
Nenadic, Z., Burdick, J.W., 2004. Spike detection using the continuous wavelet transform. IEEE transactions on Biomedical Engineering 52, 74–87.
Obeid, I., Wolf, P.D., 2004. Evaluation of spike-detection algorithms fora brain-machine interface application. IEEE Transactions on Biomedical

Engineering 51, 905–911.

Parsa, S.S., Sourizaei, M., Dehshibi, M.M., Shateri, R.E., Parsaei, M.R., 2017. Coarse-grained correspondence-based ancient sasanian coin classi-

ﬁcation by fusion of local features and sparse representation-based classiﬁer. Multimedia Tools and Applications 76, 15535–15560.

Quiroga, R.Q., Nadasdy, Z., Ben-Shaul, Y., 2004. Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural

computation 16, 1661–1687.

Rácz, M., Liber, C., Németh, E., Fiáth, R., Rokai, J., Harmati, I., Ulbert, I., Márton, G., 2020. Spike detection and sorting with deep learning.

Journal of Neural Engineering 17, 016038.

Roelofs, G., Koman, R., 1999. PNG: the deﬁnitive guide. O’Reilly & Associates, Inc.
Ryabko, B., Reznikova, Z., 1996. Using shannon entropy and kolmogorov complexity to study the communicative system and cognitive capacities

in ants. Complexity 2, 37–42.

Sablok, S., Gururaj, G., Shaikh, N., Shiksha, I., Choudhary, A.R., 2020. Interictal spike detection in eeg using time series classiﬁcation, in: 2020

4th International Conference on Intelligent Computing and Control Systems (ICICCS), IEEE. pp. 644–647.

Sadeniemi, M., Kettunen, K., Lindh-Knuutila, T., Honkela, T., 2008. Complexity of european union languages: A comparative approach. Journal

of Quantitative Linguistics 15, 185–211.

Shimazaki, H., Shinomoto, S., 2010. Kernel bandwidth optimization in spike rate estimation. Journal of computational neuroscience 29, 171–182.
Slonczewski, J., 1999. Excitation of spin waves by an electric current. Journal of Magnetism and Magnetic Materials 195, L261–L268.
Taghipour, N., Javadi, H.H.S., Dehshibi, M.M., Adamatzky, A., 2016. On complexity of persian orthography: L-systems approach. Complex

Systems 25, 127–156.

Trebacz, K., Dziubinska, H., Krol, E., 2006. Electrical signals in long-distance communication in plants, in: Communication in plants. Springer,

pp. 277–290.

Tsoi, M., Jansen, A., Bass, J., Chiang, W.C., Seck, M., Tsoi, V., Wyder, P., 1998. Excitation of a magnetic multilayer by an electric current. Physical

Review Letters 80, 4281.

Vicnesh, J., Hagiwara, Y., 2019. Accurate detection of seizure using nonlinear parameters extracted from eeg signals. Journal of Mechanics in

Medicine and Biology 19, 1940004.

Wang, Z., Wu, D., Dong, F., Cao, J., Jiang, T., Liu, J., 2020. A novel spike detection algorithm based on multi-channel of bect eeg signals. IEEE

Transactions on Circuits and Systems II: Express Briefs .

Wilson, S.B., Emerson, R., 2002. Spike detection: a review and comparison of algorithms. Clinical Neurophysiology 113, 1873–1881.
Wilson, S.B., Turner, C.A., Emerson, R.G., Scheuer, M.L., 1999. Spike detection ii: automatic, perception-based detection and clustering. Clinical

neurophysiology 110, 404–411.

Zhabotinsky, A., 1964. Periodic processes of malonic acid oxidation in a liquid phase. Bioﬁzika 9, 11.
Zhabotinsky, A.M., 2007. Belousov-zhabotinsky reaction. Scholarpedia 2, 1435.
Zimmermann, M.R., Mithöfer, A., 2013. Electrical long-distance signaling in plants, in: Long-Distance Systemic Signaling and Communication in

Plants. Springer, pp. 291–308.

Ziv, J., Lempel, A., 1977. A universal algorithm for sequential data compression. IEEE Transactions on information theory 23, 337–343.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 21 of 22

Electrical activity of fungi: Spikes detection and complexity analysis

Mohammad Mahdi Dehshibi received his PhD in Computer Science from Islamic Azad University, Tehran, Iran in 2017.
He is currently a postdoctoral research fellow at Universitat Oberta de Catalunya (UOC). He was a visiting researcher in
Unconventional Computing Lab, UWE, Bristol, U.K. He has contributed to over 50 papers published in scientiﬁc Journals
or International Conferences. His research interests include Unconventional Computing, Aﬀective Computing, and Cellular
Automata.

Andrew Adamatzky is Professor of Unconventional Computing and Director of the Unconventional Computing Labora-
tory, Department of Computer Science, University of the West of England, Bristol, UK. He does research in molecular
computing, reaction-diﬀusion computing, collision-based computing, cellular automata, slime mould computing, massive
parallel computation, applied mathematics, complexity, nature-inspired optimisation, collective intelligence and robotics,
bionics, computational psychology, non-linear science, novel hardware, and future and emergent computation. He authored
seven books, mostly notable are ‘Reaction-Diﬀusion Computing’, ‘Dynamics of Crow Minds’, ‘Physarum Machines’, and
edited twenty-two books in computing, most notable are ‘Collision Based Computing’, ‘Game of Life Cellular Automata’,
‘Memristor Networks’; he also produced a series of inﬂuential artworks published in the atlas ‘Silence of Slime Mould’.
He is founding editor-in-chief of ‘J of Cellular Automata’ and ‘J of Unconventional Computing’ and editor-in-chief of ‘J

Parallel, Emergent, Distributed Systems’ and ‘Parallel Processing Letters’.

Dehshibi and Adamatzky.: Preprint submitted to Elsevier

Page 22 of 22

View publication stats

