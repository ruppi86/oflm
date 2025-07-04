Ref. Ares(2021)200138 - 10/01/2021

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Horizon 2020

Deliverable D4.1
Intrinsic spiking of electrical potential of mycelium

Date of preparation: 2020/11/30

Revision: 1

Start date of project: 2019/12/01 Duration: 36 months

Project coordinator: UWE

Classiﬁcation: public

Partners:
lead: UWE

contribution: UWE

Project website:

http://fungar.eu/

H2020-FETopen-2019

Deliverable D4.1

Page 1 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

DELIVERABLE SUMMARY SHEET

Grant agreement number:

858132

Project acronym:

FUNGAR

Deliverable No:

Deliverable D4.1

Due date:

M12

Delivery date:

2020/11/30

Name:

Intrinsic spiking of electrical potential of mycelium

Description:

This tasks reported on here targets establishing communication
protocols with mycelium network. We designed and implemented
an experimental setup for recording electrical activity of fungi in
a controlled environment. We studied spontaneous discharges,
(ir)regularly oscillating potential, particularly distinguishing pat-
terns of spontaneous activity formed by single spike of electrical
potential and diﬀerent types of bursts according to intra-burst
ﬁring frequency.
Key ﬁndings are as following. Oyster fungi Pleurotus djamor gen-
erate actin potential like spikes of electrical potential. The trains
of spikes manifest propagation of growing mycelium in a substrate,
transportation of nutrients and metabolites and communication
processes in the mycelium network. We propose original tech-
niques for detecting and classifying the spiking activity of fungi.
Using these techniques, we analyse the information-theoretic com-
plexity of the fungal electrical activity. The results pave ways for
future research on sensorial fusion and decision making of fungi.

Partners owning:

UWE

Partners contributed:

UWE

Made available to:

public

Page 2 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Contents

1 Electrical activity of fungi

2 Experimental interface

3 Proposed method

3.1 Slicing fungi electrical activity . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3.2 Detecting time-localised events by Morse-based wavelets . . . . . . . . . . . . . .
3.3 Analytical signal envelope for locating spike pattern . . . . . . . . . . . . . . . .

4 Experimental results
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
4.1 Objective analysis
4.2 Complexity Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

5 Complexity of fungal language

1 Electrical activity of fungi

3

5

6
7
7
9

15
15
20

24

Excitation is an essential property of all living creatures, from bacteria [35], Protists [10, 15, 24],
fungi [36] and plants [19, 54, 61] to vertebrates [8, 12, 25, 37]. Waves of excitation could be also
found in various physical [21, 29, 51, 55], chemical [9, 59, 60] and social systems [16, 17].

Not only neurons spike. Action potential-like spikes of electrical potential have been discov-
ered using intracellular recording of mycelium of Neurospora crassa [50] and further conﬁrmed in
intra-cellular recordings of action potential in hypha of Pleurotus ostreatus and Armillaria bul-
bosa [40] and in extracellular recordings of fruit bodies of and substrates colonized by mycelium

Figure 1: Example of electrical spiking activity recorded from a hemp substrate colonised by
mycelium of P. ostreatus.

Deliverable D4.1

Page 3 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

of Pleurotus ostreatus [4]. While the exact nature of the travelling spikes remains uncertain we
can speculate, by drawing analogies with oscillations of electrical potential of slime mould [3],
that the spikes in fungi are triggered by calcium waves, reversing of cytoplasmic ﬂow, translo-
cation of nutrients and metabolites (Fig. 1). Studies of electrical activity of higher plants can
bring us even more clues [19]. Thus, the plants use the electrical spikes for a long-distance com-
munication aimed to coordinate an activity of their bodies. The spikes of electrical potential
in plants relate to a motor activity, responses to changes in temperature, osmotic environment
and mechanical stimulation.
In experiments with Pleurotus ostreatus5 we demonstrated that
fruit bodies of oyster fungi exhibit trains of action-like spike of extracellularly recorded electrical
potential. We observed two types of spikes: high-frequency spikes, duration nearly 3 min, and
low-frequency spikes, duration nearly 14 min. The spikes are observed in trains of 10-30 spikes.
The depolarisation and repolarisation rates of both types of spikes are the same. Refractory
period of a high-frequency spike is one sixth of the spike’s period, and of a low-frequency spike
one third of the spike’s period. We showed that fruit bodies respond with spikes of electrical
potential in response to physical, chemical and thermal stimulation; not only a simulated body
responds with a spike but other fruit bodies of the cluster respond as well. We believe the spikes
of electrical potential travelling in mycelium networks play the same roles of information carriers
as action potential travelling along neural pathways in e.g. human brains.

When recorded with diﬀerential electrodes, a propagating excitation wave is manifested by
spike. In our recent studies [5, 6], we demonstrated that the oyster fungi Pleurotus djamor gen-
erate action potential like impulses of electrical potential. We observed trains of the spontaneous
spikes1 with two types of activity, i.e., high-frequency (period 2.6 min) and low-frequency (pe-
riod 14 min). Appropriate utilisation of this information is, however, subject to the accurate
extraction of the EC spike waveform, separating it from the background activity of neighbouring
cells, and sorting the characteristics.

Lack of an algorithmic framework for exhaustive characterisation of the electrical activity
of a substrate colonised by mycelium of oyster fungi Pleurotus djamor motivated us to develop
this framework to extract spike patterns, quantify the diversity of spiking events, and measure
the complexity of fungal electrical communication. We evidenced the spiking activity of the
mycelium (see an example in Fig. 2), which will enable us to develop an experimental prototype
of fungi-based information processing devices.

We evaluated the proposed framework in comparison to the existing, in neuroscience, tech-
niques of spike detection [38, 49], and observed considerable improvement in extracting spike
activity periods. Evaluation of the proposed method for detecting spikes events compared to the
determined spikes’ arrival time by an expert shows true-positive and false-positive rates of 76%
and 16%, respectively. We found that the average dominant duration of an action-potential-like
spike is 402 sec. The spikes’ amplitude varies from 0.5 mV to 6 mV and depends on the location
of the electrical activity source (the position of electrodes). We observed that the Kolmogorov
complexity of fungal spiking varies from 11×10−4 to 57×10−4. This might indicate mycelium
sub-networks in diﬀerent parts of the substrate have been transmitting diﬀerent information
to other parts of the mycelium network, i.e., more extended propagation of excitation wave
corresponds to higher values of complexity.

1Calling the spikes spontaneous means that they are not invoked by an intentional external stimulation.

Otherwise, the spikes indeed reﬂect physiological and morphological processes ongoing in mycelial networks.

Page 4 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

Figure 2: The electrical activity of the mycelium of the grey oyster fungi. (a) Example of a
dynamics of electrical potentials recorded from eight channels of the same cluster during 63 hours.
(b) Three channels are zoomed in the inserts to show the rich combination of slow (hours) drift
of base electrical potential combined with relatively fast (minutes) oscillations of the potential.
(c) All ‘classical’ parts of a spike, i.e., depolarisation, repolarisation and refractory period, can
be found in this exemplar spike. This spike has a period of 220 s, from base-level potential to
refractory-like period, and refractory period of 840 s. The depolarisation and repolarisation rates
are 0.03 and 0.009 mV/s, respectively.

2 Experimental interface

Extracellular (EC) recordings of action potentials have been widely used to record and measure
neural activity in a number of species. The broad functionality of this method has been shown for
studying neural activity in several applications, ranging from single nerve ﬁbres in invertebrate
sensory organs to cortical neurons involved in cognition, learning and memory [20, 42, 53].

A wood shavings substrate was colonised by the mycelium of the grey oyster fungi, Pleurotus
ostreatus (Ann Miller’s Speciality Mushrooms Ltd, UK). The substrate was placed in a hydro-
ponic growing tent with a silver Mylar lightproof inner lining (Green Box Tents, UK). Figure 3
shows three examples of the experimental set-up.

We inserted pairs of iridium-coated stainless steel sub-dermal needle electrodes (Spes Medica
SRL, Italy), with twisted cables into the colonised substrate to obtain electrical activity. Using an
ADC-24 (Pico Technology, UK) high-resolution data logger with a 24-bit A/D converter, galvanic
isolation and software-selectable sample rates all contribute to a superior noise-free resolution.
We recorded electrical activity one sample per second, where the minimum and maximum logging
times were 60.04 and 93.45 hours, respectively. During the recording, the logger makes as many
measurements as possible (typically up to 600 per second) and saves the average value. We set
the acquisition voltage range to 156 mV with an oﬀset accuracy of 9 µV at 1 Hz to maintain a gain
error of 0.1%. Each electrode pair was considered independently with the noise-free resolution of
17 bits and conversion time of 60 ms. In our experiments, electrode pairs were arranged in one
of two conﬁgurations: random placement or in lines. Distance between electrodes was 1-2 cm.
In each cluster, we recorded 5–16 electrode pairs (channels) simultaneously.

Deliverable D4.1

Page 5 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

Figure 3: Three examples of the experimental set-up with (a) in lines placement of electrodes
(1 cm distance), (b) in lines placement of electrodes (2 cm distance), (c) random electrode
placement.

3 Proposed method

A spike event can be formally deﬁned as an extracellular signal that exceeds a simple amplitude
threshold and passes through a subsequent pair of user-speciﬁed time-voltage boxes. The spike,
which includes depolarisation, repolarisation, and refractory periods, reﬂects physiological and
morphological processes ongoing in mycelial networks. To extract spike events, we proposed an
unsupervised method which consists of three major steps.

In the ﬁrst step, we split the whole recording period, F (t), into k chunks, fk(t), with respect
to the signal’s transitions. To determine the transitions, we estimated the state levels of the
signal by its histogram and identiﬁed all regions that cross the upper-state boundary of the low
state and the lower-state boundary of the high state. Then, we calculated scale-to-frequency
conversions of the analytic signal in each chunk using Morse wavelet basis [32]. To assess the
presence of spike-like events, we scaled the wavelet coeﬃcients at each frequency and obtained
the sum of the scales that were less than the threshold deﬁned in Algorithm 1. Finally, we
selected regions of interest (ROI) enclosed between a consecutive local minimum and maximum
whose lengths were greater than 30 sec.

Page 6 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

In the second step, we calculated the envelopes of the analytic signal using spline interpolation
over local maximums. To determine the analytical signal, we ﬁrst applied the discrete approx-
imation of Laplace’s diﬀerential operator to fk(t) to obtain a ﬁnite sequence of equally-spaced
samples. Then, we converted this ﬁnite sequence into a same-length sequence of equally-spaced
samples of the discrete-time Fourier transform. From the average signal envelope, we extracted
regions that fall in a consecutive local minimum and maximum. These regions created constraints
that observing them led to the identiﬁcation of spike events.

In the third step, we preserved ROIs from the ﬁrst step where satisﬁed constraints obtained
in the second step. The signal envelope could guide wavelet decomposition in an unsupervised
way to cluster signal into the spike, pseudo-spike, and background activity of neighbouring cells.
We detailed the proposed method in the following sub-sections.

3.1 Slicing fungi electrical activity

To split the fungi electrical activity, F (t), with a length of t second into k chunks fk(t), 1 ≤
k ≤ t − 1, we used signal transitions that compose each pulse. To determine the transitions, we
estimated the state levels of F (t) by a histogram method [1]. Then, we identiﬁed all regions that
cross the upper-state boundary of the low state and the lower-state boundary of the high state.
To estimate the states of the signal, we followed the following steps.

1. Determining the minimum, maximum and range of amplitudes.

2. Sorting amplitude values into the histogram bins and determining the bin width by dividing

the amplitude range to the number of bins.

3. Identifying the lowest- and highest-indexed histogram bins, hblow, hbhigh, with non-zero

counts.

4. Dividing the histogram into two sub-histograms, where the indices of the lower and upper
2 (hbhigh − hblow) ≤ hb ≤

2 (hbhigh − hblow) and hblow + 1

histogram bins are hblow ≤ hb ≤ 1
hbhigh, respectively.

5. Calculating the mean of the lower and upper histogram to compute the state levels.

Each chunk is then enclosed between the last negative-going transitions of every positive-polarity
pulse and the next positive-going transition. Figure 4 shows slicing results for two channels.

3.2 Detecting time-localised events by Morse-based wavelets

The electrical activity of the mycelium exhibits modulated behaviour with variation in amplitude
and frequency over time. This feature hints that the signal can be analysed with analytic wavelets,
which are naturally grouped into even or cosine-like and odd or sine-like pairs, allowing them to
capture phase variability. A wavelet ψ(t) is a ﬁnite energy function which projects the f (t) onto
a family of time-scale waveforms by translation and scaling. The Morse wavelet, ψβ,γ(t), is an
analytic wavelet whose Fourier transforms is supported only on the positive real axis [30, 32].
This wavelet is deﬁned in the frequency domain for β ≥ 0 and γ > 0 using Eq. 1

Deliverable D4.1

Page 7 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 4: Slicing electrical potential recordings for two channels.

ψβ,γ(t) =

1
2π

(cid:90) ∞

−∞

Ψβ,γ(ω) eiωt dω,

Ψβ,γ(ω) ≡ aβ,γ ωβ e−ωγ

×






1 ω > 0
1
2 ω = 0
0 ω < 0

.

(1)

where ω is the angular frequency and aβ,γ ≡ 2
is the amplitude coeﬃcient used as a
real-valued normalised constant. Here, e is Euler’s number, β characterises the low-frequency
behaviour, and γ deﬁnes the high-frequency decay. We can rewrite Eq. 1 in the Fourier domain,
parameterised by β and γ as Eq. 2.

(cid:17) 1

γ

(cid:16) eγ
β

φβ,γ(τ, s) ≡

(cid:90) ∞

−∞

1
s

ψ∗

β,γ(

t − τ
s

)f (t) dt =

1
2π

(cid:90) ∞

−∞

eiωτ Ψ∗

β,γ(sω)F(ω) dω.

(2)

where F (ω) is the Fourier transform of f (t), and ∗ denotes the complex conjugate. When
Ψ∗
β,γ(ω) is real-valued, the conjugation may be omitted. The scale variable s causes stretching
In order to reﬂect the energy of f (t) and normalise
or compression of the wavelet in time.
s is usually used. However, we used 1
1√
the time-domain wavelets to preserve constant energy,
s
instead, since we describe time-localised signals by the amplitude. To recover the time-domain
(cid:82) ∞
representation, we can use the inverse Fourier transform by f (t) = 1
−∞ eiωt F (ω) dω and
2π
ψβ,γ(t) = (cid:82) ∞
−∞ eiωt dt = 2πδ(ω), where δ(ω) is the Dirac delta function.

The representation of Morse wavelets can be more oscillatory when both β and γ increase,
and more localised with impulses when these parameters decrease. On the other hand, increasing
β and keeping γ ﬁxed broaden the central portion of the wavelet and increase the long-time decay
rate. Whereas, increasing γ by keeping β constant expands the wavelet envelope without aﬀecting
the long-time decay rate. Following explanations given in [31], we set the symmetry parameter
γ to 3 and the time-bandwidth product P 2 = βγ to 60. We also used L1 normalisation to have

Page 8 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

an equal magnitude in the wavelets when we have equal amplitude oscillatory components at
diﬀerent scales. Figure 5 shows two randomly selected 3000-second chunks of the fungi electrical
activity (namely, Slice1 and Slice2) with their Morse wavelet scalograms.

(a)

(b)

Figure 5: Annotated spikes by the expert over with the Morse wavelet scalogram for (a) Slice1
and (b) Slice2. We added black arrows to point to the spike identiﬁed by the expert. The
scalogram is plotted as a function of time and frequency in which the maximum absolute value
at each frequency is used to normalise coeﬃcient. Frequency axis is displayed on a linear scale.

We observed that using the maximum absolute value at each frequency (level) to normalise
coeﬃcients can help in identifying events that may contain spikes. Hence, we proposed to use
Eq. 3 for normalisation and subsequently set all zero entries to 1.

(cid:18)

gβ,γ(τ, s) =

η ×

κβ,γ(τ, s) = |φβ,γ(τ, s)|(cid:124),
(cid:19)(cid:124)

κβ,γ(τ, s) − mins(κβ,γ(τ, s))
maxs(κβ,γ(τ, s))

.

(3)

where | • | and (•)(cid:124) return the absolute value and the matrix transpose, respectively. Here, η is
a scaling factor that we empirically set it to 240. We used gβ,γ(τ, s) in Algorithm 1 to extract
candidate ROIs, which are shown in Fig. 6.

As shown in Fig. 6(c,d), some of the detected regions are either too short2 or lack repolar-
isation and depolarisation periods that should be removed from B. We proposed Algorithm 2
to remove these regions, which we called them pseudo-spike and inﬂection regions, respectively.
Figure 7 shows the results.

Applying Algorithm 2 led to the loss of two spikes in Slice1 (see Fig. 7(a)) and failure to
remove two pseudo-spike and two inﬂection regions in Slice2 (see Fig. 7(b)). We found that
assessing the analytic signal by its envelope can increase the accuracy of spike detection.

3.3 Analytical signal envelope for locating spike pattern

To obtain the signal envelope, ξ, we calculated the magnitude of its analytic signal. The analytic
signal is found using the discrete Fourier transform as implemented in Hilbert transform. To

2We observed in our previous studies [5, 6] that minimum spike length was 5 mins.

Deliverable D4.1

Page 9 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Algorithm 1: Detecting candidate regions for time-localised events.

Input : gβ,γ(τ, s) – Scaled wavelets coeﬃcients.
Output: B – set of candidate regions.

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

(cid:15) = 0.05 × (max(gβ,γ(τ, s)) − min(gβ,γ(τ, s)));
maxg ← set of all LocalMaximum(gβ,γ(τ, s), (cid:15));
// LocalMaximum() returns τ ∗ if ∀τ ∈ (τ ∗ ± (cid:15)), gβ,γ(τ ∗, s) ≥ gβ,γ(τ, s).
ming ← set of all LocalMinimum(gβ,γ(τ, s), (cid:15));
U ← sort(ming
n = card(U);
// card(A) returns number of entries in A.
if n ≡ 1 (mod 2) then

(cid:83) maxg);

slack ← mean(diﬀerence of two consecutive entries);
Add min(Un + slack, τ ) to U;
n = n + 1;

end
B ← (Ui, Ui+1), ∀i ∈ {1, 3, · · · , n − 1}

12
13 end

14 return B

(a)

(b)

intensify eﬀective peaks in the signal and, speciﬁcally, inﬂection regions ineﬀective, we calculated
the second numerical derivation of the signal as L = ∂2f
4∂t2 .

A frequency-domain approach to approximately generate a discrete-time analytic signal is
proposed in [34]. In this approach, the negative frequency half of each spectral period is set to
zero, resulting in a periodic one-sided spectrum. The speciﬁc procedures for creating a complex-
valued N -point (N is even) discrete-time analytic signal F (ω) from a real-valued N -point discrete
time signal L[n] are as follows:

1. Compute the N -point discrete-time Fourier transform using F (ω) = T (cid:80)N −1

n=0 L[n]e−i2πωT n,
where |ω| ≤ 1/2T Hz and L[n] for 0 ≤ n ≤ N − 1 is obtained by sampling a band-limited
real-valued continuous-time signal L(nT ) = L[n] at periodic time intervals of T seconds to

Page 10 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(c)

(d)

Figure 6: (a,b) Identiﬁed local maxima and minima over gβ,γ(τ, s) in (a) Slice1 and (b) Slice2.
The second-row of the plot is the inverse of the ﬁrst row; therefore, the marked maximums are
identical to the local minima. (c,d) Candidate regions of interest which are alternately coloured
purple and green to ease visual tracking.

Algorithm 2: Excluding pseudo-spike and inﬂation regions form candidate ROI.

Input : B — set of ROI, i.e., Algorithm 1 output,

f — Electrical potential.

Output: C — set of wavelet-based ROIs,

D — set of pseudo-spike and inﬂection regions.

1 begin
2

for i = 1 to card(B) do

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

lb ← B(i, 1);
ub ← B(i, 2);
if (ub − lb) > 30 then
chunk = f [lb · · · ub];
minima = min(isLocalMinimum(chunk ));
// isLocalMinimum() and isLocalMaximum() use spline interpolation

in locating local extreme [23].
maxima = max(isLocalMaximum(chunk ));
if f (minima) < min(f (lb), f (ub)) or f (maxima) > max(f (lb), f (ub)) then

C ← [lb, ub];

else

D ← [lb, ub];

end

end

end

15
16 end

17 return C, D

Deliverable D4.1

Page 11 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 7: Results of applying Algorithm 2 to (a) Slice1 and (b) Slice2. Two spike events are
missed in Slice1. Two pseudo-spike and two inﬂection regions still remain in Slice2.

prevent aliasing.

2. Form the N -point one-sided discrete-time analytic signal transform:

Z[m] =





F [0],
2F [m],
F [ N
2 ],
0,

for m = 0
for 1 ≤ m ≤ N
for m = N
2
for N

2 − 1

2 + 1 ≤ m ≤ N − 1.

(4)

3. Compute the N -point inverse discrete-time Fourier transform to obtain the complex discrete-

time analytic signal of same sample rate as the original L[n]

z[n] =

1
N T

N −1
(cid:88)

m=0

Z[m]e

i2πmn
N

(5)

Obtaining analytic signal in this way can satisfy two properties: (1) The real part is identical
to the original discrete-time sequence; (2) the real and imaginary components are orthogonal.
Calculating the magnitude of this analytic signal yields signal envelope, ξ[n], containing the
upper, ξH [n], and lower, ξL[n], envelopes of L[n] (Eq. 6).

ξ[n] = |z[n]|

(6)

Envelopes are determined using spline interpolation over local maxima separated by at least
np = 60 samples. We considered np = 60 since we did not witness in our previous studies [5, 6]
fungal spikes of electrical potential shorter than 60 seconds3. We proposed Algorithm 3 to locate
candidate regions using signal envelope.

3This threshold can be changed with respect to the context of experiments.

Page 12 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Algorithm 3: Detecting candidate spike region from signal envelope.

Input : ξ[n] — Envelope of signal L[t],

np = 60 — Minimum distance between two consecutive local extreme.

Output: R — set of envelope-based ROIs.

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

ξM [n] = (ξH [n] + ξL[n]) /2;
[valmin, indmin] = isLocalMinimum(ξM [n], np);
[valmax, indmax] = isLocalMaximum(ξM [n], np);
// isLocalMinimum() and isLocalMaximum() locate local minimum and

maximum, respectively.

j ← index of the ﬁrst local maximum whose value is greater than the value of the
ﬁrst local minimum;
for i = 1 to card(indmin) do
if j ≤ card(indmax) then

∆ ← valmax(j) − valmin(i);
Add (indmin(i), indmax(j), ∆) to R;
j ← j + 1;

end

end
// R has j rows and 3 columns, as R1, R2, and R3.
ρ = mean(R3) − std(R3);
// mean() and std() calculate the mean and standard deviation,

respectively.

Remove the kth entry from R where R3(k) < ρ – see Fig. 8(b);

14
15 end

16 return R

Deliverable D4.1

Page 13 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

(d)

(e)

(f)

Figure 8: Results of applying Algorithm 3 to Slice1 (ﬁrst row) and Slice2 (second row). (a,d)
Candidate regions by ﬁnding local minima and maxima of the analytic signal envelope. The
regions with arrows are also highlighted in red on the bar chart. (b,e) The absolute prominence
diﬀerence of consecutive local minimum and maximum. Regions that do not satisfy R3(k) < ρ
are coloured in red. (c,f) Regions of Interest in R. Gray rectangle with dash edge shows the
correct spike, including repolarisation, depolarisation, and refractory periods. The purple dashed
rectangle shows the region whose refractory period attached to a pseudo-spike event.

Figure 8(a,d) shows candidate regions in R before applying Step 13. At this stage, although R
includes regions that do not observe the spike deﬁnition (pointed by arrow in plot), the correctly
identiﬁed spikes are consonant with our ﬁndings in [4, 5]. To eliminate non-spike regions, which
are highlighted in red in Fig. 8(b,e), we applied Steps 13 and 14. Nevertheless, the output of
Algorithm 3 (see Fig .8(c,f)) still contains regions that either belong to pseudo-spike/inﬂection
regions or their refractory periods attached to a pseudo-spike region.

To resolve issues in Algorithms 2 and 3, we proposed Algorithm 4 in which regions belong to
(C ∪ D) are used in updating R. If any ROI in R is a subset of (C ∪ D), it is added to the spike
event set, Fs, with an updated length. If any ROI in (C ∪ D) is a subset of R, it is added to
the pseudo-spike set, Fp. In a case of having intersection without observing subset condition, we
split the concatenation of ROIs from the intersection point into two slices. Then, the slice with
the minimum length is added to Fp. Finally, regions with a length of fewer than 60 seconds are
removed from Fs and Fp. Figure 9 shows results of applying Algorithm 4.

Page 14 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Algorithm 4: Extracting fungi spike and pseudo-spike events.

Input : C, D, R — Regions of interest.
Output: Fs, Fp — Fungi spike and pseudo-spike events, respectively.

1 begin
2

foreach re ∈ R do
chunke ← [r1
foreach rw ∈ (C ∪ D) do
w];

e · · · r2
e];

chunkw ← [r1
switch chunkw, chunke do

w · · · r2

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

case chunke ⊂ chunkw do

chunkw(end) = chunke(end);
Fs ← chunkw;

end
case chunkw ⊂ chunke do

Fp ← chunkw;

end
case intersect(chunkw, chunke) do

// intersect() checks if two chunks have an intersection

point.

Split the concatenation of chunkw and chunke from intersection point
into two sub-Chunks;
Fp ← sub-Chunks;

end

end

end

end

foreach r ∈ (Fs ∪ Fp) do
Remove r if |r| < 60;

end

23
24 end

25 return Fs, Fp

4 Experimental results

This section comprises of objective and complexity analyses. In the objective analysis, we showed
the eﬃciency of the spike event detection method in comparison with the existing, in neuroscience,
techniques of spike detection [38, 49] and the expert opinion in locating spikes’ arrival time. In
the complexity analysis, we selected complexity measures used in previous studies [7, 11, 48, 52]
to quantify activity patterns that are spatio-temporally integrated and diﬀerentiated.

4.1 Objective analysis

Various methods have been proposed to detect and sort spike events in EC recordings [18, 22,
33, 38, 39, 41, 43, 46, 56, 57, 58]. However, only a few of these methods do not require auxiliary
information like the construction of templates and the supervised setting of thresholds to detect
and sort spike events [38, 49]. Nenadic and Burdick [38] developed an unsupervised method

Deliverable D4.1

Page 15 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 9: Results of applying Algorithm 4 to (a) Slice1 and (b) Slice2. We alternatively used
red/purple for colouring spike events and blue/cyan for colouring pseudo-spike events.

to detect and localise spikes in noisy neural recordings. This method beneﬁts from continues
wavelet transform. They applied multi-scale decomposition of the signal using ‘bior1.3,’ ‘bior1.5,’
‘Haar,’ or ‘db2’ wavelet basis. To assess the presence of spikes, they separated the signal and
noise at each scale and performed Bayesian hypothesis testing. Finally, they combined decisions
at diﬀerent scales to estimate the arrival times of individual spikes.

Shimazaki and Shinomoto [49] proposed an optimisation technique for selecting the bin width
of the time-histogram. This optimisation minimised the mean integrated square in the kernel
density estimation. This method beneﬁted from variable kernel width, which allowed grasping
non-stationary phenomena, and stiﬀness constant to avoid possible overﬁtting due to excessive
freedom in the bandwidth variability. The estimated bandwidth was then used to ﬁlter spike
event regions from the signal. Figure 10 shows the results of applying these methods to two
chunks with a length of 3000 seconds.

Both methods could not correctly detect all spike events that were located by the expert.
Wavelet-based method could locate three spikes in Fig. 10(b) without detecting any spike in
Fig. 10(a). The adaptive bandwidth kernel-based method could detect one spike in Fig. 10(b)
and one pseudo-spike in Fig. 10(a) and (b). While our proposed method wrongly introduced one
spike event in Fig. 9(a) and had a total-error (wrong- or non-detection) of three in Fig. 9(b).

We also compared the proposed method with the expert opinion on a randomly selected
36,000-second chunk, i.e., 10 hours of electrical activity recordings. In this quantitative compar-
ison, the proposed method could correctly locate 21 spikes, introduce four pseudo-spike events,
overestimate two refractory periods; resulting in the true-positive and false-positive rates of 76%
and 16%, respectively. Figure 11(a) shows located spikes by the expert and Fig. 11(b) indicates
the results of the proposed spike detection method.

We applied the proposed method to six experiments where the statistical results are shown in
Figs. 12 – 13 and summarised in Table 1. It should be noted that the placement of the electrodes
in two experiments was in lines with a distance of 1 cm, in two experiments it was in lines with a
distance of 2 cm, and in two experiments it was random with a distance of approximately 2 cm.
The implementation in MATLAB R2020a and details of experiments are available at [13].

Page 16 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 10: Results of applying proposed algorithms in [38, 49] to (a) Slice1 and (b) Slice2.
Note that the wavelet-based method can only locate spike arrival time. The kernel bandwidth
optimisation can, however, extract the spike region.

Table 1: The dominant value and bandwidth for the spike’s length and amplitude in each ex-
periment across all recording channels. The duration and amplitude of spikes are estimated via
probability density function (PDF) and adaptive bandwidth kernel (ABK) [49]. The bold-face
blue and red entries indicate the absolute minimum and maximum values, respectively. We
considered the absolute value since we have bidirectional changes in potential.

3*

3*#Channels

3*#Spikes

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

PDF

ABK

PDF

ABK

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

565
447
124
951
573
862

Deliverable D4.1

Page 17 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

Figure 11: (a) Spike arrival time located by the expert. Here we used augmented pink arrow to
point to these spikes. (b) Spike regions extracted by the proposed method. Spike regions are
alternatively coloured in orange and violet. The green areas point to pseudo-spike regions that
are mistaken for spikes. Blue rectangles with dash edge show overestimated refractory periods.
We used black arrows to point to the missed spikes.

Page 18 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

(d)

(e)

(f)

Figure 12: Distribution of spike event lengths with superimposed Gaussian and Adaptive band-
width kernels [49]. (a,b) In-line electrode arrangements with a distance of 1 cm. (c,d) In-line
electrode arrangements with a distance of 2 cm. (e,f) Random electrode arrangements with an
approximate distance of 2 cm.

Deliverable D4.1

Page 19 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

(d)

(e)

(f)

Figure 13: Distribution of spike maximum amplitudes with superimposed Gaussian and Adaptive
bandwidth kernels [49] for (a,b) in lines electrode placement with a distance of 1 cm. (c,d) in
lines electrode placement with a distance of 2 cm. (e,f) random electrode placement with an
approximate distance of 2 cm.

These ﬁndings are aligned with the previously reported results on electrical activity of Physarum

polycephalum [2, 4] in which we reported that Physarum spike lengths are in the range of 60-120
seconds. In terms of growth, Physarum is faster than fungi. Therefore, we can now hypothesise
that fungal spikes can not be less than 60-120 seconds, with more observations.

4.2 Complexity Analysis

To quantify the complexity of the electrical signalling recorded, we used the following measure-
ments:

1. The Shannon entropy, H, is calculated as H = − (cid:80)

w∈W (ν(w)/η · ln(ν(w)/η)), where ν(w)
is a number of times the neighbourhood conﬁguration w is found in conﬁguration W , and
η is the total number of spike events found in all channels of an experiment.

2. Simpson’s diversity, S, is calculated as S = (cid:80)

w∈W (ν(w)/η)2. It linearly correlates with
Shannon entropy for H < 3 and the relationships becomes logarithmic for higher values of
H. The value of S ranges between 0 and 1, where 1 represents inﬁnite diversity and 0, no
diversity.

3. Space ﬁlling, D, is the ratio of non-zero entries in W to the total length of string.

4. Expressiveness, E, is calculated as the Shannon entropy H divided by space-ﬁlling ratio

D, the expressiveness reﬂects the ‘economy of diversity’.

Page 20 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

(d)

(e)

(f)

Figure 14: Barcode-like representation of spike events in diﬀerent channels for (a,b) in-line
electrode arrangements with a distance of 1 cm, (c,d) in-line electrode arrangements with a
distance of 2 cm, and (e,f) random electrode arrangements with an approximate distance of
2 cm.

Deliverable D4.1

Page 21 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

5. Lempel–Ziv complexity (compressibility), LZ, is evaluated by the size of binary string, n,
and used to assess temporal signal diversity. Here, we represented the spiking behaviour of
mycelium with a binary string where ‘1s’ indicates the presence of a spike and ‘0s’ otherwise
(see Fig. 14).

6. Perturbation complexity index P CI = LZ/H.

To calculate Lempel–Ziv complexity, we saved each signal as a PNG image (see two examples
in Fig. 15), where the ‘deﬂation’ algorithm used in PNG lossless compression [14, 26, 44] is a
variation of the classical LZ77 algorithm [62]. We employed this approach as the recorded signal
is a non-binary string. We take the largest PNG ﬁle size to normalise this measurement.

(a)

(b)

Figure 15: Two samples from input channels, which are saved in black and white PNG format
without axes and annotations.

To assess the signal diversity across all channels and observations, we represented each ex-
periment as a matrix with binary entries with a row for each channel and a column for each
observation. This binary matrix is then concatenated observation-by-observation to form one
binary string. We applied Kolmogorov complexity algorithm [28] to calculate the across channels
Lempel–Ziv complexity, LZc. LZc captures temporal signal diversity of single channels as well
as spatial signal diversity across channels as the result of the observation-by-observation concate-
nation of the binarised data matrix. We also normalise LZc by dividing the raw value by the
value obtained for the same binary input sequence randomly shuﬄed. Since the value of LZ for
a binary sequence of ﬁxed length is maximal if the sequence is entirely random, the normalised
values indicate the level of signal diversity on a scale from 0 to 1. Results of calculating these
complexity measurements for all six setups are illustrated in Fig. 16, and summarised in Tab. 2.

Table 2: The mean of complexity measurements for six experiments.

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

Page 22 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

(a)

(b)

(c)

(d)

(e)

(f)

Figure 16: (a) Shannon entropy, (b) Simpson’s diversity, (c) Space ﬁlling, (d) Expressiveness, (e)
Lempel–Ziv complexity, and (f) Perturbation complexity index. All measurements are scaled to
the range of [0, 1].

Deliverable D4.1

Page 23 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

In order to clarify the communication complexity in the mycelium substrate, we also calcu-
lated the mentioned complexity measurements for the communications in the forms of (i) pieces
of news4, (ii) a random sequence of alphanumeric5, and (iii) a periodic sequence of alphanumeric
converted to binary strings by applying Huﬀman coding [27] (see barcodes in Fig. 17). Results
of comparing the complexity of fungi electrical activity with these three forms are reported in
Tab. 3.

(a)

(b)

(c)

Figure 17: Binary representation of (a) pieces of news,(b) random sequence of alphanumeric,
and (c) periodic sequence of alphanumeric after applying Huﬀman coding.

Table 3: The complexity measurements for pieces of news, a random sequence of alphanumeric,
a periodic sequence of alphanumeric along with three chunks randomly selected from our exper-
iments.

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

5 Complexity of fungal language

We developed algorithmic framework for exhaustive characterisation of electrical activity of a
substrate colonised by mycelium of oyster fungi Pleurotus djamor. We evidenced spiking activity
of the mycelium. We found that average dominant duration of an action-potential like spike is
402 sec. The spikes amplitudes’ depends on the location of the source of electrical activity related
to the position of electrodes, thus the amplitudes provide less useful information. The amplitudes
vary from 0.5 mV to 6 mV. This is indeed low compared to 50-60 mV of intracellular recording,
nevertheless understandable due to the fact the electrodes are inserted not even in mycelium
strands but in the substrate colonised by mycelium. The spiking events have been characterised
with several complexity measures. Most measures, apart of Kolmogorov complexity shown a

4https://www.sciencemag.org/news/2020/07/meet-lizard-man-reptile-loving-biologist-tackling-some-biggest-

questions-evolution

5We used available service at https://www.random.org/

Page 24 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

Figure 18: Lempel-Ziv complexity of European languages (data from [47]) with average com-
plexity of fungal (‘fu’) electrical activity language added.

low degree of variability between channels (diﬀerent sites of the recordings). The Kolmogorov
complexity of fungal spiking varies from 11×10−4 to 57×10−4. This might indicated mycelium
sub-networks in diﬀerent parts of the substrate have been transmitting diﬀerent information to
other parts of the mycelium network. This is somehow echoes experimental results on commu-
nication between ants analysed with Kolmogorov complexity: longer paths communicated ants
corresponds to higher values of complexity [45].

LZ complexity of fungal language (Tab. 2) is much higher than of news, random or periodic
sequences (Tab. 3). The same can be observed for Shannon entropy. Kolmogorov complexity of
the fungal language is much lower than that of news sampler or random or periodic sequences.
Complexity of European languages based on their compressibility [47] is shown in Fig. 18, French
having lowest LZ complexity 0.66 and Finnish highest LZ complexity 0.79. Fungal language of
electrical activity has minimum LZ complexity 0.61 and maximum 0.91 (media 0.85, average
0.83). Thus, we can speculate that a complexity of fungal language is higher than that of human
languages (at least for European languages).

References

[1] IEEE standard on transitions, pulses, and related waveforms. IEEE Std 181-2003, pages

1–60, 2003.

[2] Andrew Adamatzky. Tactile bristle sensors made with slime mold. IEEE Sensors journal,

14(2):324–332, 2013.

[3] Andrew Adamatzky.

logic gates and sensors. Philosophical
Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences,
373(2046):20140216, 2015.

Slime mould processors,

[4] Andrew Adamatzky. On spiking behaviour of oyster fungi Pleurotus djamor. Scientiﬁc

reports, 8(1):1–7, 2018.

[5] Andrew Adamatzky. Towards fungal computer. Interface focus, 8(6):20180029, 2018.

[6] Andrew Adamatzky. Plant leaf computing. Biosystems, 182:59–64, 2019.

Deliverable D4.1

Page 25 of 29

LZ complexity0.600.650.700.750.800.85LanguagefresptgaitenslnlmtdaelsvlvdeplltsketﬁfuEU-H2020 FET grant agreement no. 858132 — fungal architectures

[7] Andrew Adamatzky and Mohammad Mahdi Dehshibi. Exploring tehran with excitable
In Andrew Adamatzky, Selim Akl, and Georgios Ch. Sirakoulis, editors, From

medium.
Parallel to Emergent Computing, chapter 22, pages 475–488. CRC Press, 2019.

[8] David J Aidley and DJ Ashley. The physiology of excitable cells, volume 4. Cambridge

University Press Cambridge, 1998.

[9] Boris P Belousov. A periodic reaction and its mechanism. Compilation of Abstracts on

Radiation Medicine, 147(145):1, 1959.

[10] MS Bingley. Membrane potentials in amoeba proteus. Journal of Experimental Biology,

45(2):251–267, 1966.

[11] Adenauer G Casali, Olivia Gosseries, Mario Rosanova, M´elanie Boly, Simone Sarasso, Ka-
rina R Casali, Silvia Casarotto, Marie-Aur´elie Bruno, Steven Laureys, Giulio Tononi, et al.
A theoretically based index of consciousness independent of sensory processing and behavior.
Science translational medicine, 5(198):198ra105–198ra105, 2013.

[12] Jorge M Davidenko, Arcady V Pertsov, Remy Salomonsz, William Baxter, and Jos´e Jal-
ife. Stationary and drifting spiral waves of excitation in isolated cardiac muscle. Nature,
355(6358):349, 1992.

[13] Mohammad Mahdi Dehshibi and Andrew Adamatzky. Supplementary material for “Elec-
trical activity of fungi: Spikes detection and complexity analysis”. https://doi.org/10.
5281/zenodo.3997031, 08 2020. (Accessed on 24/08/2020).

[14] Peter Deutsch and J Gailly. Zlib compressed data format speciﬁcation version 3.3. Technical

report, RFC 1950, May, 1996.

[15] Roger Eckert and Paul Brehm.

Ionic mechanisms of excitation in paramecium. Annual

review of biophysics and bioengineering, 8(1):353–383, 1979.

[16] I Farkas, Dirk Helbing, and T Vicsek. Human waves in stadiums. Physica A: statistical

mechanics and its applications, 330(1-2):18–24, 2003.

[17] Ill´es Farkas, Dirk Helbing, and Tam´as Vicsek. Social behaviour: Mexican waves in an

excitable medium. Nature, 419(6903):131, 2002.

[18] Felix Franke, Michal Natora, Clemens Boucsein, Matthias HJ Munk, and Klaus Obermayer.
An online spike detection and spike classiﬁcation algorithm capable of instantaneous reso-
lution of overlapping spikes. Journal of computational neuroscience, 29(1-2):127–148, 2010.

[19] J¨org Fromm and Silke Lautner. Electrical signals and their physiological signiﬁcance in

plants. Plant, cell & environment, 30(3):249–257, 2007.

[20] Marianne Fyhn, Sturla Molden, Menno P Witter, Edvard I Moser, and May-Britt Moser.
Spatial representation in the entorhinal cortex. Science, 305(5688):1258–1264, 2004.

[21] LM Gorbunov and VI Kirsanov. Excitation of plasma waves by an electromagnetic wave

packet. Sov. Phys. JETP, 66(290-294):40, 1987.

[22] J Gotman and LY Wang. State-dependent spike detection: concepts and preliminary results.

Electroencephalography and clinical Neurophysiology, 79(1):11–19, 1991.

Page 26 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[23] Charles A Hall and W Weston Meyer. Optimal error bounds for cubic spline interpolation.

Journal of Approximation Theory, 16(2):105–122, 1976.

[24] Helen G Hansma. Sodium uptake and membrane excitation in paramecium. The Journal

of cell biology, 81(2):374–381, 1979.

[25] Alan L Hodgkin and Andrew F Huxley. A quantitative description of membrane current and
its application to conduction and excitation in nerve. The Journal of physiology, 117(4):500–
544, 1952.

[26] Paul Glor Howard. The Design and Analysis of Eﬃcient Lossless Data Compression Sys-

tems. PhD thesis, Citeseer, 1993.

[27] David A Huﬀman. A method for the construction of minimum-redundancy codes. Proceed-

ings of the IRE, 40(9):1098–1101, 1952.

[28] F Kaspar and HG Schuster. Easily calculable measure for the complexity of spatiotemporal

patterns. Physical Review A, 36(2):842, 1987.

[29] Ch Kittel. Excitation of spin waves in a ferromagnet by a uniform rf ﬁeld. Physical Review,

110(6):1295, 1958.

[30] Jonathan M Lilly. Element analysis: a wavelet-based method for analysing time-localized
events in noisy time series. Proceedings of the Royal Society A: Mathematical, Physical and
Engineering Sciences, 473(2200):20160776, 2017.

[31] Jonathan M Lilly and Soﬁa C Olhede. Higher-order properties of analytic wavelets. IEEE

Transactions on Signal Processing, 57(1):146–160, 2008.

[32] Jonathan M Lilly and Soﬁa C Olhede. Generalized morse wavelets as a superfamily of
analytic wavelets. IEEE Transactions on Signal Processing, 60(11):6036–6041, 2012.

[33] Zuozhi Liu, Xiaotian Wang, and Quan Yuan. Robust detection of neural spikes using sparse
coding based features. Mathematical Biosciences and Engineering, 17(4):4257, 2020.

[34] Lawrence Marple. Computing the discrete-time” analytic” signal via ﬀt. IEEE Transactions

on signal processing, 47(9):2600–2603, 1999.

[35] Elisa Masi, Marzena Ciszak, Luisa Santopolo, Arcangela Frascella, Luciana Giovannetti,
Emmanuela Marchi, Carlo Viti, and Stefano Mancuso. Electrical spiking in bacterial
bioﬁlms. Journal of The Royal Society Interface, 12(102):20141036, 2015.

[36] Ann M. McGillviray and Neil A.R. Gow. The transhyphal electrical current of Neuruspua

crassa is carried principally by protons. Microbiology, 133(10):2875–2881, 1987.

[37] Phillip G Nelson and Melvyn Lieberman. Excitable cells in tissue culture. Springer Science

& Business Media, 2012.

[38] Zoran Nenadic and Joel W Burdick. Spike detection using the continuous wavelet transform.

IEEE transactions on Biomedical Engineering, 52(1):74–87, 2004.

[39] Iyad Obeid and Patrick D Wolf. Evaluation of spike-detection algorithms fora brain-machine
interface application. IEEE Transactions on Biomedical Engineering, 51(6):905–911, 2004.

[40] S Olsson and BS Hansson. Action potential-like activity found in fungal mycelia is sensitive

to stimulation. Naturwissenschaften, 82(1):30–31, 1995.

Deliverable D4.1

Page 27 of 29

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[41] R Quian Quiroga, Zoltan Nadasdy, and Yoram Ben-Shaul. Unsupervised spike detection and
sorting with wavelets and superparamagnetic clustering. Neural computation, 16(8):1661–
1687, 2004.

[42] Rodrigo Quian Quiroga, Alexander Kraskov, Christof Koch, and Itzhak Fried. Explicit
encoding of multimodal percepts by single neurons in the human brain. Current Biology,
19(15):1308–1313, 2009.

[43] Melinda R´acz, Csaba Liber, Erik N´emeth, Rich´ard Fi´ath, J´anos Rokai, Istv´an Harmati,
Istv´an Ulbert, and Gergely M´arton. Spike detection and sorting with deep learning. Journal
of Neural Engineering, 17(1):016038, 2020.

[44] Greg Roelofs and Richard Koman. PNG: the deﬁnitive guide. O’Reilly & Associates, Inc.,

1999.

[45] Boris Ryabko and Zhanna Reznikova. Using shannon entropy and kolmogorov complexity to
study the communicative system and cognitive capacities in ants. Complexity, 2(2):37–42,
1996.

[46] Shlok Sablok, Githali Gururaj, Naushaba Shaikh, I Shiksha, and Antara Roy Choudhary.
Interictal spike detection in eeg using time series classiﬁcation. In 2020 4th International
Conference on Intelligent Computing and Control Systems (ICICCS), pages 644–647. IEEE,
2020.

[47] Markus Sadeniemi, Kimmo Kettunen, Tiina Lindh-Knuutila, and Timo Honkela. Com-
plexity of european union languages: A comparative approach. Journal of Quantitative
Linguistics, 15(2):185–211, 2008.

[48] Michael M Schartner, Robin L Carhart-Harris, Adam B Barrett, Anil K Seth, and Suresh D
Muthukumaraswamy. Increased spontaneous meg signal diversity for psychoactive doses of
ketamine, lsd and psilocybin. Scientiﬁc reports, 7:46421, 2017.

[49] Hideaki Shimazaki and Shigeru Shinomoto. Kernel bandwidth optimization in spike rate

estimation. Journal of computational neuroscience, 29(1-2):171–182, 2010.

[50] Cliﬀord L Slayman, W Scott Long, and Dietrich Gradmann. “action potentials” in neu-
rospora crassa, a mycelial fungus. Biochimica et Biophysica Acta (BBA)-Biomembranes,
426(4):732–744, 1976.

[51] JC Slonczewski. Excitation of spin waves by an electric current. Journal of Magnetism and

Magnetic Materials, 195(2):L261–L268, 1999.

[52] Nassim Taghipour, Hamid Haj Seyyed Javadi, Mohammad Mahdi Dehshibi, and Andrew
Adamatzky. On complexity of persian orthography: L-systems approach. Complex Systems,
25(2):127–156, 2016.

[53] Caterina Trainito, Constantin von Nicolai, Earl K Miller, and Markus Siegel. Extracellular
spike waveform dissociates four functionally distinct cell classes in primate cortex. Current
Biology, 29(18):2973–2982, 2019.

[54] Kazimierz Trebacz, Halina Dziubinska, and Elzbieta Krol. Electrical signals in long-distance
communication in plants. In Communication in plants, pages 277–290. Springer, 2006.

[55] M Tsoi, AGM Jansen, J Bass, W-C Chiang, M Seck, V Tsoi, and P Wyder. Excitation of

a magnetic multilayer by an electric current. Physical Review Letters, 80(19):4281, 1998.

Page 28 of 29

Deliverable D4.1

EU-H2020 FET grant agreement no. 858132 — fungal architectures

[56] Zimeng Wang, Duanpo Wu, Fang Dong, Jiuwen Cao, Tiejia Jiang, and Junbiao Liu. A novel
spike detection algorithm based on multi-channel of bect eeg signals. IEEE Transactions on
Circuits and Systems II: Express Briefs, 2020.

[57] Scott B Wilson and Ronald Emerson. Spike detection: a review and comparison of algo-

rithms. Clinical Neurophysiology, 113(12):1873–1881, 2002.

[58] Scott B Wilson, Christine A Turner, Ronald G Emerson, and Mark L Scheuer. Spike
detection ii: automatic, perception-based detection and clustering. Clinical neurophysiology,
110(3):404–411, 1999.

[59] AM Zhabotinsky. Periodic processes of malonic acid oxidation in a liquid phase. Bioﬁzika,

9(306-311):11, 1964.

[60] Anatol M Zhabotinsky. Belousov-zhabotinsky reaction. Scholarpedia, 2(9):1435, 2007.

[61] Matthias R Zimmermann and Axel Mith¨ofer. Electrical long-distance signaling in plants. In
Long-Distance Systemic Signaling and Communication in Plants, pages 291–308. Springer,
2013.

[62] Jacob Ziv and Abraham Lempel. A universal algorithm for sequential data compression.

IEEE Transactions on information theory, 23(3):337–343, 1977.

Deliverable D4.1

Page 29 of 29

