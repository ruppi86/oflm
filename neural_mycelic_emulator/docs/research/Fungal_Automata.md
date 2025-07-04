See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/340022050

Fungal Automata

Preprint · March 2020

DOI: 10.48550/arXiv.2003.08168

CITATIONS
0

6 authors, including:

Andrew Adamatzky

University of the West of England, Bristol

922 PUBLICATIONS   14,614 CITATIONS   

SEE PROFILE

Genaro J. Martinez

National Polytechnic Institute

127 PUBLICATIONS   1,227 CITATIONS   

SEE PROFILE

READS
553

Eric Goles

Adolfo Ibáñez University

295 PUBLICATIONS   4,378 CITATIONS   

SEE PROFILE

Michail-Antisthenis Tsompanas

University of the West of England, Bristol

116 PUBLICATIONS   895 CITATIONS   

SEE PROFILE

All content following this page was uploaded by Genaro J. Martinez on 24 August 2020.

The user has requested enhancement of the downloaded file.

0
2
0
2

r
a

M
8
1

]

G
C
.
n
i
l
n
[

1
v
8
6
1
8
0
.
3
0
0
2
:
v
i
X
r
a

Fungal Automata

Andrew Adamatzky1, Eric Goles1,2, Genaro J. Mart´ınez1,3, Michail-Antisthenis
Tsompanas1, Martin Tegelaar4, and Han A. B. Wosten4

1Unconventional Computing Laboratory, UWE, Bristol, UK
2Faculty of Engineering and Science, University of Adolfo Ib´a˜nez, Santiago, Chile
3High School of Computer Science, National Polytechnic Institute, Mexico
4Microbiology Department, University of Utrecht, Utrecht, The Netherlands

Abstract
We study a cellular automaton (CA) model of information dynamics on a single hypha of a fungal
mycelium. Such a ﬁlament is divided in compartments (here also called cells) by septa. These
septa are invaginations of the cell wall and their pores allow for ﬂow of cytoplasm between com-
partments and hyphae. The septal pores of the fungal phylum of the Ascomycota can be closed
by organelles called Woronin bodies. Septal closure is increased when the septa become older
and when exposed to stress conditions. Thus, Woronin bodies act as informational ﬂow valves.
The one dimensional fungal automata is a binary state ternary neighbourhood CA, where every
compartment follows one of the elementary cellular automata (ECA) rules if its pores are open
and either remains in state ‘0’ (ﬁrst species of fungal automata) or its previous state (second
species of fungal automata) if its pores are closed. The Woronin bodies closing the pores are also
governed by ECA rules. We analyse a structure of the composition space of cell-state transition
and pore-state transitions rules, complexity of fungal automata with just few Woronin bodies, and
exemplify several important local events in the automaton dynamics.

Keywords: fungi, ascomycete, Woronin body, cellular automata

1

Introduction

The fungal kingdom represents organisms colonising all ecological niches [11] where they play a key role
[18, 15, 35, 12]. Fungi can consist of a single cell, can form enormous underground networks [38] and
can form microscopic fruit bodies or fruit bodies weighting up to half a ton [16]. The underground
mycelium network can be seen as a distributed communication and information processing system
linking together trees, fungi and bacteria [10]. Mechanisms and dynamics of information processing in
mycelium networks form an unexplored ﬁeld with just a handful of papers published related to space
[37, 34, 1] and potential use
exploration by mycelium [20, 19], patterns of electrical activity of fungi
of fungi as living electronic and computing devices [2, 3, 4].

Filamentous fungi grow by means of hyphae that grow at their tip and that branch sub-apically.
Hyphae may be coenocytic or divided in compartments by septa. Filamentous fungi in the phylum
Ascomycota have porous septa that allow for cytoplasmic streaming [31, 24]. Woronin bodies plug
the pores of these septa after hyphal wounding to prevent excessive bleeding of cytoplasm [44, 13, 22,
42, 39, 29]. In addition, they plug septa of intact growing hyphae to maintain intra- and inter-hyphal
heterogeneity [8, 7, 40, 40, 41].

Woronin bodies can be located in diﬀerent hyphal positions (Fig. 1a). When ﬁrst formed, Woronin
bodies are generally localised to the apex [30, 43, 5]. Subsequently, Woronin bodies are either trans-
ported to the cell cortex (Neurospora crassa, Sordaria ﬁmicola) or to the septum (Aspergillus oryzae,

1

 
 
 
 
 
 
(a)

(b)

Figure 1: (a) A biological scheme of a fragment of a fungal hypha of an ascomycete, where we can see
septa and associated Woronin bodies. (b) A scheme representing states of Woronin bodies: ‘0’ open,
‘1’ closed.

Aspergillus nidulans, Aspergillus fumigatus, Magnaporthe grisea, Fusarium oxysporum, Zymoseptoria
tritici ) where they are anchored with a leashin tether and largely immobile until they are translocated
to the septal pore due to cytoplasmic ﬂow or ATP depletion [33, 40, 39, 30, 29, 43, 45, 23, 6]. Woronin
bodies that are not anchored at the cellular cortex or the septum, are located in the cytoplasm and
are highly mobile (Aspergillus fumigatus, Aspergillus nidulans, Zymoseptoria tritici ) [5, 30, 40]. Septal
pore occlusion can be induced by bulk cytoplasmic ﬂow [40] or developmental [9] and environmental
cues, like puncturing of the cell wall, high temperature, carbon and nitrogen starvation, high osmolar-
ity and low pH. Interestingly, high environmental pH reduces the proportion of occluded apical septal
pores [41].

Aiming to lay a foundation of an emerging paradigm of fungal intelligence — distributed sensing
and information processing in living mycelium networks — we decided to develop a formal model of
mycelium and investigate a role of Woronin bodies in potential information dynamics in the mycelium.
The paper is structured as follows. We introduce fungal automata in Sect. 2. Properties of the
composition of cell state transition and Woronin body state transition functions are studied in Sect. 3.
Complexity of space-time conﬁguration of fungal automata, where just few cells have Woronin bodies is
studied in Sect. 4. Section 5 exempliﬁes local events, which could be useful for computation with fungal
automata, happening in fungal automata with sparsely but regularly positioned cells with Woronin
bodies. The paper concludes with Sect. 6.

2 Fungal automata M

A fungal automaton is a one-dimensional cellular automaton with binary cell states and ternary,
including central cell, cell neighbourhood, governed by two elementary cellular automata (ECA) rules,
namely the cell state transition rule f and the Woronin bodies adjustment rule g: M = (cid:104)N, u, Q, f, g(cid:105).
Each cell xi has a unique index i ∈ N. Its state is updated from Q = {0, 1} in discrete time depending of
its current state xt
i+1 and the state of cell x’s Woronin
body w. Woronin bodies take states from Q: wt = 1 means Woronin bodies (Fig. 1) in cell x blocks
the pores and the cell has no communication with its neighbours, and wt = 0 means that Woronin
bodies in cell x do not block the pores. Woronin bodies update their states g(·), wt+1 = g(u(x)t),
depending on the state of neighbourhood u(x)t. Cells x update their states by function f (·) if their
Woronin bodies do not block the pores.

i−1 and right neighbours xt

i, the states of its left xt

Two species of mycelium automata are considered M1, where each cell updates its state as following:

xt+1 =

(cid:40)
0
f (u(x)t)

if wt = 1
otherwise

2

Septum associated Woronin body~50umseptumleashinCytoplasmicWoronin body``````   1       1     0      0      1     0     0      1      0(a) M1, ρf = 133, ρg = 116

(b) M2, ρf = 133, ρg = 116

(c) M1, ρf = 73, ρg = 128

(d) M2, ρf = 73, ρg = 128

(e) M1, ρf = 61, ρg = 132

(f) M2, ρf = 61, ρg = 132

(g) M1, ρf = 57, ρg = 98

(h) M2, ρf = 57, ρg = 98

(i) M1, ρf = 26, ρg = 84

(j) M2, ρf = 26, ρg = 84

(k) M1, ρf = 125, ρg = 105

(l) M2, ρf = 125, ρg = 105

Figure 2: Examples of space-time dynamics of M. The automata are 103 cells each. Initial conﬁgu-
ration is random with probability of a cell x to be in state ‘1’, x0 = 1, equals 0.01. Each automaton
evolved for 103 iterations. Binary values of ECA rules f and g are shown in sub-captions. Rule g
is applied to every iteration starting from 200th. Cells in state ‘0’ are white, in state ‘1’ are black,
cells with Woronin bodies blocking pores are red. Indexes of cells increase from the left to the right,
iterations are increasing from the to the bottom.

3

and M2 where each cell updates its state as following:

xt+1 =

(cid:40)

xt
f (u(x)t)

if wt = 1
otherwise

where wt = g(u(x)t).

State ‘1’ in the cells of array x symbolises metabolites, signals exchanged between cells. Where

pores in a cell are open the cell updates its state by ECA rule f : {0, 1}3 → {0, 1}.

In automaton M1, when Woronin bodies block the pores in a cell, the cell does not update its
state and remains in the state ‘0’ and left and right neighbours of the cells can not detect any ‘cargo’
in this cell. In automaton, M2, where Woronin bodies block the pores in a cell, the cell does not
update its state and remains in its current state. In real living mycelium glucose and possibly other
metabolites [7] can still cross the septum even when septa are closed by Woronin bodies, but we can
ignore this fact in present abstract model.

Both species are biologically plausible and, thus, will be studied in parallel. The rules for closing
and opening Woronin bodies are also ECA rules g : {0, 1}3 → {0, 1}. If g(u(x)t) = 0 this means that
pores are open, if g(u(x)t) = 1 Woronin bodies block the pores. Examples of space-time conﬁgurations
of both species of M are shown in Fig. 2.

3 Properties of composition f ◦ g

Predecessor sets

Let F = {h : {0, 1}3 → {0, 1}} be a set of all ECA functions. Then for any composition f ◦ g,
where f, g ∈ F, can be converted to a single function h ∈ F. For each h ∈ F we can construct a set
P(h) = {f ◦ g ∈ F × F | f ◦ g → h}. The sets P(h) for each h ∈ F are available online1.

A size of P(h) for each h is shown in Fig. 3c. The functions with largest size of P(h) are rule 0 in
automaton M1 and rule 51 (only neighbourhood conﬁgurations (010, 011, 110, 111 are mapped into
1) in M2.

Size σ of P(h) vs a number γ of functions h having set P(h) of size σ is shown for automata M1

and M2 in Table 1a.

With regards to Wolfram classiﬁcation [47], sizes of P(h) for rules from Class III vary from 9 to
729 in M1 (Tab. 1b). Rule 126 would be the most diﬃcult to obtain in M1 by composition two ECA
rules chosen at random, it has only 9 ‘predecessor’ f ◦ g pairs. Rule 18 would be the easiest, for Class
III rules, to be obtained, it has 729 predecessors, in both M1 (Tab. 1b) and M2 (Tab. 1d). In M1,
one rule, rule 41, from the class IV has 243 f ◦ g predecessors, and all other rules in that class have
81 (Tab. 1c). From Class IV rule 54 has the largest number of predecessors in M2, and thus can be
considered as most common (Tab. 1d).

Diagonals

In automaton M1 for any f ∈ F f ◦ f = 0. Assume f : {0, 1}3 → 1 then Woronin bodies close the
pores and, thus, second application of f produces state ‘0’. If f : {0, 1}3 → 0 then Woronin bodes
does not close pores but yet second application of the f produce state ‘0’.

For automaton M2 a structure of diagonal mapping f ◦ f → h, where f, h ∈ F is shown in
Tab. 2. The set of the diagonal outputs f ◦ f consists of 16 rules: (0, 1, 2, 3), (16, 17, 18, 19),
(32, 33, 34, 35), (48, 49, 40, 51). These set of rules can be reduced to the following rule. Let
C(xt) = [u(x)t = (111)] ∨ [u(x)t = (111)] and B(xt) = [u(x)t = (011)] ∨ [u(x)t = (010)]. Then xt = 1
if C(x)t ∨ C(x)t ∧ B(xt).

1https://figshare.com/s/b7750ee3fe6df7cbe228

4

(a)

(b)

(c)

Figure 3: Mapping F × F → F for automaton M1 (a) and M2 (b) is visualised as an array of pixels,
P = (p)0≤ρf ≤255,0≤ρf ≤255. An entry at the intersection of any ρf and ρg is a coloured as follows: red
if pρf ρg = pρgρf , blue if ρg = pρgρf , green if ρf = pρgρf . (c) Sizes of P(h) sets for M1, circle, and M2,
solid discs, are shown for every function h apart of rule 0 (M1) and rule 51 (M2).

5

κ(ρf)0500100015002000ρf020406080100120140160180200220240(a) Rules per
|P(h)|
σ
1
3
9
27
81
243
729
2187
6561

γ
1
8
28
56
70
56
28
8
1

(c) M1: Class IV rules
Rule
41
54, 106, 110

σ
243
81

(b) M1: Class III rules
Rule
18
22, 146
30, 45, 60,
90, 105, 150
122
126

σ
729
243
81

27
9

(d) M2: Class III rules
Rule
18
22, 146
30, 45, 60 90,
105, 150
122
126

σ
729
243
81

243
81

(e) M2: Class IV rules
Rule
41
54
106
110

σ
243
729
81
27

Table 1: Characterisations of automaton mapping F × F → F. (a) Size σ of P(h) vs a number γ
of functions h having set P(h) of size σ. T (b) Sizes of sets P(h) for rules from Wolfram class III.
(b) Sizes of sets P(h) for rules from Wolfram class IV.

f ◦ f
0
1
2
3
16
17
18
19
32
33
34
35
48
49
50
51

f
0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51
128, 129, 130, 131, 144, 145, 146, 147, 160, 161, 162, 163, 176, 177, 178, 179
64, 65, 66, 67, 80, 81, 82, 83, 96, 97, 98, 99, 112, 113, 114, 115
192, 193, 194, 195, 208, 209, 210, 211, 224, 225, 226, 227, 240, 241, 242, 243
8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59
136, 137, 138, 139, 152, 153, 154, 155, 168, 169, 170, 171, 184, 185, 186, 187
72, 73, 74, 75, 88, 89, 90, 91, 104, 105, 106, 107, 120, 121, 122, 123
200, 201, 202, 203, 216, 217, 218, 219, 232, 233, 234, 235, 248, 249, 250, 251
4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55
132, 133, 134, 135, 148, 149, 150, 151, 164, 165, 166, 167, 180, 181, 182, 183
68, 69, 70, 71, 84, 85, 86, 87, 100, 101, 102, 103, 116, 117, 118, 119
196, 197, 198, 199, 212, 213, 214, 215, 228, 229, 230, 231, 244, 245, 246, 247
12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63
140, 141, 142, 143, 156, 157, 158, 159, 172, 173, 174, 175, 188, 189, 190, 191
76, 77, 78, 79, 92, 93, 94, 95, 108, 109, 110, 111, 124, 125, 126, 127
204, 205, 206, 207, 220, 221, 222, 223, 236, 237, 238, 239, 252, 253, 254, 255

Table 2: Diagonals of automaton M2.

6

Commutativity

In automaton M1, for any f, g ∈ F f ◦ g (cid:54)= g ◦ f only if f (cid:54)= g. In automaton M2 there are 32768
pairs of function which ◦ is commutative, their distribution visualised in red in Fig. 3b.

Identities and zeros

In automaton M1 there are no left or right identities, neither right zeros in (cid:104)F, F, ◦(cid:105). The only left
zero is the rule 0. In automaton M2 there are no identities or zeros at all.

Associativity

In automaton M1 there 456976 triples (cid:104)f, g, h(cid:105) on which operation ◦ is associative: (f ◦ g) ◦ h =
f ◦ (g ◦ h). The ratio of associative triples to the total number of triples is 0.027237892. There are
104976 associative triples in M2, a ratio of 0.006257057.

4 Tuning Complexity: Rule 110

To evaluate on how introduction of Woronin bodies could aﬀect complexity of automaton evolution,
we undertook two series of experiments. In the ﬁrst series we used fungal automaton where just one
cell has a Woronin body (Fig. 5). In the second series we employed fungal automaton where regularly
positioned cells (but not all cells of the array) have Woronin bodies.

State transition functions g of Woronin bodies were varied across the whole diapason but the state
transition function f of a cell was Rule 110, ρf = 110. We have chosen Rule 110 because the rule is
proven to be computationally universal [25, 14], P-complete [32], the rules belong to Wolfram class
IV renown for exhibiting complex and non-trivial interactions between travelling localisation [46], rich
families of gliders can be produce in collisions with other gliders [26, 27, 28].

We wanted to check how an introduction of Woronin bodies aﬀect dynamics of most complex
space-time developed of Rule 110 automaton. Thus, we evolved the automata from all possible initial
conﬁgurations of 8 cells placed near the end of n = 1000 cells array of resting cells and allowing
to evolve for 950 iterations. Lempel–Ziv complexity (compressibility) LZ was evaluated via sizes of
space-time conﬁgurations saved as PNG ﬁles. This is suﬃcient because the ’deﬂation’ algorithm used
in PNG lossless compression [36, 21, 17] is a variation of the classical Lempel–Ziv 1977 algorithm [48].
Estimates of LZ complexity for each of 8-cell initial conﬁgurations are shown in Fig. 4a. The initial
conﬁgurations with highest estimated LZ complexity are 10110001 (decimal 177), 11010001 (209),
10000011 (131), 11111011 (253), see example of space-time dynamics in Fig. 4b.

We assumed that a cell in the position n − 100 has a Woronin body which can be activated (Fig. 5),
i.e. start updating its state by rule f , after 100th iteration of the automaton evolution. We then
run 950 iteration of automaton evolution for each of 256 Woronin rules and estimated LZ complexity.
In experiments with M1 we found that 128 rules, with even decimal representations, do not aﬀect
space time dynamics of evolution and 128 rules, with even decimal representations, reduce complexity
of the space-time conﬁguration. The key reasons for the complexity reduction (compare Fig. 4b and
c) are cancellation of three gliders at c. 300th iteration and simpliﬁcation of the behaviour of glider
guns positioned at the tail of the propagating wave-front. In experiments with M2 128 rules, with
even decimal representations, do not change the space-time conﬁguration of the author. Other 128
rules reduce complexity and modify space-time conﬁguration by re-arranging the structures of glider
guns and establishing one oscillators at the site surrounding position of the cell with Woronin body
(Fig. 4d).

In second series of experiments we regularly positioned cells with Woronin bodies along the 1D
array: every 50th cell has a Woronin body. Then we evolved fungal automata M1 and M2 from
exactly the same initial random conﬁguration with density of ‘1’ equal to 0.3. Space-time conﬁguration
of the automaton without Woronin bodies is shown in Fig. 6a. Exemplar of space-time conﬁgurations

7

(a)

(b)

(c)

(d)

Figure 4: (a) Estimates of LZ complexity of space-time conﬁgurations of ECA Rule 110 without
Woronin bodies. (b) A space-time conﬁguration of ECA Rule 110 evolving from initial conﬁguration
10110001 (177), no Woronin bodies are activated. (c) A space-time conﬁguration of M1 Rule 110
evolving from initial conﬁguration 10110001 (177), Woronin body is governed by rule 43; red lines
indicate time when the body was activated and position of the cell with the body. In (bcd), a pixel in
position (i, t) is black if xt

i = 1.

8

Size, kB203040506070Initial conﬁguration, decimal encoding050100150200250Figure 5: Only one cell has Woronin body.

of automata with Woronin bodies are shown in Fig. 6b–h. As seen in Fig. 7 both species of fungal
automata show similar dynamics of complexity along the Woronin transition functions ordered by their
decimal values. The automaton M1 has average LZ complexity 82.2 (σ = 24.6) and the automaton
M2 78.4 (σ = 22.1). Woronin rules g which generate most LZ complex space-time conﬁgurations are
ρg = 133 in M)1 (Fig. 6b) and ρg = 193 in M)2 (Fig. 6e). The space-time dynamics of the automaton
is characterised by a substantial number of gliders guns and gliders (Fig. 6b). Functions being in the
middle of the descending hierarchy of LZ complexity produce space-time conﬁgurations with declined
number of travelling localisation and growing domains of homogeneous states (Fig. 6cg). Automata
with Woronin functions at the bottom of the complexity hierarchy quickly (i.e. after 200-300 iterations)
evolve towards stable, equilibrium states (Fig. 6dh).

5 Local events

Let us consider some local events happening in the fungal automata discussed in Sect. 4: every 50th
cell of an array has a Woronin body.

Retaining gliders. A glider can be stopped and converted into a station localisation by a cell
with Woronin body. As exempliﬁed in Fig. 8a, the localisation travelling left was stopped from further
propagation by a cell with Woronin body yet the localisation did not annihilate but remained stationary.
Register memory. Diﬀerent substrings of input string (initial conﬁguration) might lead to diﬀer-
ent equilibrium conﬁgurations achieved in the domains of the array separated by cells with Woronin
bodies. When there is just two types of equilibrium conﬁgurations they be seen as ‘bit up’ and ‘bit
down’ and therefore such fungal automaton can be used a memory register (Fig. 8b).

Reﬂectors. In many cases cells with Woronin bodies induce local domains of stationary, sometimes
time oscillations, inhomogeneities which might act as reﬂectors for travelling localisations. An example
is shown in Fig. 8c where several localisations are repeatedly bouncing between two cells with Woronin
bodies.

Modiﬁers. Cells with Woronin bodies can act as modiﬁers of states of gliders reﬂected from them
and of outcomes of collision between travelling localizations. In Fig. 8d we can see how a travelling
localisation is reﬂected from the vicinity of Woronin bodies three times: every time the state of the
localisation changes. On the third reﬂection the localisation becomes stationary.
In the fragment
(Fig. 8e) of space-time conﬁguration of automaton with Woronin bodies governed by ρg = 201 of the
fragment we can see how two localisations got into contact with each in the vicinity of the Woronin
body and an advanced structure is formed two breathing stationary localisations act as mirror, and
there are streams of travelling localisations between them. A multi-step chain reaction can be observed
in Fig. 8f: there are two stationary, breathing, localisations at the sites of the cells with Woronin bodies.
A glider is formed on the left stationary localisation. The glider travel to the right and collide into
right breather. In the result of the collision the breath undergoes structural transitions, emits a glider
travelling left and transforms itself into a pair of stationary breathers. Meantime the newly born glider
collided into left breather and changes its state.

6 Discussion

As a ﬁrst step towards formalisation of fungal intelligence we introduced one-dimensional fungal au-
tomata operated by two local transition function: one, g, governs states of Woronin bodies (pores are
open or closed), another, f , governs cells states: ‘0’ and ‘1’. We provided a detailed analysis of the

9

``````   1       1     0      0      1     0     0      1      0(a)

(b) M1, ρg = 133

(c) M1, ρg = 29

(d) M1, ρg = 49

(e) M2, ρg = 193

(f) M2, ρg = 5

(g) M2, ρg = 221

(h) M2, ρg = 174

Figure 6: (a) ECA Rule 110, no Woronin bodies. Space-time evolution of M∞ (bcd) and M∈ (e–h)
for Woronin rules shown in subcaption. LZ complexity of space-time conﬁgurations decreases from (b)
to (d) and from (e) to (h). Every 50th cell has a Woronin body.

10

Figure 7: Estimations of LZ complexity of space-time, 500 cells by 500 iterations, conﬁgurations of
M1, discs, and M1, circles, for all Woronin functions g.

(a) M1,
ρg = 2

(b) M1, ρg = 15

(c) M1,
ρg = 21

(d) M1,
ρg = 31

(e) M1, ρg =
29

(f) M1,
ρg = 201

(a) Localisation travelling left was stopped by the Woronin body. (b) Analog of a memory
Figure 8:
register. (c) Reﬂections of travelling localisations from cells with Woronin bodies. (d) Modiﬁcation of
glider state in the vicitinity of Woronin bodies. (e) A fragment of conﬁguration of automaton with
ρg = 29, left cell states, right Woronin bodies states. (f) Enlarged sub-fragment of the fragment (d)
where Wonorin body tunes the outcome of the collision. For both automata ρf = 110.

11

Size, kB406080100120140Woronin rule ρg, decimal encoding050100150200250Figure 9: An example of 5-inputs-7-outputs collision in M2, ρf = 110, ρg = 40. Every 50th cell has a
Woronin body. Cells state transitions are shown on the left, Woronin bodies state transitions on the
i = 1, left, or wt
right. A pixel in position (i, t) is black if xt

i = 1, right.

12

magma (cid:104)f, g, ◦(cid:105), results of which might be useful for future designs of computational and language
recognition structures with fungal automata. The magma as a whole does not satisfy any other prop-
erty but closure. Chances are high that there are subsets of the magma which might satisfy conditions
of other algebraic structures. A search for such subsets could be one of the topics of further studies.

Another topic could be an implementation of computational circuits in fungal automata. For
certain combination of f and g we can ﬁnd quite sophisticated families of stationary and travelling
localisations and many outcomes of the collisions and interactions between these localisations, an
illustration is shown in Fig. 9. Thus the target could be, for example, to construct a n-binary full
adder which is as compact in space and time as possible.

The theoretical results reported show that by controlling just a few cells with Woronin bodies it is
possible to drastically change dynamics of the automaton array. Third direction of future studies could
be in implemented information processing in a single hypha. In such a hypothetical experimental setup
input strings will be represented by arrays of illumination and outputs could be patterns of electrical
activity recorded from the mycelium hypha resting on an electrode array.

Acknowledgement

AA, MT, HABW have received funding from the European Union’s Horizon 2020 research and in-
novation programme FET OPEN “Challenging current thinking” under grant agreement No 858132.
EG residency in UWE has been supported by funding from the Leverhulme Trust under the Visiting
Research Professorship grant VP2-2018-001.

References

[1] Andrew Adamatzky. On spiking behaviour of oyster fungi Pleurotus djamor. Scientiﬁc reports,

7873, 2018.

[2] Andrew Adamatzky. Towards fungal computer. Interface focus, 8(6):20180029, 2018.

[3] Andrew Adamatzky, Martin Tegelaar, Han AB Wosten, Anna L Powell, Alexander E Beasley,
and Richard Mayne. On boolean gates in fungal colony. arXiv preprint arXiv:2002.09680, 2020.

[4] Alexander E Beasley, Anna L Powell, and Andrew Adamatzky. Memristive properties of mush-

rooms. arXiv preprint arXiv:2002.06413, 2020.

[5] Julia Beck and Frank Ebel. Characterization of the major Woronin body protein hexa of the
human pathogenic mold aspergillus fumigatus. International Journal of Medical Microbiology,
303(2):90–97, 2013.

[6] Michael W Berns, James R Aist, William H Wright, and Hong Liang. Optical trapping in animal
and fungal cells using a tunable, near-infrared titanium-sapphire laser. Experimental cell research,
198(2):375–378, 1992.

[7] Robert-Jan Bleichrodt, Marc Hulsman, Han AB W¨osten, and Marcel JT Reinders. Switching from
a unicellular to multicellular organization in an aspergillus niger hypha. MBio, 6(2):e00111–15,
2015.

[8] Robert-Jan Bleichrodt, G Jerre van Veluw, Brand Recter, Jun-ichi Maruyama, Katsuhiko Kita-
moto, and Han AB W¨osten. Hyphal heterogeneity in aspergillus oryzae is the result of dynamic
closure of septa by Woronin bodies. Molecular microbiology, 86(6):1334–1344, 2012.

[9] Robert-Jan Bleichrodt, Arman Vinck, Nick D Read, and Han AB W¨osten. Selective transport
between heterogeneous hyphal compartments via the plasma membrane lining septal walls of
aspergillus niger. Fungal Genetics and Biology, 82:193–200, 2015.

13

[10] Paola Bonfante and Iulia-Andra Anca. Plants, mycorrhizal fungi, and bacteria: a network of

interactions. Annual review of microbiology, 63:363–383, 2009.

[11] Michael John Carlile, Sarah C Watkinson, and Graham W Gooday. The fungi. Gulf Professional

Publishing, 2001.

[12] Martha Christensen. A view of fungal ecology. Mycologia, 81(1):1–19, 1989.

[13] Annette J Collinge and Paul Markham. Woronin bodies rapidly plug septal pores of severedpeni-

cillium chrysogenum hyphae. Experimental mycology, 9(1):80–85, 1985.

[14] Matthew Cook. Universality in elementary cellular automata. Complex systems, 15(1):1–40, 2004.

[15] Roderic C Cooke, Alan DM Rayner, et al. Ecology of saprotrophic fungi. Longman, 1984.

[16] Yu-Cheng Dai and Bao-Kai Cui. Fomitiporia ellipsoidea has the largest fruiting body among the

fungi. Fungal biology, 115(9):813–814, 2011.

[17] Peter Deutsch and Jean-Loup Gailly. Zlib compressed data format speciﬁcation version 3.3.

Technical report, 1996.

[18] David Michael Griﬃn et al. Ecology of soil fungi. Ecology of soil fungi., 1972.

[19] Marie Held, Clive Edwards, and Dan V Nicolau. Examining the behaviour of fungal cells in
microconﬁned mazelike structures. In Imaging, Manipulation, and Analysis of Biomolecules, Cells,
and Tissues VI, volume 6859, page 68590U. International Society for Optics and Photonics, 2008.

[20] Marie Held, Clive Edwards, and Dan V Nicolau. Fungal intelligence; or on the behaviour of
microorganisms in conﬁned micro-environments. In Journal of Physics: Conference Series, volume
178, page 012005. IOP Publishing, 2009.

[21] Paul Glor Howard. The Design and Analysis of Eﬃcient Lossless Data Compression Systems.

PhD thesis, Citeseer, 1993.

[22] Gregory Jedd and Nam-Hai Chua. A new self-assembled peroxisomal vesicle required for eﬃcient

resealing of the plasma membrane. Nature cell biology, 2(4):226–231, 2000.

[23] Yannik Leonhardt, Sara Carina Kakoschke, Johannes Wagener, and Frank Ebel. Lah is a trans-
membrane protein and requires spa10 for stable positioning of Woronin bodies at the septal pore
of aspergillus fumigatus. Scientiﬁc reports, 7:44179, 2017.

[24] Roger R Lew. Mass ﬂow and pressure-driven hyphal extension in neurospora crassa. Microbiology,

151(8):2685–2692, 2005.

[25] Kristian Lindgren and Mats G. Nordahl. Universal computation in simple one-dimensional cellular

automata. Complex Systems, 4(3):299–318, 1990.

[26] Genaro J. Mart´ınez, Harold V McIntosh, and Juan C. Seck-Tuoh Mora. Production of gliders by
collisions in rule 110. In European Conference on Artiﬁcial Life, pages 175–182. Springer, 2003.

[27] Genaro J. Mart´ınez, Harold V McIntosh, and Juan C. Seck-Tuoh Mora. Gliders in rule 110.

International Journal of Unconventional Computing, 2(1):1, 2006.

[28] Genaro J. Mart´ınez, Juan C. Seck-Tuoh Mora, and Sergio V.C. Vergara. Rule 110 objects and

other collision-based constructions. Journal of Cellular Automata, 2:219–242, 2007.

[29] Jun-ichi Maruyama, Praveen Rao Juvvadi, Kazutomo Ishi, and Katsuhiko Kitamoto. Three-
dimensional image analysis of plugging at the septal pore by woronin body during hypotonic
shock inducing hyphal tip bursting in the ﬁlamentous fungus aspergillus oryzae. Biochemical and
biophysical research communications, 331(4):1081–1088, 2005.

14

[30] Michelle Momany, Elizabeth A Richardson, Carole Van Sickle, and Gregory Jedd. Mapping

Woronin body position in aspergillus nidulans. Mycologia, 94(2):260–266, 2002.

[31] Royall T Moore and James H McAlear. Fine structure of mycota. 7. observations on septa of

ascomycetes and basidiomycetes. American Journal of Botany, 49(1):86–94, 1962.

[32] Turlough Neary and Damien Woods. P-completeness of cellular automaton rule 110. In Interna-

tional Colloquium on Automata, Languages, and Programming, pages 132–143. Springer, 2006.

[33] Seng Kah Ng, Fangfang Liu, Julian Lai, Wilson Low, and Gregory Jedd. A tether for Woronin
body inheritance is associated with evolutionary variation in organelle positioning. PLoS genetics,
5(6), 2009.

[34] S Olsson and BS Hansson. Action potential-like activity found in fungal mycelia is sensitive to

stimulation. Naturwissenschaften, 82(1):30–31, 1995.

[35] Alan DM Rayner, Lynne Boddy, et al. Fungal decomposition of wood. Its biology and ecology.

John Wiley & Sons Ltd., 1988.

[36] Greg Roelofs and Richard Koman. PNG: the deﬁnitive guide. O’Reilly & Associates, Inc., 1999.

[37] Cliﬀord L Slayman, W Scott Long, and Dietrich Gradmann. “Action potentials” in Neurospora
crassa, a mycelial fungus. Biochimica et Biophysica Acta (BBA)-Biomembranes, 426(4):732–744,
1976.

[38] Myron L Smith, Johann N Bruhn, and James B Anderson. The fungus Armillaria bulbosa is

among the largest and oldest living organisms. Nature, 356(6368):428, 1992.

[39] Shanthi Soundararajan, Gregory Jedd, Xiaolei Li, Marilou Ramos-Pamplo˜na, Nam H Chua, and
Naweed I Naqvi. Woronin body function in magnaporthe grisea is essential for eﬃcient pathogen-
esis and for survival during nitrogen starvation stress. The Plant Cell, 16(6):1564–1574, 2004.

[40] Gero Steinberg, Nicholas J Harmer, Martin Schuster, and Sreedhar Kilaru. Woronin body-based

sealing of septal pores. Fungal Genetics and Biology, 109:53–55, 2017.

[41] Martin Tegelaar, Robert-Jan Bleichrodt, Benjamin Nitsche, Arthur FJ Ram, and Han AB W¨osten.
Subpopulations of hyphae secrete proteins or resist heat stress in aspergillus oryzae colonies.
Environmental microbiology, 22(1):447–455, 2020.

[42] Karen Tenney, Ian Hunt, James Sweigard, June I Pounder, Chadonna McClain, Emma Jean
Bowman, and Barry J Bowman. Hex-1, a gene unique to ﬁlamentous fungi, encodes the major
protein of the woronin body and functions as a plug for septal pores. Fungal Genetics and Biology,
31(3):205–217, 2000.

[43] Wei Kiat Tey, Alison J North, Jose L Reyes, Yan Fen Lu, and Gregory Jedd. Polarized gene
expression determines Woronin body formation at the leading edge of the fungal colony. Molecular
biology of the cell, 16(6):2651–2659, 2005.

[44] APJ Trinci and Annette J Collinge. Occlusion of the septal pores of damaged hyphae ofneurospora

crassa by hexagonal crystals. Protoplasma, 80(1-3):57–67, 1974.

[45] William P Wergin. Development of Woronin bodies from microbodies infusarium oxysporum f.

sp. lycopersici. Protoplasma, 76(2):249–260, 1973.

[46] Stephen Wolfram. Universality and complexity in cellular automata. Physica D: Nonlinear Phe-

nomena, 10(1-2):1–35, 1984.

15

[47] Stephen Wolfram. Cellular automata and complexity: collected papers. Addison-Wesley Pub. Co.,

1994.

[48] Jacob Ziv and Abraham Lempel. A universal algorithm for sequential data compression. IEEE

Transactions on information theory, 23(3):337–343, 1977.

16

View publication stats

