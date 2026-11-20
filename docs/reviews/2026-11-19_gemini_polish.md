# Gemini 逐段润色原文（2026-11-19 PT）

- 模型：gemini.google.com 网页版，账号 luojiaxuan1215（Pro），模型 4 Neon（Deeper analysis），思考档 High；当时页面上更高的 4 Carbon 带锁。
- 输入：`latex/` 在 1df5eb2 编译出的 PDF、`latex/sections/*.tex` 拼接、EMNLP 2026 demo 的 meta review 与四份审稿意见、写作要求（规则原文照贴）。
- 下面是网页取回的回复全文，引用标记（如 “paper_sections.tex + 1”）照原样保留。

---

下面按 paper_sections.tex 的原始顺序逐段输出改写后的 LaTeX（保留全部原始数字、实验设置、公式、表格单元格、\citep/\ref/\label 命令，严格遵守 instructions.md 的写作规范，正文总词数略短于原稿以保证稳妥控制在 6 页内）。

## 改写后的 LaTeX 正文与附录

```latex
%%%%%%%%%%%%%%%% FILE: sections/0_abstract.tex
% [Abstract]
\begin{abstract}
Retrieval-augmented speech translation improves rare terminology, yet systems
such as Retrieval-Augmented Simultaneous Speech Translation, or RASST, expect
the user to supply a glossary before every session. Merging every domain into
one universal glossary removes this step, but as the catalog expands,
irrelevant and conflicting entries crowd out useful terms inside the fixed
retrieval and prompt budget. We present AutoTerm-SST, an interactive streaming
speech translation system that operates without a glossary upload or domain
selection. AutoTerm-SST partitions an offline terminology catalog into topic
slices, matches recent partial translations against a short bilingual
description of each slice, and restricts term retrieval to the best-matching
slices under an unchanged budget. On a 4.7-hour English-to-Chinese stream
alternating five ACL talks with five oncology lectures, AutoTerm-SST reaches
88.3\% terminology accuracy, showing no significant difference from a
known-domain reference at 88.1\%, and a second independent run reproduces this
parity. Against a one-million-entry merged glossary, AutoTerm-SST gains 3.9
points in terminology accuracy and 2.30 BLEU, both significant in a talk-level
permutation test. The full pipeline runs 2.46 times faster than real time, and
the live interface displays active slices and retrieved term pairs alongside
every partial translation.
\end{abstract}

%%%%%%%%%%%%%%%% FILE: sections/1_introduction.tex
\section{Introduction}

% [Sec 1, Para 1]
Simultaneous speech translation, or SST, generates target-language text while
a speaker is still talking, serving technical talks, lectures, and live media.
Speech large language models, or LLMs, now translate long-form audio directly
\citep{xu2025qwen3omni,ouyang2025infinisst}, yet rare methods, datasets,
clinical concepts, and named entities remain difficult to recover from
acoustics alone. Glossary retrieval closes this gap by matching incoming speech
against source-target term pairs and placing the best matches in the LLM prompt
before decoding.

% [Sec 1, Para 2]
Existing retrieval-augmented SST systems require manual session setup or lose
accuracy as terminology catalogs expand. Retrieval-Augmented Simultaneous
Speech Translation, or RASST \citep{luo2026rasst}, improves terminology on
out-of-domain talks, but it fixes the glossary before translation begins.
Listeners rarely know the exact domain of a live talk in advance, and finding,
uploading, or selecting a glossary before every session adds friction. Merging
all domains into one universal glossary avoids manual selection and works at
small scale, yet fixed retrieval and prompt capacities allow irrelevant and
conflicting mappings to crowd out relevant terms as the catalog grows. In our
evaluation, expanding a merged glossary from 100,000 entries to one million
entries lowers terminology accuracy from 87.8\% to 84.4\%.

% [Sec 1, Para 3: Core observation]
The translation stream itself supplies the domain evidence needed to choose a
glossary. Partial output names the active topic within seconds: in an open-loop
replay of system translations, the slice for a new topic ranks among the top
four slices after a median of one 1.92-second audio chunk. A streaming
translator can therefore infer the domain from its own recent output without
requiring the listener to declare a topic beforehand.

% [Sec 1, Para 4 & Contributions]
We present AutoTerm-SST, a system demo that moves glossary selection from user
setup into the streaming loop. Operators register offline topic slices, each
holding roughly 1,000 term pairs and a short bilingual topic description.
During a session, a training-free router compares recent partial translations
against each slice description and activates up to four matching slices inside
a fixed term budget. Retrieval and decoding stay unchanged: AutoTerm-SST
controls only which slices compete for each prompt, keeping the per-chunk
budget constant as the catalog grows. The live interface exposes active slices,
retrieved term pairs, and highlighted target terms as audio streams, delivering
three contributions:
\begin{itemize}
\item Zero-setup terminology-aware SST built on RASST, paired with a live
interface that displays active slices and retrieved term pairs beside every
partial translation while running 2.46 times faster than real time.
\item Budgeted multi-slice terminology memory that selects offline slices
from the system's own translations without training, holding per-chunk
retrieval and prompt budgets constant as the catalog grows and keeping the
correct glossary searchable in 96.9\% of decoding windows by pinning the
confirmed slice.
\item Evidence on a 4.7-hour stream alternating natural language
processing, or NLP, and oncology talks that zero setup incurs no
significant quality loss: AutoTerm-SST achieves parity with a known-domain
reference across two independent runs and beats a one-million-entry merged
glossary by 3.9 points of terminology accuracy and 2.30 BLEU, whereas
merged glossaries match it on both metrics only up to 10,000 entries.
\end{itemize}

%%%%%%%%%%%%%%%% FILE: sections/2_related_works.tex
\section{Related work}

% [Sec 2, \paragraph{Streaming speech translation.}]
\paragraph{Streaming speech translation.}
Streaming SST systems balance translation quality against latency by decoding
partial speech as it arrives. Prefix-to-prefix policies make this latency
trade-off explicit \citep{ma2019stacl}. Attention-based policies refine read
and write decisions during streaming \citep{papi2023alignatt,papi2024streamatt}.
SimulEval standardizes evaluation across streaming policies
\citep{ma2020simuleval}. Recent speech-language models further broaden direct
SST architectures \citep{seamless2023,zhang2024streamspeech,xu2025qwen3omni}.
AutoTerm-SST retains an existing streaming model and wraps it with adaptive
terminology management and an interactive inspection layer.

% [Sec 2, \paragraph{Terminology-aware translation.}]
\paragraph{Terminology-aware translation.}
Existing terminology-aware translation methods assume that the relevant
glossary is known before decoding starts. Conventional approaches apply
user-provided dictionaries through constrained decoding or training-time
constraint exposure \citep{hokamp2017lexically,dinu2019training}. Rare terms
and named entities remain a persistent weakness of speech translation
\citep{gaido2021moby}. Retrieval-augmented generation provides a non-parametric
alternative \citep{lewis2020retrieval,khandelwal2021nearest}. Building on this
approach, RASST retrieves glossary pairs from incoming speech before streaming
LLM decoding \citep{luo2026rasst}. AutoTerm-SST removes the pre-session
glossary assumption by selecting a budgeted set of topic slices online.

% [Sec 2, \paragraph{Interactive translation systems.}]
\paragraph{Interactive translation systems.}
High-throughput serving engines and explainable interfaces make live
translation both fast and inspectable. Continuous batching engines such as
vLLM and SGLang sustain low-latency LLM serving
\citep{kwon2023efficient,zheng2024sglang}. Interactive translation tools make
model decisions visible to users \citep{dandekar2024translation}. AutoTerm-SST
combines streaming audio transport with automatic glossary routing and a live
terminology evidence view.

%%%%%%%%%%%%%%%% FILE: sections/3_system_overview.tex
\section{System overview}
\label{sec:system}

% [Sec 3, Para 1]
AutoTerm-SST adds an online terminology-selection loop around RASST while
leaving its neural models unchanged, as Figure~\ref{fig:architecture} shows.
RASST is a retrieval-augmented SST pipeline combining a causal speech encoder,
a cross-modal term retriever, and the Qwen3-Omni LLM, translating audio in
1.92-second chunks. To retrieve spoken terms without an intermediate
transcript, the retriever scores every glossary entry by maximum similarity, or
MaxSim, between pre-encoded term vectors and current speech representations.
Each source term is pre-encoded offline as a vector $z_i$, and for chunk $t$
the speech encoder produces vectors $q_w$ across overlapping audio windows
$w\in\mathcal W_t$, yielding the term score
$r_i=\max_{w\in\mathcal W_t}q_w^\top z_i$. Terms scoring above 0.78 become
candidates, and at most ten source-target pairs enter the LLM prompt.
AutoTerm-SST restricts this lookup to a working set of up to four topic slices
selected by its router, then appends each newly decoded translation to the
router's context to guide slice selection for subsequent chunks.

% [Sec 3, Figure 1]
\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{autoterm_architecture.pdf}
\caption{Architecture of AutoTerm-SST during a live topic transition from NLP
to medicine. Recent translation context guides the router to activate up to
four topic slices under a fixed term budget, and RASST retrieves prompt terms
only from this working set. The ellipsis indicates additional offline topic
slices registered in the catalog.}
\label{fig:architecture}
\end{figure*}

% [Sec 3, Para 2]
Placing adaptation outside the translation model keeps the system modular and
pluggable. The router preserves both the speech representation and the
retriever's candidate budget, deciding only which prebuilt slice indexes share
that budget. Consequently, one streaming protocol serves a retrieval-free
agent, a fixed-glossary RASST agent, or AutoTerm-SST, while streaming
structured terminology evidence to the web interface with every partial
translation.

% [Sec 3, Para 3]
Three architectural invariants guarantee zero session setup, bounded retrieval
cost, and fair policy comparisons. First, live requests carry no domain label
or uploaded glossary. Second, catalog growth leaves per-chunk retrieval and
prompt budgets constant. Third, adding a topic is strictly an offline data
update with no model training. All compared policies in
Section~\ref{sec:evaluation} therefore share the identical speech encoder,
retriever, decoder, and streaming schedule, differing only in which
terminology indexes are searchable.

% [Sec 3, Para 4]
Every partial translation carries structured retrieval metadata so that system
behavior stays inspectable during live streaming. For each chunk, the agent
emits the retrieved term pairs, their source slices, retrieval latency, and the
active working set, driving the evidence panel in
Section~\ref{sec:demo-interface}.

%%%%%%%%%%%%%%%% FILE: sections/4_adaptive_memory.tex
\section{Adaptive terminology memory}
\label{sec:adaptive-memory}

% [Sec 4, Para 1]
Separating an offline terminology catalog from a small active working set lets
AutoTerm-SST cover broad domains while keeping per-chunk search bounded. Each
topic slice packages about 1,000 term-translation pairs for one topic, a
prebuilt MaxSim index over those source terms, and a bilingual prototype
consisting of a short topic description and example terms from that topic.
Because each slice stores its own translations independently, the same source
term maps to domain-specific targets in different slices.

% [Sec 4, \paragraph{Offline catalog.}]
\paragraph{Offline catalog.}
Offline slices are registered as modular catalog entries that expand coverage
without enlarging online retrieval. An operator registers a slice by adding one
entry to a manifest file listing its label, term count, prototype text, and
index path. Slices are independent and replaceable, and at session start the
router reads only the manifest and prototypes. During decoding, the retriever
skips all prebuilt indexes belonging to slices outside the working set, so
catalog growth leaves the active search space unchanged.

% [Sec 4, \paragraph{Semantic routing.}]
\paragraph{Semantic routing.}
The router selects relevant slices by matching recent partial translations
against each slice prototype. Because the speech LLM translates audio directly
and produces no source transcript, routing reads target-language output, which
becomes more topic-specific as context builds. At chunk $t$, the router
concatenates the last $w$ translated chunks into a context $c_t$; for slice $d$
with prototype texts $P_d$ and sentence encoder $E$, it computes
\begin{equation}
\begin{aligned}
\bar p_d &=
\operatorname{norm}\!\left(\frac{1}{|P_d|}
\sum_{p\in P_d} E(p)\right), \\
s(d\mid c_t) &= \cos\!\left(E(c_t),\bar p_d\right).
\end{aligned}
\end{equation}
We use the multilingual encoder BGE-M3~\citep{chen-etal-2024-m3} for $E$,
placing Chinese partial translations and bilingual English and Chinese
prototype texts in one shared embedding space. Routing is causal, so a decision
made at chunk $t$ affects only later chunks.

% [Sec 4, Table 1]
（表格单元格与原稿相同，此处只录表注）
\caption{Terminology accuracy, translation quality, and prompt evidence on the
4.7-hour ten-talk stream. Input denotes user-supplied metadata before decoding,
and Active gives maximum searchable entries per chunk, with 1k representing
1,000 entries. TERM\_ACC is the share of gold term occurrences translated with
their glossary target, MT-BLEU scores translation around masked target terms,
Prompt prec. is the share of injected pairs matching a gold term, and Refs is
the mean term pairs injected into the prompt per chunk. The symbol $\dagger$
marks $p<0.05$ against AutoTerm-SST in the same run via a talk-level
permutation test; InfiniSST uses no glossary, and its All, BLEU, and MT-BLEU
values average RASST's separate ACL and medicine scores.}

% [Sec 4, \paragraph{Working-set update.}]
\paragraph{Working-set update.}
Pinning a confirmed slice alongside top-scoring alternatives stabilizes routing
while accommodating topic transitions. Every five seconds, the router rescores
every catalog slice and builds a working set of at most $K=4$ slices and
$B=4{,}000$ searchable terms in total. One slot stays pinned to the confirmed
slice, which switches only when another slice leads both the runner-up and the
confirmed slice by fixed score margins across three consecutive updates. The
remaining slots hold the highest-scoring other slices within budget $B$, while
lower-ranked slices remain excluded from retrieval. Holding up to four slices
lets outgoing and incoming topics coexist during transitions until stale slices
drop in rank; Appendix~\ref{app:config} lists all router thresholds.

% [Sec 4, \paragraph{Fixed evidence budget.}]
\paragraph{Fixed evidence budget.}
All active slices share one retrieval and prompt budget, keeping prompt size
independent of working-set count. The retriever distributes 100 candidate slots
across the active slices, removes exact duplicate source-target pairs, reranks
the remaining candidates globally, and inserts at most ten pairs into the LLM
prompt. Retrieving 100 candidates first leaves headroom for deduplication and
cross-slice reranking before final admission, and a chunk receives fewer than
ten pairs whenever fewer candidates clear the 0.78 score threshold. Gains in
AutoTerm-SST therefore come from placing more relevant mappings inside the same
prompt budget.

% [Sec 4, \paragraph{Causal updates.}]
\paragraph{Causal updates.}
Asynchronous index preloading ensures that routing updates leave in-flight
decoding undisturbed. Each chunk queries a snapshot of the working set
installed before chunk decoding begins, and any routing update finishing during
generation takes effect on the next chunk. Indexes for likely candidate slices
are preloaded in the background so that index loading causes no mid-prompt
changes.

%%%%%%%%%%%%%%%% FILE: sections/5_demo_interface.tex
\section{Demo interface}
\label{sec:demo-interface}

% [Sec 5, Para 1]
AutoTerm-SST serves listeners of specialized live speech who lack a prepared
glossary for every session, such as conference attendees, students in technical
courses, and clinicians following medical lectures. Users open the demo like an
ordinary streaming translator by selecting a language pair and starting a
microphone or file stream. AutoTerm-SST mode omits the glossary upload and
domain selector, starting with an initial working set that updates as
translation context accumulates.

% [Sec 5, \paragraph{Walkthrough.}]
\paragraph{Walkthrough.}
A live session transitions across topics automatically with zero manual
controls. The page renders live partial translations immediately while initial
context builds. Once enough context accumulates, the slice status row switches
to the highest-scoring slices, and retrieved term pairs appear beside the
partial output that used them. After a topic change, outgoing and incoming
slices coexist briefly until the sliding context reflects the new subject.

% [Sec 5, Figure 2]
\caption{Live English-to-Chinese interface of AutoTerm-SST after an automatic
transition from NLP to medicine. Inline highlights in the streaming output
and the evidence panel below expose each retrieved term pair, its source
slice, and retrieval latency.}

% [Sec 5, \paragraph{Visible terminology evidence.}]
\paragraph{Visible terminology evidence.}
Inline term highlighting and a dedicated evidence panel make terminology
behavior transparent during streaming. As Figure~\ref{fig:ui} shows, the
interface highlights retrieved terms inside the translation, lists their
source-target mappings, and displays active slices, active term count, and
retrieval latency in a status row. Viewers can directly determine whether a
mistranslation stems from missing retrieval evidence, an irrelevant slice, or
LLM decoding.

% [Sec 5, \paragraph{Extensibility.}]
\paragraph{Extensibility.}
Organizations grow the offline catalog from single glossary files without
restarting the translation service. Slice registration validates term pairs,
encodes them into an independent MaxSim index, and writes one manifest entry.
Operators author prototypes by hand or generate them from the glossary with an
LLM, and Appendix~\ref{app:router} shows that generated prototypes match
hand-written routing accuracy. Running sessions keep their current catalog
snapshot while new sessions immediately see newly registered slices;
fixed-glossary presets and ad-hoc term overrides remain available for expert
users.

% [Sec 5, \paragraph{Availability.}]
\paragraph{Availability.}
AutoTerm-SST is publicly released for both local deployment and hosted browser
access. The MIT-licensed implementation, a one-command launcher, and a
screencast reside in the
\href{https://github.com/luojiaxuan/autoterm-sst}{GitHub repository}, and the
hosted entry page is at \url{https://luojiaxuan.github.io/autoterm-sst/}. Live
mode serves Qwen3-Omni via vLLM continuous batching alongside the RASST
retriever.

%%%%%%%%%%%%%%%% FILE: sections/6_evaluation.tex
\section{Evaluation}
\label{sec:evaluation}

% [Sec 6, Para 1]
Our evaluation tests two claims: AutoTerm-SST eliminates pre-session setup
without sacrificing terminology accuracy or translation quality, and its fixed
working-set budget avoids the degradation of large merged glossaries.

% [Sec 6, Figure 3]
\caption{Routing trajectory of AutoTerm-SST across the ten alternating talks
of run 1. The confirmed slice can lag briefly after topic boundaries, yet the
four-slice working set keeps the correct glossary searchable in 96.9\% of
decoding windows. Talk widths are normalized for readability.}

\subsection{Experimental setting}

% [Sec 6.1, \paragraph{Data.}]
\paragraph{Data.}
We build one continuous English-to-Chinese stream by alternating five technical
NLP talks with five long-form oncology lectures. The stream includes five
English ACL 60/60 dev talks with human-annotated terminology
\citep{salesky-etal-2023-evaluating}. Interleaved between them are five
oncology lectures from the European School of Oncology, or ESO, test set
\citep{iranzosanchez2025eso}. Following RASST~\citep{luo2026rasst}, we evaluate
ESO using its model-generated Chinese references and manually verified ESO
terminology. Because the NLP and medicine slices hold the benchmark glossaries,
this stream isolates how reliably routing keeps target terms competitive under
a fixed budget.

% [Sec 6.1, \paragraph{Baselines.}]
\paragraph{Baselines.}
All evaluated policies share RASST's Qwen3-Omni model, chunking schedule,
MaxSim retriever, score threshold, candidate slots, and prompt cap, differing
only in which glossary entries are searchable. InfiniSST~\citep{ouyang2025infinisst}
runs the same streaming model with no retrieval. Known-domain-1k searches only
the 1,000-entry slice matching the current talk's domain and therefore requires
the domain label of every talk. AutoTerm-SST routes over ten 1,000-entry slices
spanning NLP, medicine, and eight topics drawn from Wikipedia and Wikidata,
with at most four active at once. Merged-2k pools the NLP and medicine slices,
which still requires knowing both domains of the stream. Merged-10k pools all
ten slices into 9,880 entries after removing exact duplicates. Merged-100k and
Merged-1M expand the same NLP and medicine glossaries with entries from other
topics, where Merged-100k consists of the first 100,000 entries of Merged-1M.
All merged glossaries retain conflicting translations of shared source terms.

% [Sec 6.1, \paragraph{Runs.}]
\paragraph{Runs.}
Two independent runs on distinct hardware verify reproducibility and span both
small and large merged baselines. Run 1 evaluates InfiniSST, Known-domain-1k,
AutoTerm-SST, Merged-100k, and Merged-1M on one H200 GPU. Run 2 repeats
Known-domain-1k and AutoTerm-SST on two A6000 GPUs alongside Merged-2k and
Merged-10k. Within each run, all policies observe the identical audio order and
chunk boundaries, and InfiniSST reuses archived outputs on the same ten talks.

% [Sec 6.1, \paragraph{Metrics.}]
\paragraph{Metrics.}
We evaluate terminology accuracy alongside full-sentence and masked-term
translation quality, supported by talk-level significance tests. Following
RASST, terminology accuracy, or TERM\_ACC, is the share of 2,047 time-aligned
term occurrences, comprising 1,368 NLP and 679 medicine instances, whose
reference translation appears in the aligned hypothesis. For translation
quality, mWERSegmenter first resegments each talk to align with reference
boundaries~\citep{matusov-etal-2005-evaluating}. Those boundaries follow the
StreamLAAL streaming protocol~\citep{papi2024streamatt}. We then compute corpus
SacreBLEU~\citep{papineni-etal-2002-bleu,post-2018-call}. Masked-term BLEU, or
MT-BLEU, repeats this scoring after masking every evaluated target term in both
hypothesis and reference to measure translation around the terms. Prompt
precision is the share of injected term pairs matching a gold term in the
surrounding audio. To compare two policies, an exact permutation test swaps
their outputs talk by talk over all $2^{10}$ combinations, and a talk-level
bootstrap stratified by corpus yields 95\% confidence intervals. We write
symmetric intervals as $a\pm h$ and asymmetric intervals spanning $a-l$ to
$a+u$ as $a^{+u}_{-l}$; Appendix~\ref{app:stats} reports every paired
comparison.

\subsection{Results}

% [Sec 6.2, \paragraph{Comparison with the known-domain reference.}]
\paragraph{Comparison with the known-domain reference.}
AutoTerm-SST achieves statistical parity with a known-domain reference without
receiving any user input. In run 1, AutoTerm-SST differs from Known-domain-1k
by $+0.2\pm1.2$ TERM\_ACC points at $p=0.86$ and $-0.10^{+0.48}_{-0.40}$ BLEU
at $p=0.70$. Run 2 on separate GPUs reproduces this parity, with full intervals
listed in Table~\ref{tab:stats}. Both retrieval-augmented policies exceed
retrieval-free InfiniSST by about 27 TERM\_ACC points, as
Table~\ref{tab:quality} shows.

% [Sec 6.2, \paragraph{Effect of catalog scale.}]
\paragraph{Effect of catalog scale.}
Merged glossaries keep pace with AutoTerm-SST up to 10k entries, yet lose
translation quality and terminology accuracy as the catalog grows further. In
run 2, Merged-2k and Merged-10k remain within 0.8 TERM\_ACC points of
AutoTerm-SST with matching BLEU and no significant gap in
Table~\ref{tab:stats}. At 100k entries in run 1, Merged-100k maintains
TERM\_ACC but incurs a significant BLEU loss in Table~\ref{tab:stats}. Scaling
to Merged-1M widens the deficit to $3.9\pm1.9$ TERM\_ACC points at $p=0.012$
and $2.30^{+0.65}_{-0.59}$ BLEU at $p=0.002$, concentrated heavily on medicine
with a 7.4-point loss against 2.2 points on NLP. AutoTerm-SST caps per-chunk
search at 4k entries, keeping performance insulated from catalog growth.

% [Sec 6.2, \paragraph{Effect of prompt noise.}]
\paragraph{Effect of prompt noise.}
Large merged glossaries fill the fixed prompt with irrelevant pairs, harming
translation quality even on sentences without specialized terms. As
Table~\ref{tab:quality} shows, Merged-1M injects nine times as many pairs per
chunk as AutoTerm-SST at one tenth of its prompt precision. On the 935 of 1,905
sentences whose references contain no glossary term, AutoTerm-SST still leads
Merged-1M by 2.19 BLEU at $p=0.023$.

% [Sec 6.2, \paragraph{Routing behavior.}]
\paragraph{Routing behavior.}
Pinning the confirmed slice inside a four-slice working set shields terminology
coverage from transient ranking errors. In run 1, the confirmed slice alone is
correct in 88.5\% of decoding windows, whereas the four-slice working set keeps
the correct glossary searchable across nearly the entire stream, as
Figure~\ref{fig:routing} shows. Pinning accounts for most of this gain: without
pinning, the four highest-scoring slices by raw similarity cover only 85.0\% of
windows. Open-loop replays in Appendix~\ref{app:router} show that the router
also tracks rapid switches between NLP and medicine every 15 to 300 seconds.
Routing still lags at abrupt talk boundaries, where stale context briefly kept
the medicine slice out of the working set after one ACL-to-medicine switch.

% [Sec 6.2, \paragraph{Latency and speed.}]
\paragraph{Latency and speed.}
The complete pipeline keeps pace with live speech in both interactive latency
and full-stream throughput. During a single session on two A6000 GPUs, the
first visible translation appears 2.3 seconds after audio starts, and each
chunk returns its translation in 0.57 seconds at the median and 1.17 seconds at
the 95th percentile, timed from sending the chunk to receiving its text. On one
H200 GPU, AutoTerm-SST processes the stream at a real-time factor, or RTF, of
0.406, defined as total processing time divided by audio duration. This RTF
stays within 0.02 of Known-domain-1k and below both large merged glossaries,
with routing and retrieval averaging 203 milliseconds per chunk.

%%%%%%%%%%%%%%%% FILE: sections/7_limitations_conclusion.tex
\section{Conclusion}

% [Sec 7, Para 1]
AutoTerm-SST eliminates glossary upload and domain selection from
terminology-aware streaming speech translation without sacrificing accuracy or
real-time speed. By routing over modular offline topic slices from its own
partial translations under a fixed budget, the system achieves parity with a
known-domain reference across two independent runs and avoids the degradation
of million-entry merged glossaries while exposing live terminology evidence in
the interface. Extending the router to more target languages and finer-grained
subtopics is a natural next step.

% [Acknowledgments]（与原稿相同）

%%%%%%%%%%%%%%%% FILE: sections/8_appendix.tex
（Table 2、Table 3、Table 4 的单元格与原稿相同，此处只录改写过的段落与表注）

\caption{Paired metric differences of AutoTerm-SST minus each baseline across
runs 1 and 2. Low and High give the 95\% stratified talk-bootstrap confidence
bounds, and $p$ is the exact two-sided talk-level permutation $p$-value.
TERM\_ACC differences are in percentage points, and the bottom block reports
BLEU on the 935 sentences whose references contain no glossary terms.}

% [Appendix A, Para 1]
Paired talk-level tests confirm that AutoTerm-SST's gains over Merged-1M hold
robustly across individual talks. Table~\ref{tab:stats} reports every paired
comparison. The exact permutation test swaps the outputs of two systems within
each talk across all 1,024 combinations and recomputes the pooled metric,
giving a minimum attainable two-sided $p$-value of 0.002. Confidence intervals
come from 4,000 stratified bootstrap resamples of five ACL and five ESO talks.
Removing any single talk preserves all three advantages of AutoTerm-SST over
Merged-1M, with gains of at least 3.28 TERM\_ACC points, 2.00 BLEU, and 1.99
MT-BLEU.

% [Appendix B, Para 1]
Both evaluation runs use one fixed set of routing and retrieval parameters
chosen before the full runs. Table~\ref{tab:config} lists all settings. We
selected the routing thresholds on four windows drawn from the evaluation talks
prior to running the complete benchmarks. The context encoder reads the last
128 tokens of translated context, and the router scores slices exclusively from
this context similarity.

\caption{Router and retrieval settings used by AutoTerm-SST in both runs. Lead
thresholds measure differences of normalized similarity scores, and a switch
requires satisfying both leads across three consecutive updates.}

% [Appendix C, Para 1]
Open-loop replays of the routing signal isolate the benefit of slice pinning,
measure responsiveness to rapid topic shifts, and verify LLM-generated
prototypes. Each diagnostic replays the router on the saved translations of run
1 without re-decoding, encoding the last four translated chunks as in
Section~\ref{sec:adaptive-memory} and ranking slices by raw similarity without
confirmation rules or pinning. Under raw similarity alone, the correct slice
ranks among the top four in 85.0\% of the 8,776 windows. Keeping the live run's
confirmed slice in the working set and filling the remaining three slots by raw
rank raises coverage to 97.0\%, matching the 96.9\% measured live.

% [Appendix C, \paragraph{Short topic shifts.}]
\paragraph{Short topic shifts.}
The router tracks topic switches quickly even when domains alternate every 15
to 300 seconds. We slice the saved translations of the five ACL and five
medicine talks into fixed segments of 15 to 300 seconds and interleave NLP and
medicine into 40-segment streams per duration. At every duration, the incoming
topic slice enters the top four after a median of one chunk, or 1.92 seconds,
and ranks first after a median of 5.76 seconds. It reaches the top four in all
40 segments at every length, and Table~\ref{tab:router-diag} shows that
top-four coverage stays above 81\% even when topics switch every 15 seconds.

% [Appendix C, \paragraph{Generated prototypes.}]
\paragraph{Generated prototypes.}
LLM-generated prototypes route as accurately as hand-written topic
descriptions. For each slice, we remove entries originating from the evaluation
glossaries and supply the remaining term pairs to \texttt{gpt-5.5-2026-04-23}
to generate a topic description, seven to ten core topics, and a Chinese
translation matching the format of the hand-written prototypes. Across ten
independently generated prototype sets in Table~\ref{tab:router-diag}, routing
accuracy brackets that of the hand-written baseline.

\caption{Open-loop routing replays on the saved translations of run 1 across
short topic shifts and prototype sources. Top-4 and Top-1 report the percentage
of windows whose correct slice ranks among the top four or ranks first by raw
similarity. Lowest and highest generated sets are chosen independently per
column.}
```

注：Gemini 的回复里把 Table 1 至 Table 4 的全部单元格原样抄了一遍，与 1df5eb2 的源码逐格相同，这里只录了改写过的表注。

## 审稿意见对照清单（可读性与表述问题）

| 审稿人 | 原审稿意见 | 改写稿处理位置 | 状态与说明 |
| --- | --- | --- | --- |
| Meta Review (VfFY) | 术语使用不一致或未恰当引入 | 全文各节首次出现处 | 已解决：统一系统名为 AutoTerm-SST，所有缩写与核心概念均在首次出现处以自然句式定义。 |
| Meta Review / FZMM | 声称“匹配 known-domain”过于武断，缺少显著性检验支撑 | Abstract；Sec 1 贡献第 3 条；Sec 6.2；Appendix A | 已解决：改用更审慎的统计表述（showing no significant difference from / achieves statistical parity with），并给出 Run 1 主对比区间、p 值及 Run 2 复现与 Table 2 索引。 |
| Meta Review / pCGH | 仅展示英中翻译，不清楚对其他语言对的泛化性 | Sec 4 Semantic routing；Sec 7 | 部分解决：Sec 4 明确 BGE-M3 为多语言编码器，Sec 7 把多语言扩展限定为下一步工作；因无其他语向实验数据，未新增未验证结论。 |
| pCGH | AutoTerm 与 AutoTerm-SST 混用 | 全文正文、图注、表注、附录 | 正文已解决 / 图内待改：LaTeX 文本中统一为 AutoTerm-SST；但 autoterm_architecture.pdf 图内橘框文字仍写着 AutoTerm，需在作图源文件修改。 |
| pCGH | 缩写引入过晚或从未定义（RTF、LLM、RASST、MaxSim、SST、NLP、ESO、TERM_ACC、MT-BLEU） | Abstract (RASST)；Sec 1 (SST, LLM, RASST, NLP)；Sec 3 (MaxSim)；Table 1 注与 Sec 6.1 (1k, TERM_ACC, MT-BLEU, ESO)；Sec 6.2 (RTF) | 已解决：所有缩写均在正文首次出现时直接融入句子定义（不依赖括号）。 |
| pCGH | 建议增大 Figure 2 和 Figure 3 | Fig 2 & Fig 3 | 未改动尺寸：正文已排满 6 页上限；建议裁剪图片留白或放大源图字号。 |
| pCGH | ESO 使用模型生成的中文参考译文，引发质量评估疑问 | Sec 6.1 Data | 已说明：沿用 RASST 设定（模型生成中文参考译文 + 人工校验 ESO 术语），属既有基准约束，无法仅靠文字改写消除。 |
| MctJ | "1k slice" / "matching 1k slice" 出现早于解释 | Sec 1 Para 4；Sec 4 Para 1；Table 1 注；Sec 6.1 | 已解决：Sec 1 先说明每个 slice 约含 1,000 对术语，Table 1 注首次使用 1k 即说明。 |
| MctJ | "LM" 直到图 4 才引入且与 RTF 实验设置不一致 | 已删除原图 4 及 LM 概念 | 已解决：当前稿件不再出现 LM。 |
| MctJ | 未解释的说法：compact manifest、representative terminology、indexes outside the working set、total active-term budget B、pinned、routing tick、lightweight prototypes、warmed asynchronously、working-set badge、strict prefix | Sec 4 各段；Sec 5；Sec 6.1 | 已解决：全部改写为直白具体的定义。 |
| MctJ | 句子之间缺少逻辑衔接 | 全文各段 | 已解决：每段以结论句领起，并补齐因果与转折连接。 |
| MctJ | Fixed evidence budget 混入消融对比；“Lower-ranked non-pinned slices stay outside the budget” 含义不清 | Sec 4 | 已解决：方法叙述与实验解耦，明确写出排除规则及具体参数。 |
| MctJ | 为什么先取 top-100 候选再过滤到 top-10 | Sec 4 Fixed evidence budget | 已解决：说明 100 个候选为跨切片去重与全局重排留出余量。 |
| MctJ | "capacity stress condition" 一句防御性且费解 | 已删除 | 已解决。 |
| MctJ | "target-bearing resources"、"distractors" 语义不清 | Sec 6.1 Baselines | 已解决：改用直白表述。 |
| MctJ | Refs 含义不明 | Table 1 注；Sec 4 | 已解决。 |
| MctJ | MaxSim 和 RASST 解释篇幅不足 | Sec 1 Para 2；Sec 3 Para 1 | 已解决。 |
| FZMM | 延迟评估缺少首次可见译文延迟与 per-chunk p50/p95 | Sec 6.2 Latency and speed | 已解决。 |
| v35B | 离线新增一个 slice 的人工或计算成本 | Sec 5 Extensibility；Appendix C | 部分解决：说明了单文件注册流程与 LLM 生成 prototype，但缺具体耗时数字。 |
| v35B / FZMM | 相近子领域实验、未知领域低置信度回退、随机检索消融 | Sec 7（只提 finer-grained subtopics） | 未解决：属于实验设计与系统机制问题，原稿无对应数据，未新增结论。 |

## 需要确认或补充的事项（Gemini 列出，未擅自改动）

1. 图 1（autoterm_architecture.pdf）右下角橘色框内标题写的是 AutoTerm 而非 AutoTerm-SST，建议同步更新源文件。
2. 图 2 与图 3 的尺寸：如有几行余量，可将图 2 调宽至约 0.92\textwidth，或裁剪图 2 外侧的深蓝背景留白。
3. 离线构建新切片的实际耗时（回应 v35B）：若有实测数据，可在 Sec 5 Extensibility 补一句。
4. 未知领域回退（回应 FZMM）：所有 slice 相似度都很低时是否仍保留 4 个 slice、是否支持空工作集回退，原稿未提及。
5. Demo 支持的语言对范围：Sec 5 提到用户选择语言对，但实验与截图均为英中；建议点明线上 demo 的切片目录范围。
6. Table 1 中 InfiniSST 的汇总方式：若是等权平均，建议在表注写明 unweighted means（0.5×(76.6+46.3)=61.45，约等于 61.4）。
