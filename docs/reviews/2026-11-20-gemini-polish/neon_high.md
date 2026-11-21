# Gemini 4 Neon (High), paragraph polish, 2026-11-20

Conversation: https://gemini.google.com/u/1/app/b91b088e54a7006e (Gemini Pro account).
Input: the 10-page PDF built at 9160971 (Table 1 in one block, run labels still present) and the concatenated LaTeX of every section.
Sent 17:54 PT, retrieved 18:10 PT. Math in the notes was rendered twice by the page (as symbols and as text); only the text form is kept.

## Prompt

Same as `hydrogen_high.md`.

## Response (verbatim from the page text)

```latex
%%%%% FILE: sections/0_abstract.tex
\begin{abstract}
AutoTerm-SST brings zero-setup terminology adaptation to streaming speech
translation, removing both pre-session glossary uploads and domain selection.
Existing retrieval-augmented translators such as Retrieval-Augmented
Simultaneous Speech Translation, or RASST, expect a user-supplied glossary
before decoding begins. Pooling all domains into one universal glossary avoids
manual setup, yet fixed retrieval and prompt budgets cause irrelevant and
conflicting entries to crowd out relevant terms as catalogs scale.
AutoTerm-SST partitions an offline terminology catalog into topic slices, uses
a small language model to score each slice against recent partial translations,
and maintains a posterior with a sticky topic filter so term retrieval searches
only the top-ranked slices under an unchanged budget. On a 4.7-hour
English-to-Chinese stream alternating ACL talks and oncology lectures,
AutoTerm-SST achieves 87.7\% terminology accuracy, matching a known-domain
reference at 87.5\%, while outperforming a one-million-entry merged glossary by
3.3 accuracy points and 2.31 BLEU. On ten-domain RealSI streams, the router
keeps the active domain searchable in 89.7\% of audio chunks, compared with
78.8\% for a similarity router with tuned confirmation rules. A live interface
exposes the active slices and retrieved term pairs alongside each partial
translation.
\end{abstract}

%%%%% FILE: sections/1_introduction.tex
\section{Introduction}

Rare domain terminology remains a primary failure mode in simultaneous speech
translation, or SST, even for long-form speech large language models, or LLMs
\citep{xu2025qwen3omni,ouyang2025infinisst}. Direct streaming models generate
target text as speech unfolds in technical talks, lectures, and live media, yet
acoustic evidence alone leaves rare method names, clinical concepts, and named
entities hard to recover. Glossary retrieval addresses this gap by matching
incoming speech against source-target term pairs and injecting top candidates
into the decoding prompt.

Current retrieval-augmented SST systems force a trade-off between manual
session setup and catalog-scale degradation. Retrieval-Augmented Simultaneous
Speech Translation, or RASST \citep{luo2026rasst}, boosts out-of-domain
terminology accuracy, yet expects a fixed glossary before decoding begins.
Listeners in live talks seldom know the exact domain beforehand, making manual
glossary selection a barrier to adoption. Merging all domains into a single
universal glossary removes setup friction and works at small scale, yet fixed
candidate and prompt budgets let irrelevant and conflicting mappings crowd out
valid terms as the catalog expands. Scaling a merged glossary from 100,000 to
one million entries drops terminology accuracy from 87.8\% to 84.4\% in
Table~\ref{tab:quality}.

The streaming translation itself provides sufficient topical signal to select
the glossary online. Short windows of partial output reveal the active subject:
on ten-domain RealSI streams, a 4-billion-parameter LLM reading the last four
translated chunks ranks a new talk's glossary first after a median of 15.6
seconds. A streaming translator can therefore infer the domain directly from
its own partial hypotheses with zero listener input.

AutoTerm-SST moves glossary selection from pre-session setup into the streaming
loop while keeping per-chunk search bounded. Operators register offline topic
slices, each containing roughly 1,000 term pairs and a short topic prototype.
During decoding, a small LLM scores every prototype against recent partial
translations, a sticky hidden Markov filter accumulates those scores into a
topic posterior, and up to four top-ranked slices form the active working set
inside a fixed term budget. Retrieval and decoding remain unchanged, holding
per-chunk cost constant as the catalog grows. Our contributions are threefold:
\begin{itemize}
\item A zero-setup terminology-aware SST demonstration built on RASST, paired
with a live interface exposing active slices and retrieved term pairs alongside
every partial translation.
\item Training-free language-model routing with a sticky topic filter
parameterized by a single switch rate, keeping the correct glossary searchable
in 89.7\% of RealSI audio chunks compared with 78.8\% for a similarity router
with tuned confirmation rules.
\item Empirical evidence on a 4.7-hour stream alternating natural language
processing, or NLP, and oncology talks that online routing incurs no
statistically significant quality loss relative to a known-domain reference and
outperforms a one-million-entry merged glossary by 3.3 terminology accuracy
points and 2.31 BLEU.
\end{itemize}

%%%%% FILE: sections/2_related_works.tex
\section{Related work}

\paragraph{Streaming speech translation.}
Streaming SST systems optimize the quality-latency trade-off through dedicated
decoding policies and unified speech-language models, while leaving terminology
management largely orthogonal. Prefix-to-prefix schedules formalize this
trade-off \citep{ma2019stacl}, attention-guided policies refine read and write
decisions \citep{papi2023alignatt,papi2024streamatt}, and SimulEval
standardizes benchmarking \citep{ma2020simuleval}. Direct speech-language
models further unify long-form streaming architectures
\citep{seamless2023,zhang2024streamspeech,xu2025qwen3omni}. AutoTerm-SST
preserves the underlying streaming model and wraps it with online terminology
management and live inspection.

\paragraph{Terminology-aware translation.}
Existing terminology-aware translation methods rely on a glossary known before
decoding, leaving dynamic domain discovery unaddressed. Rare terms and named
entities remain persistent error sources in speech translation
\citep{gaido2021moby}. Prior work injects user-supplied dictionaries via
constrained decoding or training-time exposure
\citep{hokamp2017lexically,dinu2019training}. Retrieval-augmented generation
provides a non-parametric alternative
\citep{lewis2020retrieval,khandelwal2021nearest} that RASST adapts to streaming
LLM decoding \citep{luo2026rasst}. AutoTerm-SST eliminates pre-session glossary
setup by selecting topic slices online, surfaces those decisions like
interactive inspection tools \citep{dandekar2024translation}, and runs over a
continuous-batching serving engine
\citep{kwon2023efficient,zheng2024sglang}.

%%%%% FILE: sections/3_system_overview.tex
\section{System overview}
\label{sec:system}

AutoTerm-SST wraps RASST in an online terminology-selection loop while leaving
every translation and retrieval parameter unchanged, as shown in
Figure~\ref{fig:architecture}. RASST decodes 1.92-second audio chunks using a
speech encoder, a cross-modal term retriever, and the Qwen3-Omni LLM: the
encoder maps chunk $t$ to acoustic features $h_t$, the retriever constructs a
term map $G_t$, and the LLM translates $h_t$ conditioned on $G_t$. Operating
directly on speech in the absence of a transcript, the retriever scores
glossary source terms by maximum similarity, or MaxSim. Each source term has a
pre-encoded vector $z_i$, chunk $t$ yields window vectors $q_v$ for
$v\in\mathcal V_t$, and the term score is
$r_i=\max_{v\in\mathcal V_t}q_v^\top z_i$. Candidates exceeding 0.78 are
retained, and up to ten source-target pairs enter $G_t$. AutoTerm-SST modifies
only the searchable subset: its router limits lookup to a working set of at
most four topic slices, and each emitted translation updates the router context
for subsequent chunks.

\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{autoterm_architecture.pdf}
\caption{Live topic transition from NLP to medicine in AutoTerm-SST. The
adaptive terminology memory in orange scores slices against recent translations
and activates up to four slices covering at most 4,000 terms; MaxSim builds
each term map $G_t$ strictly from this working set, and the LLM translates
features $h_t$ using $G_t$. Ellipses denote additional offline slices.}
\label{fig:architecture}
\end{figure*}

Keeping adaptation outside the decoding model preserves a modular serving
architecture. A single streaming protocol supports retrieval-free translation,
fixed-glossary RASST, and AutoTerm-SST alike, attaching the retrieved pairs,
their source slices, and the active working set to every partial translation
sent to the interface.

%%%%% FILE: sections/4_adaptive_memory.tex
\section{Adaptive terminology memory}
\label{sec:adaptive-memory}

Separating an offline terminology catalog from a bounded active working set
lets AutoTerm-SST span diverse domains at constant per-chunk retrieval cost.
Each topic slice bundles roughly 1,000 term-translation pairs, a prebuilt
MaxSim index over its source terms, and a bilingual prototype summarizing the
topic with example terms. Storing translations per slice allows identical source
terms to take domain-specific target translations across slices.

\paragraph{Offline catalog.}
Slice registration decouples catalog storage from active retrieval. Operators
register a slice through a single manifest entry specifying its label, term
count, prototype, and index path; the router reads only manifest metadata and
prototypes, and retrieval bypasses all inactive indexes.

\paragraph{Language-model routing.}
A small instruction-tuned LLM scores every catalog slice directly from recent
target-language output in two training-free forward passes. Because direct
speech translation emits no source transcript, routing inspects target
hypotheses as topical context accumulates. At chunk $t$, the router
concatenates the last $w=4$ translated chunks into context $c_t$ and constructs
a multiple-choice prompt mapping each slice prototype to an option letter. Each
prototype $P_d$ comprises a one-line description and an English sentence listing
core topics and representative terms. The router LLM $\pi$, 4-billion-parameter
Qwen3-4B-Instruct~\citep{yang2025qwen3}, scores slice $d$ by its first-token
probability on the letter assigned to $d$. Averaging across the forward order
$\sigma$ and reversed order $\bar\sigma$ cancels position bias:
\begin{equation}
o_t(d)=\tfrac12\bigl[\pi(\ell_\sigma(d)\mid c_t)
+\pi(\ell_{\bar\sigma}(d)\mid c_t)\bigr],
\end{equation}
where $\ell_\sigma(d)$ denotes the letter of $d$ under order $\sigma$. Scoring
requires zero token generation or fine-tuning.

\paragraph{Topic filtering.}
A sticky hidden Markov model smooths noisy per-update scores into a stable
topic posterior $b_t$ using a single free parameter. Between updates, the
active topic persists with probability $1-\rho$ and transitions uniformly to
the remaining $N-1$ slices otherwise, treating $o_t$ as the observation
likelihood:
\begin{equation}
\begin{aligned}
b_t(d)\propto{}&\Bigl[(1-\rho)\,b_{t-1}(d)\\
&+\tfrac{\rho}{N-1}\bigl(1-b_{t-1}(d)\bigr)\Bigr]\,o_t(d)^{\beta_t}.
\end{aligned}
\end{equation}
To avoid double-counting overlapping windows across consecutive contexts, the
exponent $\beta_t=\min(1,n_t/w)$, where $n_t$ counts chunks translated since
the last update, weights each chunk once. We fix the switch rate at
$\rho=10^{-3}$; Appendix~\ref{app:router} shows that top-four coverage shifts
by at most 0.5 points as $\rho$ spans $10^{-5}$ to $10^{-1}$. Causal updates
ensure decisions at chunk $t$ influence only future chunks.

\paragraph{Working-set update.}
Periodic posterior ranking caps active memory while allowing outgoing and
incoming topics to overlap across transitions. Every five seconds, the router
rescores all slices, updates the posterior, and selects up to $K=4$ top-ranked
slices within a budget of $B=4{,}000$ searchable terms, excluding lower-ranked
slices from retrieval. Appendix~\ref{app:config} summarizes all router
settings.

\paragraph{Fixed evidence budget.}
All active slices share a single retrieval and prompt budget, ensuring gains
arise from higher relevance within identical capacity. The retriever partitions
100 candidate slots evenly across active slices, removes exact duplicates,
reranks the remainder globally, and injects up to ten pairs exceeding the 0.78
threshold into the prompt.

\paragraph{Non-blocking updates.}
Asynchronous background updates keep routing off the live decoding path. A
worker batches every session with updated context, each chunk retrieves against
the working set active at decoding start, and newly computed working sets take
effect on the next chunk. Indexes for probable slices preload concurrently in
the background.

%%%%% FILE: sections/5_demo_interface.tex
\section{Demo interface}
\label{sec:demo-interface}

AutoTerm-SST delivers terminology-aware streaming translation through a
standard zero-setup interface built for conference attendees, technical
students, and clinicians following specialized live talks. Users pick a
language pair and start a microphone or file stream directly, requiring zero
glossary uploads or manual domain selection.

\paragraph{Walkthrough.}
A live session tracks domain shifts autonomously from the first utterance.
Partial translations stream immediately; as context builds, the top status row
displays the highest-posterior slice while the evidence panel lists retrieved
term pairs per output. Across topic shifts, outgoing and incoming slices
briefly coexist in the working set.

\begin{figure*}[t]
\centering
\includegraphics[width=0.64\textwidth]{ui_evidence_panel.png}
\caption{English-to-Chinese replay interface during an oncology lecture,
roughly 19 minutes into run 3. The status row reports the top-ranked medicine
slice and 4,000 active terms; retrieved terms appear highlighted inline, and
the bottom panel lists recent pairs with similarity scores and source
slices.}
\label{fig:ui}
\end{figure*}

\paragraph{Visible terminology evidence.}
Inline highlighting and a live evidence panel expose every retrieval decision
for immediate inspection. In Figure~\ref{fig:ui}, the interface highlights
matched terms inside the translation and lists their source-target mappings;
the status row displays the top slice, its posterior probability, and active
term count, while the panel header reports active slices and retrieval latency.
Viewers can immediately distinguish whether a mistranslated term went
unretrieved, originated from an off-topic slice, or went unused during
decoding.

\paragraph{Extensibility.}
Adding a new domain requires a single glossary file and zero service downtime.
Registration validates term pairs, builds the MaxSim index, and appends a
manifest entry, with optional LLM prototype generation evaluated in
Appendix~\ref{app:router}. Active sessions retain their catalog snapshot, and
fixed-glossary presets remain available.

\paragraph{Availability.}
AutoTerm-SST ships as an MIT-licensed open-source release with a one-command
launcher and screencast in the
\href{https://github.com/luojiaxuan/autoterm-sst}{GitHub repository} and a
hosted entry page at \url{https://luojiaxuan.github.io/autoterm-sst/}.
Alongside the recorded replay shown in Figure~\ref{fig:ui}, live serving runs
Qwen3-Omni with vLLM continuous batching and the RASST retriever.

%%%%% FILE: sections/6_evaluation.tex
\section{Evaluation}
\label{sec:evaluation}

Our evaluation validates three core properties of AutoTerm-SST: zero-setup
routing performs on par with a known-domain reference, a bounded working set
avoids the degradation of large merged glossaries, and language-model routing
tracks frequent domain shifts across ten topics.

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{routing_states_realsi.pdf}
\caption{Per-chunk routing states across the ten equal-width talks of the
first RealSI stream order. Each row marks whether the talk's slice ranks
first, ranks lower in the working set, or is absent. The language-model
router ranks a new domain first within 21 seconds in nine of ten talks,
compared with three for similarity routing.}
\label{fig:routing}
\end{figure*}

\subsection{Experimental setting}

\paragraph{Data.}
Two multi-talk English-to-Chinese streams evaluate closed-loop translation
quality and ten-domain routing responsiveness. For translation quality, we
alternate five ACL 60/60 dev talks with human-annotated
terminology~\citep{salesky-etal-2023-evaluating} as the NLP portion and five
long-form oncology lectures from the European School of Oncology, or ESO, test
set~\citep{iranzosanchez2025eso} as the medicine portion, forming one
continuous 4.7-hour stream. Following RASST~\citep{luo2026rasst}, ESO
evaluation uses model-generated Chinese references and manually verified
terminology; because both slices include the benchmark glossaries, this stream
directly tests whether routing keeps gold terms searchable. For multi-domain
routing, we concatenate the ten English-to-Chinese RealSI
talks~\citep{cheng2024clasi}, each lasting 3.4 to 5.2 minutes with a human
domain label mapping to one slice, into three 45-minute streams in random
orders.

[Table 1: tabular unchanged from the input.]
\caption{Translation and terminology results on the ten-talk stream. Input
gives pre-session input, Active the search cap, and Pairs mean pairs per chunk;
Similarity routing uses Appendix~\ref{app:config}. TERM\_ACC and prompt
precision are percentages; MT-BLEU masks target terms. $^*$Run 1 on identical
chunks; $^\dagger$$p<0.05$ against AutoTerm-SST by permutation; InfiniSST
averages both corpora.}
\label{tab:quality}

[Table 2: tabular unchanged except the last header, Never top -> No top-1.]
\caption{Routing accuracy and speed across 30 RealSI talks. Columns report
chunk coverage and top-1 rate in percent, median delay to first rank one, and
talks failing to reach rank one.}
\label{tab:realsi}

\paragraph{Baselines.}
All evaluated systems share RASST's model, chunking, retriever, threshold,
candidate slots, and prompt cap, isolating the effect of which entries remain
searchable; each baseline name reflects its per-chunk search scope.
InfiniSST~\citep{ouyang2025infinisst} disables retrieval completely.
Known-domain-1k searches only the 1,000-entry slice matching the active talk,
requiring ground-truth domain labels. AutoTerm-SST routes over ten 1,000-entry
slices covering NLP, medicine, and eight Wikipedia and Wikidata topics,
activating at most four slices per chunk. Similarity routing retains those
slices and budgets using the Appendix~\ref{app:config} embedding router.
Merged-2k pools the NLP and medicine slices under known domains, whereas
Merged-10k pools all ten slices into 9,880 deduplicated entries. Merged-1M
augments the NLP and medicine glossaries with external topic entries, and
Merged-100k takes its first 100,000 entries; merged baselines retain
conflicting translations for shared source terms.

\paragraph{Runs.}
Identical audio sequences and chunk boundaries allow talk-level pairing across
all hardware runs. Run 3 evaluates AutoTerm-SST alongside Known-domain-1k,
similarity routing, Merged-2k, and Merged-10k on two H200 GPUs, which also host
the RealSI streams. Merged-100k and Merged-1M come from run 1 on one H200 GPU
and pair talk by talk with AutoTerm-SST. Runs 1 and 2, the latter on two A6000
GPUs, additionally compare similarity routing against Known-domain-1k.

\paragraph{Metrics.}
Our evaluation tracks terminology accuracy, translation fluency, prompt
cleanliness, and domain coverage. Following RASST, terminology accuracy, or
TERM\_ACC, measures the share of 2,047 time-aligned term occurrences, 1,368 NLP
and 679 medicine, whose reference translation appears in the aligned
hypothesis. Translation quality uses
mWERSegmenter~\citep{matusov-etal-2005-evaluating} to align each talk to
StreamLAAL reference boundaries~\citep{papi2024streamatt} before scoring corpus
SacreBLEU~\citep{papineni-etal-2002-bleu,post-2018-call}. Masked-term BLEU, or
MT-BLEU, repeats this score with evaluated target terms masked in both
hypothesis and reference. Prompt precision measures the fraction of injected
pairs matching a gold term in the source chunk, and RealSI coverage measures
the fraction of chunks keeping the correct slice active.

\paragraph{Statistics.}
Exact talk-level permutation tests and stratified bootstrap intervals quantify
uncertainty for every paired comparison, all listed in
Appendix~\ref{app:stats}. Each permutation test swaps system outputs talk by
talk across all $2^{10}$ assignments, while a corpus-stratified talk bootstrap
yields 95\% confidence intervals, written $a\pm h$ when symmetric and
$a^{+u}_{-l}$ otherwise, for AutoTerm-SST minus each baseline.

\subsection{Results}

\paragraph{Known-domain parity.}
AutoTerm-SST matches a known-domain reference within sampling noise using zero
user input. In run 3, AutoTerm-SST minus Known-domain-1k is
$+0.2^{+0.8}_{-1.0}$ TERM\_ACC points at $p=0.82$ and $-0.05^{+0.44}_{-0.55}$
BLEU at $p=0.85$. Similarity routing similarly tracks the reference across runs
1 and 2 in Table~\ref{tab:stats}, and every glossary-equipped run substantially
outperforms retrieval-free InfiniSST in Table~\ref{tab:quality}.

\paragraph{Catalog scaling.}
Merged glossaries match routed retrieval up to 10k entries, yet lose both
fluency and terminology accuracy at larger catalog scales. Merged-2k and
Merged-10k stay within 0.4 TERM\_ACC points of AutoTerm-SST at $p\ge0.34$ with
comparable BLEU in Table~\ref{tab:quality}. Merged-100k maintains TERM\_ACC
parity, yet trails AutoTerm-SST by $1.38^{+0.53}_{-0.70}$ BLEU at $p=0.010$. At
one million entries, Merged-1M drops behind by $3.3^{+1.5}_{-1.9}$ TERM\_ACC
points at $p=0.031$ and $2.31^{+0.63}_{-0.82}$ BLEU at $p=0.002$; medicine
accounts for most of that terminology loss, falling 7.4 points against 1.2 on
NLP. Prompt crowding drives the broader quality drop: Merged-1M injects nine
times the pairs per chunk at one tenth the prompt precision, and AutoTerm-SST
retains a 1.85 BLEU gain at $p=0.047$ across the 935 sentences whose references
contain no glossary term.

\paragraph{Ten-domain routing.}
Language-model routing tracks multi-domain transitions that heuristic
confirmation rules miss. On RealSI in Table~\ref{tab:realsi}, it raises
working-set coverage from 78.8\% to 89.7\% over similarity routing and promotes
a new talk's slice to rank one after a median of 15.6 seconds. Because
similarity routing requires three consecutive leads and a 30-second cooldown,
its top slice fails to match the active domain in 19 of the 30 talks,
illustrated for the first stream order in Figure~\ref{fig:routing}. On the
slower two-domain stream, both routers keep the active slice searchable in
roughly 98\% of chunks, yielding equivalent translation accuracy.
Appendix~\ref{app:router} extends this comparison to faster switches and 24
close-domain slices.

\paragraph{Processing speed.}
Routing and retrieval add minimal latency relative to generation. AutoTerm-SST
matches Known-domain-1k in overall real-time factor in Table~\ref{tab:speed},
processing audio in roughly a fifth of its duration, compared with 0.042
seconds per chunk measured for RASST retrieval alone on identical GPUs. The
first partial translation appears 2.3 seconds after audio onset.

[Table 3: tabular unchanged.]
\caption{Per-chunk compute time on one ACL talk across three A6000 GPUs for
1.92-second chunks. Retrieval covers speech encoding and routing; the
real-time factor, RTF, divides total seconds by chunk duration.}
\label{tab:speed}

%%%%% FILE: sections/7_limitations_conclusion.tex
\section{Conclusion}

AutoTerm-SST removes glossary upload and domain selection from
terminology-aware streaming translation by routing partial translations across
offline topic slices. It performs on par with a known-domain reference, avoids
the quality losses of a million-entry merged glossary, and tracks ten-domain
shifts. Extending routing to finer-grained subtopics remains a natural next
step.

\section*{Ethics statement}
All experiments rely strictly on published research corpora: ACL 60/60, the ESO
test set, and RealSI under its CC BY 4.0 license, with zero audio
redistribution. We used Claude, GPT, and Gemini for coding, data processing,
and editing; the authors verified all outputs and assume full responsibility.

%%%%% FILE: sections/8_appendix.tex
[Table 4: tabular unchanged from the input, including the run 1 and run 2 block.]
\caption{Paired metric differences between each routed system and baseline. Low
and High report 95\% stratified talk-bootstrap bounds, and $p$ gives the exact
two-sided talk-level permutation $p$-value. TERM\_ACC differences are
percentage points; $^*$ denotes run 1 baselines.}
\label{tab:stats}

\section{Statistical comparisons}
\label{app:stats}

Paired talk-level tests confirm that AutoTerm-SST's advantages over Merged-1M
hold robustly across individual talks, with all paired comparisons detailed in
Table~\ref{tab:stats}. Our exact permutation test swaps system outputs within
each talk across all 1,024 combinations and recomputes the pooled metric,
yielding a minimum two-sided $p$-value of 0.002 on which all significance
claims rest. Confidence intervals derive from 4,000 stratified bootstrap
resamples of five ACL and five ESO talks; with ten talks, percentile intervals
can diverge from exact permutation tests near the boundary, excluding zero
while $p\ge0.08$ for run 2 Merged-10k and similarity-routing MT-BLEU.
Leave-one-talk-out analysis preserves every AutoTerm-SST advantage over
Merged-1M by at least 2.57 TERM\_ACC points, 2.04 BLEU, and 1.70 MT-BLEU.

\section{Router configuration}
\label{app:config}

Language-model routing adds one hyperparameter, the switch rate $\rho$, to the
shared settings in Table~\ref{tab:config}, whereas the similarity baseline
requires five tuned confirmation thresholds. The similarity router embeds each
context, truncated to the last 128 tokens of $c_t$, alongside Chinese-augmented
prototypes using BGE-M3~\citep{chen-etal-2024-m3}. It pins a confirmed slice in
the working set and updates it only when a challenger beats both the runner-up
and the confirmed slice by fixed margins across three consecutive updates
outside a 30-second cooldown. Those thresholds were calibrated on four short
segments sampled from the evaluation talks prior to running full streams.
Because both routers operate on wall-clock intervals, each ran exclusively on
its host server.

[Table 5: tabular unchanged.]
\caption{Router and retrieval hyperparameters. Lead thresholds represent
normalized similarity margins, and similarity-router switches require
satisfying both margins across every consecutive agreeing update.}
\label{tab:config}

\section{Router diagnostics}
\label{app:router}

Open-loop replays compare both routers on identical cached translations. Each
replay feeds saved run 1 translations to both routers using the four-chunk
context of Section~\ref{sec:adaptive-memory} and refreshes scores at most once
every 5 seconds of audio; the similarity router applies its confirmation and
pinning rules, while the language-model router executes one filter step with
$\rho=10^{-3}$. Table~\ref{tab:router-diag} reports four-slice working-set
coverage across all conditions. Replayed on run 1 emission timestamps, the
similarity router reproduces its live behavior closely, executing 19 switches
as in live decoding and achieving 96.2\% chunk coverage compared with 96.9\%
live.

\paragraph{Topic filter.}
Posterior filtering turns noisy per-update scores into stable rankings with no
threshold rules. Under per-chunk updates, raw letter probabilities keep the
correct slice in the top four for 90.5\% of chunks, rising to 98.7\% after
filtering. Coverage holds between 98.4\% and 98.9\% as $\rho$ spans $10^{-5}$
to $10^{-1}$, indicating low sensitivity on this stream. Raw similarity covers
85.0\% under identical updates, increasing to 97.0\% with confirmation and
pinning.

\paragraph{Short topic shifts.}
Posterior filtering tracks topic alternations spanning 15 to 300 seconds. For
each duration in Table~\ref{tab:router-diag}, we slice saved NLP and medicine
translations into fixed-length segments and interleave them into one 40-segment
stream. Across all segment lengths, the language-model router enters the median
segment with the incoming slice already in the working set and promotes it to
rank one after a median of 7.7 seconds. Constrained by three consecutive leads
and a 30-second cooldown, the similarity router confirms the incoming slice in
segments of one minute or longer only after a median of 27 to 46 seconds.

\paragraph{Other target languages.}
Open-loop routing coverage transfers to German and Japanese outputs on the
ten-talk stream. Replaying German and Japanese RASST translations of the same
ten talks requires changing only the target language named in the prompt.

\paragraph{Close domains.}
Finer-grained slices widen the gap between the two routers. Using 400
non-evaluation terms per catalog, the same 4B model proposes eight subtopics
each for NLP and medicine and assigns every term by letter probability,
yielding 24 slices alongside the eight background topics. Across the 2,047 gold
term occurrences, the target term's slice stays active for 82.7\% under
language-model routing and 52.2\% under similarity routing, compared with
97.3\% for an oracle selecting the best four slices per talk in hindsight.

\paragraph{Generated prototypes and late updates.}
Automatically generated prototypes closely match hand-written descriptions, and
one-chunk update latency causes negligible coverage loss. GPT-5.5 generated one
prototype per slice from glossaries stripped of all evaluation terms; under
per-chunk updates, language-model top-four coverage reaches 97.9\% on the
ten-talk stream compared with 98.7\% using hand-written prototypes. RealSI
experiments adopt these generated prototypes because the hand-written NLP
description overlaps with the RealSI technology talk. Whereas the closed-loop
runs of Section~\ref{sec:evaluation} completed each routing update before
advancing chunks, the asynchronous worker can apply an update one chunk late
during faster-than-real-time audio arrival; delaying every update by one chunk
reduces language-model coverage by at most 0.33 points across
Table~\ref{tab:router-diag}.

[Table 6: tabular unchanged.]
\caption{Open-loop coverage of both deployed routers at the live update rate.
Cells report percentages of chunks keeping the correct slice in the four-slice
working set; the close-domain row counts gold term occurrences with an active
slice.}
\label{tab:router-diag}
```

The tabular bodies above are marked as unchanged; Gemini returned them in full and they match the input exactly.

## Gemini's notes (verbatim)

Five Most Important Changes

Conclusion-first topic sentences across every section: Rewrote procedural and lab-log openings (the Abstract, Introduction paragraph 1, Offline catalog, Language-model routing, Working-set update, Data, Runs, Metrics, Statistics, and Appendix B) so every paragraph leads with its core claim before supplying implementation or experimental detail.

Tighter statistical and empirical calibration: Softened overbroad claims about zero quality loss, universal parameter insensitivity, live non-blocking execution, and multilingual transfer so every statement stays strictly within the bounds of the measured ten-talk and open-loop data.

Unpacked dense numerical sentences and tightened captions: Split the stacked Merged-1M result sentence in Section 6.2 into clear single-point clauses, tightened Table 1's caption from five rendered lines down to roughly three and a half, and standardized all nine captions to begin with short noun phrases within the 3-to-4-line budget.

Eliminated banned phrasing, defensive negations, and punctuation violations: Removed all occurrences of never (including Table 2's Never top header, renamed No top-1), many, and vague a few phrasing, replaced defensive constructions (as opposed to, need not, neither ... nor), and converted compound en dashes (--) to hyphens across all running text.

Reduced total prose length across both main text and appendix: Trimmed roughly 125 words from the main text and 30 words from the appendix while keeping every citation, label, reference, interval, and p-value intact, ensuring the 6-page main and 2-page appendix limits remain comfortable.

Sentences in the Original Draft That Overclaim

Section 6 opening (We test three claims: zero setup costs no accuracy or translation quality...) and Introduction bullet 3 (zero setup costs no significant quality): A non-significant permutation test on ten talks (p=0.82 for TERM_ACC, p=0.85 for BLEU) cannot establish zero loss; Table 4's 95% confidence intervals allow drops of up to 0.83 TERM_ACC points and 0.60 BLEU. Revised to performs on par with a known-domain reference and incurs no statistically significant quality loss.

Section 4 (No chunk waits for routing.): Appendix C reveals that the closed-loop evaluation runs in Section 6 actually waited for each routing update to finish before advancing chunks, whereas only the deployed background worker runs asynchronously. Revised to scope the claim explicitly to the live decoding path (Asynchronous background updates keep routing off the live decoding path.).

Section 6.2 (AutoTerm-SST processes audio in about a fifth of its duration, as fast as Known-domain-1k in Table 3): Table 3 shows AutoTerm-SST's retrieval-plus-routing time is 0.094 s versus 0.059 s for Known-domain-1k on a single ACL talk; overall RTF ties at 0.22 only because generation happened to be slightly faster (0.333 s vs. 0.359 s). Revised to specify matches Known-domain-1k in overall real-time factor.

Appendix C (Coverage stays between 98.4% and 98.9% as \rho ranges from 10^{-5} to 10^{-1}, so the single free parameter needs no tuning.): Sensitivity to rho was tested only on the two-domain ten-talk open-loop replay, where both domains fit easily inside the four-slice working set, not on fast-switching ten-domain RealSI streams or 24 close-domain slices. Revised to indicating low sensitivity on this stream.

Appendix C (The router reads German and Japanese output as well as Chinese.): The experiment tests only two-domain open-loop working-set coverage on the ten-talk stream (where even similarity routing exceeds 97%), not closed-loop translation quality or multi-domain routing in German or Japanese. Revised to Open-loop routing coverage transfers to German and Japanese outputs on the ten-talk stream.

Table 2 (To top: 68.1 s, Never top: 19 out of 30 talks) — methodological caveat to verify: If 19 of 30 talks fail to reach rank one under similarity routing, an unconditional median across all 30 talks is undefined; 68.1 s must be the median conditioned on the talks that do reach rank one. I kept the exact wording and numbers per your hard rules, but consider clarifying median delay among talks reaching rank one in Table 2's caption if that matches your computation. Also note one minor arithmetic check in Section 6.2: Table 1 lists NLP TERM_ACC as 88.4 vs. 87.1 (a 1.3-point difference from rounded table values), while the prose reports 1.2 (likely from unrounded decimals)—worth double-checking for consistency.
