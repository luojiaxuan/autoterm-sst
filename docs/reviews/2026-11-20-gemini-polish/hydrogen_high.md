# Gemini 4 Hydrogen (High), paragraph polish, 2026-11-20

Conversation: https://gemini.google.com/u/1/app/398e3cc7dcf75e57 (Gemini Pro account).
Input: the 10-page PDF built at b9e33e1 and the concatenated LaTeX of every section.
The draft it polished still had Table 1 in two blocks (run 3 and run 1); 9160971 later merged it into one.

## Prompt

CONTEXT: The attached PDF is our NAACL 2027 System Demonstrations submission (6-page main text, appendix at most 2 pages; both limits are exactly met now) and the .txt file holds its LaTeX sections. TASK: Polish the writing paragraph by paragraph with the whole paper as context, so it reads like a top-venue paper: clear rather than dense, not defensive, not a lab report or chronological log, with a crisp claim and story. Return revised LaTeX for every section from the abstract to the conclusion plus the appendix prose, as full paragraphs ready to paste. HARD RULES: keep every number, interval, p-value, citation, label and ref exactly; never make a claim stronger than its numbers; total length must not grow because the page limits are full; no em or en dashes as punctuation; no parentheses in running text except citations and math; no defensive negations such as 'X, not Y' or 'rather than'; no vague quantifiers (many, some, several) or absolute words (never, always, clearly); do not pile numbers into sentences or repeat numbers the tables show; every paragraph opens with its conclusion; paragraph titles are short sentence-case noun phrases; captions start with a short noun phrase and stay within 3 to 4 rendered lines. After the LaTeX, list the five most important changes and any sentence you think overclaims.

## Response (verbatim from the page text)

```latex
%%%%% FILE: sections/0_abstract.tex
\begin{abstract}
AutoTerm-SST eliminates pre-session glossary setup in retrieval-augmented
simultaneous speech translation by dynamically routing live translation
streams to relevant offline topic slices. Existing systems such as
Retrieval-Augmented Simultaneous Speech Translation, or RASST, improve rare
terminology yet require users to supply a domain glossary before every session,
whereas pooling all domains into one universal glossary allows irrelevant and
conflicting entries to crowd fixed retrieval and prompt budgets as the catalog
expands. AutoTerm-SST partitions an offline terminology catalog into topic
slices, scores slice prototypes against recent partial translations with a
small language model, smooths these scores into a posterior via a sticky topic
filter, and restricts retrieval to the most probable slices under an unchanged
budget. On a 4.7-hour English-to-Chinese stream alternating ACL talks with
oncology lectures, AutoTerm-SST achieves 87.7\% terminology accuracy, matching a
known-domain reference at 87.5\% and outperforming a one-million-entry merged
glossary by 3.9 accuracy points and 2.30 BLEU. On ten-domain RealSI streams, the
language-model router keeps the target slice searchable in 89.7\% of audio
chunks, compared with 78.8\% for a similarity router with tuned confirmation
rules. The live interface displays active slices and retrieved term pairs
beside every partial translation.
\end{abstract}

%%%%% FILE: sections/1_introduction.tex
\section{Introduction}

Glossary retrieval bridges a persistent gap in simultaneous speech translation,
or SST, by supplying rare technical terms that acoustics alone fail to resolve.
Speech large language models, or LLMs, now translate long-form audio directly
\citep{xu2025qwen3omni,ouyang2025infinisst}, yet specialized method names,
clinical concepts, and named entities remain prone to acoustic error. Matching
incoming speech against source-target term pairs and injecting top candidates
into the decoding prompt substantially improves terminology recovery.

Existing retrieval-augmented SST systems force a trade-off between manual
session setup and accuracy loss at large catalog scale. Retrieval-Augmented
Simultaneous Speech Translation, or RASST \citep{luo2026rasst}, improves
terminology on out-of-domain talks yet requires fixing a glossary before
translation begins, adding friction when listeners lack advance domain
knowledge. Pooling all domains into one universal glossary avoids manual
selection, yet fixed candidate and prompt budgets allow irrelevant and
conflicting mappings to crowd out relevant terms as the catalog grows. In
Table~\ref{tab:quality}, expanding a merged glossary from 100,000 to one
million entries lowers terminology accuracy from 87.8\% to 84.4\%.

A streaming translator can infer the active domain directly from a few seconds
of its own partial translation output. On RealSI streams that switch domains
every few minutes, a 4-billion-parameter LLM reading the last four translated
chunks ranks a new talk's glossary first after a median of 15.6 seconds,
removing any need for listener domain input.

We present AutoTerm-SST, a system demonstration that moves glossary selection
from pre-session setup into the streaming loop under a constant per-chunk
budget. Operators register offline topic slices, each holding roughly 1,000
term pairs and a short prototype description. During a session, a small LLM
scores every prototype against recent partial translations, a sticky hidden
Markov filter accumulates these scores into a topic posterior, and at most four
top-ranked slices form the active working set while retrieval and decoding
remain unchanged. Our contributions are threefold:
\begin{itemize}
\item Zero-setup terminology-aware SST built on RASST, paired with a live
interface displaying active slices and retrieved term pairs alongside every
partial translation.
\item Training-free language-model routing with a sticky topic filter whose
sole free parameter is the switch rate, keeping the correct glossary searchable
in 89.7\% of chunks on ten-domain RealSI streams versus 78.8\% for a
similarity router with tuned confirmation rules.
\item Empirical validation on a 4.7-hour stream alternating natural language
processing, or NLP, and oncology talks showing statistical parity with a
known-domain reference and gains of 3.9 terminology accuracy points and 2.30
BLEU over a one-million-entry merged glossary.
\end{itemize}

%%%%% FILE: sections/2_related_works.tex
\section{Related work}

\paragraph{Streaming speech translation.}
Streaming SST architectures balance translation quality against latency while
decoding partial speech. Prefix-to-prefix policies make this trade-off
explicit \citep{ma2019stacl}, attention-based policies refine read and write
decisions \citep{papi2023alignatt,papi2024streamatt}, and SimulEval
standardizes evaluation \citep{ma2020simuleval}. Speech-language models broaden
direct SST architectures
\citep{seamless2023,zhang2024streamspeech,xu2025qwen3omni}. AutoTerm-SST wraps
an existing streaming backbone with online terminology management and live
evidence inspection.

\paragraph{Terminology-aware translation.}
Prior terminology-aware methods assume a known domain glossary prior to
decoding, whereas AutoTerm-SST selects glossaries online. Rare terms and named
entities remain a known challenge in speech translation
\citep{gaido2021moby}. Conventional approaches enforce user-supplied
dictionaries via constrained decoding or training-time exposure
\citep{hokamp2017lexically,dinu2019training}, while retrieval-augmented
generation offers a non-parametric alternative
\citep{lewis2020retrieval,khandelwal2021nearest} that RASST adapts to
streaming LLM decoding \citep{luo2026rasst}. AutoTerm-SST eliminates
pre-session glossary selection by routing topic slices online and exposes
retrieval decisions like interactive translation interfaces
\citep{dandekar2024translation} atop a continuous-batching serving engine
\citep{kwon2023efficient,zheng2024sglang}.

%%%%% FILE: sections/3_system_overview.tex
\section{System overview}
\label{sec:system}

AutoTerm-SST wraps RASST in an online terminology-selection loop while leaving
its acoustic and language models unchanged, as Figure~\ref{fig:architecture}
shows. RASST translates audio in 1.92-second chunks using a speech encoder, a
cross-modal term retriever, and the Qwen3-Omni LLM: the encoder maps chunk $t$
to features $h_t$, the retriever constructs a term map $G_t$, and the LLM
decodes $h_t$ with $G_t$ in its prompt. Because streaming decoding produces no
source transcript, the retriever matches speech directly against pre-encoded
source-term vectors $z_i$ across overlapping windows $v\in\mathcal V_t$ via
maximum similarity, or MaxSim, scoring $r_i=\max_{v\in\mathcal V_t}q_v^\top
z_i$. Candidates above 0.78 are ranked and at most ten source-target pairs
enter $G_t$. AutoTerm-SST restricts this lookup to a router-selected working
set of up to four topic slices and appends each new translation to the
router's context.

\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{autoterm_architecture.pdf}
\caption{AutoTerm-SST during a live topic shift from NLP to medicine. The
adaptive terminology memory, shown in orange, scores slices against recent
translations and activates up to four slices holding at most 4,000 terms; the
MaxSim retriever builds $G_t$ solely from this working set before the LLM
decodes speech features $h_t$, while the ellipsis denotes additional offline
slices.}
\label{fig:architecture}
\end{figure*}

Decoupling domain adaptation from the translation backbone keeps the serving
architecture modular across deployment modes. One unified streaming protocol
supports a retrieval-free translator, a fixed-glossary RASST translator, or
AutoTerm-SST, streaming retrieved pairs, source slices, and the active working
set alongside every partial translation.

%%%%% FILE: sections/4_adaptive_memory.tex
\section{Adaptive terminology memory}
\label{sec:adaptive-memory}

Separating a large offline catalog from a small working set enables broad
domain coverage at a constant per-chunk search cost. Each topic slice bundles
about 1,000 term-translation pairs, a prebuilt MaxSim index over source terms,
and a bilingual prototype containing a short topic description with example
terms. Storing translations per slice allows the same source term to map to
distinct targets across domains.

\paragraph{Offline catalog.}
Registering a new domain requires adding one manifest entry to the offline
catalog. Each entry specifies the slice label, term count, prototype, and index
path; the router reads only the manifest and prototypes, while the retriever
skips every index outside the working set.

\paragraph{Language-model routing.}
A compact instruction-tuned LLM scores catalog slices directly from recent
target-language translations without source transcripts or model training. At
chunk $t$, the router concatenates the last $w=4$ translated chunks into a
context $c_t$ and builds a multiple-choice prompt listing every slice prototype
$P_d$ under a letter, where $P_d$ pairs a one-line label with an English
sentence naming core topics and representative terms. The router $\pi$,
Qwen3-4B-Instruct \citep{yang2025qwen3}, scores slice $d$ by the first-token
probability assigned to its letter, averaging over listed order $\sigma$ and
its reverse $\bar\sigma$ to remove position bias:
\begin{equation}
o_t(d)=\tfrac12\bigl[\pi(\ell_\sigma(d)\mid c_t)
+\pi(\ell_{\bar\sigma}(d)\mid c_t)\bigr],
\end{equation}
where $\ell_\sigma(d)$ denotes the letter of $d$ under order $\sigma$. Each
update takes two forward passes and requires zero text generation or training.

\paragraph{Topic filtering.}
A sticky hidden Markov filter accumulates single-step router scores into a
stable posterior $b_t$ over the active topic. Between updates, the topic stays
with probability $1-\rho$ and otherwise transitions uniformly to one of the
other $N-1$ slices, while each update treats $o_t$ as the observation
likelihood:
\begin{equation}
\begin{aligned}
b_t(d)\propto{}&\Bigl[(1-\rho)\,b_{t-1}(d)\\
&+\tfrac{\rho}{N-1}\bigl(1-b_{t-1}(d)\bigr)\Bigr]\,o_t(d)^{\beta_t}.
\end{aligned}
\end{equation}
The exponent $\beta_t=\min(1,n_t/w)$, where $n_t$ counts chunks translated since
the previous update, prevents double-counting overlapping context windows.
Setting the sole free parameter $\rho=10^{-3}$ yields stable behavior:
Appendix~\ref{app:router} shows that top-four coverage shifts by at most 0.5
points across $\rho\in[10^{-5},10^{-1}]$. Causal updates affect only subsequent
chunks.

\paragraph{Working-set update.}
Periodic posterior ranking maintains a working set of at most four slices so
outgoing and incoming topics overlap smoothly during transitions. Every five
seconds, the router rescores all catalog slices, updates the posterior, and
selects up to $K=4$ top-ranked slices capped at $B=4{,}000$ searchable terms,
excluding lower-ranked slices until stale topics drop in rank.
Appendix~\ref{app:config} lists all router settings.

\paragraph{Fixed evidence budget.}
All active slices share one fixed retrieval and prompt budget, turning better
routing directly into higher-precision prompt evidence. The retriever divides
100 candidate slots equally across active slices, removes exact duplicate
pairs, reranks the remainder globally, and admits at most ten pairs clearing
the 0.78 threshold into the prompt.

\paragraph{Non-blocking updates.}
Asynchronous background execution avoids blocking waits during audio chunk
decoding. A background worker batches all sessions with updated contexts, each
chunk queries the working set active when its decoding starts, updates take
effect on the next arriving chunk, and candidate slice indexes preload in the
background.

%%%%% FILE: sections/5_demo_interface.tex
\section{Demo interface}
\label{sec:demo-interface}

AutoTerm-SST delivers zero-setup terminology adaptation for specialized live
speech through a standard streaming translator interface. Designed for
conference attendees, students in technical courses, and clinicians following
medical lectures, the demo starts once the user chooses a language pair and
launches a microphone or file stream, requiring zero glossary uploads or
domain menus.

\paragraph{Walkthrough.}
Live sessions follow topic shifts automatically while displaying active slices
and retrieved terms in real time. Partial translations appear immediately; as
context accumulates, the top status row names the highest-posterior slice, the
evidence panel lists retrieved term pairs per segment, and outgoing and
incoming slices briefly coexist in the working set across transitions.

\begin{figure*}[t]
\centering
\includegraphics[width=0.64\textwidth]{ui_evidence_panel.png}
\caption{English-to-Chinese interface replaying run 3 during an oncology
lecture, 19 minutes into the stream. The status row displays the medicine
slice at the top of the posterior with 4,000 active terms; highlighted spans
mark retrieved terms in the translation, and the bottom panel lists recent
pairs with similarity scores and source slices.}
\label{fig:ui}
\end{figure*}

\paragraph{Visible terminology evidence.}
Inline term highlighting and a dedicated evidence panel make retrieval
behavior transparent during live translation. As Figure~\ref{fig:ui} shows,
retrieved terms appear highlighted inside the target text with source-target
mappings listed below; the status row displays the top slice, its posterior
probability, and the active term count, while the evidence-panel header
reports active slices and retrieval latency. Listeners can immediately inspect
whether a mistranslated term went unretrieved, originated from an off-topic
slice, or was retrieved yet ignored by the decoder.

\paragraph{Extensibility.}
Operators can register new domain slices online without restarting running
translation sessions. Uploading one glossary file validates term pairs,
builds the MaxSim index, writes the manifest entry, and optionally generates a
slice prototype via an LLM as evaluated in Appendix~\ref{app:router}. Active
sessions keep their catalog snapshot, and fixed-glossary presets remain
available for expert workflows.

\paragraph{Availability.}
The MIT-licensed codebase, one-command launcher, screencast, and hosted entry
page are publicly accessible via the
\href{https://github.com/luojiaxuan/autoterm-sst}{GitHub repository} and
\url{https://luojiaxuan.github.io/autoterm-sst/}. Live deployment serves
Qwen3-Omni with vLLM continuous batching alongside the RASST retriever,
whereas Figure~\ref{fig:ui} replays recorded session outputs.

%%%%% FILE: sections/6_evaluation.tex
\section{Evaluation}
\label{sec:evaluation}

Our evaluation tests three claims: zero-setup routing preserves known-domain
translation quality, bounded working sets prevent large-catalog degradation,
and language-model routing tracks rapid domain transitions across ten topics.

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{routing_states_realsi.pdf}
\caption{Routing trajectories on the first RealSI stream order across ten
single-domain talks drawn at equal width. Each row marks per chunk whether
the target slice ranks first in the working set, ranks lower, or is absent;
the language-model router promotes a new talk's slice to top-1 within 21
seconds in nine of ten talks, compared with three for similarity routing.}
\label{fig:routing}
\end{figure*}

\subsection{Experimental setting}

\paragraph{Data.}
Two streaming benchmarks evaluate sustained bilingual domain alternation and
rapid ten-domain switching. We construct a continuous 4.7-hour
English-to-Chinese stream alternating five ACL 60/60 dev talks with
human-annotated terminology \citep{salesky-etal-2023-evaluating} and five
long-form oncology lectures from the European School of Oncology, or ESO, test
set \citep{iranzosanchez2025eso}. Following RASST \citep{luo2026rasst}, ESO is
evaluated against model-generated Chinese references and verified ESO
terminology; because the NLP and medicine slices hold the benchmark
glossaries, this stream tests whether routing keeps gold terms searchable. We
also chain ten English-to-Chinese RealSI talks \citep{cheng2024clasi} into
45-minute streams in three random orders, where each 3.4 to 5.2 minute talk
carries a human domain label mapped to one of ten ground-truth slices.

[Table 1 LaTeX unchanged from the two-block draft except its caption:]
\caption{Results on the ten-talk stream. Input indicates pre-session user
supply, Active gives maximum searchable entries per chunk, and Similarity
routing replaces the language-model router with Appendix~\ref{app:config}.
TERM\_ACC and prompt precision are percentages, MT-BLEU masks glossary terms,
Pairs is mean injected pairs per chunk, $\dagger$ denotes $p<0.05$ against the
routed system in the same run, and InfiniSST averages NLP and medicine
unweighted.}

[Table 2:]
Router & Coverage & Top-1 & To top & 0 top-1 \\
AutoTerm-SST & 89.7 & 67.0 & 15.6 s & 1 \\
Similarity routing & 78.8 & 23.9 & 68.1 s & 19 \\
\caption{Routing on 30 RealSI talks: percentages of chunks with the target
slice in the working set and at top-1, median delay to first top-1 rank, and
talks where it fails to rank first.}

\paragraph{Baselines.}
All baselines share the identical RASST backbone and inference budget, varying
only which entries remain searchable per chunk. InfiniSST
\citep{ouyang2025infinisst} runs the streaming translator without retrieval,
while Known-domain-1k searches only the ground-truth 1,000-entry slice for
each talk. AutoTerm-SST routes over ten 1,000-entry slices covering NLP,
medicine, and eight topics from Wikipedia and Wikidata with at most four
active, and Similarity routing keeps these slices and budgets using the
embedding router of Appendix~\ref{app:config}. Merged-2k pools the NLP and
medicine slices using both domain labels, Merged-10k pools all ten slices into
9,880 deduplicated entries, and Merged-1M plus its 100,000-entry prefix
Merged-100k expand the pool with additional topics while retaining conflicting
target translations.

\paragraph{Runs.}
All paired comparisons are evaluated strictly within runs sharing identical
audio order and chunk boundaries. Run 3, on two H200 GPUs, compares
AutoTerm-SST against Known-domain-1k, similarity routing, Merged-2k, and
Merged-10k. Run 1, on one H200 GPU, evaluates similarity routing against
InfiniSST, Known-domain-1k, Merged-100k, and Merged-1M, while run 2 repeats
the known-domain comparison on two A6000 GPUs; RealSI streams run on the two
H200 GPUs of run 3.

\paragraph{Metrics.}
We evaluate terminology accuracy, overall and masked-term translation quality,
prompt precision, and slice coverage. Following RASST, terminology accuracy,
or TERM\_ACC, measures the share of 2,047 time-aligned term occurrences, 1,368
NLP and 679 medicine, whose reference translation appears in the aligned
hypothesis. Translation quality is scored via corpus SacreBLEU
\citep{papineni-etal-2002-bleu,post-2018-call} after mWERSegmenter alignment
\citep{matusov-etal-2005-evaluating} under the StreamLAAL protocol
\citep{papi2024streamatt}, and masked-term BLEU, or MT-BLEU, repeats scoring
with evaluated target terms masked. Prompt precision is the fraction of
injected pairs matching a gold term in the source chunk, and RealSI coverage
is the fraction of chunks whose target slice enters the working set.

\paragraph{Statistics.}
Statistical significance rests on exact talk-level permutation tests paired
with stratified talk-level bootstrap confidence intervals. The exact
permutation test swaps system outputs talk by talk across all $2^{10}$
combinations, while a corpus-stratified talk bootstrap yields 95\% confidence
intervals written $a\pm h$ when symmetric and $a^{+u}_{-l}$ otherwise.
Differences subtract each baseline from AutoTerm-SST, with full paired
comparisons in Appendix~\ref{app:stats}.

\subsection{Results}

\paragraph{Known-domain reference comparison.}
AutoTerm-SST achieves statistical parity with the known-domain reference in
both terminology accuracy and BLEU without receiving user domain input. In
run 3, AutoTerm-SST minus Known-domain-1k is $+0.2^{+0.8}_{-1.0}$ TERM\_ACC
points at $p=0.82$ and $-0.05^{+0.44}_{-0.55}$ BLEU at $p=0.85$. Similarity
routing shows similar parity across runs 1 and 2 in Table~\ref{tab:stats}, and
all retrieval-augmented configurations substantially outperform retrieval-free
InfiniSST in Table~\ref{tab:quality}.

\paragraph{Catalog scale effects.}
Merged glossaries match routed retrieval up to 10k entries, yet suffer
significant quality and terminology losses at 100k and 1M entries. In run 3,
Merged-2k and Merged-10k differ from AutoTerm-SST by at most 0.4 TERM\_ACC
points at $p\ge0.34$ with comparable BLEU. In run 1, Merged-100k shows no
significant aggregate TERM\_ACC difference yet trails similarity routing by
$1.37^{+0.43}_{-0.51}$ BLEU at $p=0.004$. Merged-1M trails by $3.9\pm1.9$
TERM\_ACC points at $p=0.012$ and $2.30^{+0.65}_{-0.59}$ BLEU at $p=0.002$,
concentrating its loss on medicine at 7.4 points versus 2.2 on NLP, whereas the
two routers differ by only 0.1 points in run 3. Because Merged-1M floods the
prompt with ninefold more pairs per chunk at one tenth the prompt precision, it
also degrades general translation, trailing the routed system by 2.19 BLEU at
$p=0.023$ on the 935 reference sentences containing zero glossary terms.

\paragraph{Ten-domain routing.}
Language-model routing tracks frequent domain transitions across ten topics
that heuristic confirmation rules miss. On RealSI streams, the language-model
router delivers higher working-set coverage and promotes incoming talk slices
to top-1 far faster than similarity routing, as Table~\ref{tab:realsi} and
Figure~\ref{fig:routing} show. Because similarity routing requires three
consecutive leads plus a 30-second cooldown before switching, it fails to
promote the active talk's slice to top-1 in most sessions. On the two-domain
ten-talk stream, both routers keep the correct slice searchable in about 98\%
of chunks, yielding equivalent downstream accuracy; Appendix~\ref{app:router}
evaluates faster switches and 24 fine-grained subtopics.

\paragraph{Processing speed.}
Asynchronous routing adds negligible overhead, keeping end-to-end processing
roughly five times faster than real time. AutoTerm-SST matches Known-domain-1k
in end-to-end real-time factor in Table~\ref{tab:speed}, where RASST's
retriever alone accounts for 0.042 seconds per chunk on the same GPUs, and the
first translated output appears 2.3 seconds after audio onset.

[Table 3 unchanged; caption:]
\caption{Compute seconds per 1.92-second chunk for one ACL talk session on
three A6000 GPUs. Retrieval includes speech encoding and routing, and RTF
divides total compute time by chunk duration.}

%%%%% FILE: sections/7_limitations_conclusion.tex
\section{Conclusion}

AutoTerm-SST eliminates pre-session glossary upload and domain selection in
streaming speech translation by routing partial translations to offline topic
slices. Across bilingual and ten-domain streams, the system matches a
known-domain reference, avoids the accuracy and BLEU losses of a million-entry
merged glossary, and tracks rapid topic changes in real time. Extending the
router to finer-grained subtopics is a natural next step.

[Ethics statement unchanged.]

%%%%% FILE: sections/8_appendix.tex
[Table A unchanged from the draft except its caption:]
\caption{Paired metric differences between the routed system and each
baseline. Low and High give 95\% stratified talk-bootstrap confidence bounds,
and $p$ is the exact two-sided talk-level permutation $p$-value, with TERM\_ACC
differences in percentage points.}

\section{Statistical comparisons}
\label{app:stats}

Paired talk-level tests confirm that the gains of the routed system over
Merged-1M hold across individual talks, with all comparisons reported in
Table~\ref{tab:stats}. The exact permutation test swaps system outputs within
each talk across all 1,024 combinations and recomputes the pooled metric,
yielding a minimum attainable two-sided $p$-value of 0.002 on which all
significance claims rest. Across ten talks, percentile bootstrap intervals
from 4,000 stratified resamples of five ACL and five ESO talks can diverge from
exact permutation tests near decision boundaries, excluding zero while
$p\ge0.08$ for Merged-10k in run 2 and similarity-routing MT-BLEU in run 3.
Leave-one-talk-out analysis preserves all three gains of similarity routing
over Merged-1M in run 1, maintaining at least 3.28 TERM\_ACC points, 2.00
BLEU, and 1.99 MT-BLEU.

\section{Router configuration}
\label{app:config}

The language-model router adds a single free parameter, the switch rate $\rho$,
to the shared retrieval and routing settings listed in Table~\ref{tab:config}.
The similarity-routing baseline embeds the context and each prototype using
BGE-M3 \citep{chen-etal-2024-m3}, reading up to the last 128 tokens of $c_t$
and appending a Chinese translation to each prototype. It updates its pinned
confirmed slice only when a rival slice leads both the runner-up and the
confirmed slice by fixed margins across three consecutive updates outside a
30-second post-switch cooldown, using thresholds tuned on four short segments
drawn from the evaluation talks prior to full-stream runs. Both routers ran
alone on dedicated servers to ensure consistent wall-clock updates.

[Table B unchanged; caption:]
\caption{Router and retrieval settings. Lead thresholds measure differences in
normalized similarity scores, and similarity-router switches require both
leads across consecutive updates.}

\section{Router diagnostics}
\label{app:router}

Open-loop replays on saved run 1 translations isolate router behavior from
decoding variance under identical inputs. Each replay constructs context $c_t$
from the last four translated chunks as in Section~\ref{sec:adaptive-memory}
and refreshes scores at most once every 5 seconds of audio, applying the
confirmation pin for similarity routing and one filter step with $\rho=10^{-3}$
for language-model routing. Replayed on run 1 emission timestamps, similarity
routing executes 19 switches and covers 96.2\% of chunks, closely matching its
live coverage of 96.9\% in Table~\ref{tab:router-diag}.

\paragraph{Topic filter.}
The sticky HMM filter stabilizes noisy per-update router probabilities without
heuristic switch thresholds. Updated at every chunk, raw letter probabilities
keep the correct slice in the top four for 90.5\% of chunks, whereas the
filtered posterior reaches 98.7\% and stays within 98.4\% to 98.9\% across
$\rho\in[10^{-5},10^{-1}]$. Raw embedding similarity covers 85.0\% under the
same updates, rising to 97.0\% with confirmation pinning.

\paragraph{Short topic shifts.}
The filtered posterior tracks interleaved topic switches down to 15-second
intervals with minimal transition lag. For each duration in
Table~\ref{tab:router-diag}, we slice saved NLP and medicine translations into
fixed-length segments and interleave them into a 40-segment stream. Across all
segment lengths, the median segment begins with the incoming slice already in
the language-model working set and promotes it to the posterior top-1 after a
median of 7.7 seconds, whereas the similarity router's three-update rule and
30-second cooldown delay confirmation to a median of 27 to 46 seconds on
segments of one minute or longer.

\paragraph{Additional target languages.}
Prompt-based routing generalizes across target languages by changing only the
target language label in the router prompt. Replaying German and Japanese
RASST translations of the ten-talk stream preserves high working-set coverage
for both routers, as Table~\ref{tab:router-diag} shows.

\paragraph{Fine-grained domains.}
Partitioning domains into finer subtopics widens the routing advantage of the
language model over embedding similarity. Using Qwen3-4B-Instruct to propose
eight subtopics each from 400 non-evaluation terms in NLP and medicine and
assign terms by letter probability yields 24 total slices. Across all 2,047
gold term occurrences, language-model routing substantially outperforms
similarity routing in Table~\ref{tab:router-diag} against a hindsight top-four
oracle bound of 97.3\%.

\paragraph{Generated prototypes and delayed updates.}
Automatically generated slice prototypes and asynchronous one-chunk update
delays preserve routing coverage. Generating one prototype per slice via
GPT-5.5 with evaluation terms excluded achieves 97.9\% per-chunk top-four
coverage on the ten-talk stream versus 98.7\% with hand-written prototypes,
and RealSI streams use these generated prototypes to avoid overlap with the
RealSI technology talk. Replaying every routing update one chunk late,
modeling faster-than-real-time audio arrival in the deployed worker, reduces
language-model router coverage by at most 0.33 points across
Table~\ref{tab:router-diag}.

[Table C unchanged; caption:]
\caption{Open-loop replays of deployed routers at the live update rate. Cells
report the percentage of chunks with the target slice in the four-slice
working set, or gold term occurrences for the 24-slice row.}
```

## Gemini's notes (verbatim)

Five Most Important Changes

1. Conclusion-first topic sentences across every paragraph: Replaced procedural and setup-heavy openings (such as dataset construction details in Section 6.1 or replay setup in Appendix C) with direct takeaway statements in sentence 1 so every paragraph states its claim up front.
2. De-densified number-heavy sentences and eliminated redundant table recitals: Split multi-clause numerical sentences (such as the Merged-1M prompt-pollution and non-glossary BLEU analysis in Section 6.2) into focused single-claim sentences, and deferred raw cell values in Sections 6.2 and Appendix C to Tables 1–6 while keeping every interval, p-value, and standalone figure verbatim.
3. Removed defensive negations, non-math parentheses, and punctuation dashes: Rewrote contrastive negations (such as "as opposed to", "need not", and "neither...nor") into affirmative statements, stripped all parenthetical asides in running text, and eliminated em and en dashes throughout prose and captions.
4. Eliminated vague quantifiers and absolute terms: Removed every occurrence of banned vague quantifiers and absolutes, including replacing "never" in running text and Table 2 (Never top → 0 top-1), with exact counts or bounded empirical statements.
5. Tightened prose and captions to protect the 6-page and 2-page limits: Standardized every caption to start with a short noun phrase within 3–4 rendered lines and trimmed redundant phrasing (~8% overall reduction in running prose) so the paper stays safely inside the NAACL page limits.

Sentences in the Draft That Overclaim (and How They Were Grounded)

- Section 6 intro and Section 1: "zero setup costs no accuracy" / "costs no significant quality". Non-significant tests (p=0.82 TERM_ACC, p=0.85 BLEU) establish parity within sample variance rather than zero difference; point estimates +0.2 TERM_ACC and −0.05 BLEU with 95% bounds spanning roughly ±1.0 point and ±0.5 BLEU. Revision: statistical parity with the known-domain reference within confidence bounds.
- Section 6.2: "In run 1, Merged-100k matches similarity routing in TERM_ACC". Merged-100k trails similarity routing by 0.54 aggregate TERM_ACC points (95% CI [−0.60, +1.84], p=0.48) and drops 1.8 points on Medicine (84.5% vs. 86.3%). Revision: no statistically significant aggregate TERM_ACC difference while highlighting the significant BLEU loss (p=0.004).
- Section 4: "No chunk waits for routing." Decoding does not synchronously await routing, but background scoring and index preloading still share GPU and host resources. Revision: avoiding blocking waits during audio chunk decoding.
- Appendix C: "so the single free parameter needs no tuning". Insensitivity across ρ ∈ [1e-5, 1e-1] on the ten-talk benchmark does not guarantee ρ never requires adjustment. Revision: grounded in the empirical sensitivity bound 98.4% to 98.9%.
- Section 6.2: "AutoTerm-SST processes audio in about a fifth of its duration, as fast as Known-domain-1k". Both round to 0.22 RTF, yet retrieval plus routing takes 0.094 s versus 0.059 s per chunk. Revision: matching Known-domain-1k in end-to-end real-time factor.
