# Readability review, Opus 5.5 Extra High, 2026-11-20

A fresh model with no context of this paper read the 9-page PDF built at `a9175c6` and its comment-free LaTeX, using `~/.claude/skills/readability-review/prompt.md`. Report verbatim.

# AutoTerm-SST: area-chair readability report

I read all 9 pages in order. The main text runs from the Abstract through Section 7 and the Ethics statement (pages 1–6). The references fill pages 6–8, and Appendices A–C run from page 8 to page 9. Every quote below is copied from the LaTeX source. Each rewrite keeps the original's numbers, claims and references. Where a rewrite needs a fact only the authors know, I mark the gap with [brackets].

---

## 1. Story

**One-sentence claim.** A small LLM reads AutoTerm-SST's own partial translations and, through a sticky topic filter, picks which offline topic slices the term retriever may search. So a terminology-aware streaming translator needs no glossary setup, matches a reference that is told each talk's domain, and avoids the losses of a million-entry merged glossary.

**Does the main text tell it as one line?** Mostly. The introduction states the problem (setting up a glossary before each session is friction, and merged glossaries get worse as they grow), then the insight (the translation itself shows the domain), then the system. Sections 3–4 give the mechanism and Section 5 the interface. Section 6 opens with three claims that line up one-to-one with its result paragraphs, and Section 7 restates them.

**Where the line breaks, jumps or repeats:**

- **Result used as motivation.** Introduction paragraph 3 gives the system's final RealSI result (15.6 s, the AutoTerm-SST row of Table 2, which already includes the topic filter) as a motivating observation. It comes before the method exists, and RealSI has not been introduced yet.
- **Sections 3, 4 and 5 are in an odd order.** Section 3 ends by sending data "to the interface", but Section 4 (how slices are chosen) comes next and the interface only in Section 5. Section 5 then opens on the audience, with no link from Section 4.
- **The router comparison has to be pieced together.** Contribution 2 rests on a baseline ("a similarity router with tuned confirmation rules") that the main text describes only by pointing to Appendix B. The Results also say the two routers tie on the quality stream (about 98% coverage each). So the router's advantage depends on Table 2, Figure 3 and Appendix C, and the reader has to assemble this from three places.
- **Repetition.**
  - The mechanism (slices, prototype, small LLM, filter, working set) is described four times: abstract, introduction paragraph 4, the end of Section 3, and Section 4.
  - The headline numbers 89.7/78.8 and 3.3/2.31 appear in the abstract, the contributions and the Results.
  - "Adding a domain" appears in both Section 4 (Offline catalog) and Section 5 (Extensibility).
  - The status row and evidence panel are described in both Walkthrough and Visible terminology evidence.
- **Related work leaves out what is new.** It does not cover domain or topic identification, or multi-domain routing, which is exactly what the router does. A reader cannot tell what is new about the router compared with domain classification.
- **Expected sections or metrics are missing.** There is no Limitations section; ACL-family venues usually require one, so check the demo-track call for papers. The Metrics paragraph names the StreamLAAL protocol but reports no latency metric, so a streaming-translation reader will look for one.

**Transitions.**

| Transition | Verdict |
|---|---|
| Section 2 → 3 | Causal: "keeps an existing streaming model and adds … around it" leads into the overview. |
| Section 3 → 4 | Weak: Section 3 ends on the interface. |
| Section 4 → 5 | Abrupt. |
| Section 5 → 6 | Abrupt, but Section 6 recovers with an explicit list of claims. |
| Within Section 6 | Clean. "Processing speed" is a fourth result the opening does not announce. |

---

## 2. Main text, paragraph by paragraph

### Abstract, "Retrieval-augmented speech translation improves…"

- **Score:** 4.
- **Connection:** none before it; leads into the introduction.
- **Role:** the whole story in miniature.
- **What made me stop:**
  - The second headline number is compared against "a similarity router with tuned confirmation rules", which is not explained anywhere before Appendix B.
  - "keeps the correct glossary searchable": the abstract has just introduced *topic slices*, so "glossary" makes me ask whether this is something different.
  - The results sentence carries four numbers and three comparisons.
- **Quote:** "On a 4.7-hour English-to-Chinese stream alternating ACL talks with oncology lectures, AutoTerm-SST reaches 87.7\% terminology accuracy, on par with a reference told each talk's domain at 87.5\%, and it beats a one-million-entry merged glossary by 3.3 accuracy points and 2.31 BLEU. On ten-domain RealSI streams, the router keeps the correct glossary searchable in 89.7\% of audio chunks, against 78.8\% for a similarity router with tuned confirmation rules."
- **Rewrite:**
  > On a 4.7-hour English-to-Chinese stream alternating ACL talks with oncology lectures, AutoTerm-SST reaches 87.7% terminology accuracy, on par with the 87.5% of a reference that is told each talk's domain. It also beats a one-million-entry merged glossary by 3.3 accuracy points and 2.31 BLEU. On ten-domain RealSI streams, the router keeps the correct topic slice searchable in 89.7% of audio chunks, against 78.8% for an embedding-similarity router with tuned confirmation rules.

### 1 Introduction, P1 "Simultaneous speech translation, or SST…"

- **Score:** 4.
- **Connection:** good; it ends on glossary retrieval, which paragraph 2 picks up.
- **Role:** the task and the terminology gap.
- **What made me stop:** "Speech large language models, or LLMs" ties the abbreviation LLM to *speech* models. Later "a small LLM" is the text-only router, so the abbreviation changes meaning.
- **Quote:** "Speech large language models, or LLMs, now translate long-form audio directly (Qwen Team, 2025; Ouyang et al., 2025), yet rare method and dataset names, clinical concepts, and named entities remain difficult to recover from acoustics alone."
- **Rewrite:**
  > Large language models, or LLMs, that take speech as input now translate long-form audio directly (Qwen Team, 2025; Ouyang et al., 2025), yet rare method and dataset names, clinical concepts, and named entities remain difficult to recover from acoustics alone.

### 1 Introduction, P2 "Existing retrieval-augmented SST systems…"

- **Score:** 4.
- **Connection:** good.
- **Role:** the two ways prior systems fail (manual setup; loss of accuracy as glossaries grow), backed by a forward reference to Table 1.
- **What made me stop:** in the merged-glossary sentence, "the prompt a fixed number of term pairs" leaves out its verb, and the sentence has three clauses.
- **Quote:** "Merging all domains into one universal glossary avoids manual selection and works at small scale, yet retrieval keeps a fixed number of candidates and the prompt a fixed number of term pairs, so irrelevant and conflicting mappings crowd out relevant terms as the catalog grows."
- **Rewrite:**
  > Merging all domains into one universal glossary avoids manual selection and works at small scale. Retrieval, however, keeps a fixed number of candidates, and the prompt holds a fixed number of term pairs, so as the catalog grows, irrelevant and conflicting mappings crowd out relevant terms.

### 1 Introduction, P3 "The translation stream itself supplies…"

- **Score:** 3.
- **Connection:** this is the causal turning point of the introduction, and it works as a bridge.
- **Role:** the key insight.
- **What made me stop:**
  - RealSI appears here for the first time, with no citation or description.
  - The 15.6-second figure reads like a pilot study of a bare LLM. In fact it is the full system's result (Table 2, which includes the topic filter), quoted before the method is introduced.
  - "glossary" stands for what Table 2 calls a slice.
- **Quote:** "On RealSI streams that change domain every 3.4 to 5.2 minutes, a 4-billion-parameter LLM reading the last four translated chunks ranks a new talk's glossary first after a median of 15.6 seconds."
- **Rewrite:**
  > On streams of RealSI talks (Cheng et al., 2024) that change domain every 3.4 to 5.2 minutes, our router, a 4-billion-parameter LLM reading the last four translated chunks, ranks a new talk's glossary first after a median of 15.6 seconds (Table 2).

### 1 Introduction, P4 "We present AutoTerm-SST, a system demo…"

- **Score:** 4.
- **Connection:** good; it answers paragraph 2's problem with paragraph 3's insight.
- **Role:** the system in one paragraph.
- **What made me stop:** two appositions stacked inside each other ("…slices, each holding … a prototype, a short description…").
- **Quote:** "Operators register offline topic slices, each holding roughly 1,000 term pairs and a prototype, a short description of the topic."
- **Rewrite:**
  > Operators register offline topic slices; each slice holds roughly 1,000 term pairs and a prototype, which is a short description of the topic.

### 1 Introduction, contributions list

- **Score:** 4.
- **Connection:** good; the three bullets match the three claims in Section 6, although in a different order.
- **Role:** the claims.
- **What made me stop:**
  - Bullet 2 uses the undefined "similarity router with tuned confirmation rules".
  - Bullet 3 defines NLP in the middle of the bullet (acceptable).
- **Quote:** "On ten-domain RealSI streams it keeps the correct glossary searchable in 89.7\% of chunks, against 78.8\% for a similarity router with tuned confirmation rules."
- **Rewrite:**
  > On ten-domain RealSI streams it keeps the correct slice searchable in 89.7% of chunks, against 78.8% for an embedding-similarity router with tuned confirmation rules (Appendix B).

### 2 Related work, "Streaming speech translation."

- **Score:** 4.
- **Connection:** good; the last sentence places AutoTerm-SST.
- **Role:** background on streaming translation policies.
- **What made me stop:** "broaden … architectures" is vague.
- **Quote:** "Speech-language models broaden direct SST architectures (Seamless Communication, 2023; Zhang et al., 2024; Qwen Team, 2025)."
- **Rewrite:**
  > Direct SST architectures now also build on speech-language models (Seamless Communication, 2023; Zhang et al., 2024; Qwen Team, 2025).

### 2 Related work, "Terminology-aware translation."

- **Score:** 3.
- **Connection:** adequate.
- **Role:** places the work relative to glossary methods, RAG, and interfaces.
- **What made me stop:**
  - The topic sentence states a limitation, then the paragraph steps back to motivation (rare terms), then forward again.
  - The last sentence joins three unrelated ideas: online selection, visible decisions, and a serving engine.
  - There is no prior work on domain or topic identification.
- **Quote:** "AutoTerm-SST removes the pre-session glossary by selecting topic slices online, and like interactive translation tools (Dandekar et al., 2024) it makes these decisions visible, on top of a continuous-batching serving engine (Kwon et al., 2023; Zheng et al., 2024)."
- **Rewrite:**
  > Rare terms and named entities remain a weakness of speech translation (Gaido et al., 2021). Terminology-aware methods take the relevant glossary as given before decoding: prior work applies user-provided dictionaries through constrained decoding or training-time exposure (Hokamp and Liu, 2017; Dinu et al., 2019). Retrieval-augmented generation offers a non-parametric alternative (Lewis et al., 2020; Khandelwal et al., 2021), which RASST applies to streaming LLM decoding (Luo et al., 2026). AutoTerm-SST removes the pre-session glossary by selecting topic slices online. Like interactive translation tools (Dandekar et al., 2024), it makes these decisions visible, and it runs on a continuous-batching serving engine (Kwon et al., 2023; Zheng et al., 2024).

### 3 System overview, P1 "AutoTerm-SST adds an online terminology-selection loop…"

- **Score:** 3.
- **Connection:** good from Section 2.
- **Role:** the RASST pipeline and the single point where AutoTerm-SST intervenes.
- **What made me stop:**
  - Seven sentences and six symbols ($h_t, G_t, z_i, q_v, \mathcal V_t, r_i$); four of the symbols are never used again.
  - "Because there is no transcript" does not say whose transcript.
  - The paragraph's point ("changes only which entries this lookup sees") arrives at the end.
- **Quote:** "Because there is no transcript, the retriever matches speech directly against glossary source terms by maximum similarity, or MaxSim."
- **Rewrite:**
  > AutoTerm-SST adds an online terminology-selection loop around RASST without changing its models, as Figure 1 shows. RASST translates audio in 1.92-second chunks in three steps: a speech encoder turns chunk $t$ into features $h_t$, a cross-modal term retriever builds a term map $G_t$, and the Qwen3-Omni LLM translates $h_t$ with $G_t$ in its prompt. Because the model produces no source transcript, the retriever matches speech directly against glossary source terms by maximum similarity, or MaxSim: each source term is pre-encoded as a vector $z_i$, the speech of chunk $t$ is encoded into vectors $q_v$ over overlapping windows $v\in\mathcal V_t$, and the term score is $r_i=\max_{v\in\mathcal V_t}q_v^\top z_i$. Terms scoring above 0.78 become candidates, and at most ten source–target pairs enter $G_t$. AutoTerm-SST changes only which entries this lookup sees: its router restricts the lookup to a working set of up to four topic slices, and each new translation joins the router's context for later chunks.

### 3 System overview, P2 "Placing adaptation outside the translation model…"

- **Score:** 4.
- **Connection:** weak forward link; it ends on "the interface", but Section 4 comes next.
- **Role:** pluggability and the data passed to the interface.
- **What made me stop:** two unrelated claims share one sentence.
- **Rewrite:**
  > Placing adaptation outside the translation model keeps the system pluggable: one streaming protocol serves a retrieval-free translator, a fixed-glossary RASST translator, or AutoTerm-SST. Every partial translation carries the retrieved pairs, their source slices, and the working set to the interface (Section 5); Section 4 first describes how the working set is chosen.

### 4 Adaptive terminology memory, opening "Separating an offline catalog…"

- **Score:** 4.
- **Connection:** good.
- **Role:** defines the catalog and the slice.
- **What made me stop:**
  - The section title "adaptive terminology memory" is never defined in terms of catalog, router and working set.
  - "bilingual prototype" is unclear. Later the prototype is "a one-line description followed by an English sentence", and Appendix B "extends each prototype with its Chinese translation", so I cannot tell which part is in which language.
- **Quote:** "…and a bilingual prototype: a short topic description with example terms."
- **Rewrite:**
  > The adaptive terminology memory separates an offline catalog from a small working set, which lets AutoTerm-SST cover broad domains while keeping per-chunk search bounded. Each topic slice packages about 1,000 term–translation pairs, a prebuilt MaxSim index over the source terms, and a bilingual prototype: a short topic description with example terms [state which part is in which language].

### 4, "Offline catalog."

- **Score:** 4.
- **Connection:** fine.
- **Role:** how slices are registered.
- **What made me stop:** "Adding a domain" uses a third word for a slice; the paragraph also repeats Section 5's Extensibility paragraph.
- **Rewrite:**
  > Adding a topic slice touches only the catalog manifest: an operator registers the slice with one entry holding its label, term count, prototype, and index path. The router reads only the manifest and prototypes, and the retriever skips every index outside the working set.

### 4, "Language-model routing."

- **Score:** 4.
- **Connection:** good.
- **Role:** the scoring function, Eq. (1).
- **What made me stop:**
  - The definition of $P_d$ interrupts the scoring sentences.
  - "the listed order $\sigma$" does not say whether the order is fixed, random, or redrawn at each update.
  - "At chunk $t$" conflicts with routing "every five seconds" two paragraphs later.
- **Quote:** "Averaging over the listed order $\sigma$ and its reverse $\bar\sigma$ removes the preference for early or late positions"
- **Rewrite:**
  > At chunk $t$, the router concatenates the last $w=4$ translated chunks into a context $c_t$ and builds a multiple-choice prompt that lists the prototype $P_d$ of every slice $d$ under a letter; $P_d$ is a one-line description followed by an English sentence naming the slice's core topics and representative terms. The score of slice $d$ is the probability that the router LLM $\pi$, the 4-billion-parameter Qwen3-4B-Instruct (Yang et al., 2025), assigns to the letter of $d$ as the first answer token. Averaging over the listed order $\sigma$ [fixed/random] and its reverse $\bar\sigma$ removes the preference for early or late positions:

### 4, "Topic filtering."

- **Score:** 3.
- **Connection:** good.
- **Role:** the sticky hidden Markov model filter, Eq. (2), and its only free parameter.
- **What made me stop:**
  - $N$ is used without definition.
  - $t$ meant a chunk in the previous paragraph, but here $b_{t-1}$ is the *previous update*.
  - To follow the $\beta_t$ sentence, the reader needs to know that updates come more often than the $w$-chunk context, a fact that appears only in the next paragraph.
  - "sticky" is jargon.
- **Quote:** "Consecutive contexts can share chunks, so the exponent $\beta_t=\min(1,n_t/w)$, with $n_t$ the number of chunks translated since the previous update, counts each chunk's evidence once."
- **Rewrite:**
  > A sticky hidden Markov model, one whose topic rarely changes between updates, accumulates these noisy scores into a stable posterior $b_t$ over the $N$ catalog slices. … Updates run every five seconds while each context spans $w=4$ chunks, so consecutive contexts can share chunks; the exponent $\beta_t=\min(1,n_t/w)$, with $n_t$ the number of chunks translated since the previous update, counts each chunk's evidence once.
- **Also:** index updates with a separate letter, for example $u$, instead of $t$.

### 4, "Working-set update."

- **Score:** 4.
- **Connection:** good.
- **Role:** the $K$ and $B$ limits.
- **What made me stop:** "once it drops in rank" does not say below what.
- **Quote:** "Lower-ranked slices are excluded from retrieval, and a stale slice leaves once it drops in rank."
- **Rewrite:**
  > Slices that do not fit are excluded from retrieval, so a stale slice leaves as soon as it ranks below the slices that fit.

### 4, "Fixed evidence budget."

- **Score:** 4.
- **Connection:** good.
- **Role:** shows the comparison is fair.
- **What made me stop:** "Gains … inside the same budget" does not say gains over what, or the same budget as whose.
- **Quote:** "Gains of AutoTerm-SST therefore come from placing more relevant mappings inside the same budget."
- **Rewrite:**
  > Every configuration in Section 6 shares this retrieval and prompt budget, so gains of AutoTerm-SST come from placing more relevant mappings inside the same budget.

### 4, "Non-blocking updates."

- **Score:** 3.
- **Connection:** good, but Appendix C later says the Section 6 runs did *not* use this timing.
- **Role:** keeping routing off the decoding path.
- **What made me stop:** "the working set current when its decoding begins" is a garden-path phrase, and "It scores, in one batch, every session" breaks up its clause.
- **Quote:** "It scores, in one batch, every session whose context has changed; each chunk retrieves from the working set current when its decoding begins, and an update takes effect at the first chunk that starts after it finishes."
- **Rewrite:**
  > In one batch, it scores every session whose context has changed. Each chunk retrieves from the working set that is current when its decoding begins, so an update takes effect at the first chunk that starts after the update finishes.

### 5 Demo interface, opening "AutoTerm-SST serves listeners…"

- **Score:** 5.
- **Role:** the audience and the starting state.
- **Note:** the move from Section 4 is abrupt, but the paragraph reads cleanly.

### 5, "Walkthrough."

- **Score:** 4.
- **Connection:** overlaps the next paragraph.
- **Role:** the user flow.
- **What made me stop:**
  - "names the slice on top of the router's posterior" sits next to "at the top of the interface", so "top" reads spatially.
  - The evidence panel is named but not located.
- **Rewrite:**
  > Once context accumulates, the status row at the top of the interface names the slice with the highest posterior probability, and the evidence panel below the translation lists the pairs retrieved for each output.

### 5, "Visible terminology evidence."

- **Score:** 4.
- **Connection:** good.
- **Role:** the diagnostic value of the interface; this is the paper's best user-facing argument.
- **What made me stop:**
  - One sentence lists three regions of the interface and five fields.
  - The words in the text do not match the labels on the screenshot ("Confidence", "Active Glossary", "Retrieved terms").
- **Rewrite:**
  > As Figure 2 shows, the interface highlights retrieved terms inside the translation, and the evidence panel ("Retrieved terms") lists their source–target mappings. The status row shows the top slice ("Active Glossary"), its posterior probability ("Confidence"), and the active term count; the evidence-panel header names the active slices and the retrieval latency.

### 5, "Extensibility."

- **Score:** 4.
- **Connection:** repeats the Offline catalog paragraph.
- **Role:** adding topics.
- **What made me stop:** the first sentence chains three registration steps, an LLM option and a pointer; the second joins two unrelated facts.
- **Rewrite:**
  > Adding a topic takes one glossary file and no restart. Registration validates the pairs, builds the index, and writes the manifest entry; an LLM can write the prototype from the glossary, as Appendix C evaluates. Running sessions keep their catalog snapshot, and expert users can still choose fixed-glossary presets.

### 5, "Availability."

- **Score:** 5.
- **Note:** in print, the link text "GitHub repository" hides the URL. Print the URL, as you already do for the entry page.

### 6 Evaluation, opening "We test three claims…"

- **Score:** 5.
- **Connection:** excellent; the three claims map onto the Results.
- **Note:** Processing speed is not announced. Optionally add "and we report processing speed."

### 6.1, "Data."

- **Score:** 2.
- **Connection:** fine.
- **Role:** defines both evaluation sets; every later result depends on this paragraph.
- **What made me stop:**
  - "Two English-to-Chinese benchmarks" is followed by three corpora.
  - The phrase "alternating five English ACL 60/60 dev talks with human-annotated terminology (…), the NLP part, and five long-form oncology lectures (…), the medicine part" reads as a three-item list.
  - The stream is never named, yet every later caption calls it "the ten-talk stream", a name that also fits the RealSI streams ("the ten English-to-Chinese talks of RealSI").
  - "our ten slices" appears before Baselines introduces them.
  - The "three random orders" are what Table 2 later counts as "30 RealSI talks", but this paragraph does not say so.
- **Rewrite:**
  > Two English-to-Chinese benchmarks test translation quality and ten-domain routing. For quality, we build one continuous 4.7-hour stream, the ACL–ESO stream, that alternates two parts: five English ACL 60/60 dev talks with human-annotated terminology (Salesky et al., 2023) form the NLP part, and five long-form oncology lectures from the test set of the European School of Oncology, or ESO (Iranzo-Sánchez et al., 2025), form the medicine part. Following RASST (Luo et al., 2026), … searchable. To test routing across ten domains, we also chain the ten English-to-Chinese talks of RealSI (Cheng et al., 2024) into 45-minute streams in three random orders, 30 talks in all. Each talk lasts 3.4 to 5.2 minutes and carries a human domain label that maps to one of the ten catalog slices described under Baselines, which gives the correct slice of every chunk.
- **Also:** use "ACL–ESO stream" in every caption in place of "ten-talk stream".

### 6.1, "Baselines."

- **Score:** 3.
- **Connection:** fine.
- **Role:** the seven configurations.
- **What made me stop:**
  - Seven configurations are listed with no grouping.
  - AutoTerm-SST itself is described under "Baselines".
  - Similarity routing, the comparator for contribution 2, is defined only by a pointer to Appendix B.
  - In "Merged-100k is its first 100,000 entries", "its" refers back to Merged-1M, and the order that defines "first" is not given.
- **Quotes:** "Similarity routing keeps these slices and budgets but uses the embedding-similarity router of Appendix B." and "Merged-1M adds entries from other topics to the same NLP and medicine glossaries, and Merged-100k is its first 100,000 entries."
- **Rewrite:**
  > The configurations differ in what the user must supply: Known-domain-1k and Merged-2k need domain labels, while the other merged glossaries, Similarity routing, and AutoTerm-SST need none. … Similarity routing keeps these slices and budgets but uses the embedding-similarity router of Appendix B, which switches to a new slice only after it leads by fixed margins in three consecutive updates outside a 30-second cooldown. … Merged-1M adds entries from other topics to the same NLP and medicine glossaries, and Merged-100k keeps the first 100,000 entries of Merged-1M [in which order].

### 6.1, "Metrics."

- **Score:** 3.
- **Connection:** fine.
- **Role:** defines TERM_ACC, BLEU, MT-BLEU, prompt precision and coverage.
- **What made me stop:**
  - The first sentence nests two appositions and three numbers.
  - "reference boundaries of the StreamLAAL protocol": StreamLAAL is a latency protocol and no latency is reported, so I went looking for a latency result.
  - "MT-BLEU" reads as "machine-translation BLEU".
- **Quote:** "Following RASST, terminology accuracy, or TERM\_ACC, is the share of 2,047 time-aligned term occurrences, 1,368 NLP and 679 medicine, whose reference translation appears in the aligned hypothesis."
- **Rewrite:**
  > Following RASST, terminology accuracy, or TERM_ACC, is the share of time-aligned term occurrences whose reference translation appears in the aligned hypothesis; the stream has 2,047 such occurrences, 1,368 in NLP talks and 679 in medicine talks. For translation quality, mWERSegmenter (Matusov et al., 2005) first resegments each talk to the reference segmentation used by the StreamLAAL protocol (Papi et al., 2024), and we then compute corpus SacreBLEU (Papineni et al., 2002; Post, 2018).
- **Also:** consider renaming MT-BLEU to "masked BLEU".

### 6.1, "Statistics."

- **Score:** 4.
- **Connection:** fine.
- **Role:** the significance protocol.
- **What made me stop:** the sign convention comes last, and "all $2^{10}$ combinations" does not say combinations of what.
- **Rewrite:**
  > Differences are AutoTerm-SST minus the baseline, and Appendix A reports every paired comparison. An exact permutation test swaps the outputs of two configurations within each talk over all $2^{10}$ swap patterns of the ten talks, and a talk-level bootstrap stratified by corpus gives 95% confidence intervals, written $a\pm h$ when symmetric and $a^{+u}_{-l}$ otherwise.

### 6.2, "Comparison with the known-domain reference."

- **Score:** 4.
- **Connection:** good; this is claim 1.
- **What made me stop:** the intervals, units and $p$-values sit in one string.
- **Rewrite:**
  > AutoTerm-SST performs on par with a known-domain reference without receiving any user input: relative to Known-domain-1k, it differs by $+0.2^{+0.8}_{-1.0}$ TERM_ACC points ($p=0.82$) and $-0.05^{+0.44}_{-0.55}$ BLEU ($p=0.85$). Every glossary configuration in Table 1 lies well above retrieval-free InfiniSST.

### 6.2, "Effect of catalog scale."

- **Score:** 3.
- **Connection:** good; this is claim 2.
- **What made me stop:**
  - Ten numbers in seven sentences.
  - "irrelevant pairs, nine times as many per chunk" attaches "nine times" to *irrelevant* pairs, but Table 1's 8.70 vs 0.94 is the ratio of *all* injected pairs. Irrelevant pairs alone are about 14 times as many.
  - In the last sentence, "it" refers to Merged-1M, mentioned three sentences earlier.
- **Quote:** "Merged-1M also fills the fixed prompt with irrelevant pairs, nine times as many per chunk as AutoTerm-SST at one tenth of its prompt precision."
- **Rewrite:**
  > Merged glossaries keep pace with a routed working set up to 10k entries, yet lose translation quality and terminology accuracy as the catalog grows further. Up to 10k entries, Merged-2k and Merged-10k lead AutoTerm-SST by at most 0.4 TERM_ACC points ($p\ge0.34$), with matching BLEU. At 100k entries, Merged-100k matches AutoTerm-SST in TERM_ACC but trails it by $1.38^{+0.53}_{-0.70}$ BLEU ($p=0.010$). At 1M entries, Merged-1M trails by $3.3^{+1.5}_{-1.9}$ TERM_ACC points ($p=0.031$) and by $2.31^{+0.63}_{-0.82}$ BLEU ($p=0.002$), and its TERM_ACC loss concentrates on medicine, a 7.4-point gap. Merged-1M also fills the fixed prompt with irrelevant pairs: it injects nine times as many pairs per chunk as AutoTerm-SST at one tenth of its prompt precision. The damage reaches sentences without terms: on the 935 sentences whose references contain no glossary term, AutoTerm-SST leads Merged-1M by 1.85 BLEU ($p=0.047$).

### 6.2, "Routing across ten domains."

- **Score:** 4.
- **Connection:** good; this is claim 3.
- **What made me stop:**
  - "the confirmation rules" have no owner and no definition yet.
  - "the two-domain ten-talk stream" makes the reader work out which stream is meant.
- **Rewrite** (first and penultimate sentences):
  > The language-model router follows topic changes that the similarity router's confirmation rules miss. … On the two-domain ACL–ESO stream, both routers keep the correct slice searchable in about 98% of chunks, so their accuracies in Table 1 match.

### 6.2, "Processing speed."

- **Score:** 3.
- **Connection:** not announced in the Section 6 opening.
- **What made me stop:**
  - The purpose of the RASST sentence is unclear. Is 0.042 s meant to be compared with the Retrieval column, which also includes speech encoding and routing?
  - The conditions behind the 2.3-second first-output time are not stated.
- **Quote:** "Its real-time factor matches that of Known-domain-1k in Table 3, and RASST measured its retriever alone at 0.042 seconds per chunk on the same GPUs."
- **Rewrite:**
  > With routing, AutoTerm-SST still processes audio about 4.5 times faster than real time, and its real-time factor matches that of Known-domain-1k in Table 3. For comparison with the Retrieval column, which also includes speech encoding and routing, RASST measured its retriever alone at 0.042 seconds per chunk on the same GPUs.

### 7 Conclusion

- **Score:** 4.
- **Connection:** a faithful recap.
- **What made me stop:** the main text never shows subtopic results, so the "next step" comes without grounding.
- **Quote:** "Extending the router to finer-grained subtopics is a natural next step."
- **Rewrite:**
  > Extending the router to finer-grained subtopics, which Appendix C tests with 24 close-domain slices, is a natural next step.

### Ethics statement

- **Score:** 4.
- **What made me stop:** "its CC BY 4.0 license" could cover all three corpora or only RealSI.
- **Rewrite:**
  > Our evaluation uses published research corpora: ACL 60/60, the ESO test set, and RealSI. We use RealSI under its CC BY 4.0 license without redistributing its audio.

---

## 3. Appendices

- **A Statistical comparisons. Score 4.** The opening sentence credits the permutation test with showing that the gains "hold across individual talks". It is the leave-one-out sentence at the end that shows this. Move that sentence up.
- **B Router configuration. Score 3.**
  - Problem: the switch-rule sentence carries five conditions.
  - Problem: "Because both routers update on wall-clock time, each ran alone on its server" skips the reason.
  - Rewrite:
    > It keeps a confirmed slice and pins it in the working set. The confirmed slice changes only when another slice leads both the runner-up and the confirmed slice by fixed margins in three consecutive updates, outside a 30-second cooldown. … Because both routers update on wall-clock time, each ran alone on its server [so that load from the other could not shift its update times].
- **C, opening "Open-loop replays…". Score 3.**
  - Problem: "open-loop" and "closed-loop" are never defined.
  - Problem: "the same 19 switches" collides with the "19 of the 30 talks" in Table 2, which is a different quantity on a different stream.
  - Rewrite:
    > Open-loop replays compare the two deployed routers on identical inputs: each replay feeds saved translations of the ACL–ESO stream to both routers, so routing cannot change the translations, and builds the context from the last four translated chunks as in Section 4. … Replayed on the live emission times, the similarity router makes the same 19 slice switches on this stream as it did live.
- **C, "Topic filter." Score 3.**
  - Problem: "Updated at every chunk" differs from the 5-second rate in Section 4 and Table 5.
  - Problem: the 97.0% here equals the 97.0 in Table 5, which is stated to use the live rate, so I cannot tell which schedule produced which number.
  - Rewrite:
    > On the ACL–ESO stream with an update at every chunk, rather than every five seconds, letter probabilities alone keep the correct slice among the top four in 90.5% of chunks, and the filtered posterior in 98.7%.
- **C, "Short topic shifts." Score 3.**
  - Problem: "the median segment starts with the incoming slice already in the working set" is hard to parse as a statistic.
  - Rewrite:
    > At every length, at least half of the segments start with the incoming slice already in the language-model router's working set, and the slice reaches the top of the posterior after a median of 7.7 seconds.
- **C, "Other target languages." Score 4.** There is no pointer to Table 5 and no numbers in the text. Add "(Table 5)".
- **C, "Close domains." Score 4.**
- **C, "Generated prototypes and late updates." Score 2.**
  - Problem: two unrelated results share one paragraph.
  - Problem: "The RealSI streams use them": "them" is ambiguous. It is also the only place that discloses that the main-text RealSI numbers use GPT-5.5 prototypes.
  - Problem: "the RealSI technology talk" matches no domain in Figure 3.
  - Problem: "closed-loop runs of Section 6 applied each update before the next chunk" contradicts the Non-blocking updates paragraph in Section 4.
  - Problem: "in Table 5" points at a table that has no on-time vs late comparison.
  - Rewrite, as two paragraphs:
    > **Generated prototypes.** An LLM can write the prototypes. GPT-5.5 wrote one per slice from its glossary with every evaluation term removed. With per-chunk updates, the language-model router keeps the correct slice among its top four in 97.9% of ACL–ESO chunks with these generated prototypes, against 98.7% with hand-written prototypes. The RealSI streams use the generated prototypes, because the hand-written NLP prototype names topics of the RealSI technology [NLP?] talk.
    >
    > **Late updates.** The closed-loop runs of Section 6 applied each update before the next chunk, whereas the deployed worker of Section 4 applies it one chunk later when audio arrives faster than real time. Replaying every update one chunk late lowers language-model coverage by at most 0.33 points [relative to the on-time replays in Table 5; state which timing Table 5 uses].

---

## 4. Figures and tables

### Figure 1 (fig:architecture), 4 rendered lines — flagged

The caption itself reads well, but the figure contradicts the method:

- The box inside the memory is labelled "Sliding-window context similarity". That describes the *baseline's* embedding similarity, not letter-probability scoring by the LLM plus the sticky filter.
- There is no arrow from the translations back into the memory, so the central loop (routing reads its own output) is invisible.
- "top-K, B terms" is not tied to $K=4$ and $B=4{,}000$.
- Orange marks both the memory and the medicine chunk and glossary.
- "SRL" and the Chinese outputs are not glossed.

Figure fix: relabel the box "LM router + sticky topic filter", and add an arrow from "Translation at Chunk_t" into the memory labelled "last $w=4$ chunks".

Caption rewrite (4 lines):

> Figure 1: AutoTerm-SST during a live topic transition from NLP to medicine. The adaptive terminology memory (orange box) routes recent translations to topic slices and activates up to $K=4$ slices with at most $B=4{,}000$ terms; the MaxSim retriever builds each chunk's term map $G_t$ only from this working set, and the LLM translates speech features $h_t$ with $G_t$. The ellipsis marks further offline slices; SRL is semantic role labeling.

### Figure 2 (fig:ui), 4 rendered lines — flagged

Problems:

- "the ten-talk stream" is ambiguous.
- "about 19 minutes in" does not say whether that is into the stream or into the lecture.
- The caption's wording does not match the on-screen labels ("Active Glossary", "Confidence", "Active Terms", "Retrieved terms"); "Switch Count 14" is unexplained.
- The dark screenshot is small in print.

Rewrite:

> Figure 2: The English-to-Chinese interface replaying the ACL–ESO stream during an oncology lecture, about 19 minutes into the [stream/lecture]. The status row shows the top slice of the router's posterior (Active Glossary, Confidence) and 4,000 searchable terms (Active Terms); retrieved terms are highlighted in the translation, and the Retrieved Terms panel lists the latest pairs with scores and source slices.

### Figure 3 (fig:routing), 4 rendered lines — flagged (minor)

The legend makes the colors self-contained. Problems:

- "first RealSI stream order" is jargon.
- The row label "AutoTerm-SST" does not match the caption's "language-model router".
- "drawn at equal width" hides the fact that the talks differ in length.

Rewrite:

> Figure 3: Per-chunk routing on the first of three RealSI stream orders: ten talks, one domain each, drawn at equal width although they last 3.4 to 5.2 minutes. Each row marks whether the talk's slice ranks first in the working set, ranks lower, or is absent. The language-model router (AutoTerm-SST) ranks a new talk's slice first within 21 seconds in nine of the ten talks; the similarity router ever ranks it first in only three.

### Table 1 (tab:quality), 4 rendered lines — flagged

Problems:

- "ten-talk stream" is ambiguous, and TERM_ACC is not expanded.
- The Input values "Domain" and "Domains" are not explained.
- "its All, BLEU, and MT-BLEU average the corpora" is cryptic. The arithmetic shows InfiniSST's All (61.4) is the unweighted mean of 76.6 and 46.3. Every other row's All is pooled over the 2,047 occurrences: AutoTerm-SST's 87.7 equals (88.4·1368 + 86.3·679)/2047. Pooled the same way, InfiniSST would be about 66.5, so the All column mixes two kinds of average. Say so explicitly, or use the same aggregation for every row.
- AutoTerm-SST and Similarity routing have identical prompt precision (35.7) and pairs (0.94). Readers will check for a copy error, so a one-line note would help.

Rename the Input entries in the table to "Talk domain" and "Both domains". Rewrite (4 lines):

> Table 1: Results on the 4.7-hour ACL–ESO stream. Input is what the user supplies before the session, and Active the most entries searched per chunk; Similarity routing swaps in the router of Appendix B. TERM_ACC (terminology accuracy) and prompt precision are %, MT-BLEU masks glossary terms, and Pairs counts injected pairs per chunk. †: $p<0.05$ against AutoTerm-SST. InfiniSST uses no glossary; its All, BLEU, and MT-BLEU are unweighted means of the two corpora.

### Table 2 (tab:realsi), 5 rendered lines — flagged (over the limit)

Problems:

- "percentages of chunks whose correct slice is in the working set and ranks first" reads as one condition, but it describes two columns.
- "Never top" has no unit; it counts talks out of 30.
- 30 = three orders × ten talks, which is not stated.
- The row names do not match Table 5.

Rename the columns to "In set (%)", "First (%)", "Median to first", "Talks never first". Rewrite (3 lines):

> Table 2: Routing on 30 RealSI talks (three orders of ten). Percentages are of chunks; the median delay is over talks whose correct slice ever ranks first.

### Table 3 (tab:speed), 4 rendered lines — not flagged

Self-contained and readable. Optionally add "lower is better; RTF below 1 is faster than real time."

### Table 4 (tab:stats), 3 rendered lines — flagged (minor)

Problems: "ten-talk stream" is ambiguous, and the caption does not say that positive values favor AutoTerm-SST.

Rewrite:

> Table 4: AutoTerm-SST minus each baseline on the ACL–ESO stream; positive values favor AutoTerm-SST. Low and High give the 95% stratified talk-bootstrap confidence bounds, and $p$ is the exact two-sided talk-level permutation $p$-value. TERM_ACC differences are in percentage points.

### Table 5 (tab:router-diag), 3 rendered lines — flagged

Problems:

- "Open-loop" is undefined.
- The caption does not say which stream each column group uses.
- It does not say whether updates are on time or one chunk late, even though the text cites the late-update result "in Table 5".
- The row names ("Similarity", "Language model") differ from Tables 1–2.

Rewrite (4 lines):

> Table 5: Router coverage in replays of saved translations, so routing cannot change the output: % of chunks whose correct slice is in the four-slice working set, at the live update rate [on time / one chunk late]. Columns: the ACL–ESO stream; 40-segment NLP–medicine streams switching every 15–300 s; the ACL–ESO talks translated into German and Japanese; and 24 close-domain slices, where cells count gold term occurrences whose slice is in the working set.

### Table 6 (tab:config), 4 rendered lines — not flagged

Self-contained. A minor mismatch: the table says "Qwen3-4B-Instruct-2507" and the text says "Qwen3-4B-Instruct".

---

## 5. Where a first-time reader is most likely to get lost

1. **Which stream is "the ten-talk stream"?** Table 1, Table 4, Table 5, Figure 2 and Appendix C all report results on "the ten-talk stream". The Data paragraph never uses that name, and the RealSI streams are also built from ten talks. A reader can easily read Table 1 as RealSI.
   - **Fix:** in Data, name the two sets once: the "ACL–ESO stream" (4.7 h, ten talks, two domains) and the "RealSI streams" (three orders of ten talks, 30 talks, ten domains). Use these names in every caption and paragraph.

2. **The comparator for contribution 2 is defined only in Appendix B, under shifting names.** The abstract, the contributions and the Results compare against "a similarity router with tuned confirmation rules" that the main text never describes. The routers are also called "AutoTerm-SST" vs "Similarity routing" (Tables 1–2, Figure 3), "Language model" vs "Similarity" (Table 5), and "language-model router" vs "similarity router" (text).
   - **Fix:** add one sentence to Baselines: BGE-M3 embeddings of the last 128 tokens, matched against prototypes; a switch only after the new slice leads by fixed margins in three consecutive updates outside a 30-second cooldown; the confirmed slice is pinned. Then use one pair of names everywhere.

3. **Figure 1 shows the wrong mechanism.** "Sliding-window context similarity" tells the reader that the method is embedding similarity, which is the baseline. And with no arrow from translations back to the memory, the paper's key idea (routing on its own output) is invisible.
   - **Fix:** relabel the box "LM router + sticky topic filter". Add the feedback arrow labelled "last $w=4$ translated chunks". Write $K=4$ and $B=4{,}000$ in the figure.

4. **When routing updates happen, and under which conditions each number was produced, is scattered and looks contradictory.**
   - Section 4 says the router updates every five seconds, but writes its equations per chunk $t$.
   - Section 4 also says updates take effect one chunk late in the deployed system.
   - Appendix C says the Section 6 runs applied each update before the next chunk. Its topic-filter numbers use per-chunk updates, while Table 5 uses "the live update rate". A late-update result is cited "in Table 5", which shows no such comparison.
   - The main-text RealSI numbers use GPT-5.5 prototypes, and this is disclosed only in the last appendix paragraph.
   - **Fix:** add a "Run conditions" sentence to Section 6.1 that states, for each result, the update cadence, when updates take effect, and where the prototypes come from. Index updates with $u$. State the timing in the Table 5 caption.

5. **The central object goes by several names.** It is called glossary, topic slice, slice, topic, domain, catalog, working set, and adaptive terminology memory. The abstract and contribution 2 say "keeps the correct glossary searchable" while Section 6 and Table 2 say "correct slice". "Adding a domain" (Section 4) and "Adding a topic" (Section 5) describe the same action, and the section title "Adaptive terminology memory" is never defined.
   - **Fix:** define the terms once at the end of Section 3:
     - catalog = all slices;
     - slice = one topic's ~1,000-pair glossary plus its index and prototype;
     - working set = up to four active slices;
     - adaptive terminology memory = catalog + router + working set.

     Then write "slice" wherever a slice is meant.

**Counts:**

- 15 paragraphs scored below 4: 10 in the main text (Introduction P3, Related work "Terminology-aware translation", Section 3 P1, Topic filtering, Non-blocking updates, Data, Baselines, Metrics, Effect of catalog scale, Processing speed) and 5 in the appendices (B, the opening of C, Topic filter, Short topic shifts, Generated prototypes and late updates).
- 7 captions flagged: Figures 1, 2 and 3, and Tables 1, 2, 4 and 5. Only Table 2's caption runs past 4 rendered lines.