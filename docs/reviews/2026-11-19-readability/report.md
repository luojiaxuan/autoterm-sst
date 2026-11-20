<!-- 无记忆审读者：Claude Opus 5.5，effort xhigh，2026-11-19 PT；材料为 6bddc1e 编译的 PDF 与去注释的 LaTeX 源 -->

# Area-chair readability report: *AutoTerm-SST: Adaptive Terminology Memory with Zero Session-Time Setup for Streaming Speech Translation*

**Layout as rendered.** The main text runs over pp. 1–6: the abstract, Sections 1–7, then Acknowledgments. References fill pp. 7–8. Appendices A–C start in the right column of p. 8, and Tables 2–4 sit on p. 9. Table 1 is a main-text table, but it floats to the top of p. 4 inside Section 4.

---

## 1. Story

**One-sentence claim.** A retrieval-augmented streaming translator can drop pre-session glossary setup by choosing up to four offline topic slices from its own partial translations under a fixed retrieval and prompt budget. Doing so shows no significant loss against a known-domain reference and avoids the degradation of million-entry merged glossaries.

The intro tells this as one line, and the line is mostly clear. The problem is that RASST needs a glossary and merged glossaries degrade as they grow. The insight is that partial output reveals the topic within seconds. The method is slices, a router and a fixed budget. The evidence is parity with the known-domain reference, a win over Merged-1M and a prompt-noise mechanism. The line breaks in five places:

1. **Results before the setup.** Table 1 appears on p. 4 inside Section 4. Every row name in it is defined on pp. 5–6.
2. **The key insight rests on appendix evidence.** Intro ¶3 relies on an "open-loop replay" whose evidence exists only in Appendix C, and it is not cited there. It also uses "slice" one paragraph before slices are defined.
3. **Sections 3 and 4 overlap.** Section 3 already explains the working set, the 0.78 threshold and the ten-pair cap. Section 4 repeats them, and Section 3 ends with a forward pointer to Section 6 just before Section 4 steps back into design. That is a jump, not a causal hand-off.
4. **Section 5 has no causal bridge on either side.** The interface claim ("Viewers can directly determine whether a mistranslation stems from…") reappears in the conclusion's takeaway, but no result in Section 6 supports it.
5. **"Routing behavior" mixes live and replay numbers.** It sets live numbers (88.5%, 96.9%) beside an open-loop replay number (85.0%), and it says "the router" tracks 15-second switches when the replay disables the confirmation rule and pinning.

Inside Section 6 the transitions are causal and work well: catalog scale leads to prompt noise as the mechanism, which leads to routing behavior as the reason AutoTerm works. Between top-level sections they are topical rather than causal. The numbers 2.46×, 96.9%, 3.9 and 2.30 are repeated across the abstract, contributions and results, which is acceptable. The "three invariants" restate the intro's "keeping the per-chunk budget constant" a third time.

---

## 2. Main text, paragraph by paragraph

### Abstract: **4**
- **Connection:** good. **Role:** problem, method, headline numbers.
- **What stopped me:**
  - "the fixed retrieval and prompt budget" presupposes a budget the reader has not yet heard of.
  - The results sentence carries three clauses and two percentages.
  - "known-domain reference" is undefined, and "reference" collides with reference translations.
  - "ACL talks" names a venue, not a domain; ML readers outside NLP will not decode it.
- **Quotes:**
  - "Merging every domain into one universal glossary removes this step, but as the catalog expands, irrelevant and conflicting entries crowd out useful terms inside the fixed retrieval and prompt budget."
  - "On a 4.7-hour English-to-Chinese stream alternating five ACL talks with five oncology lectures, AutoTerm-SST reaches 88.3% terminology accuracy, showing no significant difference from a known-domain reference at 88.1%, and a second independent run reproduces this parity."
- **Rewrite:** "Merging every domain into one universal glossary removes this step, but because retrieval and the prompt have fixed budgets, irrelevant and conflicting entries crowd out useful terms as the catalog expands. … On a 4.7-hour English-to-Chinese stream alternating five ACL talks with five oncology lectures, AutoTerm-SST reaches 88.3% terminology accuracy. This shows no significant difference from a known-domain reference that is told each talk's domain (88.1%), and a second independent run reproduces this parity."

### §1 Introduction

**¶1 "Simultaneous speech translation, or SST…": 4**
- **Connection:** good. **Role:** task and gap.
- **What stopped me:** "rare methods, datasets" reads as "uncommon methods" rather than "names of methods". Quote: "yet rare methods, datasets, clinical concepts, and named entities remain difficult to recover from acoustics alone."
- **Rewrite:** "yet rare method and dataset names, clinical concepts, and named entities remain difficult to recover from acoustics alone."

**¶2 "Existing retrieval-augmented SST systems…": 4**
- **Connection:** good. **Role:** the two failure modes, manual setup and merged-glossary decay.
- **What stopped me:** "fixed retrieval and prompt capacities" is vague at this point. The 87.8%→84.4% figure has no pointer.
- **Quote:** "yet fixed retrieval and prompt capacities let irrelevant and conflicting mappings crowd out relevant terms as the catalog grows. In our evaluation, expanding a merged glossary from 100,000 to one million entries lowers terminology accuracy from 87.8% to 84.4%."
- **Rewrite:** "yet retrieval keeps a fixed number of candidates and the prompt a fixed number of term pairs, so irrelevant and conflicting mappings crowd out relevant terms as the catalog grows. In our evaluation (Table 1), expanding a merged glossary from 100,000 to one million entries lowers terminology accuracy from 87.8% to 84.4%."

**¶3 "The translation stream itself supplies…": 3**
- **Connection:** good causally. This is the paper's key insight. **Role:** motivates routing from output.
- **What stopped me:**
  - "slice" is used before ¶4 defines it.
  - "open-loop replay" is unexplained jargon.
  - "top four" by what measure, and why four?
  - The supporting evidence lives only in Appendix C, which is not cited.
- **Quote:** "Partial output names the active topic within seconds: in an open-loop replay of system translations, the slice for a new topic ranks among the top four slices after a median of one 1.92-second audio chunk."
- **Rewrite:** "Partial output names the active topic within seconds: when we replay saved system translations without re-decoding (Appendix C), the glossary for a new topic ranks among the four best-matching topic glossaries after a median of one 1.92-second audio chunk."

**¶4 "We present AutoTerm-SST, a system demo…": 4**
- **Connection:** good. **Role:** system summary.
- **What stopped me:** one object gets three names. Here it is "short bilingual topic description" and then "slice description". Section 4 calls it a "prototype", the word used everywhere afterwards. Introduce "prototype" now.
- **Quote:** "Operators register offline topic slices, each holding roughly 1,000 term pairs and a short bilingual topic description. During a session, a training-free router compares recent partial translations against each slice description…"
- **Rewrite:** "Operators register offline topic slices, each holding roughly 1,000 term pairs and a prototype, a short bilingual topic description. During a session, a training-free router compares recent partial translations against each prototype and activates up to four matching slices inside a fixed term budget."

**Contribution bullet 1, "Zero-setup terminology-aware SST…": 4**
- **What stopped me:** the clause "while running 2.46 times faster than real time" grammatically attaches to the interface.
- **Rewrite:** "Zero-setup terminology-aware SST built on RASST that runs 2.46 times faster than real time, paired with a live interface that displays active slices and retrieved term pairs beside every partial translation."

**Contribution bullet 2, "Budgeted multi-slice terminology memory…": 3**
- **What stopped me:**
  - "pinning", "confirmed slice" and "decoding windows" are all undefined at this point; "decoding windows" is never defined anywhere.
  - The sentence is one long noun phrase carrying three participial clauses.
- **Quote:** "holding per-chunk retrieval and prompt budgets constant as the catalog grows and keeping the correct glossary searchable in 96.9% of decoding windows by pinning the confirmed slice."
- **Rewrite:** "Budgeted multi-slice terminology memory that selects offline slices from the system's own translations without training. It holds per-chunk retrieval and prompt budgets constant as the catalog grows, and by pinning the slice the router has confirmed as the current topic, it keeps the correct glossary searchable in 96.9% of decoding windows."
- If a decoding window is one 1.92-second chunk, say "chunks" here and everywhere else.

**Contribution bullet 3, "Evidence on a 4.7-hour stream…": 3**
- **What stopped me:** "Evidence on X that Y" splits the noun from its complement with a 14-word modifier. I had to reparse it to find what the evidence shows.
- **Rewrite:** "Evidence that zero setup incurs no significant quality loss, from a 4.7-hour stream alternating natural language processing, or NLP, and oncology talks. AutoTerm-SST is on par with a known-domain reference across two independent runs and beats a one-million-entry merged glossary by 3.9 points of terminology accuracy and 2.30 BLEU, whereas merged glossaries match it on both metrics only up to 10,000 entries."

### §2 Related work

**¶1 "Streaming speech translation.": 5**
- **Connection:** good. **Role:** positions the system as a layer around an existing model.

**¶2 "Terminology-aware translation.": 4**
- **What stopped me:** the second sentence joins two unrelated facts with "and": what the methods do, and a weakness of speech translation.
- **Quote:** "They apply user-provided dictionaries through constrained decoding or training-time exposure (Hokamp and Liu, 2017; Dinu et al., 2019), and rare terms and named entities remain a weakness of speech translation (Gaido et al., 2021)."
- **Rewrite:** "Rare terms and named entities remain a weakness of speech translation (Gaido et al., 2021). Terminology-aware translation methods apply user-provided dictionaries through constrained decoding or training-time exposure (Hokamp and Liu, 2017; Dinu et al., 2019), and so assume the relevant glossary is known before decoding."
- Then continue with the retrieval sentence and the AutoTerm-SST sentence unchanged.

**¶3 "Interactive translation systems.": 4**
- **What stopped me:** the heading promises interactive tools, but the first clause is about LLM serving engines.
- **Rewrite:** rename the heading to "**Serving and interactive translation.**" and keep the text.

### §3 System overview

**¶1 "AutoTerm-SST adds an online terminology-selection loop…": 3**
- **Connection:** good to §2. **Role:** RASST background plus what AutoTerm-SST changes.
- **What stopped me:**
  - Four sentences of RASST internals, including an inline formula, sit between the topic sentence and the AutoTerm-specific point.
  - The symbol $w$ indexes speech windows here. In §4 and Table 3, $w$ is the number of translated chunks in the router context.
- **Quote:** "To find spoken terms without a transcript, the retriever scores every glossary entry by maximum similarity, or MaxSim: with each source term pre-encoded as a vector $z_i$ and the speech of chunk $t$ encoded into vectors $q_w$ over overlapping windows $w\in\mathcal W_t$, the term score is $r_i=\max_{w\in\mathcal W_t}q_w^\top z_i$."
- **Rewrite:** "Because there is no transcript, the retriever matches speech directly against glossary source terms by maximum similarity, or MaxSim. Each source term is pre-encoded as a vector $z_i$, the speech of chunk $t$ is encoded into vectors $q_v$ over overlapping windows $v\in\mathcal V_t$, and the term score is $r_i=\max_{v\in\mathcal V_t}q_v^\top z_i$. Terms scoring above 0.78 become candidates, and at most ten source–target pairs enter the LLM prompt. AutoTerm-SST changes only which entries this lookup can see: its router restricts the lookup to a working set of up to four topic slices, and each newly decoded translation is appended to the router's context to guide slice selection for subsequent chunks."

**¶2 "Placing adaptation outside the translation model…": 3**
- **Connection:** weak. "therefore" follows the three invariants, but the shared-components conclusion follows from pluggability, which comes two sentences earlier.
- **Role:** the design contract and a promise of a controlled comparison.
- **What stopped me:**
  - "agent" is SimulEval jargon.
  - "Three invariants hold." is followed by a comma-separated list the reader must count.
  - "policies" means read/write policies in §2 but glossary configurations here.
- **Quote:** "Three invariants hold. Live requests carry no domain label or glossary, catalog growth leaves per-chunk retrieval and prompt budgets constant, and adding a topic is an offline data update with no model training. All compared policies in Section 6 therefore share…"
- **Rewrite:** "Placing adaptation outside the translation model keeps the system pluggable: one streaming protocol serves a retrieval-free translator, a fixed-glossary RASST translator, or AutoTerm-SST, and every partial translation carries the retrieved pairs, their source slices, and the working set to the interface. All compared policies in Section 6 therefore share the speech encoder, retriever, decoder, and streaming schedule, and differ only in which terminology indexes are searchable. The design also keeps three invariants: (i) live requests carry no domain label or glossary; (ii) catalog growth leaves per-chunk retrieval and prompt budgets constant; and (iii) adding a topic is an offline data update with no model training."

### §4 Adaptive terminology memory

**Opener "Separating an offline catalog from a small working set…": 5**
- **Role:** defines the slice and the prototype. It reads cleanly.
- Its first sentence restates §3. A causal opener would help, for example: "The working set introduced above rests on one design choice:…"

**¶ "Offline catalog.": 5**
- Clear.
- It overlaps with §5 "Extensibility". Consider moving the registration details here and cross-referencing.

**¶ "Semantic routing.": 4**
- **Connection:** good. **Role:** the routing score.
- **What stopped me:**
  - $P_d$ is "prototype texts" (plural), but the prototype was introduced as one description. Which texts make up $P_d$?
  - "norm" is undefined.
  - $w$ clashes with §3, as noted above.
- **Quote:** "for slice $d$ with prototype texts $P_d$ and sentence encoder $E$, it computes"
- **Rewrite:** "At chunk $t$, the router concatenates the last $w$ translated chunks into a context $c_t$. For slice $d$, let $P_d$ be its prototype texts (state which: the description and each example term, in both languages?) and $E$ a sentence encoder. The router averages the prototype embeddings, normalizes the mean with $\operatorname{norm}(\cdot)$ (state: L2), and scores the slice by cosine similarity to the context:"

**¶ "Working-set update.": 3**
- **Connection:** good. **Role:** the stabilizing rule behind the 96.9% result.
- **What stopped me:**
  - The first sentence uses "pinning" and "confirmed slice" before either is defined.
  - The definition of the confirmed slice is buried in a relative clause with a double comparison.
  - The main text omits the warm-up (6 s) and cooldown (30 s) listed in Table 3.
- **Quote:** "One slot stays pinned to the confirmed slice, which switches only when another slice leads both the runner-up and the confirmed slice by fixed score margins across three consecutive updates."
- **Rewrite:** "Pinning a confirmed slice alongside top-scoring alternatives stabilizes routing while accommodating topic transitions. The confirmed slice is the router's current topic choice: it changes only when another slice leads both the runner-up and the confirmed slice by fixed score margins in three consecutive updates. Every five seconds, the router rescores every catalog slice and builds a working set of at most $K=4$ slices and $B=4{,}000$ searchable terms in total. One slot is always pinned to the confirmed slice. The remaining slots hold the highest-scoring other slices within budget $B$, and lower-ranked slices remain excluded from retrieval. Holding up to four slices lets outgoing and incoming topics coexist during transitions until stale slices drop in rank; Appendix B lists all router thresholds."

**¶ "Fixed evidence budget.": 4**
- **What stopped me:**
  - "distributes 100 candidate slots across the active slices" does not say how: equally or by score?
  - One sentence joins two unrelated facts.
- **Quote:** "Retrieving 100 candidates first leaves headroom for deduplication and cross-slice reranking before final admission, and a chunk receives fewer than ten pairs whenever fewer candidates clear the 0.78 score threshold."
- **Rewrite:** "Retrieving 100 candidates first leaves headroom for deduplication and cross-slice reranking before final admission. A chunk receives fewer than ten pairs whenever fewer candidates clear the 0.78 score threshold."

**¶ "Causal updates.": 4**
- **What stopped me:** a missing step. The snapshot already guarantees the prompt cannot change mid-decoding, so it is unclear what preloading adds.
- **Quote:** "Indexes of likely slices are preloaded in the background, so loading never changes a prompt mid-decoding."
- **Rewrite:** "Indexes of likely slices are preloaded in the background, so loading happens outside decoding and never changes a prompt mid-decoding."

### §5 Demo interface

**Opener "AutoTerm-SST serves listeners…": 5**
- **Role:** the user and the zero-setup entry point.
- **Connection from §4:** weak, since there is no bridge. Consider: "The router and budget run invisibly; the interface makes their decisions visible."

**¶ "Walkthrough.": 4**
- **What stopped me:**
  - "the slice status row" is mentioned before Figure 2 is referenced.
  - "beside the output that used them" does not match the screenshot, where the pairs sit in a panel below the translation.
- **Quote:** "once context accumulates, the slice status row switches to the highest-scoring slices, and retrieved pairs appear beside the output that used them."
- **Rewrite:** "Once context accumulates, the status row at the top of the interface (Figure 2) switches to the highest-scoring slices, and the retrieved pairs used for each output appear in the evidence panel."

**¶ "Visible terminology evidence.": 4**
- **What stopped me:** the text says the *status row* shows "the active slices … and retrieval latency". In Figure 2, the status row shows only the confirmed slice (`medicine_core_1k`) and the term count. The four active slices and "retrieve 62.4ms" appear in the evidence-panel header.
- **Quote:** "and displays the active slices, active term count, and retrieval latency in a status row."
- **Rewrite:** "As Figure 2 shows, the interface highlights retrieved terms inside the translation and lists their source–target mappings; a status row shows the confirmed slice and active term count, and the evidence-panel header names the active slices and the retrieval latency."

**¶ "Extensibility.": 4**
- **What stopped me:** "their snapshot" is ambiguous. §4 used "snapshot" for the per-chunk working set; here it seems to mean the catalog.
- **Quote:** "Running sessions keep their snapshot while new sessions see new slices at once"
- **Rewrite:** "Running sessions keep the catalog snapshot taken at session start, while new sessions see new slices at once"

**¶ "Availability.": 4**
- **What stopped me:** "Live mode" is undefined. Is it the opposite of the replay used for Figure 2?
- **Rewrite:** "In live mode, as opposed to replaying recorded outputs as in Figure 2, the demo serves Qwen3-Omni with vLLM continuous batching alongside the RASST retriever."

### §6 Evaluation

**Opener "Our evaluation tests two claims…": 5**
- A clear map. Results ¶1 tests claim 1, and ¶¶2–3 test claim 2.

**¶ "Data.": 4**
- **What stopped me:** "keeps target terms competitive" — competitive against what?
- **Quote:** "Because the NLP and medicine slices hold the benchmark glossaries, this stream measures how reliably routing keeps target terms competitive under a fixed budget."
- **Rewrite:** "Because the NLP and medicine slices contain the benchmark glossaries, the gold terms are in the catalog, and this stream measures how reliably routing keeps them competitive for the fixed retrieval and prompt budget."

**¶ "Baselines.": 3**
- **What stopped me:**
  - Six system names arrive in eight sentences with no naming convention.
  - "entries from other topics" has no stated source.
  - The first sentence repeats the end of §3.
- **Quote:** "Merged-100k and Merged-1M expand the same NLP and medicine glossaries with entries from other topics, and Merged-100k consists of the first 100,000 entries of Merged-1M."
- **Rewrite:** "All policies share RASST's model, chunks, retriever, threshold, candidate slots, and prompt cap, and differ only in which entries are searchable; each baseline's name gives the number of entries it searches per chunk. InfiniSST (Ouyang et al., 2025) runs the same streaming model without retrieval. Known-domain-1k searches only the 1,000-entry slice matching the current talk's domain and therefore requires the domain label of every talk. AutoTerm-SST routes over ten 1,000-entry slices, covering NLP, medicine, and eight topics drawn from Wikipedia and Wikidata, with at most four active at once. The merged glossaries search all their entries for every chunk and retain conflicting translations of shared source terms. Merged-2k pools the NLP and medicine slices, which still requires knowing both domains of the stream. Merged-10k pools all ten slices into 9,880 entries after removing exact duplicates. Merged-1M expands the same NLP and medicine glossaries with entries from other topics (state the source), and Merged-100k consists of its first 100,000 entries."

**¶ "Runs.": 4**
- **What stopped me:** the paragraph never says why there are two runs, or why AutoTerm-SST scores 88.3 in run 1 and 87.3 in run 2.
- **Rewrite:** "Run 1, on one H200 GPU, evaluates InfiniSST, Known-domain-1k, AutoTerm-SST, Merged-100k, and Merged-1M. Run 2, on two A6000 GPUs, repeats Known-domain-1k and AutoTerm-SST as an independent check of their parity and adds Merged-2k and Merged-10k. Within a run, all policies see the same audio order and chunk boundaries, so all comparisons are made within a run."
- If known, add one clause on the source of run-to-run variation.

**¶ "Metrics.": 3**
- **What stopped me:**
  - Seven concepts in one paragraph: TERM_ACC, BLEU pipeline, MT-BLEU, prompt precision, permutation test, bootstrap and interval notation.
  - The TERM_ACC sentence nests its counts as an apposition.
  - "reference boundaries of the StreamLAAL protocol" is jargon.
  - "in the surrounding audio" is vague. Which span?
  - The sign convention (AutoTerm-SST minus baseline) appears only in the Table 2 caption, in the appendix.
- **Quotes:**
  - "Following RASST, terminology accuracy, or TERM_ACC, is the share of 2,047 time-aligned term occurrences, 1,368 NLP and 679 medicine, whose reference translation appears in the aligned hypothesis."
  - "To compare two policies, an exact permutation test swaps their outputs talk by talk over all $2^{10}$ combinations, and a talk-level bootstrap stratified by corpus yields 95% confidence intervals."
- **Rewrite as two paragraphs:**
  - "**Metrics.** Following RASST, terminology accuracy, or TERM_ACC, is the share of 2,047 time-aligned term occurrences whose reference translation appears in the aligned hypothesis; 1,368 occurrences are NLP and 679 are medicine. For translation quality, we resegment each talk with mWERSegmenter to the reference boundaries of the StreamLAAL protocol and compute corpus SacreBLEU. Masked-term BLEU, or MT-BLEU, repeats this scoring after masking every evaluated target term in both hypothesis and reference, so it measures the translation around the terms. Prompt precision is the share of injected term pairs that match a gold term in the surrounding audio (state the span)."
  - "**Statistics.** To compare two policies, an exact permutation test swaps their outputs talk by talk over all $2^{10}$ combinations, and a talk-level bootstrap stratified by corpus yields 95% confidence intervals. Differences are AutoTerm-SST minus the baseline. We write symmetric intervals as $a\pm h$ and asymmetric intervals spanning $a-l$ to $a+u$ as $a^{+u}_{-l}$; Appendix A reports every paired comparison."

**¶ "Comparison with the known-domain reference.": 4**
- **What stopped me:**
  - The sign of "+0.2" is ambiguous in the main text.
  - The run-2 parity claim, which the abstract headlines, sends the reader to an appendix table on p. 9.
  - "Both retrieval-augmented policies" needs naming.
- **Quote:** "In run 1, AutoTerm-SST differs from Known-domain-1k by $+0.2\pm1.2$ TERM_ACC points at $p=0.86$ … Run 2 on separate GPUs again shows no significant difference, as Table 2 reports. Both retrieval-augmented policies exceed…"
- **Rewrite:** "In run 1, AutoTerm-SST minus Known-domain-1k is $+0.2\pm1.2$ TERM_ACC points at $p=0.86$ and $-0.10^{+0.48}_{-0.40}$ BLEU at $p=0.70$. Run 2 on separate GPUs again shows no significant difference (Table 2). Both retrieval-augmented policies, AutoTerm-SST and Known-domain-1k, exceed retrieval-free InfiniSST by about 27 TERM_ACC points (Table 1)."

**¶ "Effect of catalog scale.": 3**
- **What stopped me:**
  - One sentence carries two intervals, two p-values and two point losses.
  - "the deficit": whose?
  - "maintains TERM_ACC": relative to what?
  - "the largest gap, to Merged-10k" hides that Merged-10k is *ahead*.
  - Table 2 gives that gap as −0.83, which exceeds "within 0.8", and its bootstrap interval [−1.58, −0.25] excludes zero. A careful reader will notice both.
- **Quote:** "Scaling to Merged-1M widens the deficit to $3.9\pm1.9$ TERM_ACC points at $p=0.012$ and $2.30^{+0.65}_{-0.59}$ BLEU at $p=0.002$, concentrated on medicine with a 7.4-point loss against 2.2 points on NLP."
- **Rewrite:** "In run 2, Merged-2k and Merged-10k remain within 0.8 TERM_ACC points of AutoTerm-SST with matching BLEU; the largest gap, where Merged-10k leads, has $p=0.08$. In run 1, Merged-100k matches AutoTerm-SST in TERM_ACC but trails it by $1.37^{+0.43}_{-0.51}$ BLEU at $p=0.004$. Merged-1M trails by $3.9\pm1.9$ TERM_ACC points at $p=0.012$ and by $2.30^{+0.65}_{-0.59}$ BLEU at $p=0.002$. Its TERM_ACC loss is concentrated on medicine: 7.4 points, versus 2.2 points on NLP."
- Reconcile "0.8" with Table 2's 0.83.

**¶ "Effect of prompt noise.": 4**
- **What stopped me:** the reason the no-term subset is the right test is left implicit.
- **Quote:** "On the 935 of 1,905 sentences whose references contain no glossary term, AutoTerm-SST still leads Merged-1M by 2.19 BLEU at $p=0.023$."
- **Rewrite:** "On the 935 of 1,905 sentences whose references contain no glossary term, where reference-term accuracy plays no role, AutoTerm-SST still leads Merged-1M by 2.19 BLEU at $p=0.023$."

**¶ "Routing behavior.": 3**
- **What stopped me:**
  - "nearly the entire stream" replaces the 96.9% that the reader is looking for.
  - "this gain" has no clear referent.
  - The 85.0% comes from an open-loop replay but is set beside live numbers.
  - "the router also tracks switches … every 15 to 300 seconds" describes a replay that disables the confirmation rule and pinning; with the 30 s cooldown in Table 3, the live router could not.
  - The tense shifts to "kept".
- **Quote:** "Pinning accounts for most of this gain: without pinning, the four highest-scoring slices by raw similarity cover only 85.0% of windows. Open-loop replays in Appendix C show that the router also tracks switches between NLP and medicine every 15 to 300 seconds."
- **Rewrite:** "In run 1, the confirmed slice alone is correct in 88.5% of decoding windows, whereas the four-slice working set keeps the correct glossary searchable in 96.9% of them (Figure 3). Pinning, rather than the extra slots, accounts for most of this gain: in an open-loop replay that ranks slices by raw similarity on the saved translations (Appendix C), the four highest-scoring slices without pinning cover only 85.0% of windows. The same replays show that raw-similarity ranking also tracks switches between NLP and medicine that occur every 15 to 300 seconds. Routing still lags at abrupt talk boundaries: after one ACL-to-medicine switch, stale context briefly kept the medicine slice out of the working set."

**¶ "Latency and speed.": 3**
- **What stopped me:**
  - Two sentences hold seven numbers.
  - The definition of real-time factor sits mid-sentence.
  - The 2.46× from the abstract is never connected to 0.406.
- **Quote:** "On one H200 GPU, AutoTerm-SST processes the stream at a real-time factor of 0.406, the processing time divided by the audio duration, within 0.02 of Known-domain-1k; routing and retrieval average 203 milliseconds per chunk."
- **Rewrite:** "In a single session on two A6000 GPUs, the first translation appears 2.3 seconds after audio starts. Each chunk then returns its text 0.57 seconds after it is sent at the median and 1.17 seconds at the 95th percentile. On one H200 GPU, AutoTerm-SST runs at a real-time factor (processing time divided by audio duration) of 0.406, that is, 2.46 times faster than real time and within 0.02 of Known-domain-1k. Routing and retrieval average 203 milliseconds per chunk."

### §7 Conclusion, and Acknowledgments

**Conclusion: 4**
- **What stopped me:** the second sentence chains "By…, … and … while …".
- **Rewrite:** "It routes over modular offline topic slices using its own partial translations under a fixed budget. With this routing, the system is on par with a known-domain reference across two independent runs and avoids the degradation of million-entry merged glossaries, while the interface exposes live terminology evidence."

**Acknowledgments: 5**

---

## 3. Appendices

**A, "Paired talk-level tests show…": 4**
- Clear.
- The topic sentence covers only Merged-1M, but the paragraph introduces all of Table 2.

**B, "Both evaluation runs use one fixed set…": 3**
- **What stopped me:**
  - "four windows" is a fourth meaning of "window", and its length is not given.
  - "the last 128 tokens" conflicts with, or refines, the four-chunk context $w$ of §4 and Table 3, and the reader cannot tell which.
  - Tuning on "windows drawn from the evaluation talks" will make reviewers stop. Say how large these windows are relative to the 4.7 hours.
- **Rewrite:** "We selected the routing thresholds on four [length]-second segments drawn from the evaluation talks before running the complete streams. The context encoder reads the last 128 tokens of the translated context $c_t$ [state whether this truncates the four-chunk context $w$], and the router scores slices only by this context similarity."

**C, opener "Open-loop replays of the routing signal…": 4**
- Clear, and it defines open-loop properly. The main text should borrow this definition.
- "8,776 windows" is again an undefined unit.

**C, "Short topic shifts.": 3**
- **What stopped me:**
  - "segments of 15 to 300 seconds" reads as variable-length segments, when it means five fixed lengths.
  - Two different "top four" measures sit side by side: entering the top four at least once per segment, and the share of windows.
  - "The router" here is raw similarity only.
- **Rewrite:** "The router's raw-similarity ranking follows topic switches quickly even when domains alternate every 15 to 300 seconds. For each length in Table 4 (15, 30, 60, 120, and 300 seconds), we cut the saved translations of the five ACL and five medicine talks into segments of that length and interleave NLP and medicine segments into one 40-segment stream. At every length, the incoming topic's slice enters the top four after a median of one chunk, or 1.92 seconds, ranks first after a median of 5.76 seconds, and reaches the top four in all 40 segments. Table 4 shows that the share of windows with the correct slice in the top four stays above 81% even when topics switch every 15 seconds."

**C, "Generated prototypes.": 3**
- **What stopped me:**
  - "brackets" is jargon.
  - "routing accuracy" does not say whether it means Top-4 or Top-1.
  - Generated prototypes contain "seven to ten core topics", but §4 defines prototypes as containing "example terms". Are these the same?
- **Quote:** "Across ten independently generated prototype sets in Table 4, routing accuracy brackets that of the hand-written prototypes."
- **Rewrite:** "Across the ten independently generated prototype sets in Table 4, the lowest and highest sets bracket the hand-written prototypes on both Top-4 and Top-1."
- Also align "core topics" with "example terms".

---

## 4. Figures and tables

**Figure 1 (`fig:architecture`), caption 3 lines. Not self-contained.**
- **Problems:**
  - The figure labels $h_t$ and $G_t$ ("Speech feature", "Term map"), which the text never defines.
  - No box is labeled "router". The router is presumably "Sliding-window context similarity" inside the box titled "AutoTerm", not "AutoTerm-SST".
  - $K$ and $B$ appear before §4 defines them.
  - "NLP-focused set" can be confused with the working set.
  - The green colour and "SRL" are unexplained.
  - The feedback from translations to the router, which is the paper's core idea and is named in the caption, has no arrow.
- **Rewrite (≤4 lines):** "AutoTerm-SST during a live topic transition from NLP to medicine. The router (orange AutoTerm box) scores slices against recent translations and activates up to K = 4 slices with at most B = 4,000 terms; the MaxSim retriever builds each chunk's term map G_t only from this working set, and the LLM translates speech features h_t with G_t. Green marks terms (SRL: semantic role labeling); the ellipsis marks further offline slices."
- Also add the translation→router arrow.

**Figure 2 (`fig:ui`), caption 4 lines. Partly self-contained.**
- **Problems:**
  - "run 1" and "confirmed" need the main text.
  - Status-row fields are unexplained ("Confidence 0.90", "Switch Count", "Router hybrid_window_t…").
  - At 0.72 textwidth the status-row text is about 4 pt and unreadable in print. Crop or enlarge it.
  - The text describes the panel differently from the screenshot; see §5 above.
- **Rewrite:** "The English-to-Chinese interface replaying run-1 outputs about 20 minutes into the stream, during an oncology lecture. The status row shows the confirmed medicine slice (the router's current topic) and the 4,000 searchable terms. Retrieved terms are highlighted in the translation; the bottom panel lists the latest retrieved pairs with scores and source slices, under a header naming the active slices and retrieval latency."

**Figure 3 (`fig:routing`), caption 3 lines. Not self-contained.**
- **Problems:**
  - The three rows are not explained.
  - The "Other" (gray) and "Absent" (pink) markers are not explained.
  - The row label "Selected slice" does not match "confirmed slice" in the caption and text.
  - "decoding windows" is undefined.
  - "Talk widths are normalized" does not say the x-axis is not time.
- **Rewrite:** "Routing trajectory of AutoTerm-SST across the ten alternating talks of run 1, each talk drawn at equal width. Rows show the talk domain, the confirmed slice (gray: a slice of another topic), and whether the correct slice is in the four-slice working set (green) or absent (pink). The confirmed slice can lag briefly after topic boundaries, yet the working set keeps the correct glossary searchable in 96.9% of decoding windows."
- Rename the row label to "Confirmed slice".

**Table 1 (`tab:quality`), caption 5 lines. Over the limit and not self-contained.**
- **Problems:**
  - MT-BLEU is undefined in the caption.
  - The baseline naming convention is missing.
  - "Refs" is an odd name for injected pairs; rename it "Pairs/chunk".
  - "ACL" in the caption does not match the "NLP" column header.
  - The table floats to p. 4, before any of its rows are defined. Move the table environment into §6.
- **Rewrite (≈4 lines):** "Results on the 4.7-hour ten-talk stream. Input: what the user supplies before the session; Active: most entries searched per chunk (1k = 1,000); TERM_ACC and prompt precision in %; MT-BLEU: BLEU with glossary terms masked; Refs: mean term pairs injected per chunk. †: p < 0.05 vs. AutoTerm-SST in the same run (talk-level permutation test). InfiniSST uses no glossary; its All, BLEU, and MT-BLEU average its ACL and medicine scores unweighted."

**Table 2 (`tab:stats`), caption 4 lines. Acceptable.**
- The sign convention, bounds, p and units are all given.
- Minor issues:
  - MT-BLEU is undefined.
  - Low and High are bounds here but offsets in the text notation.
  - The main text cites this appendix table for a headline claim.
- Optional: add "(positive favors AutoTerm-SST)".

**Table 3 (`tab:config`), caption 4 lines. Acceptable.**
- It is self-contained.
- "Warm-up" and "Cooldown" never appear in the main text, so mention them in §4 "Working-set update".

**Table 4 (`tab:router-diag`), caption 6 lines. Over the limit.**
- **Rewrite (≤4 lines):** "Open-loop router replays on run-1 translations. Top-4 and Top-1: % of windows whose correct slice ranks in the top four or first by raw similarity. Lowest and highest generated sets are chosen per column."

---

## 5. Where a first-time reader is most likely to get lost (most severe first)

1. **Table 1 appears before its vocabulary exists.** It sits on p. 4 in §4, while Known-domain-1k, Merged-100k, the runs, MT-BLEU, prompt precision and "Refs" are defined on pp. 5–6. The reader either stalls or skips it and forgets it.
   - **Fix:** move the table environment after §6.1 "Metrics", expand the caption as above, and rename "Refs".
2. **"Routing behavior" mixes live and replay evidence.** It sets the live 88.5% and 96.9% beside a replay-only 85.0%, uses "this gain" with no referent, and credits "the router" with tracking 15-second switches when the replay switches off confirmation and pinning (Table 3 has a 30 s cooldown).
   - **Fix:** use the rewrite in §2 above. Name the replay at first use, in intro ¶3 with a pointer to Appendix C, and say "raw-similarity ranking" wherever the replay is meant.
3. **"Window" has four meanings, and the headline unit is undefined.** It means MaxSim speech windows ($w\in\mathcal W_t$, §3), the router context of $w$ chunks (§4, Table 3), "decoding windows" (contributions, Figure 3, §6.2, and the "8,776 windows" in Appendix C), and tuning windows (Appendix B). The 96.9% headline is stated in a unit the paper never defines.
   - **Fix:** rename the MaxSim index to $v$. Define "decoding window" once, or say "chunk" if that is what it is. Call the tuning windows "segments".
4. **Naming drift for the same objects.**
   - The slice description is also called the "topic description" and the "prototype", and a prototype holds either "example terms" or "core topics".
   - The two corpora are called ACL/NLP and oncology/medicine/ESO.
   - The router's choice is the "confirmed slice" in the text but "Selected slice" in Figure 3.
   - "Policy" means a read/write policy in §2 and a glossary configuration in §§3 and 6.
   - **Fix:** one name per object, introduced once: "prototype" from intro ¶4; "NLP (ACL 60/60)" and "medicine (ESO oncology)" defined in Data and then only NLP/medicine; "confirmed slice" everywhere; "system" or "configuration" instead of "policy" outside §2.
5. **Figure 1 does not match the prose.** $h_t$ and $G_t$ are never defined, no box is called the router, $K$ and $B$ are undefined at that point, and the defining loop from translation output back to the router is not drawn.
   - **Fix:** label the router box "Router" and add the feedback arrow. Either define $h_t$ and $G_t$ in §3 or drop them from the figure. Use the rewritten caption.

**Counts:** 14 paragraphs scored below 4 (11 in the main text, 3 in the appendices) and 5 captions flagged (Figures 1, 2 and 3; Tables 1 and 4).