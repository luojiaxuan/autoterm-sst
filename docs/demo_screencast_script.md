# AutoTerm-SST screencast script (≈2.5 min)

Target: EMNLP System Demonstrations screencast (CFP: ≤2.5 min, desk-reject
without it). Reviewers of demo papers weight *system utility*, *does it run
live*, and *clarity* far above ablation depth. So the video should spend most
of its budget on the one live moment competitors cannot fake: the automatic
ACL→medicine glossary switch happening in a single streaming session with
nothing configured.

Record at 1280×800 or larger, light-on-dark UI, cursor visible. Speak over a
single continuous screen capture — do not cut mid-switch. Use the green
**ACL → Medicine** sample (one 120-second clip, domain boundary at 0:60).

Live demo entry point (put on a title card / description): the stable URL
<https://luojiaxuan.github.io/autoterm-sst/> redirects to the current GPU host;
source + mock mode at <https://github.com/luojiaxuan/autoterm-sst>.

---

## Shot list and narration

### 0:00–0:20 — The problem (title + UI idle)
*Screen: the idle page, cursor resting near the "zero-setup" tagline.*

> "Streaming speech translators mistranslate the technical terms that matter
> most — and today the only fix is to hand-build a glossary before every talk.
> AutoTerm-SST removes that step. It watches the stream and routes the right
> terminology memory in automatically. Nothing to upload, no domain to pick."

Point the cursor at the **Terminology → Auto (zero-setup)** pill and the
`Active Glossary: none` field. Do not touch Fixed.

### 0:20–0:35 — One click to start
*Screen: click the green **ACL → Medicine** sample, then the ▶ play button.*

> "I'll start a real session — an academic NLP talk that turns into an oncology
> talk halfway through. I click the sample, and press play. That's the entire
> setup."

Let translation text begin streaming in the Live Translation panel. Do not
narrate over dead air while the engine warms up (~5 s); trim that pause in edit.

### 0:35–1:05 — ACL domain: it routes to NLP on its own
*Screen: translation streaming; camera on the status strip.*

> "It's translating an NLP talk. Watch the router — Auto Topic reads
> `nlp`, the Active Glossary has snapped to `nlp_core_10k`, and the retrieved
> terms under the translation are things like *annotated corpus* and
> *language model*, each shown with its target-language term and a retrieval
> score. I never told it this was NLP."

Hover the **Retrieved terms** rows so the reviewer sees term → translation →
score. Let a few technical terms land correctly in the output.

### 1:05–1:45 — The switch (the money shot)
*Screen: keep rolling across the 0:60 clip boundary. The speaker turns to
oncology; ~15–45 s later the Active Glossary flips to `medicine_core_10k` and
Switch Count ticks to 1.*

> "Now the talk changes to medicine. No button, no reload. The router sees the
> new terminology in its own output, and… there — Active Glossary switches to
> `medicine_core_10k`, Switch Count goes to one. The retrieved terms are now
> *chemotherapy*, *carcinoma*, and the translation picks up the correct
> oncology vocabulary mid-stream."

This is the beat to hold on screen. If the switch lands late in your take,
speed the pre-switch stretch 1.5× in edit but show the switch itself at real
time.

### 1:45–2:00 — Latency control, live
*Screen: mid-session, open the Latency dropdown, switch 2x → 1x, click
Update Session (cheap path — no reinit). Point at the retrieve/gen ms in the
terms panel.*

> "Latency is a live control: I drop the chunk multiplier from two to one and
> the session keeps going, now with faster partials. Retrieval stays around
> eighty milliseconds a chunk, and the prompt never carries more than ten
> score-filtered terms — a broad glossary can't flood the model."

### 2:00–2:20 — Three languages
*Screen: split or quick cuts: (1) the En→Ja tab and (2) the En→De tab,
each already mid-stream on the same combined clip, glossary chip showing
`medicine_core_10k` / retrieved ja/de term pairs. Keep each ~8s.*

> "The same terminology memory serves three target languages — one MaxSim
> index, per-language translations. Here's the same talk going to Japanese,
> and to German — same automatic switch, no setup in any of them. Across
> ten-talk streams, automatic routing beats both fixed single-domain
> glossaries in all three languages, and thirty-two concurrent sessions stay
> real-time."

*(Practical: record the ja/de B-roll from the aries demo hosts — I bring up
En→Ja and En→De servers with their own tunnel URLs after the eval runs
finish; ask Claude for the current URLs. If they are not up at recording
time, substitute the paper's Appendix D table for this beat.)*

### 2:20–2:30 — Close
*Screen: back to the zh live UI, lower-third with the stable URL.*

> "Zero-setup terminology memory for streaming translation — live at this
> URL. Try it yourself."

---

## Practical capture notes

- **Rehearse the switch timing once.** Load ACL → Medicine, press play, and
  note the wall-clock second when Active Glossary flips (our runs: 13–55 s after
  the 0:60 boundary, so 0:73–1:45 into playback). Plan narration around your
  actual take.
- **If the switch is slow in a take,** it is still correct — the paper reports
  one delayed switch over a generic talk opening. Just re-record; don't fake it.
- **Do not click ▶ before the clip shows a duration** in the audio scrubber
  (over a slow tunnel the blob needs a second to buffer). Clicking play too
  early leaves it stuck at 0:00.
- **Keep Fixed mode out of the video** unless you explicitly contrast it — the
  pitch is zero-setup Auto.
- Export H.264 MP4, ≤2.5 min. Add a one-line lower-third with the live URL.
- After the tunnel restarts, run `scripts/update_demo_redirect.sh <new-url>` so
  the URL on the title card keeps working for reviewers.

## While recording, also grab the replacement for Figure 2

The paper's Figure 2 (`demo_paper_emnlp/latex/figures/ui_evidence_panel.png`)
is stale — it shows the old `common_terms` diagnostic glossary. During your
take, once the session has routed to a real domain, take one clean screenshot
of the evidence panel showing:
- **Active Glossary** = `nlp_core_10k` or `medicine_core_10k` (not
  `common-terms`),
- a few **Retrieved terms** rows with target-language translations and scores,
  labelled with a real source (not `diagnostic:common_terms`),
- some **Live Translation** text visible.

Save it over `ui_evidence_panel.png` (same aspect ratio, ~1230×715) and
recompile. This can only be captured from a real browser with audio playback;
an automated/headless Chrome cannot decode the sample and will sit at 0:00.
