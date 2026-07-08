Review the pushed change from 368462d to a5ed4ff on branch framework.

User request:
- ASR-text topic routing is not realistic because the demo is E2E.
- Continue solving auto-term domain switching with a true speech-window router, or target_translation_text-window-router.
- Earlier project decision: auto glossary should be domain-specific by default and should not prepend a common glossary.
- Git docs/README are the source of truth and must reflect the implementation.

Acceptance criteria:
- Production/default auto_working must not rely on source transcripts or ASR text.
- Deployable E2E route should use speech-window domain probes and generated target-translation windows.
- Source transcript routing may remain as controlled diagnostic/eval only.
- Generated target routing should not override explicit external router_text sources such as manifest_source.
- The default active prompt inventory should be domain-specific, not common_terms + domain.
- Routing-only domain probes must not change the fixed top-10 prompt candidate budget.
- Docs must not continue to claim source/ASR topic text or common_terms base is the default production strategy.

Please review for spec mismatches, hidden regressions, missing tests, and confusing docs. Pay special attention to:
- Whether generated target text is populated at the right point in the streaming loop.
- Whether using generated target text can self-reinforce wrong glossary choices without enough guardrails.
- Whether removing the default common base breaks fixed top-10 behavior or active slice metadata.
- Whether config parsing still has surprising legacy behavior around common_terms.
- Whether tests are sufficient for the new E2E routing path and domain-only inventory.
