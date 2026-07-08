Review the changes from base commit b3cd6b9 through HEAD for the AutoTerm mixed-domain switch benchmark and router update.

User goal:
- Build a benchmark with ACL 5 talks and medicine 5 speeches, randomly ordered or interleaved, to check whether auto glossary routing can switch between NLP and medicine domains.
- Do not rely on ASR/source transcript text for routing. A target-translation-text window router is acceptable; speech/domain probe guard is acceptable.
- Keep Git as the source of truth.

Acceptance criteria to challenge:
- The new benchmark must clearly distinguish target-translation-text proxy diagnostics from full E2E generated-target + real MaxSim probe evaluation.
- The router update must not accidentally make generated target text alone switch domains without domain-probe evidence.
- The benchmark metrics must not hide wrong-domain steady-state behavior by overusing transition grace windows.
- The implementation must follow repo constraints: no new environment-variable controls, focused diff, tests/docs updated.
