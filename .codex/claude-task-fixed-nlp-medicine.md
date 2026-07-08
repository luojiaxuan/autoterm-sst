User asked why fixed NLP glossary looks unexpectedly good on medicine data in the mixed ACL/medicine term accuracy benchmark. Review the latest commit only. Acceptance criteria:
- Fixed non-none glossary presets must use the same fixed prompt_k=10 surface contract as auto_working, while no_glossary/none remains empty.
- Mixed audio term scorer must keep occurrence term_acc but add unique term/type diagnostics without changing the existing denominator.
- Documentation should clearly explain why old fixed NLP medicine rows are output-centric and not prompt-channel attribution.
- Do not recommend environment-variable controls; repo policy prefers explicit config/parameters.
