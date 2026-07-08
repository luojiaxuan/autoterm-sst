Review the changes from base commit 43f2b34 through HEAD for the mixed-audio AutoTerm switch harness.

User goal:
- Continue from the ACL+medicine proxy router benchmark toward true E2E/probe replay.
- Build an eval harness that uses ACL 5 talks and medicine 5 speeches as one mixed audio playlist, random or alternating, and checks whether auto router can switch domains.
- Do not use ASR/source transcript text as routing evidence.
- Keep Git docs as source of truth and record remaining todos.

Acceptance criteria to challenge:
- The harness should stream real audio chunks through JSON WS and score runtime router metadata against playlist spans.
- It should not claim full E2E evidence before a valid hybrid_window_topic server run exists.
- Dry-run should validate path discovery/spans without requiring GPU.
- Docs should clearly state remaining todos and the current server/GPU blocker.
- Follow repo constraints: no new env-var controls, Git SoT, minimal focused changes.
