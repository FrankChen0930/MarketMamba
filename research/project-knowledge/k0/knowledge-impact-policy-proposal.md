# Knowledge-impact Policy Proposal

## Requires a knowledge update

- Semantic, authority or subsystem-boundary change.
- Active/disabled/deployed state change.
- Known limitation or evidence-class change.
- New canonical contract/artifact or a supersession.
- Deprecation/removal or a changed external runtime assumption.

The code/research change should update the affected domain note and, when relevant, `Current_State.md` or `Authority_Map.md` in the same review.

## Normally does not require one

- Formatting or comments.
- Pure refactor with unchanged behavior/boundaries.
- Tests-only strengthening with no changed contract.
- Internal rename with stable public paths and semantics.

## Review prompt

Ask: “Would this change cause a new agent to choose a different runtime, authority, data class, experiment contract or safety decision?” If yes, knowledge impact is material.
