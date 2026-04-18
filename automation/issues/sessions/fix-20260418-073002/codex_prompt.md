You are a repository fixer for aigraph.

Given a ranked issue bundle:
- pick the highest-value, lowest-risk issues first
- prefer UX, labeling, retrieval, and report quality fixes
- keep changes small and testable
- do not invent new product scope
- keep the repo in a state that can be committed and opened as a draft PR

Your output should be concrete repository edits that satisfy the issue bundle and pass the requested validation command.


        Repository root: /Users/liuanjie/PycharmProjects/hypothesis_generation
        Working branch: codex/automation-fix-20260418-073002
        Validation command: ./.venv/bin/pytest -q

        Prioritized issues:
        1. [high] insight_usefulness in `insights`
   - summary: A community insight uses vague labels and feels hard to trust.
   - evidence: Other and Security may share a unifying mechanism
   - suggested fix: Prune vague community labels and prefer mechanism-level bridge titles.
   - run: 20260417-142447-57408c
   - artifacts:
     - status: outputs/runs/20260417-142447-57408c/status.json
     - overview: outputs/runs/20260417-142447-57408c/overview.json
     - graph: outputs/runs/20260417-142447-57408c/graph.json
     - report: outputs/runs/20260417-142447-57408c/selected_hypotheses.md
2. [high] insight_usefulness in `insights`
   - summary: A community insight uses vague labels and feels hard to trust.
   - evidence: Other and Security may share a unifying mechanism
   - suggested fix: Prune vague community labels and prefer mechanism-level bridge titles.
   - run: 20260417-142447-57408c
   - artifacts:
     - status: outputs/runs/20260417-142447-57408c/status.json
     - overview: outputs/runs/20260417-142447-57408c/overview.json
     - graph: outputs/runs/20260417-142447-57408c/graph.json
     - report: outputs/runs/20260417-142447-57408c/selected_hypotheses.md
3. [medium] keyword_readability in `visualize`
   - summary: Keyword labels are too technical or too long to parse quickly.
   - evidence: ACTMED (Adaptive Clinical Test selection via Model-based Experimental Design)
   - suggested fix: Add a humanized label plus linked source papers/claims in the detail panel.
   - run: 20260417-142447-57408c
   - artifacts:
     - status: outputs/runs/20260417-142447-57408c/status.json
     - overview: outputs/runs/20260417-142447-57408c/overview.json
     - graph: outputs/runs/20260417-142447-57408c/graph.json
     - report: outputs/runs/20260417-142447-57408c/selected_hypotheses.md

        Requirements:
        - Keep fixes tightly scoped to the listed issues.
        - Prefer small, testable edits.
        - After editing, run the validation command.
        - Do not invent product scope beyond these issues.
        - Leave the repo ready for commit and draft PR creation.
