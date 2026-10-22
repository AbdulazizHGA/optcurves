# Literature check of 20 October 2026: log

Second literature check of the repository (summary in README, "Related work and literature check"). It covers the
worst-case theorem of section 4, including its μ = 0 value h²(4h − 7)/(2(2 + h²)), and again the threshold h₁(κ).
This log lists what was read and searched and what came back. The raw API responses were not stored; every query
below can be rerun. A search that finds nothing is evidence, not proof, that a result is new.

## Papers read

| Paper | Versions | Read | Finding |
|---|---|---|---|
| Barzilai, Shamir, Zamani, *Are Convex Optimization Curves Convex?*, [arXiv:2503.10138](https://arxiv.org/abs/2503.10138) | v1 (13 Mar 2025), v2, v3 (28 Jun 2025, the TMLR camera-ready source), TMLR text (06/2025) | v3 in full from the LaTeX source; v1 and v2 by diffing their document bodies against v3 | Threshold 1.75/L, Huber-type counterexample from x₀ = −1.8, gradient norms monotone for η ≤ 2/L, gradient flow convex. No strong convexity and no size of a violation in any version. |
| Le Duc Hieu, [arXiv:2509.08954](https://arxiv.org/abs/2509.08954) | v1, the only version | In full (LaTeX source) | Relative-inexact impossibility, local smoothness, quadratics convex up to 2/L, a two-step scheme, a no-go inequality. No size of a violation; "strongly" only in a bibliography title. |
| H. Le Duc, L. Liberti, [arXiv:2510.12829](https://arxiv.org/abs/2510.12829) | v3 | Optimization-curve section | Seed results and 14 LLM conjectures; strong convexity only in conjecture 3, about the implicit (backward Euler) method. |
| OpenAI, GPT-5 science report, [arXiv:2511.16072](https://arxiv.org/abs/2511.16072) | v1 | Section 1 (Bubeck) | GPT-5 Pro improved v1's 1/L to 1.5/L; internal models derived 1.75/L. Nothing on the size of a violation. |
| Nagda, Raghavan, Thakurta, [arXiv:2509.18057](https://arxiv.org/abs/2509.18057) | latest | Context of the citation | Cites the paper only as an AI-assisted result later matched without AI. |

Nearest prior result: the inequality that ends the proof of Theorem 1 of Barzilai et al., with L = 1,

−v₁ ≥ (7/8 − h/2)‖g₁ − g₀‖² + ½‖g₂ − (g₀ + g₁)/2‖².

For h > 7/4, dropping the square gives v₁ ≤ (h/2 − 7/8)‖g₁ − g₀‖² ≤ h²(4h − 7)/8 · ‖x₀ − x*‖², a constant (2 + h²)/4
times the exact one (checked in SymPy). The full inequality, with the square, holds with equality on the extremal
one-dimensional data (checked in exact arithmetic at h = 9/5, 19/10 and 39/20). The shortened bound does not: at
h = 9/5 it gives 0.0624220617 against v₁ = 0.0618320611. It loses twice, first the square and then the step
‖g₁ − g₀‖ ≤ h‖x₀ − x*‖.

## Citing works of arXiv:2503.10138

| Index | Result |
|---|---|
| Semantic Scholar Graph API | citationCount 1: Hieu. Hieu: 0 citing works. |
| OpenAlex | Paper W6966804991, cited_by_count 0, `cites:` filter 0. Hieu W7081575489, cited_by_count 0. Recent works of Zamani and Shamir: none on optimization curves. |
| Google Scholar | "Cited by 3": the GPT-5 report, Nagda–Raghavan–Thakurta, Hieu (the list had come back empty on 19 October). Le Duc–Liberti "Cited by 2": two AI surveys, unrelated. |

## Keyword searches

**arXiv API** (search_query, newest first, up to 60 results each), 25 queries. Relevant hits: only arXiv:2503.10138
and arXiv:2509.08954.

1. `all:"optimization curve"`
2. `all:"optimization curves"`
3. `abs:"optimization curve" AND abs:"strongly convex"`
4. `abs:"optimization curve" AND abs:"worst-case"`
5. `abs:"optimization curve" AND abs:"gradient descent"`
6. `all:"convex optimization curves"`
7. `abs:"convexity" AND abs:"gradient descent" AND abs:"1.75"`
8. `all:"1.75/L"`
9. `abs:"curve" AND abs:"convex" AND abs:"gradient descent" AND abs:"step size" AND abs:"concave"`
10. `abs:"second difference" AND abs:"gradient descent"`
11. `abs:"discrete convexity" AND abs:"gradient"`
12. `abs:"function values" AND abs:"convex sequence" AND abs:"gradient"`
13. `abs:"performance estimation" AND abs:"strongly convex" AND abs:"gradient descent" AND abs:"step size"`
14. `abs:"worst-case" AND abs:"strongly convex" AND abs:"gradient descent" AND abs:"convexity" AND abs:"curve"`
15. `abs:"Barzilai" AND abs:"Shamir"`
16. `abs:"decrease" AND abs:"monotone" AND abs:"gradient descent" AND abs:"function value" AND abs:"differences"`
17. `ti:"optimization curve"`
18. `abs:"loss curve" AND abs:"convex" AND abs:"gradient descent" AND abs:"step size"`
19. `abs:"convexity of the loss curve"`
20. `abs:"performance estimation" AND abs:"function values" AND abs:"convexity" AND abs:"gradient descent"`
21. `abs:"sequence of function values" AND abs:"convex" AND abs:"gradient descent" AND abs:"smooth"`
22. `abs:"objective values" AND abs:"convex sequence"`
23. `abs:"consecutive" AND abs:"decrease" AND abs:"gradient descent" AND abs:"smooth convex" AND abs:"step size"`
24. `abs:"optimization curves" AND abs:"heavy ball"`
25. `abs:"optimization curve" AND abs:"proximal"`

**Semantic Scholar search.** Through the API: "optimization curve convex gradient descent", "optimization curve
concave consecutive steps gradient descent", "worst-case convexity violation gradient descent function values";
nothing relevant beyond the two papers. Two more queries were refused by the API's rate limit: "convexity
optimization curve strongly convex" (then run on the website: nothing relevant on the first page) and "convex
optimization curves step size 1.75" (not run).

**Google Scholar.** The exact title; `"optimization curve" "gradient descent" convex "strongly convex"` (since 2025);
`"optimization curve" convex "gradient descent" "worst-case" "step size"`; `"1.75/L" "strongly convex" "gradient
descent"` (since 2025). Relevant hits: the papers above only.

**zbMATH Open.** The paper is not indexed; "convex optimization curves" and the exact title return unrelated work.

**Web search.** Three queries (optimization curve and strong convexity; follow-ups of the paper on the size of a
violation; consecutive concave steps and performance estimation): nothing on the size of a violation.

**GitHub.** Code search, with total hits: `"optimization curve" convex` (190), `"convex optimization curves"` (17),
`"optimization curve" "strongly convex"` (15), `"optimization curve" convexity PEP` (3), `"optimization curve" "worst
case"` (53), `"convex optimization curves" strongly` (14), `"optimization curve" "1.75"` (111), `"h**2*(4*h - 7)"` (0),
`repo:PerformanceEstimation/PEPit curve` (0), `repo:pepflow-lib/PEPFlow curve` (3, all about rate curves in
plots); also the arXiv ids `"2503.10138"` and `"2509.08954"`. The hits are arXiv listing mirrors, copies of the
GPT-5 report and corpus copies of Hieu's paper. Two further queries were refused by the rate limit and not rerun.
PEPit (master, last pushed 21 May 2026; release 0.5.1): no example on optimization curves and no example beyond 0.5.1.
PEPFlow (main, last pushed 28 Sep 2026): its examples (gradient descent, AGM, APPM, DRS, OGM, OGM-G, PGM, Lyapunov)
verify convergence rates only.

## Not covered

The OpenReview forum of the TMLR paper (id TZtpxselK2, reviews and discussion): both the page and the API asked for
a browser verification, which was not bypassed.

## Outcome

Nothing found states the worst-case size of a curve violation, for convex or strongly convex f, or the threshold
h₁(κ). The 18–19 October check left no log in this repository; only its summary in the README records it. The API
queries were rerun on 21 October with stored responses: [literature-check-2026-10-21.md](literature-check-2026-10-21.md).
