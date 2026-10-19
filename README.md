# optcurves: can the gradient-descent optimization curve be concave several steps in a row?

[![Lean](https://github.com/AbdulazizHGA/optcurves/actions/workflows/lean.yml/badge.svg)](https://github.com/AbdulazizHGA/optcurves/actions/workflows/lean.yml)
[![Python](https://github.com/AbdulazizHGA/optcurves/actions/workflows/python.yml/badge.svg)](https://github.com/AbdulazizHGA/optcurves/actions/workflows/python.yml)

Gradient descent with constant step size h/L on an L-smooth convex function produces values f(x₀), f(x₁), f(x₂), …
Barzilai, Shamir and Zamani ([TMLR 2025, arXiv:2503.10138](https://arxiv.org/abs/2503.10138)) showed that this
*optimization curve* is convex for every h ≤ 1.75, and that it can fail to be convex for 1.75 < h < 2. They left open
"whether the optimization curve might be concave for several consecutive steps". This repository answers that
question for gradient descent, and extends the threshold to strongly convex functions.

Write v_k = 2 f(x_k) − f(x_{k−1}) − f(x_{k+1}) (curve violation at step k; the curve is convex at k iff v_k ≤ 0).
*n consecutive violations* means v_k > 0 for n consecutive values of k.

**Tags.** `[exact]` = checked in exact rational arithmetic (SymPy/Fractions, no floating point).
`[Lean-verified]` = checked by the Lean 4 kernel with Mathlib. `[numerical]` = performance-estimation (PEP)
SDP solved in floating point. `[conjecture]` = a formula that fits numerical data, nothing more.
Throughout, L = 1 (rescaling f covers every L), steps are h/L, and "convex" means L-smooth convex.

## Results

### 1. Consecutive violations happen, but only for larger steps

| n consecutive violations | impossible at h = `[exact]` | possible at h = `[exact]` | PEP threshold h_n `[numerical]` |
|---|---|---|---|
| 1 | 1.75 (Barzilai et al.) | any h > 1.75 (Barzilai et al.) | 1.75000 |
| 2 | 1.8634 = 9317/5000 | 1.8635 = 3727/2000 (in ℝ³) | 1.86345 |
| 3 | 1.9 = 19/10 | 1.9001 = 19001/10000 (in ℝ⁴) | 1.90005 |
| 4 | 1.91955 = 38391/20000 | 1.91965 = 38393/20000 (in ℝ⁴) | 1.91960 |
| 5 | 1.93215 = 38643/20000 | 1.93225 = 38645/20000 (in ℝ⁵) | 1.93222 |
| 6 | 1.9411 = 19411/10000 | 1.9412 = 9706/5000 (in ℝ⁵) | 1.94120 |

* **"Impossible at h" `[exact]`** holds for *every* 1-smooth convex function on *any* real inner-product space,
  with or without a minimizer, and for every starting point (so for every n consecutive steps of every run).
  It is a rational certificate: weights w_k ≥ 0 summing to 1 with **w₁v₁ + … + w_nv_n ≤ 0**, written as a
  nonnegative combination of interpolation inequalities plus an explicit sum of squares
  ([certificates/](certificates/)). Exact certificates also exist at h = 3/2, 9/5, 37/20, 93/50 (n = 2) and
  h = 47/25 (n = 3).
* **"Possible at h" `[exact]`** is rational data (points, gradients, values in ℚ^d, minimizer at 0) satisfying every
  interpolation inequality and v₁, …, v_n > 0 exactly ([counterexamples/](counterexamples/)). By the interpolation
  theorem of Taylor, Hendrickx and Glineur (2017, Thm. 4), such data come from a genuine 1-smooth convex
  f: ℝ^d → ℝ with a minimizer; gradient descent from x₀ then visits exactly the stored points. The theorem is the
  only external input.
* So each threshold is pinned to an interval of width 10⁻⁴: e.g. **h₂ ∈ [1.8634, 1.8635]** and
  **h₃ ∈ [1.9, 1.9001]**, in the sense that violations are impossible at the left end and possible at the right
  end. We did *not* prove that the set of bad step sizes is an interval (monotonicity in h); the PEP curves
  in the figure below suggest it.
* `[numerical]` PEP thresholds: PEPit 0.5.1, Clarabel 0.11.1 (SCS 3.3.1 at eps = 1e-9 when Clarabel reports
  "inaccurate"), "no violation" = worst case ≤ 10⁻⁷, bisection on [1.5, 1.9999]. The window x₀, …, x_{n+1} starts
  at a free point. Two normalizations, ‖x₀ − x*‖ ≤ 1 (functions with a minimizer) and ‖∇f(x₀)‖ ≤ 1 (all
  functions), give thresholds within 3 × 10⁻⁵ of each other; the table shows the first. Both lie inside every
  exact bracket (checked in the tests).
* Answer to the open question: yes, the curve can be concave at several consecutive steps, but each additional
  step needs a larger step size; the thresholds approach 2 slowly (h₆ ≈ 1.9412).

![Worst-case min(v₁, …, v_n) against h](figures/consecutive.png)

`python -m optcurves.study consecutive && python -m optcurves.figures`

### 2. Kernel-checked theorems `[Lean-verified]`

[lean/Optcurves/Certificates.lean](lean/Optcurves/Certificates.lean) is generated from the certificates by
`python -m optcurves.lean_gen` and contains, for h = 9/5 and h = 9317/5000 (n = 2) and h = 19/10 (n = 3):

```lean
theorem no_consecutive_n2_h9_5 (x g : ℕ → E) (f : ℕ → ℝ)
    (hgd : ∀ k, x (k + 1) = x k - (9 / 5 : ℝ) • g k)
    (hI : ∀ i j, Interp (x i) (x j) (g i) (g j) (f i) (f j)) :
    ¬ (0 < (2 * f 1 - f 0 - f 2) ∧ 0 < (2 * f 2 - f 1 - f 3))
```

and the stronger weighted form (81/124)·v₁ + (43/124)·v₂ ≤ 0. `E` is any real inner-product space.
All six declarations depend only on `[propext, Classical.choice, Quot.sound]`.

* **Scope.** The hypothesis `hI` is the interpolation inequality of 1-smooth convex functions,
  f_j + ⟨g_j, x_i − x_j⟩ + ½‖g_i − g_j‖² ≤ f_i, between every two iterates. Every 1-smooth convex function satisfies
  it (Nesterov 2004, Thm. 2.1.5), but that bridge lemma is **not** formalized here.
* **Checked by:** the [Lean workflow](.github/workflows/lean.yml) on every push: Lean 4.35.0 with Mathlib
  v4.35.0, `leanprover/lean-action` with `axiom-audit` (all declarations under `Optcurves` within the three
  standard axioms), plus [scripts/check_axioms.py](scripts/check_axioms.py), which checks every `#print axioms`
  line. Also checked on live.lean-lang.org (19 Oct 2026).

### 3. A picture of two consecutive violations `[exact]`

A 1-smooth convex f: ℝ² → ℝ with minimizer x* = 0 on which gradient descent with h = 19/10 breaks convexity at
steps 1 and 2 (v₁ = 0.00513, v₂ = 0.00643, exact rationals in
[counterexamples/n2_h19_10_d2.json](counterexamples/n2_h19_10_d2.json)). The contours are those of the largest
1-smooth convex interpolant of the data, evaluated with its dual formula; it reproduces the stored values.

![Two-dimensional counterexample at h = 19/10](figures/counterexample_2d.png)

`python -m optcurves.counterexample && python -m optcurves.figures`

A second plottable example, three consecutive violations at h = 199/100 in ℝ², is in
[counterexamples/n3_h199_100_d2.json](counterexamples/n3_h199_100_d2.json). `[numerical, heuristic]` A multistart
local search (200 starts) found no strictly feasible one-dimensional example at h = 1.9 (n = 2) or h = 1.99 (n = 3);
this is not a proof that one dimension is impossible.

### 4. Strongly convex functions `[numerical]`, with a conjectured closed form

For L-smooth μ-strongly convex f (κ = L/μ), the threshold for n consecutive violations, PEP with ‖x₀ − x*‖ ≤ 1:

| κ | h₁(κ) | h₂(κ) | h₃(κ) | conjecture for h₁(κ) |
|---|---|---|---|---|
| 1.1 | 1.8551 | 1.9850 | 1.9866 | 1.8551 |
| 1.5 | 1.7929 | 1.9414 | 1.9596 | 1.7929 |
| 2 | 1.7753 | 1.9152 | 1.9477 | 1.7753 |
| 4 | 1.7596 | 1.8867 | 1.9294 | 1.7596 |
| 10 | 1.7534 | 1.8742 | 1.9138 | 1.7534 |
| 100 | 1.7503 | 1.8662 | 1.9027 | 1.7503 |
| ∞ (μ = 0) | 1.7500 | 1.8634 | 1.9001 | 1.75 |

`[conjecture]` **h₁(κ) = (1 + κ) − √((1 + κ)² − (7κ + 1)/2)**, i.e. the smaller root of
2h² − 4(1 + κ)h + 7κ + 1 = 0. It gives 7/4 as κ → ∞, exactly 9/5 at κ = 7/5, 3 − √6/2 at κ = 2, and tends to 2
as κ → 1. It matches all 30 computed values (κ from 1.05 to 10⁴) within 4.3 × 10⁻⁶, the size of the bisection
tolerance; it was found by fitting integer quadratics to the numbers and is **not proved**. Every h_n(κ) we computed
is larger than its convex counterpart and decreases in κ. Some points near κ = 1 needed the SCS fallback; the
number of such solves is stored next to every threshold in [results/strong.json](results/strong.json).

![Thresholds against the condition number](figures/strong_threshold.png)

`python -m optcurves.study strong && python -m optcurves.figures`

## Method

1. **PEP** ([src/optcurves/pep.py](src/optcurves/pep.py)). PEPit's `SmoothConvexFunction` /
   `SmoothStronglyConvexFunction`; the window x₀, …, x_{n+1} with free x₀; PEPit maximizes min(v₁, …, v_n) over
   all functions in the class.
2. **Certificates** ([src/optcurves/certificate.py](src/optcurves/certificate.py)). By minimax, max min_k v_k ≤ 0
   iff some convex combination Σ w_k v_k is ≤ 0, so a single certificate covers all cases. Linear functions make
   every term vanish, which forces S·1 = 0 for the Gram part S (facial reduction). We impose it, maximize the
   smallest eigenvalue of the reduced form in coordinates g_a − g_{n+1}, round to rationals, repair the linear
   equalities exactly, and prove positive definiteness with an exact LDLᵀ, which is also the sum of squares that
   Lean receives.
3. **Counterexamples** ([src/optcurves/counterexample.py](src/optcurves/counterexample.py)). Rounding the PEP
   optimum fails, since many inequalities are tight there. Instead we maximize the smallest slack of all
   interpolation inequalities and violations (an SDP over the Gram matrix, or a small nonlinear program in ℝ²),
   factor, round to rationals and check exactly.
4. **Lean** ([src/optcurves/lean_gen.py](src/optcurves/lean_gen.py)). Each certificate becomes a theorem whose proof
   feeds the interpolation inequalities and the squares to `linarith`.

The PEP layer is cross-checked against the two hand-built SDPs in the tests.

## Limitations

* Exact statements are about the listed step sizes only; nothing is proved for all h in an interval.
* The Lean theorems assume the interpolation inequalities (bridge lemma not formalized).
* The counterexamples rely on the interpolation theorem of Taylor, Hendrickx and Glineur (2017).
* Strongly convex results are numerical, and the closed form for h₁(κ) is a conjecture.
* Only gradient descent with constant steps. Heavy ball, Nesterov and proximal methods are not studied.

## Reproduce

Python ≥ 3.10. Every number above comes from these commands (6 minutes in total on an M1 laptop, mostly
`study strong`). On our machine a full rerun reproduced every JSON file, the Lean file and the figures
byte for byte:

```bash
pip install -e ".[test]"
./reproduce.sh          # study, certificates, counterexamples, Lean file, figures, tests
```

Single steps: `python -m optcurves.study consecutive|strong`, `python -m optcurves.certificate`,
`python -m optcurves.counterexample`, `python -m optcurves.lean_gen`, `python -m optcurves.figures`, `pytest`.
The tests check every certificate and counterexample in exact arithmetic, that tampering breaks them, that the exact
brackets have width ≤ 10⁻⁴ and contain the numerical thresholds, and that the Lean file matches the generator.

Lean (needs elan; downloads the Mathlib cache):

```bash
cd lean && lake exe cache get && lake build
```

## Related work and literature check

* G. Barzilai, O. Shamir, M. Zamani. *Are Convex Optimization Curves Convex?* TMLR 2025,
  [arXiv:2503.10138](https://arxiv.org/abs/2503.10138). Threshold 1.75 for one violation; poses the
  consecutive-steps question; does not treat strongly convex functions.
* Le Duc Hieu. *Convexity of Optimization Curves: Local Sharp Thresholds, Robustness Impossibility, and New
  Counterexamples*, [arXiv:2509.08954](https://arxiv.org/abs/2509.08954), 2025. Convex case only; its
  counterexamples break convexity at a single step.
* H. Le Duc, L. Liberti. *Mathematics with large language models as provers and verifiers*,
  [arXiv:2510.12829](https://arxiv.org/abs/2510.12829), 2025. Uses the 1.75 result as a seed problem; no
  consecutive violations.
* OpenAI, GPT-5 science report, [arXiv:2511.16072](https://arxiv.org/abs/2511.16072): GPT-5 Pro improved the bound
  in the paper's first version from 1/L to 1.5/L.
* Performance estimation: Drori and Teboulle (Math. Program. 2014); Taylor, Hendrickx and Glineur, *Smooth strongly
  convex interpolation and exact worst-case performance of first-order methods* (Math. Program. 2017); PEPit,
  Goujaud et al. (Math. Program. Comput. 2024).

Literature check on 18–19 October 2026: Semantic Scholar citations of arXiv:2503.10138 (only Hieu); OpenAlex
(0 citing works indexed); Google Scholar (shows "cited by 3", but the citing list came back empty; keyword searches
found nothing relevant); arXiv API searches for "optimization curve" with convex, consecutive, strongly convex and
Lean (only the papers above); GitHub search of repositories, code and issues, including PEPit and PEPFlow. We found
no work on consecutive violations or on the strongly convex threshold. Indexes lag, so this is evidence, not proof,
that the results are new.

## ملخص بالعربية

**السؤال:** هل يمكن أن يفقد منحنى التحسين k ↦ f(x_k) لطريقة الانحدار التدريجي بخطوة ثابتة h/L تحدّبه في عدة
خطوات متتالية؟ هذا سؤال مفتوح طرحه Barzilai وShamir وZamani (مجلة TMLR، 2025).

**ما ثبت:**
- **[دقيق]** الانكساران المتتاليان مستحيلان عند h = 1.8634، وممكنان عند h = 1.8635. أي أن العتبة h₂ تقع في
  [1.8634، 1.8635]. وبالمثل h₃ في [1.9، 1.9001]، وh₄ وh₅ وh₆ في فترات عرضها 10⁻⁴.
- **[دقيق]** الاستحالة شهادة كسرية تصح لكل دالة محدبة L-ملساء في أي فضاء ضرب داخلي. والإمكان مثال مضاد ببيانات
  كسرية، يتحول إلى دالة فعلية بمبرهنة الاستيفاء لـ Taylor وHendrickx وGlineur (2017).
- **[Lean]** ست مبرهنات (عند 9/5 و1.8634 لانكسارين، وعند 19/10 لثلاثة) تحققت منها نواة Lean مع Mathlib، ولا
  تعتمد إلا على المسلّمات الثلاث القياسية. فرضياتها متباينات الاستيفاء، ولمّة الجسر إلى الدوال غير مبرهنة.
- **[عددي]** للدوال قوية التحدب: العتبات ترتفع كلما صغر κ = L/μ. و**[حدسية]** للانكسار الواحد:
  h₁(κ) = (1 + κ) − √((1 + κ)² − (7κ + 1)/2)، وتطابق كل القيم المحسوبة بفرق أقل من 4.3 × 10⁻⁶، لكنها غير مبرهنة.

**ما بقي مفتوحاً:** برهان أن مجموعة الخطوات السيئة فترة، وصيغة مغلقة لـ h₂ وما بعدها، وبرهان الحدسية، وطرق
أخرى مثل Heavy Ball وNesterov.

## Citation and license

See [CITATION.cff](CITATION.cff). Code and data: [Apache-2.0](LICENSE).
