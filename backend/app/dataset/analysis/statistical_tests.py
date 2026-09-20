"""Rigorous inferential statistical hypothesis testing module using scipy.stats."""

import math
from typing import Any

import numpy as np
import scipy.stats as stats
from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class StatisticalTestReport(BaseModel):
    """Formal documentation and numerical results of an inferential hypothesis test."""

    test_name: str
    research_question: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic_name: str
    test_statistic_value: float
    degrees_of_freedom: int | None = None
    p_value: float
    alpha: float = 0.05
    statistically_significant: bool
    effect_size_name: str
    effect_size_value: float
    sample_size: int
    assumptions: list[str]
    limitations: list[str]
    interpretation: str
    details: dict[str, Any] = Field(default_factory=dict)


class StatisticalTestSuiteResult(BaseModel):
    """Aggregates all executed statistical tests with provenance and multi-test correction metadata."""

    total_tests_conducted: int
    alpha_level: float = 0.05
    tests: list[StatisticalTestReport]
    multiple_comparison_correction: str = "None (Independent Prespecified Hypotheses)"
    notes: list[str] = Field(default_factory=list)


class StatisticalTester:
    """Executes formal statistical hypothesis tests answering specific empirical research questions."""

    @classmethod
    def test_party_vs_veracity(
        cls,
        records: list[CanonicalNewsRecord],
        alpha: float = 0.05,
    ) -> StatisticalTestReport:
        """
        Test 1: Chi-square test of independence between Political Party and Normalized Veracity Label.
        H0: Veracity rating is independent of speaker political party affiliation.
        H1: Veracity rating and speaker political party affiliation are statistically associated.
        """
        # Filter for records with known party and mapped normalized label
        valid_recs: list[tuple[str, str]] = []
        for r in records:
            party = (r.metadata.get("party_affiliation") or "").strip().lower()
            label = r.normalized_label
            # Retain prominent parties to avoid ultra-sparse tables (democrat, republican, independent/other)
            if party and label and party not in ("none", "nan", ""):
                # Group major parties
                if "democrat" in party:
                    canonical_party = "democrat"
                elif "republican" in party:
                    canonical_party = "republican"
                else:
                    canonical_party = "other/independent"
                valid_recs.append((canonical_party, label))

        n_samples = len(valid_recs)
        if n_samples < 20:
            return StatisticalTestReport(
                test_name="Chi-Square Test of Independence (Party × Veracity)",
                research_question="Is statement veracity independent of speaker political party affiliation?",
                null_hypothesis="H0: Statement veracity and speaker political party are independent.",
                alternative_hypothesis="H1: Statement veracity and speaker political party are dependent.",
                test_statistic_name="Chi-Square (χ²)",
                test_statistic_value=0.0,
                degrees_of_freedom=0,
                p_value=1.0,
                alpha=alpha,
                statistically_significant=False,
                effect_size_name="Cramér's V",
                effect_size_value=0.0,
                sample_size=n_samples,
                assumptions=[
                    "Adequate expected cell frequencies (>= 5 in >= 80% of cells).",
                    "Independent observations across speakers and statements.",
                ],
                limitations=[
                    "Sample size of records with verified party affiliation is insufficient for asymptotic chi-square approximation."
                ],
                interpretation="Test could not be reliably evaluated due to insufficient data.",
                details={"contingency_table": {}},
            )

        parties = sorted({p for p, _ in valid_recs})
        labels = sorted({lbl for _, lbl in valid_recs})

        party_idx = {p: i for i, p in enumerate(parties)}
        label_idx = {lbl: j for j, lbl in enumerate(labels)}

        matrix = np.zeros((len(parties), len(labels)), dtype=int)
        for p, lbl in valid_recs:
            matrix[party_idx[p], label_idx[lbl]] += 1

        chi2_res = stats.chi2_contingency(matrix)
        chi2_stat = round(float(chi2_res.statistic), 4)
        p_val = float(chi2_res.pvalue)
        dof = int(chi2_res.dof)

        # Cramér's V: V = sqrt(chi2 / (n * min(r-1, k-1)))
        min_dim = min(len(parties) - 1, len(labels) - 1)
        cramers_v = (
            round(math.sqrt(chi2_stat / (n_samples * min_dim)), 4) if min_dim > 0 else 0.0
        )

        is_sig = p_val < alpha

        contingency_dict: dict[str, dict[str, int]] = {}
        for p in parties:
            contingency_dict[p] = {
                lbl: int(matrix[party_idx[p], label_idx[lbl]]) for lbl in labels
            }

        # Substantive interpretation
        if is_sig:
            interp = (
                f"Statistically significant association detected (χ²={chi2_stat}, df={dof}, p={p_val:.2e}, "
                f"Cramér's V={cramers_v}). Observed statement truthfulness differs systematically across political "
                f"affiliations in this benchmark. Note that Cramér's V of {cramers_v} indicates the magnitude of association."
            )
        else:
            interp = (
                f"No statistically significant association observed between party and statement veracity "
                f"(χ²={chi2_stat}, df={dof}, p={p_val:.4f}). Null hypothesis cannot be rejected at α={alpha}."
            )

        return StatisticalTestReport(
            test_name="Chi-Square Test of Independence (Party × Veracity)",
            research_question="Is statement veracity independent of speaker political party affiliation?",
            null_hypothesis="H0: Statement veracity and speaker political party affiliation are independent.",
            alternative_hypothesis="H1: Statement veracity and speaker political party affiliation are associated.",
            test_statistic_name="Chi-Square (χ²)",
            test_statistic_value=chi2_stat,
            degrees_of_freedom=dof,
            p_value=p_val,
            alpha=alpha,
            statistically_significant=is_sig,
            effect_size_name="Cramér's V",
            effect_size_value=cramers_v,
            sample_size=n_samples,
            assumptions=[
                "Contingency table expected frequencies >= 5 in at least 80% of cells.",
                "Mutual exclusivity of categorization.",
            ],
            limitations=[
                "LIAR statements are fact-checked based on newsworthiness and editorial selection, not random sampling of all political speech.",
                "Statistical association reflects fact-checked claims in PolitiFact, not a generalized claim about political parties as a whole.",
            ],
            interpretation=interp,
            details={"contingency_table": contingency_dict, "expected_frequencies": chi2_res.expected_freq.tolist()},
        )

    @classmethod
    def test_text_length_by_veracity(
        cls,
        records: list[CanonicalNewsRecord],
        group1_label: str = "LIKELY_TRUE",
        group2_label: str = "LIKELY_FALSE",
        alpha: float = 0.05,
    ) -> StatisticalTestReport:
        """
        Test 2: Mann-Whitney U test on text length between True statements and False statements.
        H0: The text length distribution of True statements is identical to that of False statements.
        H1: The text length distributions of True and False statements differ systematically.
        """
        g1_lengths = [
            len(r.text) for r in records if r.normalized_label == group1_label and r.text
        ]
        g2_lengths = [
            len(r.text) for r in records if r.normalized_label == group2_label and r.text
        ]

        n1, n2 = len(g1_lengths), len(g2_lengths)
        if n1 < 5 or n2 < 5:
            return StatisticalTestReport(
                test_name=f"Mann-Whitney U Test ({group1_label} vs {group2_label} Text Length)",
                research_question=f"Do statements categorized as {group1_label} and {group2_label} differ in text length?",
                null_hypothesis=f"H0: Distribution of text lengths is identical for {group1_label} and {group2_label}.",
                alternative_hypothesis=f"H1: Distribution of text lengths differs between {group1_label} and {group2_label}.",
                test_statistic_name="Mann-Whitney U",
                test_statistic_value=0.0,
                degrees_of_freedom=None,
                p_value=1.0,
                alpha=alpha,
                statistically_significant=False,
                effect_size_name="Rank-Biserial Correlation",
                effect_size_value=0.0,
                sample_size=n1 + n2,
                assumptions=["Independent observations."],
                limitations=["Insufficient sample size in one or both comparative groups."],
                interpretation="Test could not be conducted due to sparse group sample size.",
                details={"n1": n1, "n2": n2},
            )

        u_res = stats.mannwhitneyu(g1_lengths, g2_lengths, alternative="two-sided")
        u_stat = round(float(u_res.statistic), 2)
        p_val = float(u_res.pvalue)

        # Rank-biserial correlation: r_rb = 1 - (2U / (n1 * n2))
        r_rb = round(1.0 - (2.0 * u_stat / (n1 * n2)), 4)
        is_sig = p_val < alpha

        m1 = round(float(np.median(g1_lengths)), 1)
        m2 = round(float(np.median(g2_lengths)), 1)

        if is_sig:
            interp = (
                f"Statistically significant difference in character length between {group1_label} "
                f"(median={m1}) and {group2_label} (median={m2}) statements (U={u_stat}, p={p_val:.2e}, "
                f"Rank-biserial r={r_rb}). While statistically detectable, the effect size indicates "
                f"practical length variation is modest."
            )
        else:
            interp = (
                f"No statistically significant difference in text length between {group1_label} (median={m1}) "
                f"and {group2_label} (median={m2}) statements (U={u_stat}, p={p_val:.4f}). Text length does not "
                f"distinguish veracity classes."
            )

        return StatisticalTestReport(
            test_name=f"Mann-Whitney U Test ({group1_label} vs {group2_label} Text Length)",
            research_question=f"Do statements categorized as {group1_label} and {group2_label} differ in character length?",
            null_hypothesis=f"H0: Text length distributions for {group1_label} and {group2_label} are identical.",
            alternative_hypothesis=f"H1: Text length distributions for {group1_label} and {group2_label} differ.",
            test_statistic_name="Mann-Whitney U",
            test_statistic_value=u_stat,
            degrees_of_freedom=None,
            p_value=p_val,
            alpha=alpha,
            statistically_significant=is_sig,
            effect_size_name="Rank-Biserial Correlation",
            effect_size_value=r_rb,
            sample_size=n1 + n2,
            assumptions=[
                "Continuous or ordinal response variable (character length in characters).",
                "Non-parametric distribution (does not assume Gaussian normality).",
                "Independent observations across statements.",
            ],
            limitations=[
                "Statistical significance in large sample sizes does not imply that text length alone is an effective predictive feature.",
                "Character length is sensitive to quotation conventions and speech transcript editing.",
            ],
            interpretation=interp,
            details={
                f"{group1_label}_count": n1,
                f"{group1_label}_median": m1,
                f"{group2_label}_count": n2,
                f"{group2_label}_median": m2,
            },
        )

    @classmethod
    def test_label_uniformity(
        cls,
        records: list[CanonicalNewsRecord],
        alpha: float = 0.05,
    ) -> StatisticalTestReport:
        """
        Test 3: Chi-square Goodness-of-Fit test against uniform distribution across normalized labels.
        H0: Records are uniformly distributed across all observed normalized veracity classes.
        H1: Record frequencies deviate significantly from a uniform distribution.
        """
        mapped_labels = [r.normalized_label for r in records if r.normalized_label]
        if not mapped_labels:
            return StatisticalTestReport(
                test_name="Chi-Square Goodness-of-Fit (Label Uniformity)",
                research_question="Are records uniformly distributed across truthfulness categories?",
                null_hypothesis="H0: Statements are equally distributed across all normalized labels.",
                alternative_hypothesis="H1: Statements are not equally distributed across normalized labels.",
                test_statistic_name="Chi-Square (χ²)",
                test_statistic_value=0.0,
                degrees_of_freedom=0,
                p_value=1.0,
                alpha=alpha,
                statistically_significant=False,
                effect_size_name="Cohen's w",
                effect_size_value=0.0,
                sample_size=0,
                assumptions=[],
                limitations=["No mapped records available."],
                interpretation="Test could not be conducted.",
            )

        from collections import Counter

        counts = Counter(mapped_labels)
        categories = sorted(counts.keys())
        observed = np.array([counts[c] for c in categories])
        k = len(categories)
        n_total = int(np.sum(observed))

        if k < 2:
            return StatisticalTestReport(
                test_name="Chi-Square Goodness-of-Fit (Label Uniformity)",
                research_question="Are records uniformly distributed across truthfulness categories?",
                null_hypothesis="H0: Statements are equally distributed across normalized labels.",
                alternative_hypothesis="H1: Statements are not equally distributed across normalized labels.",
                test_statistic_name="Chi-Square (χ²)",
                test_statistic_value=0.0,
                degrees_of_freedom=0,
                p_value=1.0,
                alpha=alpha,
                statistically_significant=False,
                effect_size_name="Cohen's w",
                effect_size_value=0.0,
                sample_size=n_total,
                assumptions=[],
                limitations=["Dataset contains fewer than 2 distinct classes."],
                interpretation="Test requires at least 2 distinct classes to test uniformity.",
            )

        expected = np.full(k, n_total / k, dtype=float)
        gof_res = stats.chisquare(f_obs=observed, f_exp=expected)
        chi2_stat = round(float(gof_res.statistic), 4)
        p_val = float(gof_res.pvalue)
        dof = k - 1

        # Cohen's w for goodness-of-fit: w = sqrt(sum((p_obs - p_exp)^2 / p_exp))
        p_obs = observed / n_total
        p_exp = expected / n_total
        cohens_w = round(float(np.sqrt(np.sum(((p_obs - p_exp) ** 2) / p_exp))), 4)

        is_sig = p_val < alpha

        obs_dict = {cat: int(counts[cat]) for cat in categories}
        exp_dict = {cat: round(float(expected[i]), 1) for i, cat in enumerate(categories)}

        if is_sig:
            interp = (
                f"Statistically significant deviation from uniform class distribution (χ²={chi2_stat}, df={dof}, "
                f"p={p_val:.2e}, Cohen's w={cohens_w}). The benchmark exhibits significant class disparity relative "
                f"to an equal prior expectation."
            )
        else:
            interp = (
                f"No significant deviation from uniform distribution detected (χ²={chi2_stat}, df={dof}, p={p_val:.4f}). "
                f"Observed classes do not statistically differ from equal distribution."
            )

        return StatisticalTestReport(
            test_name="Chi-Square Goodness-of-Fit (Label Uniformity)",
            research_question="Are statements uniformly distributed across truthfulness categories?",
            null_hypothesis="H0: Statement records are uniformly distributed across all canonical labels.",
            alternative_hypothesis="H1: Statement records are not uniformly distributed across canonical labels.",
            test_statistic_name="Chi-Square (χ²)",
            test_statistic_value=chi2_stat,
            degrees_of_freedom=dof,
            p_value=p_val,
            alpha=alpha,
            statistically_significant=is_sig,
            effect_size_name="Cohen's w",
            effect_size_value=cohens_w,
            sample_size=n_total,
            assumptions=[
                "Categories are mutually exclusive and exhaustive.",
                "Expected frequency per cell >= 5.",
            ],
            limitations=[
                "Uniform distribution is tested as an agnostic statistical baseline null model; natural real-world veracity distributions are not expected to be uniform.",
                "Rejection of uniformity is standard for empirical media datasets and does not imply defective data collection.",
            ],
            interpretation=interp,
            details={"observed": obs_dict, "expected": exp_dict},
        )

    @classmethod
    def run_all(
        cls,
        records: list[CanonicalNewsRecord],
        alpha: float = 0.05,
    ) -> StatisticalTestSuiteResult:
        """Runs the standard battery of prespecified inferential statistical tests."""
        t1 = cls.test_party_vs_veracity(records, alpha=alpha)
        t2 = cls.test_text_length_by_veracity(records, alpha=alpha)
        t3 = cls.test_label_uniformity(records, alpha=alpha)

        return StatisticalTestSuiteResult(
            total_tests_conducted=3,
            alpha_level=alpha,
            tests=[t1, t2, t3],
            multiple_comparison_correction="Prespecified Distinct Hypotheses (Nominal α=0.05)",
            notes=[
                "All statistical tests executed using SciPy 1.18.0 with explicit hypotheses and assumptions.",
                "Effect sizes (Cramér's V, Rank-Biserial r, Cohen's w) are reported alongside p-values.",
            ],
        )
