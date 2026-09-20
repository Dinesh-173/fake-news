"""Generates publication-quality vector EDA figures directly from empirical analysis JSON artifacts."""

import argparse
import html
import json
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



class SvgChartBuilder:
    """Zero-dependency publication-quality vector SVG chart builder."""

    @staticmethod
    def create_bar_chart(
        title: str,
        subtitle: str,
        categories: list[str],
        values: list[float],
        colors: list[str],
        val_formatter: str = "{:,.0f}",
        x_label: str = "Category",
        y_label: str = "Count",
        width: int = 800,
        height: int = 460,
    ) -> str:
        margin_left = 90
        margin_right = 50
        margin_top = 80
        margin_bottom = 90

        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        max_val = max(values) if values and max(values) > 0 else 1.0
        # Round up max_val to pleasant ceiling
        order = 10 ** (len(str(int(max_val))) - 1) if max_val > 10 else 1
        y_max = ((int(max_val) // order) + 1) * order if max_val > 10 else max_val * 1.2

        n = len(categories)
        bar_step = plot_w / max(n, 1)
        bar_w = bar_step * 0.65

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
            f'style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
            f'<rect width="{width}" height="{height}" fill="#FFFFFF" />',
            f'<text x="{margin_left}" y="35" font-size="20" font-weight="700" fill="#0F172A">{html.escape(title)}</text>',
            f'<text x="{margin_left}" y="56" font-size="12" fill="#64748B">{html.escape(subtitle)}</text>',
        ]

        # Horizontal grid lines and Y-axis labels
        num_grid = 5
        for i in range(num_grid + 1):
            y_val = (y_max / num_grid) * i
            y_pos = margin_top + plot_h - (i / num_grid) * plot_h
            svg.append(
                f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{width - margin_right}" y2="{y_pos:.1f}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="3,3" />'
            )
            svg.append(
                f'<text x="{margin_left - 12}" y="{y_pos + 4:.1f}" font-size="11" fill="#64748B" text-anchor="end">{y_val:,.0f}</text>'
            )

        # Main axes
        svg.append(
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#94A3B8" stroke-width="1.5" />'
        )
        svg.append(
            f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#94A3B8" stroke-width="1.5" />'
        )

        # Bars and X-axis labels
        for idx, (cat, val) in enumerate(zip(categories, values, strict=False)):
            x_bar = margin_left + idx * bar_step + (bar_step - bar_w) / 2
            b_height = (val / y_max) * plot_h if y_max > 0 else 0
            y_bar = margin_top + plot_h - b_height
            color = colors[idx % len(colors)]

            # Bar with subtle rounded corners
            svg.append(
                f'<rect x="{x_bar:.1f}" y="{y_bar:.1f}" width="{bar_w:.1f}" height="{b_height:.1f}" rx="3" fill="{color}" />'
            )

            # Data value text above bar
            svg.append(
                f'<text x="{x_bar + bar_w / 2:.1f}" y="{y_bar - 8:.1f}" font-size="11" font-weight="600" fill="#1E293B" text-anchor="middle">{val_formatter.format(val)}</text>'
            )

            # X-axis label
            wrapped_cat = html.escape(cat)
            svg.append(
                f'<text x="{x_bar + bar_w / 2:.1f}" y="{margin_top + plot_h + 22:.1f}" font-size="11" fill="#334155" text-anchor="middle">{wrapped_cat}</text>'
            )

        # Axis Titles
        svg.append(
            f'<text x="{width / 2}" y="{height - 20}" font-size="12" font-weight="600" fill="#475569" text-anchor="middle">{html.escape(x_label)}</text>'
        )
        svg.append(
            f'<text transform="rotate(-90)" x="-{margin_top + plot_h / 2}" y="24" font-size="12" font-weight="600" fill="#475569" text-anchor="middle">{html.escape(y_label)}</text>'
        )

        svg.append("</svg>")
        return "\n".join(svg)


def generate_figure1_label_distribution(labels_json_path: Path, output_path: Path) -> Path:
    """Figure 1: Original vs TruthLens Normalized Class Distribution."""
    data = json.loads(labels_json_path.read_text(encoding="utf-8"))
    norm_dist = data.get("normalized_distribution", {})

    categories = list(norm_dist.keys())
    values = [float(norm_dist[k]) for k in categories]

    # Academic palette
    palette = ["#0284C7", "#059669", "#DC2626", "#D97706", "#7C3AED", "#64748B"]

    svg_content = SvgChartBuilder.create_bar_chart(
        title="Figure 1: TruthLens Normalized Veracity Class Distribution",
        subtitle=f"Empirical frequency distribution across {data.get('total_records', 0):,} records (Mapping Coverage: {data.get('mapping_coverage', 1.0) * 100:.1f}%)",
        categories=categories,
        values=values,
        colors=palette,
        val_formatter="{:,.0f}",
        x_label="Canonical Veracity State",
        y_label="Number of Statements",
        width=840,
        height=480,
    )
    output_path.write_text(svg_content, encoding="utf-8")
    return output_path


def generate_figure2_text_length_distribution(text_json_path: Path, output_path: Path) -> Path:
    """Figure 2: Statement Character Length Percentiles & Metrics."""
    data = json.loads(text_json_path.read_text(encoding="utf-8"))
    stats = data.get("char_length_stats", {})

    percentiles = ["P5", "P25", "Median (P50)", "Mean", "P75", "P95", "P99"]
    values = [
        float(stats.get("p5", 0)),
        float(stats.get("p25", 0)),
        float(stats.get("median", 0)),
        float(stats.get("mean", 0)),
        float(stats.get("p75", 0)),
        float(stats.get("p95", 0)),
        float(stats.get("p99", 0)),
    ]

    colors = ["#94A3B8", "#64748B", "#2563EB", "#0D9488", "#475569", "#D97706", "#DC2626"]

    svg_content = SvgChartBuilder.create_bar_chart(
        title="Figure 2: Statement Character Length Distribution & Key Percentiles",
        subtitle=f"Computed via Documented Regex Word Tokenizer across {data.get('total_records', 0):,} records (μ = {stats.get('mean', 0):.1f}, σ = {stats.get('std', 0):.1f})",
        categories=percentiles,
        values=values,
        colors=colors,
        val_formatter="{:,.0f} chars",
        x_label="Distribution Statistic",
        y_label="Length in Characters",
        width=880,
        height=480,
    )
    output_path.write_text(svg_content, encoding="utf-8")
    return output_path


def generate_figure3_source_distribution(source_json_path: Path, output_path: Path) -> Path:
    """Figure 3: Top Publisher / Speaker Frequency."""
    data = json.loads(source_json_path.read_text(encoding="utf-8"))
    top_publishers = data.get("top_publishers", [])[:8]

    categories = [p["name"] for p in top_publishers]
    values = [float(p["count"]) for p in top_publishers]

    palette = ["#1E40AF", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE", "#DBEAFE", "#EFF6FF"]

    svg_content = SvgChartBuilder.create_bar_chart(
        title="Figure 3: Prominent Statement Originators & Speaker Concentration",
        subtitle=f"Top 8 Speakers across {data.get('unique_sources', 0):,} distinct sources (Top-5 Concentration Share: {data.get('top_5_share', 0):.1f}%, HHI: {data.get('herfindahl_index', 0)})",
        categories=categories,
        values=values,
        colors=palette,
        val_formatter="{:,.0f}",
        x_label="Speaker / Originator",
        y_label="Record Count",
        width=920,
        height=480,
    )
    output_path.write_text(svg_content, encoding="utf-8")
    return output_path


def generate_figure4_modality_availability(modality_json_path: Path, output_path: Path) -> Path:
    """Figure 4: Cross-Modal Breakdown."""
    data = json.loads(modality_json_path.read_text(encoding="utf-8"))

    categories = ["Text Only", "Image Only", "Text + Img", "Text + Meta", "All Modalities"]
    values = [
        float(data.get("text_only_count", 0)),
        float(data.get("image_only_count", 0)),
        float(data.get("text_and_image_count", 0)),
        float(data.get("text_and_metadata_count", 0)),
        float(data.get("text_image_and_metadata_count", 0)),
    ]

    palette = ["#6366F1", "#EC4899", "#8B5CF6", "#14B8A6", "#10B981"]

    svg_content = SvgChartBuilder.create_bar_chart(
        title="Figure 4: Modality Representation Breakdown",
        subtitle=f"Evaluated across {data.get('total_records', 0):,} records (Local Image Binaries Available: {'YES' if data.get('image_binary_available') else 'NO'})",
        categories=categories,
        values=values,
        colors=palette,
        val_formatter="{:,.0f}",
        x_label="Modality Combination",
        y_label="Record Count",
        width=840,
        height=480,
    )
    output_path.write_text(svg_content, encoding="utf-8")
    return output_path


def generate_figure5_dataset_comparison_matrix(output_path: Path) -> Path:
    """Figure 5: Candidate Benchmark Comparison Table (SVG Matrix)."""
    width = 900
    height = 500
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
        f'style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF" />',
        '<text x="50" y="38" font-size="20" font-weight="700" fill="#0F172A">Figure 5: TruthLens Benchmark Comparison Matrix</text>',
        '<text x="50" y="60" font-size="12" fill="#64748B">Systematic empirical inspection of candidate benchmarks across research dimensions</text>',
    ]

    headers = ["Dimension", "LIAR (ACL '17)", "FakeNewsNet ('19)", "Fakeddit ('20)", "NELA-GT ('22)"]
    rows = [
        ("Acquisition Status", "AVAILABLE LOCALLY", "NOT ACQUIRED", "NOT ACQUIRED", "NOT ACQUIRED"),
        ("Verified Records", "12,836 (Measured)", "N/A (Pending)", "N/A (Pending)", "N/A (Pending)"),
        ("Primary Modality", "Textual Statements", "Articles + Social", "Multimodal Reddit", "News Articles"),
        ("Local Image Assets", "None (TSV Format)", "Pending Hydration", "Pending Download", "Text/HTML Only"),
        ("Granular Labels", "6-Class Veracity", "Binary (Real/Fake)", "2-way & 6-way", "Continuous Scores"),
        ("Speaker Metadata", "Speaker, Party, Job", "Twitter Context", "Subreddit, Author", "Outlet Identity"),
        ("Duplicate Handling", "Exact & Near Jaccard", "Unprocessed", "Unprocessed", "Unprocessed"),
        ("Leakage Protection", "Zero-Leakage Split", "Pending Split", "Pending Split", "Pending Split"),
    ]

    col_x = [50, 240, 410, 580, 750]
    row_h = 40
    y_start = 100

    # Header Row
    svg.append(f'<rect x="40" y="{y_start - 25}" width="{width - 80}" height="32" fill="#F1F5F9" rx="4" />')
    for i, h in enumerate(headers):
        svg.append(
            f'<text x="{col_x[i]}" y="{y_start - 5}" font-size="12" font-weight="700" fill="#1E293B">{h}</text>'
        )

    # Data Rows
    for r_idx, r_vals in enumerate(rows):
        y_pos = y_start + (r_idx + 1) * row_h
        bg_color = "#FFFFFF" if r_idx % 2 == 0 else "#F8FAFC"
        svg.append(
            f'<rect x="40" y="{y_pos - 25}" width="{width - 80}" height="{row_h}" fill="{bg_color}" />'
        )
        svg.append(
            f'<line x1="40" y1="{y_pos + 15}" x2="{width - 40}" y2="{y_pos + 15}" stroke="#E2E8F0" stroke-width="1" />'
        )

        dim_label = r_vals[0]
        svg.append(
            f'<text x="{col_x[0]}" y="{y_pos}" font-size="11" font-weight="600" fill="#334155">{dim_label}</text>'
        )

        for c_idx in range(1, 5):
            val = r_vals[c_idx]
            if "AVAILABLE LOCALLY" in val or "Measured" in val:
                fill_color = "#047857"
                font_weight = "600"
            elif "NOT ACQUIRED" in val or "N/A" in val:
                fill_color = "#94A3B8"
                font_weight = "400"
            else:
                fill_color = "#0F172A"
                font_weight = "400"

            svg.append(
                f'<text x="{col_x[c_idx]}" y="{y_pos}" font-size="11" font-weight="{font_weight}" fill="{fill_color}">{val}</text>'
            )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade SVG EDA figures from profiling run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to run output directory.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating publication-grade EDA figures in {fig_dir}...")

    # Figure 1
    f1 = generate_figure1_label_distribution(run_dir / "label_distribution.json", fig_dir / "fig1_label_distribution.svg")
    print(f"  [OK] {f1.name}")

    # Figure 2
    f2 = generate_figure2_text_length_distribution(run_dir / "text_statistics.json", fig_dir / "fig2_text_length_distribution.svg")
    print(f"  [OK] {f2.name}")

    # Figure 3
    f3 = generate_figure3_source_distribution(run_dir / "source_statistics.json", fig_dir / "fig3_source_party_distribution.svg")
    print(f"  [OK] {f3.name}")

    # Figure 4
    f4 = generate_figure4_modality_availability(run_dir / "modality_statistics.json", fig_dir / "fig4_modality_availability.svg")
    print(f"  [OK] {f4.name}")

    # Figure 5
    f5 = generate_figure5_dataset_comparison_matrix(fig_dir / "fig5_dataset_comparison_matrix.svg")
    print(f"  [OK] {f5.name}")

    print("\nAll 5 publication-ready EDA vector figures successfully generated.")


if __name__ == "__main__":
    main()
