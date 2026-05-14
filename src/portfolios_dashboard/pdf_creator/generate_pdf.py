# Libraries
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER


DTQ_NAVY  = colors.HexColor("#111440")
DTQ_GRAY  = colors.HexColor("#f4f5fa")
DTQ_GREEN = colors.HexColor("#1a7a4a")
DTQ_RED   = colors.HexColor("#c0392b")
DTQ_LIGHT = colors.HexColor("#e5e7ef")


def fig_to_image(fig, width=6.5*inch, height=3*inch):
    """Convert a Plotly figure to a ReportLab Image."""
    img_bytes = fig.to_image(format="png", width=900, height=420, scale=2)
    buf = io.BytesIO(img_bytes)
    return Image(buf, width=width, height=height)


def generate_pdf(
    portfolio_name: str,
    start_date: str,
    end_date: str,
    benchmark_label: str,
    ann_return: float,
    ann_std: float,
    sharpe: float,
    var_95: float,
    es: float,
    md: float,
    ced: float,
    te: float,
    daily_var: float,
    fig_cumulative,
    fig_rolling,
    fig_attribution,
    capm_risk: pd.DataFrame,
    alpha_mean: float,
    beta_mean: float,
    beta_last: float,
    sys_pct: float,
    idio_pct: float,
) -> bytes:
    """Generate a client-ready PDF report."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("title", fontSize=22, textColor=DTQ_NAVY,
                                  fontName="Helvetica-Bold", spaceAfter=4)
    subtitle_style = ParagraphStyle("subtitle", fontSize=11, textColor=colors.HexColor("#888888"),
                                     fontName="Helvetica", spaceAfter=16)
    section_style = ParagraphStyle("section", fontSize=9, textColor=DTQ_NAVY,
                                    fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6,
                                    textTransform="uppercase", letterSpacing=1.2)
    body_style = ParagraphStyle("body", fontSize=10, textColor=colors.HexColor("#444444"),
                                 fontName="Helvetica", leading=16, spaceAfter=8)
    summary_style = ParagraphStyle("summary", fontSize=11, textColor=colors.HexColor("#27500A"),
                                    fontName="Helvetica", leading=17, spaceAfter=12,
                                    backColor=colors.HexColor("#EAF3DE"),
                                    borderPad=10, borderRadius=6)

    story = []

    # ── PAGE 1: Cover ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Portfolio Attribution Report", title_style))
    story.append(Paragraph(f"{portfolio_name} &nbsp;·&nbsp; {start_date} – {end_date} &nbsp;·&nbsp; Benchmark: {benchmark_label}", subtitle_style))

    # Summary sentence
    alpha_word = "positive" if alpha_mean > 0 else "negative"
    summary_text = (
        f"This portfolio earned <b>{ann_return:.2f}% annualized</b> — of which "
        f"<b>{sys_pct*100:.1f}% came from broad market exposure</b> and "
        f"<b>{idio_pct*100:.1f}% from strategy-specific decisions.</b> "
        f"Alpha generation has been <b>{alpha_word}</b> on average "
        f"({alpha_mean*100:.3f}% daily), with a current market sensitivity "
        f"(beta) of <b>{beta_last:.2f}</b> vs. a historical average of <b>{beta_mean:.2f}</b>."
    )
    story.append(Paragraph(summary_text, summary_style))

    # ── PAGE 2: Cumulative returns + risk metrics ─────────────────────────────
    story.append(Paragraph("Cumulative Returns", section_style))
    story.append(fig_to_image(fig_cumulative, height=2.8*inch))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("Risk & Performance Metrics", section_style))

    metrics_data = [
        ["Annualized Return", f"{ann_return:.2f}%",
         "Sharpe Ratio", f"{sharpe:.4f}x",
         "Tracking Error", f"{te:.4f}%"],
        ["Annualized Volatility", f"{ann_std:.2f}%",
         "Value at Risk (95%)", f"{var_95:.2%}",
         "Max Drawdown", f"{md:.2%}"],
        ["Daily Variance", f"{daily_var:.6f}",
         "Expected Shortfall", f"{es:.2%}",
         "Cond. Exp. Drawdown", f"{ced:.2%}"],
    ]

    col_w = (6.5*inch) / 6
    metric_table = Table(metrics_data, colWidths=[col_w*1.4, col_w*0.9, col_w*1.4, col_w*0.9, col_w*1.4, col_w*0.9])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DTQ_GRAY),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("FONTNAME", (5, 0), (5, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), DTQ_NAVY),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#888888")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#888888")),
        ("TEXTCOLOR", (4, 0), (4, -1), colors.HexColor("#888888")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [DTQ_GRAY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, DTQ_LIGHT),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(metric_table)

    # ── PAGE 3: Rolling beta + decomposition ──────────────────────────────────
    story.append(Paragraph("Market Sensitivity Over Time", section_style))
    story.append(fig_to_image(fig_rolling, height=2.5*inch))

    story.append(Paragraph("Return Decomposition", section_style))

    display = capm_risk.copy()
    display.index = display.index.str.replace("_", " ").str.title()
    display.index = display.index.str.replace("Total Variance", "Total Portfolio")\
                                   .str.replace("Systematic Variance", "Market Exposure")\
                                   .str.replace("Idio Variance", "Strategy-Specific")

    decomp_data = [["Source", "Variance", "Share", "Ann. Volatility"]]
    for idx, row in display.iterrows():
        decomp_data.append([
            idx,
            f"{row['variance']:.6f}",
            f"{row['percentage']:.1%}",
            f"{row['ann_volatility']:.2%}",
        ])

    decomp_table = Table(decomp_data, colWidths=[2.5*inch, 1.5*inch, 1.2*inch, 1.3*inch])
    decomp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DTQ_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), DTQ_NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, DTQ_GRAY]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, DTQ_LIGHT),
    ]))
    story.append(decomp_table)

    # ── PAGE 4: Attribution + interpretation ──────────────────────────────────
    story.append(Paragraph("What Drove the Return?", section_style))
    story.append(fig_to_image(fig_attribution, height=2.8*inch))

    story.append(Paragraph("Interpretation", section_style))

    # Alpha
    if alpha_mean > 0:
        alpha_msg = f"The portfolio has generated <b>positive alpha</b> on average ({alpha_mean*100:.3f}% daily), suggesting the strategy adds value beyond market exposure."
    else:
        alpha_msg = f"The portfolio shows <b>negative alpha</b> on average ({alpha_mean*100:.3f}% daily), indicating returns have lagged what market exposure alone would predict."

    # Beta
    if beta_mean < 0.8:
        beta_msg = f"With an average beta of <b>{beta_mean:.2f}</b>, the portfolio is <b>defensive</b> — it moves less than the market and may offer downside protection."
    elif beta_mean <= 1.2:
        beta_msg = f"With an average beta of <b>{beta_mean:.2f}</b>, the portfolio tracks the market <b>closely</b> with similar risk exposure."
    else:
        beta_msg = f"With an average beta of <b>{beta_mean:.2f}</b>, the portfolio is <b>aggressive</b> — it amplifies market movements and carries higher systematic risk."

    # Trend
    if abs(beta_last - beta_mean) > 0.2:
        trend_msg = f"Recently, market sensitivity has shifted to <b>{beta_last:.2f}</b>, notably {'higher' if beta_last > beta_mean else 'lower'} than its historical average — a meaningful change in risk posture."
    else:
        trend_msg = f"Recent market sensitivity (<b>{beta_last:.2f}</b>) is consistent with the historical average — no significant change in risk posture detected."

    risk_msg = f"<b>{sys_pct*100:.1f}%</b> of total variance is explained by market exposure, while <b>{idio_pct*100:.1f}%</b> is specific to this strategy."

    for msg in [alpha_msg, beta_msg, trend_msg, risk_msg]:
        story.append(Paragraph(msg, body_style))

    # Footer
    story.append(Spacer(1, 0.3*inch))
    footer_style = ParagraphStyle("footer", fontSize=8, textColor=colors.HexColor("#aaaaaa"),
                                   fontName="Helvetica", alignment=TA_CENTER)
    story.append(Paragraph(
        "Alpha version — for academic and research purposes only. "
        "Not financial advice. Results should not be used for investment decisions.",
        footer_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()