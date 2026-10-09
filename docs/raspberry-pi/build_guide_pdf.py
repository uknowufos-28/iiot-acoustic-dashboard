"""Build the Raspberry Pi setup PDF from its companion Markdown guide."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import os
import tempfile
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / "README.md"
OUTPUT = SOURCE.with_name("Raspberry_Pi_Hybrid_Model_Setup_Guide.pdf")

INK = colors.HexColor("#17313A")
MUTED = colors.HexColor("#60747A")
TEAL = colors.HexColor("#176B64")
PALE = colors.HexColor("#EFF6F4")
LINE = colors.HexColor("#DCE5E3")
AMBER = colors.HexColor("#FFF7E7")
CODE_BG = colors.HexColor("#F2F5F4")


class CodePanel(Flowable):
    def __init__(self, source: str, style: ParagraphStyle):
        super().__init__()
        self.code = Preformatted(
            textwrap.dedent(source).strip("\n"),
            style,
            maxLineLength=94,
            splitChars=" /,-",
        )
        self.width = 0
        self.height = 0

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        _, content_height = self.code.wrap(avail_width - 16, avail_height)
        self.height = content_height + 16
        return self.width, self.height

    def draw(self) -> None:
        self.canv.setFillColor(CODE_BG)
        self.canv.setStrokeColor(LINE)
        self.canv.roundRect(0, 0, self.width, self.height, 5, stroke=1, fill=1)
        self.code.drawOn(self.canv, 8, 8)


class ArchitectureDiagram(Flowable):
    def __init__(self):
        super().__init__()
        self.height = 96

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        self.width = avail_width
        return self.width, self.height

    def draw(self) -> None:
        stages = [
            ("INPUT", ["Sensor driver", "or CSV upload"]),
            ("PREPROCESS", ["Validate signal", "16 kHz · 3 s", "mel features"]),
            ("HYBRID MODEL", ["CNN embedding", "32 values", "fused SVM"]),
            ("PI API", ["Authenticate", "infer · store", "latest result"]),
            ("DASHBOARD", ["GitHub Pages", "class · scores", "waveform"]),
        ]
        margin = 2
        gap = 11
        box_width = (self.width - 2 * margin - gap * (len(stages) - 1)) / len(stages)
        box_height = 59
        y = 22
        for index, (title, details) in enumerate(stages):
            x = margin + index * (box_width + gap)
            self.canv.setFillColor(PALE if index != 2 else colors.HexColor("#E3F0ED"))
            self.canv.setStrokeColor(colors.HexColor("#BFD5D0"))
            self.canv.roundRect(x, y, box_width, box_height, 5, stroke=1, fill=1)
            self.canv.setFillColor(TEAL)
            self.canv.setFont("Helvetica-Bold", 6.6)
            self.canv.drawCentredString(x + box_width / 2, y + box_height - 13, title)
            self.canv.setFillColor(INK)
            self.canv.setFont("Helvetica", 6.6)
            for line_index, detail in enumerate(details):
                self.canv.drawCentredString(
                    x + box_width / 2,
                    y + box_height - 26 - line_index * 9,
                    detail,
                )
            if index < len(stages) - 1:
                x1 = x + box_width + 1
                x2 = x + box_width + gap - 2
                mid_y = y + box_height / 2
                self.canv.setStrokeColor(TEAL)
                self.canv.setFillColor(TEAL)
                self.canv.setLineWidth(1)
                self.canv.line(x1, mid_y, x2, mid_y)
                self.canv.line(x2 - 3, mid_y + 2.5, x2, mid_y)
                self.canv.line(x2 - 3, mid_y - 2.5, x2, mid_y)
        self.canv.setFillColor(MUTED)
        self.canv.setFont("Helvetica-Oblique", 7)
        self.canv.drawCentredString(
            self.width / 2,
            8,
            "Remote browser-to-Pi traffic must use HTTPS; keep the model service behind authentication.",
        )


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "GuideTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "GuideSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=11,
        ),
        "h1": ParagraphStyle(
            "GuideH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=INK,
            spaceBefore=17,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "GuideH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=TEAL,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "GuideH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "GuideBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=13,
            textColor=INK,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "list": ParagraphStyle(
            "GuideList",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=12.3,
            textColor=INK,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "GuideQuote",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=13,
            textColor=colors.HexColor("#704B11"),
            backColor=AMBER,
            borderColor=colors.HexColor("#ECD29A"),
            borderWidth=0.5,
            borderPadding=7,
            leftIndent=7,
            rightIndent=7,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "GuideCode",
            fontName="Courier",
            fontSize=6.7,
            leading=8.7,
            textColor=colors.HexColor("#203840"),
        ),
        "table_header": ParagraphStyle(
            "GuideTableHeader",
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.2,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "GuideTableCell",
            fontName="Helvetica",
            fontSize=7.3,
            leading=9.2,
            textColor=INK,
        ),
        "cover_note": ParagraphStyle(
            "GuideCoverNote",
            fontName="Helvetica",
            fontSize=9.4,
            leading=15,
            textColor=INK,
            alignment=TA_CENTER,
            borderColor=LINE,
            borderWidth=0.6,
            borderPadding=12,
            backColor=colors.white,
            spaceBefore=6,
            spaceAfter=10,
        ),
    }


def inline_markup(value: str) -> str:
    value = escape(value, quote=False)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#176B64">\1</link>',
        value,
    )
    value = re.sub(
        r"&lt;(https?://[^&]+?)&gt;",
        r'<link href="\1" color="#176B64">\1</link>',
        value,
    )
    value = re.sub(r"`([^`]+)`", r'<font name="Courier" size="7.4">\1</font>', value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value


def parse_table(lines: list[str], styles: dict) -> Table:
    rows = []
    for row_index, raw in enumerate(lines):
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if row_index == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        style = styles["table_header"] if row_index == 0 else styles["table_cell"]
        rows.append([Paragraph(inline_markup(cell), style) for cell in cells])
    widths = [1.25 * mm * 0 + 92 * mm, None] if len(rows[0]) == 2 else None
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAF9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_story(markdown: str, styles: dict) -> list:
    lines = markdown.splitlines()
    story = []
    index = 0
    first_title = True
    first_flowchart = True

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip().lower()
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            if language == "mermaid":
                if first_flowchart:
                    story.extend([Spacer(1, 4), ArchitectureDiagram(), Spacer(1, 8)])
                    first_flowchart = False
                continue
            story.append(CodePanel("\n".join(block), styles["code"]))
            story.append(Spacer(1, 5))
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2)
            if level == 1 and first_title:
                story.append(Paragraph(inline_markup(text), styles["title"]))
                story.append(Paragraph(
                    "Practical setup, secure connection, CSV testing, and the path to live sensor data",
                    styles["subtitle"],
                ))
                story.append(ArchitectureDiagram())
                story.append(Spacer(1, 6))
                first_title = False
                first_flowchart = False
            else:
                style = styles[f"h{level}"]
                story.append(Paragraph(inline_markup(text), style))
            index += 1
            continue

        if stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.7, color=LINE, spaceBefore=4, spaceAfter=7))
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and "|" in lines[index + 1]:
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(parse_table(table_lines, styles))
            story.append(Spacer(1, 6))
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip()[1:].strip())
                index += 1
            story.append(Paragraph(inline_markup(" ".join(quote_lines)), styles["quote"]))
            continue

        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if unordered or ordered:
            list_items = []
            ordered_list = ordered is not None
            while index < len(lines):
                current = lines[index].strip()
                match = re.match(r"^(\d+)\.\s+(.+)$", current) if ordered_list else re.match(r"^[-*]\s+(.+)$", current)
                if not match:
                    break
                marker = f"{match.group(1)}." if ordered_list else "•"
                item_text = match.group(2) if ordered_list else match.group(1)
                list_items.append(Paragraph(
                    f"<b>{marker}</b> {inline_markup(item_text)}",
                    styles["list"],
                ))
                index += 1
            story.extend(list_items)
            story.append(Spacer(1, 3))
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith(("```", "#", ">", "|", "---"))
                or re.match(r"^[-*]\s+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph = Paragraph(inline_markup(" ".join(paragraph_lines)), styles["body"])
        if first_title:
            story.append(KeepTogether([paragraph, Spacer(1, 3)]))
        else:
            story.append(paragraph)

    return story


def draw_page_chrome(canvas, document) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setFillColor(TEAL)
    canvas.rect(0, page_height - 3, page_width, 3, stroke=0, fill=1)
    if canvas.getPageNumber() > 1:
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(document.leftMargin, page_height - 11 * mm, "IIoT MOTOR ACOUSTIC MONITOR")
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, 13 * mm, page_width - document.rightMargin, 13 * mm)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(document.leftMargin, 8.5 * mm, "Raspberry Pi hybrid model setup · CSV path tested; sensor hardware pending")
    canvas.drawRightString(page_width - document.rightMargin, 8.5 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Guide source not found: {SOURCE}")
    styles = make_styles()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".hybrid-guide-",
        suffix=".pdf",
        dir=OUTPUT.parent,
    )
    os.close(descriptor)
    temporary_output = Path(temporary_name)
    document = SimpleDocTemplate(
        str(temporary_output),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=18 * mm,
        bottomMargin=19 * mm,
        title="Raspberry Pi Hybrid Model Setup Guide",
        author="IIoT Motor Acoustic Monitor",
        subject="GitHub Pages dashboard and Raspberry Pi CNN-SVM integration",
    )
    story = build_story(SOURCE.read_text(encoding="utf-8"), styles)
    try:
        document.build(story, onFirstPage=draw_page_chrome, onLaterPages=draw_page_chrome)
        temporary_output.replace(OUTPUT)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
