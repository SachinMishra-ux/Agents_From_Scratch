---
name: PPT Design and Theme Formatting
description: Instructions and guidelines to plan slide blueprints first, design executive card-based typography, alternate dynamic slide layouts (image_left, image_right, table, cards, title, ending), generate targeted Gemini illustrations, and handle iterative Human-in-the-Loop feedback enhancements.
---

# PPT Design and Theme Formatting Skill (Executive Edition)

This skill defines the complete visual design system, **mandatory presentation planning workflow**, **dynamic multi-layout rules** (`title`, `image_right`, `image_left`, `table`, `cards`, `ending`), **card-based typography**, and iterative **Human-in-the-Loop (HITL)** enhancement protocol for generating executive-grade PowerPoint (`.pptx`) presentations.

---

## 1. Mandatory Step-by-Step Workflow (Plan First -> Generate Visuals -> Compile -> HITL Review)

You MUST execute tools in this exact order:

1. **Step 1 — Plan Presentation Blueprint (`plan_presentation_blueprint`)**:
   * **ALWAYS plan the entire presentation before generating any images!**
   * Decide the exact slide count, narrative flow, section tags, and **varied layout types** (`title`, `image_right`, `image_left`, `table`, `cards`, `ending`).
   * **Never place every image on the right side!** Deliberately alternate between `image_right`, `image_left`, `table` (which renders a styled comparison table + visual infographic), and `cards`.
   * Specify which slides require a Gemini illustration (`needs_image=True`) and write a tailored visual prompt (`visual_prompt`) for each planned visual slide.

2. **Step 2 — Generate Planned Slide Images (`generate_slide_image`)**:
   * Strictly follow the blueprint from Step 1 and call `generate_slide_image` (`gemini-2.5-flash-image`) for each slide marked `needs_image=True` (including `image_right`, `image_left`, and `table` visual diagrams).

3. **Step 3 — Compile Executive Deck (`create_premium_presentation`)**:
   * Pass the complete list of slide objects with rich `"Title: Detailed explanation"` bullets, `table_data` (for `"table"` slides), `key_insight` callouts, and `image_path` values.

4. **Step 4 — Submit for Human-in-the-Loop Review (`submit_presentation_for_review`)**:
   * Pause via `HumanInTheLoopMiddleware` so the user can approve or request iterative refinements.

---

## 2. Widescreen 16:9 Canvas & Executive Color System
* **Slide Width**: `13.333 inches` (`Inches(13.333)`) | **Slide Height**: `7.5 inches` (`Inches(7.5)`)
* **Base Slide Layout**: `prs.slide_layouts[6]` (Blank Layout)
* **Color Palette**:
  * **Canvas Base**: Obsidian Midnight (`RGB: 9, 13, 24`)
  * **Elevated Feature Card Fill**: Deep Executive Slate (`RGB: 20, 29, 49`)
  * **Elevated Card Border**: Crisp Slate Hairline (`RGB: 48, 68, 108`)
  * **Table Header Fill**: Royal Indigo/Cyan Accent (`RGB: 30, 48, 92`)
  * **Table Alternating Rows**: `RGB(18, 26, 44)` and `RGB(25, 36, 60)`
  * **Primary Heading Text**: Pure Snow White (`RGB: 248, 250, 252`)
  * **Body Description Text**: Bright Silver-White (`RGB: 214, 224, 238`) — high contrast, never dull gray
  * **Accent Colors**: Neon Cyan (`RGB: 0, 242, 254`), Royal Purple (`RGB: 147, 51, 234`), Emerald (`RGB: 16, 185, 129`), Gold (`RGB: 245, 158, 11`)

---

## 3. Card-Based Typography Rule (No Plain Floating Bullets!)
* **Format Every Bullet as `"Lead Title: Detailed Explanation"`**:
  * Example: `"Physical & Chemical Barriers: Epithelial tight junctions, mucosal membranes, and antimicrobial defensins block 99% of pathogen entry."`
* The rendering engine parses the text before `:` as a **Bold 14.5pt White Card Header** and the text after `:` as a **Crisp 12pt Silver-White Body Paragraph** inside an individual **Rounded Feature Card** with a **Left Neon Accent Bar** and a **Numbered Badge (`01`, `02`, `03`)**.
* Every content slide also includes a **Bottom Key Takeaway Strip** (`key_insight`) so the slide looks balanced, dense, and executive from top to bottom.

---

## 4. Predefined Multi-Background Image Resources (`skills/ppt_design/resources/`)
1. **`bg_opening_slide.png`** (`layout: "title"`): Obsidian & cyan/violet silk light waves framing the hero title card.
2. **`bg_content_text.png`** (`layout: "cards"`, `"text"`, `"table"`): Clean dark architectural slate backdrop with subtle dot-matrix texture and top/bottom neon trim.
3. **`bg_content_image.png`** (`layout: "image_right"`, `"image_left"`): Uniform dark executive backdrop supporting both left-hand and right-hand framed Gemini visuals without clashing borders.
4. **`bg_closing_slide.png`** (`layout: "ending"`): Gold, cyan, and royal violet horizon finale backdrop.

---

## 5. Dynamic Slide Layout Formats (6 Supported Layouts)

### Layout 1: Opening Hero Slide (`layout: "title"`)
* Full-bleed `bg_opening_slide.png`, centered Glassmorphism Hero Card, Neon Cyan Pill Badge, `38pt` Bold Title, `17pt` Subtitle, and bottom Executive Metadata Bar.

### Layout 2: Feature Cards Left + Gemini Image Right (`layout: "image_right"`)
* **Left Zone (`left=0.75", width=5.9"`)**: 3–4 stacked Executive Feature Cards (`01`, `02`, `03`) with left neon accent bars, bold white sub-headings, and bright body text.
* **Right Zone (`left=6.95", width=5.63"`)**: Elevated Dark Visual Frame housing the Gemini illustration + bottom visual caption bar.
* **Bottom Strip (`top=6.45"`)**: Full-width Key Takeaway Banner (`key_insight`).

### Layout 3: Gemini Image Left + Feature Cards Right (`layout: "image_left"`)
* **Left Zone (`left=0.75", width=5.63"`)**: Elevated Dark Visual Frame housing the Gemini illustration + bottom visual caption bar on the **LEFT** side.
* **Right Zone (`left=6.68", width=5.9"`)**: 3–4 stacked Executive Feature Cards (`01`, `02`, `03`) on the **RIGHT** side.
* **Bottom Strip (`top=6.45"`)**: Full-width Key Takeaway Banner (`key_insight`).

### Layout 4: Structured Comparison Table + Table Visual (`layout: "table"`)
* Renders a native styled PowerPoint Table (`slide.shapes.add_table`) using `table_data={"headers": [...], "rows": [[...], ...]}` with bold cyan/indigo header cells and alternating dark-slate data rows.
* If `image_path` is also provided on a `"table"` slide, places the styled comparison table alongside a framed Gemini comparison infographic/chart (`left=7.75", width=4.83"`) so both tabular metrics and visual comparison appear together!

### Layout 5: Multi-Card Executive Grid + Insight Panel (`layout: "cards"` or `"text"`)
* **Left Zone (`left=0.75", width=7.1"`)**: Stacked Executive Feature Cards with numbered badges and colored accent bars.
* **Right Zone (`left=8.15", width=4.43"`)**: High-impact Purple/Cyan **Executive Strategic Insight Card** (`key_insight`) + key metric highlights.

### Layout 6: Closing Summary & Q&A Slide (`layout: "ending"`)
* **Left Zone**: Numbered Executive Takeaway Cards (`01`, `02`, `03`, `04`).
* **Right Zone**: Framed Executive Summary & Next Steps Card (`key_insight`).
