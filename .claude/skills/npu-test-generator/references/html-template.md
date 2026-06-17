# HTML Report Template

Shared HTML/CSS patterns for Phase 1/2/3 output reports. Use these as the canonical reference when generating HTML — copy the CSS verbatim, adapt the HTML structure per phase.

## Design Rules (ALL phases)

1. **Server-side Python rendering** — no JavaScript data binding (`var D=[...]`). Build HTML strings in Python, write the file. Only use JS for client-side search/filter on already-rendered DOM.
2. **`render_list(arr)` helper** — every array field must go through a helper that returns `<ul>...</ul>` or `<span class="empty">--</span>` for empty arrays.
3. **`table-layout: fixed` + explicit column widths** — every `<th>` needs `nth-child(N){width:...}` and the `<th>` count must match `<td>` count.
4. **No green text** — use purple (#8e44ad) or dark (#1a1a1a) for body text. Green ONLY for converged/success badges.
5. **Single-file self-contained HTML** — inline all CSS in `<style>`, no external dependencies.

## CSS Variables & Global Reset

```css
:root {
  --bg: #f5f5f5; --card: #fff; --text: #1a1a1a;
  --muted: #666; --border: #dee2e6;
  --p0: #c0392b; --p1: #e67e22; --p2: #2962ff;
  --accent: #8e44ad;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.6;
}
.container { max-width: 100%; margin: 0 auto; padding: 16px; }
```

## Color Scheme

| Role | Hex | CSS Variable |
|------|-----|-------------|
| Background | `#f5f5f5` | `--bg` |
| Card background | `#fff` | `--card` |
| Body text | `#1a1a1a` | `--text` |
| Secondary text | `#666` | `--muted` |
| Borders | `#dee2e6` | `--border` |
| Accent / purple | `#8e44ad` | `--accent` |
| P0 / critical / red | `#c0392b` | `--p0` |
| P1 / feature / orange | `#e67e22` | `--p1` |
| P2 / edge / blue | `#2962ff` | `--p2` |

## Badges

```css
.badge { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; }
/* Priority */
.bp0 { background: #fdecea; color: var(--p0); }
.bp1 { background: #fef5e7; color: var(--p1); }
.bp2 { background: #eef2fd; color: var(--p2); }
/* Iteration (Phase 1) */
.bi1 { background: #e8e0f0; color: var(--accent); }
.bi2 { background: #d5f5e3; color: #1e8449; }
.bi3 { background: #fff3e0; color: #e65100; }
/* Filled badges (Phase 2) */
.badge-p0 { background: #c0392b; color: #fff; }
.badge-p1 { background: #e67e22; color: #fff; }
.badge-p2 { background: #2962ff; color: #fff; }
.badge-iter { background: #8e44ad; color: #fff; }
```

## Empty/null rendering

```python
def render_list(arr):
    if not arr:
        return '<span class="empty">--</span>'
    return "<ul>" + "".join(f"<li>{item}</li>" for item in arr) + "</ul>"
```

```css
.empty { color: var(--muted); font-style: italic; font-size: 11px; }
.hidden { display: none; }
```

## Component: Header

```html
<div class="header">
  <h1>{phase_title}</h1>
  <p class="sub">{module_name} &middot; Phase {N} &middot; Platform &middot; {iterations} iterations &middot; {convergence_status}</p>
  <div class="meta">
    <div class="mc"><div class="v">{count}</div><div class="l">{label}</div></div>
    <!-- more stat cards -->
  </div>
</div>
```

```css
.header { background: var(--card); border-radius: 12px; padding: 20px 24px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.header h1 { font-size: 22px; }
.sub { color: var(--muted); font-size: 13px; }
.meta { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.mc { background: #fafafa; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; text-align: center; min-width: 80px; }
.mc .v { font-size: 20px; font-weight: 700; color: var(--accent); }
.mc .l { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
```

## Component: Convergence Timeline

Horizontal nodes connected by arrows. Best/converged node gets `.tn.best` (green background).

```html
<div class="iter-section"><h3>Convergence Trajectory</h3>
<div class="timeline">
  <div class="tn"><div class="er" style="color:#c0392b">s=15.1</div><div class="lb">Iter 1: 23 errors</div></div>
  <span class="ta">&rarr;</span>
  <div class="tn"><div class="er" style="color:#e67e22">s=2.4</div><div class="lb">Iter 2: 3 errors</div></div>
  <span class="ta">&rarr;</span>
  <div class="tn best"><div class="er" style="color:#8e44ad">s=0</div><div class="lb">Iter 3: CONVERGED</div></div>
</div></div>
```

```css
.timeline { display: flex; gap: 6px; align-items: center; font-size: 13px; margin: 6px 0; flex-wrap: wrap; }
.tn { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; text-align: center; min-width: 100px; }
.tn.best { border-color: #8e44ad; background: #d5f5e3; }
.tn .er { font-size: 20px; font-weight: 700; }
.tn .lb { font-size: 10px; color: var(--muted); }
.ta { font-size: 14px; color: var(--muted); margin: 0 4px; }
```

**Color keys for s-values:**
- s >= 8 (high severity): `#c0392b` (red)
- s >= 5: `#e67e22` (orange)
- s > 0 and improving: `#1e8449` (green)
- s == 0 (converged): `#8e44ad` (purple)

## Component: Iteration Breakdown (Phase 1)

```html
<div class="iter-section"><h3>Iteration N: Detection</h3>
<div class="iter-card">
  <h4>N errors found by Sensor E subagent (s=X.X)</h4>
  <div class="err-list">
    <span class="et et-c">error_type &times;N</span>
    <span class="et et-m">error_type &times;N</span>
    <span class="et et-n">error_type &times;N</span>
  </div>
  <div class="detail">Key findings and corrections applied.</div>
</div></div>
```

```css
.iter-section { margin-bottom: 14px; }
.iter-section h3 { font-size: 13px; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; }
.iter-card { background: var(--card); border-radius: 10px; padding: 14px 18px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.04); border-left: 4px solid var(--accent); }
.iter-card.cvg { border-left-color: #8e44ad; }
.iter-card h4 { font-size: 13px; margin-bottom: 3px; }
.iter-card .detail { font-size: 12px; color: var(--muted); }
.err-list { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
.et { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.et-c { background: #fdecea; color: var(--p0); }   /* critical */
.et-m { background: #fef5e7; color: var(--p1); }   /* moderate */
.et-n { background: #eef2fd; color: var(--p2); }   /* minor */
```

## Component: Iteration Table (Phase 2)

```html
<div class="iter-breakdown"><h2>Iteration Detail</h2>
<table class="iter-table">
  <thead><tr><th>Iter</th><th>Errors Found</th><th>Corrections Applied</th><th>s_t</th></tr></thead>
  <tbody>
    <tr><td>0</td><td>Initial generation</td><td>—</td><td>—</td></tr>
    <tr><td>1</td><td>error types + counts</td><td>fixes applied</td><td>s → s</td></tr>
  </tbody>
</table></div>
```

```css
.iter-breakdown { background: var(--card); margin: 0 36px 20px; border-radius: 12px; padding: 24px 32px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
.iter-breakdown h2 { font-size: 17px; color: #3d1f5c; margin-bottom: 16px; }
.iter-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.iter-table th { background: linear-gradient(180deg, #f8f4fc 0%, #f0e8f5 100%); color: #5b2c8e; padding: 10px; text-align: left; border-bottom: 2px solid #e0d0e8; font-weight: 600; font-size: 12px; text-transform: uppercase; }
.iter-table td { padding: 10px; border-bottom: 1px solid #f5f5f5; vertical-align: top; color: #444; }
.iter-table th:nth-child(1) { width: 6%; text-align: center; }
.iter-table th:nth-child(2) { width: 35%; }
.iter-table th:nth-child(3) { width: 45%; }
.iter-table th:nth-child(4) { width: 14%; text-align: center; }
```

## Component: Data Table (Phase 1)

```html
<div class="controls"><input type="text" id="s" placeholder="Search..." oninput="f()"></div>
<div class="table-wrap"><table>
  <thead><tr>
    <th>ID</th><th>Iter</th><!-- ... N columns --><th>Last Column</th>
  </tr></thead>
  <tbody>
    <tr class="i{iteration}"><td>...</td><!-- exactly N td cells --></tr>
  </tbody>
</table></div>
```

```css
.controls { display: flex; gap: 10px; margin-bottom: 12px; }
.controls input { padding: 8px 14px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; flex: 1; min-width: 180px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 12px; background: var(--card); border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
th { background: #fafafa; padding: 8px 10px; text-align: left; font-weight: 600; font-size: 11px; color: var(--muted); text-transform: uppercase; border-bottom: 2px solid var(--border); }
td { padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; word-wrap: break-word; overflow-wrap: break-word; }
/* MUST match th count: */
th:nth-child(1) { width: 60px; } th:nth-child(2) { width: 40px; } /* ... */
tr:hover { background: #fafafa; }
/* Iteration-colored rows: */
tr.i2 { background: #f1e9f5; } tr.i2:hover { background: #e0d4eb; }
tr.i3 { background: #e8f5e9; } tr.i3:hover { background: #c8e6c9; }
/* Merged (deduplicated) rows: */
tr.merged { background: #fafafa; color: #999; text-decoration: line-through; }
tr.merged:hover { background: #f0f0f0; }
tr.merged td { opacity: 0.6; }
/* Confidence cells: */
.conf-strong { font-weight: 700; color: #27ae60; text-align: center; font-size: 13px; }
.conf-adequate { font-weight: 600; color: #8e44ad; text-align: center; font-size: 13px; }
.conf-weak { font-weight: 600; color: #e67e22; text-align: center; font-size: 13px; }
.conf-insufficient { font-weight: 700; color: #c0392b; text-align: center; font-size: 13px; }
.conf-na { color: var(--muted); text-align: center; font-style: italic; font-size: 11px; }
td[class^="conf-"] { cursor: help; }
```

## Component: Test Case Cards (Phase 2)

Each test case rendered as a card with priority color left-border.

```html
<div class="controls">
  <input type="text" id="search" placeholder="Search by ID, title, description, or steps..." oninput="filterCards()">
  <select id="priorityFilter" onchange="filterCards()">
    <option value="all">All Priorities</option>
    <option value="P0">P0 — Critical</option>
    <option value="P1">P1 — Feature</option>
    <option value="P2">P2 — Edge</option>
  </select>
  <span id="resultCount"></span>
</div>

<div class="cards" id="cards">
<div class="card p{N}">
  <div class="card-header">
    <span class="badge badge-p{N}">P{N}</span>
    <span class="badge badge-iter">Iter {N}</span>
    <span style="font-size:11px;color:#bbb;">{TC-ID} → {TP-ID}</span>
    <div class="card-title">{title}</div>
  </div>
  <div class="card-desc">{description}</div>

  <div class="card-section">
    <h4>Preconditions</h4>
    {render_list(preconditions)}
  </div>
  <div class="card-section">
    <h4>Test Data</h4>
    {render_test_data(test_data)}
  </div>
  <div class="card-section">
    <h4>Steps</h4>
    <ol>{steps_list}</ol>
  </div>
  <div class="card-section">
    <h4>Expected Results</h4>
    {render_expected_results(expected_results)}
  </div>
  <div class="card-section">
    <h4>Teardown</h4>
    {render_list(teardown)}
  </div>
</div>
</div>
```

```css
.cards { padding: 0 36px 40px; display: flex; flex-direction: column; gap: 14px; }
.card { background: #fff; border: 1px solid #e8e8e8; border-radius: 10px; padding: 22px 24px; transition: box-shadow .2s; }
.card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.08); }
.card.p0 { border-left: 4px solid #c0392b; }
.card.p1 { border-left: 4px solid #e67e22; }
.card.p2 { border-left: 4px solid #2962ff; }
.card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; flex-wrap: wrap; gap: 6px; }
.card-title { font-size: 15px; font-weight: 600; color: #1a1a1a; margin-top: 4px; }
.card-desc { color: #888; font-size: 13px; margin-bottom: 12px; }
.card-section { margin-top: 12px; }
.card-section h4 { font-size: 11px; color: #8e44ad; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; font-weight: 600; }
.card-section ul, .card-section ol { padding-left: 20px; font-size: 13px; color: #555; }
.card-section li { margin-bottom: 3px; }
/* Test data + expected results sub-tables: */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 4px; }
.data-table td { padding: 6px 10px; border-bottom: 1px solid #f5f5f5; vertical-align: top; word-break: break-word; }
.data-table .td-key { font-weight: 600; color: #6c3483; width: 35%; background: #fdfbff; font-size: 12px; }
.data-table .assertion { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 12px; color: #2c3e50; }
```

## Client-side Search/Filter JS (minimal)

```html
<script>
// Phase 1: table row filter
function f() {
  const q = document.getElementById('s').value.toLowerCase();
  document.querySelectorAll('tbody tr').forEach(r => {
    r.classList.toggle('hidden', q && !r.textContent.toLowerCase().includes(q));
  });
}

// Phase 2: card filter with priority
function filterCards() {
  const q = document.getElementById('search').value.toLowerCase();
  const prio = document.getElementById('priorityFilter').value;
  const cards = document.querySelectorAll('.card');
  let visible = 0;
  cards.forEach(c => {
    const searchText = c.getAttribute('data-search').toLowerCase();
    const cardPrio = c.getAttribute('data-priority');
    const match = (!q || searchText.includes(q)) && (prio === 'all' || cardPrio === prio);
    c.style.display = match ? '' : 'none';
    if (match) visible++;
  });
  document.getElementById('resultCount').textContent = 'Showing ' + visible + ' of ' + cards.length;
}
document.getElementById('resultCount').textContent = 'Showing all ' + document.querySelectorAll('.card').length;
</script>
```

## Phase-Specific Guidance

### Phase 1 — Test Points Table

- Render as a `<table>` with fixed columns
- Columns: ID, Iter(badge), Dim, Feature, Key Factors, Source, Pri(badge), **Conf**, Preconditions, Expected Behavior, Boundary, Error Paths
- **Conf column (REQUIRED)**: read per-TP confidence from `test-out/phase<N>/confidence.json`. Render each cell as the aggregate score (mean of 5 dimensions, rounded to 2 decimals) with color coding and the per-dimension breakdown as a `title` tooltip:
  - `>= 0.85` → `conf-strong` (green `#27ae60`)
  - `0.70–0.84` → `conf-adequate` (purple `#8e44ad`)
  - `0.50–0.69` → `conf-weak` (orange `#e67e22`)
  - `< 0.50` → `conf-insufficient` (red `#c0392b`)
  - Missing data → `conf-na` (grey italic `--`)
- Iteration-colored rows (`tr.i1`, `tr.i2`, `tr.i3`)
- **Merged TPs** (those with `"merged_into"` field): render as `tr.merged` with greyed-out text, strikethrough, reduced opacity. Append a note in the Feature column: "(merged into TP-XYZ)". These rows should still appear in the table (audit trail) but be visually de-emphasized. Their Conf cell shows the surviving TP's score in parentheses.
- Convergence section: timeline + per-iteration cards with error tags
- Priority badge: P0/P1/P2
- Iteration badge: I1/I2/I3

### Phase 2 — Test Case Cards

- Render as card components (`div.card`) with priority left-border
- Each card: title, description, preconditions, test_data (sub-table), numbered steps, expected results (sub-table), teardown
- Search filters by ID/title/description/steps text content
- Priority filter dropdown (P0/P1/P2/all)
- Convergence section: timeline (Phase 1 style) + iteration detail table
- Summary header: total cases, covered TPs, priority distribution, iterations, final s_t
- **Confidence cards**: If `confidence.json` exists for Phase 2, render per-TC confidence scores in each card (same color coding as Phase 1)

### Phase 3 — Test Script

Generate a **minimal summary HTML** report. Output: `test-out/phase3/test-script.html`.

**Sections:**

1. **Header**: module name, platform, Phase 3, convergence status, summary stat cards (methods, assertions, syntax result, classes)
2. **Convergence timeline**: same `.tn` / `.ta` component as Phase 1/2
3. **Iteration detail table**: same `.iter-table` with errors found + corrections applied + s_t
4. **Test method table**: columns: Method, TC-ID, Class, Assertions (count), Config Match (pass/fail/none), Oracle (pass/fail/none)
5. **Code preview**: syntax-highlighted `<pre>` block of the generated script (first ~100 lines linked to full file)

**CSS**: Reuse `:root` variables, header, meta cards, timeline, and iteration table from Phase 1/2 templates.

**Checklist:**
- [ ] Header with stat cards (methods, assertions, classes, syntax check result)
- [ ] Convergence timeline with iterations
- [ ] Iteration detail table
- [ ] Test method table with per-method assertion counts
- [ ] Syntax check result prominently displayed
- [ ] Valid HTML5

## Python Generation Checklist

**PRE-REQUISITE**: Write the HTML generation logic as a **standalone `.py` file** (e.g., `generate_html.py`) and execute it with `python generate_html.py`. Do NOT inline the Python code inside `bash -c "python -c '...'"`. Long inline scripts with nested quotes will fail on bash due to quote-escaping limits. After generation, delete the temporary script.

- [ ] HTML generator is a standalone `.py` file, not inline in bash

- [ ] CSS uses `:root` variables or matches the color scheme table above
- [ ] All arrays rendered through `render_list()` or equivalent — empty arrays show `--`
- [ ] Table `<th>` count == `<td>` count; column widths set via `nth-child`
- [ ] Card `data-search` attribute includes id, title, description, steps text
- [ ] Card `data-priority` attribute set for priority filter
- [ ] Timeline uses `.tn` nodes with `.ta` arrows; best node has `.tn.best`
- [ ] No JS data binding — all content rendered server-side in Python
- [ ] Search/filter bar above table: text input + priority checkboxes + dimension checkboxes + live count
- [ ] Filter JS: pure DOM walk on `#tptable tbody tr` (Phase 1) or `#tctable tbody tr` (Phase 2)
- [ ] Filter JS pattern (copy into a `<script>` at end of `<body>`):
  ```
  function filter() {
    const q = document.getElementById("search").value.toLowerCase();
    const pris = new Set([...document.querySelectorAll("[data-pri]:checked")].map(c=>c.dataset.pri));
    const dims = new Set([...document.querySelectorAll("[data-dim]:checked")].map(c=>c.dataset.dim));
    document.querySelectorAll("#tptable tbody tr").forEach(r => {
      const txt = r.textContent.toLowerCase();
      const ok = (!q || txt.includes(q)) && pris.has(r.cells[2].textContent.trim()) && dims.has(r.cells[3].textContent.trim());
      r.style.display = ok ? "" : "none";
    });
    // update count element
  }
  filter(); // run on load
  ```
- [ ] Priority badges match the color scheme (red P0, orange P1, blue P2)
- [ ] Confidence column present (Phase 1/2): reads `confidence.json`, renders color-coded aggregate with per-dimension tooltip
- [ ] Merged TPs (Phase 1): rows with `merged_into` field rendered grey/strikethrough with "(merged into TP-XYZ)" note; Conf cell shows surviving TP's score
- [ ] File is valid HTML5 (check `<!DOCTYPE html>`, charset meta, no unclosed tags)

## Confidence Overview Panel

When Phase 1/2 Sub writes `confidence.json`, read it in the report generator and render a confidence summary section before the main table.

### File format (`test-out/phase<N>/confidence.json`)

```json
{
  "TP-001": {"source_grounding": 0.9, "observability": 0.8, "specification_completeness": 0.7, "platform_fidelity": 0.9, "logical_coherence": 0.8},
  "TP-002": {"source_grounding": 0.8, "observability": 0.6, "specification_completeness": 0.8, "platform_fidelity": 0.9, "logical_coherence": 0.7}
}
```

### CSS for confidence badges

```css
.conf-panel { background: var(--card); padding: 16px 20px; border-radius: 8px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.conf-panel h2 { font-size: 14px; margin-bottom: 10px; }
.conf-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.conf-card { background: var(--bg); padding: 10px 12px; border-radius: 6px; }
.conf-card .id { font-family: monospace; font-weight: 600; font-size: 12px; }
.conf-card .score { font-size: 20px; font-weight: 700; }
.conf-card .dims { font-size: 10px; color: var(--muted); margin-top: 4px; }
.conf-card.conf-strong .score { color: #27ae60; }   /* aggregate >= 0.85 */
.conf-card.conf-adequate .score { color: #8e44ad; } /* aggregate 0.70-0.84 */
.conf-card.conf-weak .score { color: #e67e22; }     /* aggregate 0.50-0.69 */
.conf-card.conf-insufficient .score { color: #c0392b; } /* aggregate < 0.50 */
.conf-mean { background: var(--card); padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; display: inline-block; }
.conf-mean .big { font-size: 24px; font-weight: 700; }
```

### Rendering pattern (Python)

```python
import json, os

conf_path = 'test-out/phase1/confidence.json'
if os.path.exists(conf_path):
    conf = json.load(open(conf_path, encoding='utf-8'))
    # Compute per-TP aggregate
    cards = []
    for tp_id, dims in conf.items():
        agg = round(sum(dims.values()) / 5, 2)
        if agg >= 0.85: css = 'conf-strong'
        elif agg >= 0.70: css = 'conf-adequate'
        elif agg >= 0.50: css = 'conf-weak'
        else: css = 'conf-insufficient'
        cards.append((tp_id, agg, css, dims))
    # Sort by aggregate ascending (worst first)
    cards.sort(key=lambda x: x[1])
    # Mean across all
    mean_agg = round(sum(c[1] for c in cards) / len(cards), 2)
    # Render conf-mean badge + conf-grid with conf-card elements
```

### Checklist

- [ ] Confidence panel rendered when `confidence.json` exists; gracefully omitted when absent
- [ ] Cards sorted worst-first (low aggregate = top of grid)
- [ ] Color coding matches the 4-tier rating scale
- [ ] Mean aggregate shown prominently
- [ ] Dimension breakdown visible on hover or as small text per card
