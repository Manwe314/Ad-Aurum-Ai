# bbb/utils/logger_html.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import datetime as _dt
import html as _html

_Color = Optional[str]

# === Theme (fg, bg) ===
_DEFAULT_THEME: Dict[str, Tuple[_Color, _Color]] = {
    # category -> (fg, bg)
    "info":        ("#F07A37", "#38F0EE"),
    "debug":       ("#FFFFFF", "#E00B99"),
    "success":     ("#D12469", "#30F03A"),
    "warn":        ("#FDD639", "#BD0707"),
    "error":       ("#51EC2F", "#EF1FBD"),
    "battle":      ("#017236", "#EF3E2C"),
    "card_play":   ("#3DF9FF", "#571469"),
    "bet":         ("#7925E6", "#F0ED34"),
    "equalize":    ("#2356E2", "#F0B629"),
    "resolution":  ("#2F83F0", "#F0CE8D"),
    "section":     ("#F0C52B", "#3548EF"),
}

_ICON: Dict[str, str] = {
    "info": "ℹ️", "debug": "🔍", "success": "✅", "warn": "⚠️", "error": "❌",
    "battle": "⚔️", "card_play": "🃏", "bet": "💰", "equalize": "⚖️", "resolution": "🏁", "section": "📘",
}

@dataclass
class _Event:
    cat: Optional[str]
    text_html: str
    fg: _Color
    bg: _Color
    player: Optional[str]
    stats: Optional[Dict[str, Any]] = None
    # rendering group
    section_id: int = 0

@dataclass
class _Section:
    sec_id: int
    title: str

class HtmlLogger:
    """
    HTML logger with:
      - Category-based colors (set_theme)
      - Player columns + Global last column (set_players)
      - Collapsible sections with toggle buttons
      - Convenience calls: log_cat, log_player, section
      - Optional per-event stats blocks (stats=True, player_obj=...)
    """
    def __init__(self, path: str, title: str = "BBB Report"):
        self.path = path
        self.title = title
        self._theme: Dict[str, Tuple[_Color, _Color]] = dict(_DEFAULT_THEME)
        self._players: List[str] = []
        self._events: List[_Event] = []
        self._sections: List[_Section] = []
        self._cur_section_id: int = 0
        self._started = _dt.datetime.now()

    # ---------- Configuration ----------
    def set_theme(self, mapping: Dict[str, Tuple[_Color, _Color]]):
        self._theme.update(mapping or {})

    def set_players(self, players: List[str]):
        self._players = list(players or [])

    # ---------- High-level logging ----------
    def section(self, title: str):
        self._cur_section_id += 1
        self._sections.append(_Section(self._cur_section_id, title))

    def log_cat(self, category: str, text: str, *, player: Optional[str] = None,
                stats: bool = False, player_obj: Any = None):
        fg, bg = self._theme.get(category, (None, None))
        icon = _ICON.get(category, "")
        msg = f"{icon} {text}" if icon else text
        self._events.append(_Event(
            cat=category, text_html=_esc(msg), fg=fg, bg=bg, player=player,
            stats=_extract_stats(player_obj) if stats else None,
            section_id=self._cur_section_id
        ))

    def log_player(self, player: str, text: str, *, category: Optional[str] = None,
                   stats: bool = False, player_obj: Any = None):
        if category:
            self.log_cat(category, text, player=player, stats=stats, player_obj=player_obj)
        else:
            self.log(text, player=player, stats=stats, player_obj=player_obj)

    # ---------- Low-level logging ----------
    def log(self, text: str, *, fg: _Color = None, bg: _Color = None, player: Optional[str] = None,
            stats: bool = False, player_obj: Any = None):
        self._events.append(_Event(
            cat=None, text_html=_esc(text), fg=fg, bg=bg, player=player,
            stats=_extract_stats(player_obj) if stats else None,
            section_id=self._cur_section_id
        ))

    # ---------- Save ----------
    def save(self):
        html = self._render()
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(html)

    # ---------- Internal ----------
    def _render(self) -> str:
        # Columns: players first, Global LAST
        cols = list(self._players) + ["Global"]
        ncols = len(cols)

        head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{_esc(self.title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{_CSS(ncols)}
</head><body>
<div class="container">
  <div class="header">
    <div class="h1">{_esc(self.title)}</div>
    <div class="meta">Generated at {_esc(self._started.isoformat(timespec="seconds"))}</div>
  </div>
"""

        body = ['<div class="log">']
        # One fixed header row at top
        body.append('<table class="grid"><thead><tr>')
        for name in cols:
            body.append(f'<th class="col">{_esc(name)}</th>')
        body.append('</tr></thead><tbody>')

        # Group events by section id in order:
        # If no explicit sections were added, everything belongs to section 0; render one implicit section.
        section_ids_in_order = [s.sec_id for s in self._sections]
        if 0 not in section_ids_in_order:
            section_ids_in_order = [0] + section_ids_in_order

        # Build mapping sec_id -> title
        sec_titles: Dict[int, str] = {0: "Log"}
        for s in self._sections:
            sec_titles[s.sec_id] = s.title

        # Prepare events per section
        events_by_sec: Dict[int, List[_Event]] = {}
        for ev in self._events:
            events_by_sec.setdefault(ev.section_id, []).append(ev)

        # Render each section as a header row + collapsible tbody
        sec_counter = 0
        for sec_id in section_ids_in_order:
            sec_counter += 1
            sec_dom_id = f"sec{sec_counter}"
            title = sec_titles.get(sec_id, f"Section {sec_id}")

            # Close current tbody, insert section bar, start new tbody with id
            body.append('</tbody></table>')
            body.append(_section_bar(title, sec_dom_id, self._theme["section"]))
            body.append('<table class="grid"><tbody id="{}" class="section-body">'.format(_esc(sec_dom_id)))

            # Rows for this section
            for ev in events_by_sec.get(sec_id, []):
                # Choose column: player col or Global (last)
                if ev.player is not None and ev.player in self._players:
                    col_idx = self._players.index(ev.player)
                else:
                    col_idx = len(self._players)  # Global

                body.append('<tr>')
                for i in range(ncols):
                    if i == col_idx:
                        style = _style_cell(ev.fg, ev.bg)
                        content = ev.text_html
                        if ev.stats:
                            content += _render_stats(ev.stats)
                        body.append(f'<td class="cell" style="{style}">{content}</td>')
                    else:
                        body.append('<td class="cell empty"></td>')
                body.append('</tr>')

        # Close last table
        body.append('</tbody></table></div>')

        tail = f"""
</div>
<script>
document.addEventListener('click', function(e) {{
  const btn = e.target.closest('.sec-toggle');
  if (!btn) return;
  const target = btn.getAttribute('data-target');
  const tbody = document.getElementById(target);
  if (!tbody) return;
  const hidden = tbody.classList.toggle('collapsed');
  btn.textContent = hidden ? 'Expand' : 'Collapse';
}});
</script>
</body></html>
"""
        return head + "".join(body) + tail


# ---------- Helpers ----------

def _esc(s: Any) -> str:
    return _html.escape(str(s))

def _style_cell(fg: _Color, bg: _Color) -> str:
    parts = []
    if bg: parts.append(f"background:{bg}")
    if fg: parts.append(f"color:{fg}")
    return ";".join(parts)

def _section_bar(title: str, dom_id: str, theme_pair: Tuple[_Color, _Color]) -> str:
    fg, bg = theme_pair
    return f"""
<div class="section">
  <div class="section-title">{_esc(title)}</div>
  <button class="sec-toggle" data-target="{_esc(dom_id)}">Collapse</button>
</div>
"""

def _extract_stats(player_obj: Any) -> Optional[Dict[str, Any]]:
    if player_obj is None:
        return None
    try:
        behind = getattr(player_obj, "coins", None)
        front  = getattr(player_obj, "front_coins", None)
        hand   = len(getattr(player_obj, "cards", []) or [])
        won    = len(getattr(player_obj, "cards_won", []) or [])
        rounds = getattr(player_obj, "rounds_won", None)
        return {
            "behind": behind if behind is not None else "-",
            "front":  front  if front  is not None else "-",
            "hand":   hand,
            "won":    won,
            "rounds": rounds if rounds is not None else "-",
        }
    except Exception:
        return None

def _render_stats(stats: Dict[str, Any]) -> str:
    return f"""
<div class="stats">
  <div class="stat"><span class="k">behind</span><span class="v">{_esc(stats.get('behind', '-'))}</span></div>
  <div class="stat"><span class="k">front</span><span class="v">{_esc(stats.get('front', '-'))}</span></div>
  <div class="stat"><span class="k">hand</span><span class="v">{_esc(stats.get('hand', '-'))}</span></div>
  <div class="stat"><span class="k">won</span><span class="v">{_esc(stats.get('won', '-'))}</span></div>
  <div class="stat"><span class="k">rounds</span><span class="v">{_esc(stats.get('rounds', '-'))}</span></div>
</div>
"""

def _CSS(ncols: int) -> str:
    return f"""
<style>
:root {{
  --bg: #0b0f16;
  --fg: #e6edf3;
  --muted: #94a3b8;
  --panel: #0f172a;
  --border: #1f2937;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); }}
.container {{ max-width: 1700px; margin: 0 auto; padding: 20px; }}
.header .h1 {{ font-size: 22px; margin-bottom: 4px; }}
.header .meta {{ color: var(--muted); font-size: 12px; margin-bottom: 12px; }}

.section {{
  margin: 18px 0 10px 0; padding: 10px 12px; border-left: 4px solid #3b82f6;
  background: #0b1220; color:#c7d2fe; font-weight:600; border-radius: 6px;
  display:flex; align-items:center; justify-content:space-between;
}}
.section-title {{ font-size: 14px; }}
.section .sec-toggle {{
  appearance:none; border:1px solid #334155; background:#0e172a; color:#cbd5e1;
  padding:6px 10px; border-radius:6px; font-family:var(--mono); cursor:pointer;
}}
.section .sec-toggle:hover {{ background:#101a2e; }}

.log {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
table.grid {{ width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; }}
thead th {{
  position: sticky; top: 0;
  background: #111827; color:#9ca3af; font-weight:600; font-size:12px;
  border-bottom: 1px solid var(--border);
  padding: 8px 8px;
}}
th.col, td.cell {{ border-right: 1px solid var(--border); }}
th.col:last-child, td.cell:last-child {{ border-right: none; }}
td.cell {{ padding: 6px 8px; font-family: var(--mono); font-size: 13px; line-height:1.4; }}
td.cell.empty {{ background: #0f172a; color: transparent; }}
tbody tr:nth-child(2n+1) td.cell.empty {{ background: #0d1522; }}
tbody tr:hover td {{ background: #101826; }}

.section-body.collapsed {{ display: none; }}

.stats {{
  margin-top: 6px;
  padding: 6px 4px;                 
  border: 1px dashed #334155;
  border-radius: 6px;
  background: #0b1322;

  display: flex;
  gap: 6px;
  overflow-x: auto;                 
  overflow-y: hidden;
  scrollbar-width: thin;           
}}
.stats::-webkit-scrollbar {{
  height: 6px;
}}
.stats::-webkit-scrollbar-thumb {{
  background: #22314e;
  border-radius: 3px;
}}

.stats .stat {{
  flex: 0 0 auto;                   /* don’t shrink; keep pills side-by-side */
  display: inline-flex;
  align-items: center;
  justify-content: space-between;

  background: #0e172a;
  border: 1px solid #1f2a44;
  border-radius: 6px;

  padding: 4px 6px;                 /* tighter pill */
  min-width: 90px;                  /* compact but readable */
}}

.stats .k {{
  color: #93c5fd;
  font-size: 11px;                  /* slightly smaller */
  margin-right: 6px;
  opacity: 0.9;
}}
.stats .v {{
  color: #e5e7eb;
  font-weight: 600;
  font-size: 12px;
}}
</style>
"""
