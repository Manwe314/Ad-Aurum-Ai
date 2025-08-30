# bbb/utils/logger_html.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import datetime as _dt
import html as _html
import os

_Color = Optional[str]

_DEFAULT_THEME: Dict[str, Tuple[_Color, _Color]] = {
    # category -> (fg, bg)
    "info":        ("#F07A37", "#38F0EE"),
    "debug":       ("#C7D2FE", "#1B2437"),
    "success":     ("#9B5370", "#30F03A"),
    "warn":        ("#FDE68A", "#3B2F1A"),
    "error":       ("#579B48", "#EF1FBD"),
    "battle":      ("#4F9B73", "#EF3E2C"),
    "card_play":   ("#F5D0FE", "#33193A"),
    "bet":         ("#74569B", "#F0ED34"),
    "equalize":    ("#4E629B", "#F0B629"),
    "resolution":  ("#84B3F0", "#F0CE8D"),
    "section":     ("#F0C52B", "#3548EF"),
}

_ICON: Dict[str, str] = {
    "info": "ℹ️", "debug": "🔍", "success": "✅", "warn": "⚠️", "error": "❌",
    "battle": "⚔️", "card_play": "🃏", "bet": "💰", "equalize": "⚖️", "resolution": "🏁", "section": "📘",
}

@dataclass
class _Event:
    ts: str
    cat: Optional[str]
    text_html: str
    fg: _Color
    bg: _Color
    player: Optional[str]
    is_section: bool = False
    section_title: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None  # {"behind":...,"front":...,"hand":...,"won":...,"rounds":...}

class HtmlLogger:
    """
    HTML logger with:
      - Category-based colors (set_theme)
      - Player columns (set_players)
      - Chronological rows aligned across Global + per-player columns
      - Convenience calls: log_cat, log_player, section
      - Optional per-event stats blocks
    """
    def __init__(self, path: str, title: str = "BBB Report"):
        self.path = path
        self.title = title
        self._theme: Dict[str, Tuple[_Color, _Color]] = dict(_DEFAULT_THEME)
        self._players: List[str] = []
        self._events: List[_Event] = []
        self._started = _dt.datetime.now()

    # ---------- Configuration ----------
    def set_theme(self, mapping: Dict[str, Tuple[_Color, _Color]]):
        self._theme.update(mapping or {})

    def set_players(self, players: List[str]):
        self._players = list(players or [])

    # ---------- High-level logging ----------
    def section(self, title: str):
        self._events.append(_Event(
            ts=self._ts(), cat="section", text_html=_esc(title),
            fg=self._theme["section"][0], bg=self._theme["section"][1],
            player=None, is_section=True, section_title=title
        ))

    def log_cat(self, category: str, text: str, *, player: Optional[str] = None,
                stats: bool = False, player_obj: Any = None):
        fg, bg = self._theme.get(category, (None, None))
        icon = _ICON.get(category, "")
        msg = f"{icon} {text}" if icon else text
        self._events.append(_Event(
            ts=self._ts(), cat=category, text_html=_esc(msg), fg=fg, bg=bg,
            player=player, stats=_extract_stats(player_obj) if stats else None
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
            ts=self._ts(), cat=None, text_html=_esc(text), fg=fg, bg=bg,
            player=player, stats=_extract_stats(player_obj) if stats else None
        ))

    # ---------- Save ----------
    def save(self):
        # ensure directories exist
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        html = self._render()
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(html)

    # ---------- Internal ----------
    def _ts(self) -> str:
        return _dt.datetime.now().strftime("%H:%M:%S")

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
        body.append('<table class="grid"><thead><tr>')
        for name in cols:
            body.append(f'<th class="col">{_esc(name)}</th>')
        body.append('</tr></thead><tbody>')
    
        for ev in self._events:
            if ev.is_section:
                body.append('</tbody></table>')
                body.append(f'<div class="section">{_esc(ev.section_title or "")}</div>')
                body.append('<table class="grid"><thead><tr>')
                for name in cols:
                    body.append(f'<th class="col">{_esc(name)}</th>')
                body.append('</tr></thead><tbody>')
                continue
            
            # Place event in the right column:
            # - if player specified & known -> that player's column (index in self._players)
            # - otherwise (GLOBAL) -> last column
            if ev.player is not None and ev.player in self._players:
                col_idx = self._players.index(ev.player)
            else:
                col_idx = len(self._players)  # Global = last column
    
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
    
        body.append('</tbody></table></div>')
        tail = '</div></body></html>'
        return head + "".join(body) + tail


# ---------- Helpers ----------

def _esc(s: Any) -> str:
    return _html.escape(str(s))

def _style_cell(fg: _Color, bg: _Color) -> str:
    parts = []
    if bg: parts.append(f"background:{bg}")
    if fg: parts.append(f"color:{fg}")
    return ";".join(parts)

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
    # roomier + responsive auto-fit (wraps to 2 rows on narrow cells)
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
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

.header .h1 {{ font-size: 22px; margin-bottom: 4px; }}
.header .meta {{ color: var(--muted); font-size: 12px; margin-bottom: 12px; }}

.section {{
  margin: 18px 0 10px 0; padding: 10px 12px; border-left: 4px solid #3b82f6;
  background: #0b1220; color:#c7d2fe; font-weight:600; border-radius: 6px;
}}

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

td.cell {{ padding: 8px 10px; font-family: var(--mono); font-size: 13px; line-height:1.5; }}
td.cell.empty {{ background: #0f172a; color: transparent; }}
tbody tr:nth-child(2n+1) td.cell.empty {{ background: #0d1522; }}
tbody tr:hover td {{ background: #101826; }}

.stats {{
  margin-top: 8px; padding: 8px; border: 1px dashed #334155; border-radius: 8px;
  background: #0b1322;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));  /* roomier & wraps */
  gap: 8px;
}}
.stats .stat {{
  display:flex; align-items:center; justify-content:space-between;
  background:#0e172a; border:1px solid #1f2a44; border-radius:8px; padding: 6px 8px;
}}
.stats .k {{ color:#93c5fd; font-size:13px; opacity:0.95; }}
.stats .v {{ color:#e5e7eb; font-weight:700; font-size:13px; }}
</style>
"""
