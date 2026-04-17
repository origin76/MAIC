#!/usr/bin/env python3
"""Generate vector architecture diagrams for the thesis progress PPT."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "figures" / "generated"


PALETTE = {
    "bg": "#f7f8fc",
    "ink": "#182230",
    "muted": "#5b6472",
    "line": "#7a8697",
    "actor": "#d8ecff",
    "critic": "#ffe7cc",
    "comm": "#dff6e3",
    "attack": "#ffd9d9",
    "move": "#dbe8ff",
    "accent": "#1e6bd6",
    "accent2": "#0e8f5b",
    "accent3": "#cc5a14",
    "warning": "#fff2bf",
    "white": "#ffffff",
}


class SvgCanvas:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.title = title
        self.elements: list[str] = []

    def add(self, element: str) -> None:
        self.elements.append(element)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(self.elements)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(self.title)}</title>
<desc id="desc">Vector architecture diagram for thesis progress presentation.</desc>
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="{PALETTE["line"]}"/>
  </marker>
  <marker id="arrow-accent" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="{PALETTE["accent"]}"/>
  </marker>
</defs>
<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="{PALETTE["bg"]}"/>
{payload}
</svg>
"""
        path.write_text(svg, encoding="utf-8")


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#7a8697", stroke_width: int = 2, rx: int = 18, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str | None = None, width: int = 3, accent: bool = False, dash: str | None = None) -> str:
    marker = "url(#arrow-accent)" if accent else "url(#arrow)"
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color or PALETTE["line"]}" stroke-width="{width}" '
        f'stroke-linecap="round" marker-end="{marker}"{dash_attr}/>'
    )


def text(x: float, y: float, content: str, size: int = 22, fill: str | None = None, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill or PALETTE["ink"]}" text-anchor="{anchor}">{escape(content)}</text>'
    )


def multiline(x: float, y: float, lines: list[str], size: int = 19, fill: str | None = None, weight: str = "400", line_gap: int = 24) -> str:
    return "\n".join(text(x, y + i * line_gap, line, size=size, fill=fill, weight=weight) for i, line in enumerate(lines))


def node(canvas: SvgCanvas, x: float, y: float, w: float, h: float, title_text: str, body_lines: list[str], fill: str, stroke: str | None = None, accent_bar: str | None = None, dashed: bool = False) -> None:
    canvas.add(rect(x, y, w, h, fill=fill, stroke=stroke or PALETTE["line"], dash="10 8" if dashed else None))
    if accent_bar:
        canvas.add(rect(x, y, w, 14, fill=accent_bar, stroke=accent_bar, stroke_width=0, rx=18))
    canvas.add(text(x + 18, y + 34, title_text, size=24, weight="700"))
    canvas.add(multiline(x + 18, y + 66, body_lines, size=18, fill=PALETTE["muted"], line_gap=23))


def pill(canvas: SvgCanvas, x: float, y: float, w: float, h: float, label: str, fill: str, stroke: str | None = None) -> None:
    canvas.add(rect(x, y, w, h, fill=fill, stroke=stroke or fill, stroke_width=1, rx=h // 2))
    canvas.add(text(x + w / 2, y + h / 2 + 7, label, size=18, weight="700", anchor="middle"))


def draw_backbone() -> None:
    canvas = SvgCanvas(1600, 920, "Officialish MAPPO backbone")
    canvas.add(text(70, 62, "Officialish MAPPO Backbone", size=34, weight="700"))
    canvas.add(text(70, 96, "Shared recurrent actor + agent-wise centralized critic + PPO stabilizers", size=20, fill=PALETTE["muted"]))

    node(canvas, 70, 150, 220, 120, "Actor Input", ["local obs_i", "+ optional agent id", "obs_last_action=False on SMAC"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 340, 150, 210, 120, "Encoder", ["Linear(input, 64)", "ReLU"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 600, 150, 220, 120, "Temporal Core", ["GRUCell(64, 64)", "shared across all agents"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 870, 150, 250, 120, "Policy Head", ["Linear(64,64) + ReLU", "Linear(64, n_actions)"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 1170, 150, 280, 120, "Execution Output", ["mask unavailable actions", "softmax over logits", "decentralized action sampling"], PALETTE["actor"], accent_bar=PALETTE["accent"])

    for x1, x2 in [(290, 340), (550, 600), (820, 870), (1120, 1170)]:
        canvas.add(line(x1, 210, x2, 210))

    node(canvas, 170, 390, 320, 150, "Critic Input (training only)", ["global state_t", "+ local obs_i", "+ last_action_{t-1}", "+ agent_id"], PALETTE["critic"], accent_bar=PALETTE["accent3"])
    node(canvas, 560, 390, 220, 150, "Centralized Critic", ["MLP: 256 -> 256 -> 1", "predict V_i(s, o_i, a_{t-1})"], PALETTE["critic"], accent_bar=PALETTE["accent3"])
    node(canvas, 850, 390, 300, 150, "Value Targets", ["GAE(lambda=0.95)", "ValueNorm", "Huber loss + value clipping"], PALETTE["critic"], accent_bar=PALETTE["accent3"])
    node(canvas, 1210, 390, 250, 150, "Optimization", ["critic lr = 7e-4", "actor lr = 4e-4", "grad clip = 5"], PALETTE["critic"], accent_bar=PALETTE["accent3"])
    for x1, x2 in [(490, 560), (780, 850), (1150, 1210)]:
        canvas.add(line(x1, 465, x2, 465, color=PALETTE["accent3"], accent=True))

    node(canvas, 70, 640, 1480, 180, "PPO Stabilizers in the Chosen Backbone", [
        "policy active masks and value active masks",
        "clipped surrogate objective with clip = 0.2 and up to 6 PPO epochs",
        "critic value clipping with value_clip_param = 0.2",
        "linear actor LR decay to 0.1 x base LR, critic LR kept high",
        "KL early stop with target_kl = 0.02",
        "warm-start from step 1,404,757 of the best 1.5M officialish run",
    ], PALETTE["white"], accent_bar=PALETTE["accent2"], dashed=True)

    pill(canvas, 1170, 292, 250, 42, "CTDE: centralized training, decentralized execution", PALETTE["warning"], stroke="#d0b25c")
    canvas.save(OUTPUT_DIR / "ppt_mappo_backbone_structure.svg")


def draw_timeline() -> None:
    canvas = SvgCanvas(1700, 520, "Algorithm evolution timeline")
    canvas.add(text(70, 62, "Progress Timeline: from join1 validation to dual-stream communication", size=34, weight="700"))
    canvas.add(text(70, 96, "Each stage kept the MAPPO backbone fixed as much as possible and changed only one structural factor at a time.", size=20, fill=PALETTE["muted"]))

    stages = [
        ("join1", "Validate that communication can help,\nbut can also collapse without constraints.", PALETTE["warning"]),
        ("Backbone", "Build a reliable MAPPO line:\nofficialish learner + warm-start.", PALETTE["actor"]),
        ("V1", "Minimal residual adapter:\nlow bandwidth, zero-init, low gain.", PALETTE["comm"]),
        ("V2", "Sharpen routing only:\nattention entropy penalty / faster comm.", PALETTE["comm"]),
        ("V3", "Replace hidden feature carrier\nwith action intention carrier.", PALETTE["comm"]),
        ("V4", "Route relation-aware messages and\nmodify attack logits only.", PALETTE["attack"]),
        ("V5", "Split attack and move streams\nfor subspace-specific coordination.", PALETTE["move"]),
    ]

    x = 70
    y = 170
    w = 210
    h = 180
    gap = 25
    for idx, (name, desc, fill) in enumerate(stages):
        body = desc.split("\n")
        node(canvas, x, y, w, h, name, body, fill, accent_bar=PALETTE["accent"] if idx < 2 else PALETTE["accent2"])
        if idx < len(stages) - 1:
            canvas.add(line(x + w, y + h / 2, x + w + gap, y + h / 2))
        x += w + gap

    node(canvas, 70, 390, 600, 90, "Narrative to emphasize in the PPT", [
        "communication was not assumed to be useful; it was validated, constrained, and then injected on top of a strong local policy",
    ], PALETTE["white"], accent_bar=PALETTE["accent3"])
    node(canvas, 760, 390, 600, 90, "Current status", [
        "attack semantics are already learnable; move coordination remains the main open problem",
    ], PALETTE["white"], accent_bar=PALETTE["accent2"])
    canvas.save(OUTPUT_DIR / "ppt_algorithm_evolution_timeline.svg")


def draw_v1_v3_family() -> None:
    canvas = SvgCanvas(1600, 980, "Residual adapter family")
    canvas.add(text(70, 62, "V1-V3 Residual Adapter Family", size=34, weight="700"))
    canvas.add(text(70, 96, "One backbone, three increasingly semantic communication carriers", size=20, fill=PALETTE["muted"]))

    node(canvas, 70, 180, 230, 120, "Shared Backbone", ["obs_i -> FC -> GRU", "produce agent hidden h_i"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 360, 90, 260, 120, "Queries / Keys", ["q_i = W_q h_i", "k_j = W_k h_j", "top-k sparse attention"], PALETTE["comm"], accent_bar=PALETTE["accent2"])
    node(canvas, 360, 260, 260, 180, "Value Carrier", ["V1: value from hidden h_j", "V2: same carrier as V1", "V3: value from action probs/logits", "optional detach for backbone or intent"], PALETTE["comm"], accent_bar=PALETTE["accent2"])
    node(canvas, 690, 180, 230, 120, "Message", ["alpha_ij * value_j", "flatten heads", "LayerNorm"], PALETTE["comm"], accent_bar=PALETTE["accent2"])
    node(canvas, 980, 180, 270, 120, "Residual Adapter", ["gate(h_i, m_i)", "zero-init residual projection", "small residual_comm_scale"], PALETTE["comm"], accent_bar=PALETTE["accent2"])
    node(canvas, 1310, 180, 220, 120, "Policy Output", ["fused hidden = h_i + delta_i", "MLP -> logits"], PALETTE["actor"], accent_bar=PALETTE["accent"])

    canvas.add(line(300, 240, 360, 150))
    canvas.add(line(300, 240, 360, 350))
    canvas.add(line(620, 150, 690, 240))
    canvas.add(line(620, 350, 690, 240))
    canvas.add(line(920, 240, 980, 240))
    canvas.add(line(1250, 240, 1310, 240))

    node(canvas, 70, 520, 430, 150, "V1: Minimal Residual Communication", [
        "single-head, top-k=2, comm_value_dim=4",
        "gate bias = -2.5, residual projection zero initialized",
        "goal: verify that a tiny adapter can be attached without breaking the backbone",
    ], PALETTE["white"], accent_bar=PALETTE["accent"])
    node(canvas, 560, 520, 430, 150, "V2: Sharpened Routing", [
        "keep the same network structure as V1",
        "add tiny attention-entropy regularization or faster comm learning",
        "goal: force attention to stop averaging neighbors blindly",
    ], PALETTE["white"], accent_bar=PALETTE["accent3"])
    node(canvas, 1050, 520, 480, 150, "V3: Action-Intention Sharing", [
        "the key structural change is the message carrier",
        "send action intention instead of abstract hidden features",
        "goal: make messages physically meaningful and easier for the actor to exploit",
    ], PALETTE["white"], accent_bar=PALETTE["accent2"])

    node(canvas, 70, 740, 1460, 150, "Talk Track for the Slide", [
        "V1 asks whether communication can be injected safely.",
        "V2 asks whether the same branch becomes useful once routing is sharpened.",
        "V3 asks whether the problem is not routing itself, but the semantic content of the message.",
    ], PALETTE["warning"], accent_bar="#c79a00")

    canvas.save(OUTPUT_DIR / "ppt_v1_v3_residual_family.svg")


def draw_v4() -> None:
    canvas = SvgCanvas(1600, 980, "Targeted attack-subspace fusion")
    canvas.add(text(70, 62, "V4 Attack-Subspace Targeted Fusion", size=34, weight="700"))
    canvas.add(text(70, 96, "Communication no longer modifies the whole hidden state. It only biases the attack logits.", size=20, fill=PALETTE["muted"]))

    node(canvas, 70, 170, 250, 120, "Backbone Actor", ["obs_i -> FC -> GRU", "policy head outputs local logits"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 380, 170, 240, 120, "Logit Split", ["move/base logits", "attack logits"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 690, 70, 330, 180, "Attack Intent Message", ["attack_probs_j", "can_attack_j", "top1_mass_j", "value_proj -> message value"], PALETTE["attack"], accent_bar="#d24b4b")
    node(canvas, 690, 300, 330, 180, "Relation-Aware Routing", ["query from h_i", "key from [h_j, relation_ij]", "top-1 routing", "optional no-comm token"], PALETTE["comm"], accent_bar=PALETTE["accent2"])
    node(canvas, 1080, 170, 240, 120, "Attack Fusion", ["LayerNorm(message)", "attack_gate(h_i, m_i)", "delta_attack(h_i, m_i)"], PALETTE["attack"], accent_bar="#d24b4b")
    node(canvas, 1370, 170, 170, 120, "Final Output", ["move logits unchanged", "attack logits corrected", "concat -> final logits"], PALETTE["actor"], accent_bar=PALETTE["accent"])

    canvas.add(line(320, 230, 380, 230))
    canvas.add(line(620, 230, 690, 160))
    canvas.add(line(620, 230, 690, 390))
    canvas.add(line(1020, 160, 1080, 230))
    canvas.add(line(1020, 390, 1080, 230))
    canvas.add(line(1320, 230, 1370, 230))

    node(canvas, 70, 560, 1460, 250, "What changes relative to V3", [
        "messages are no longer generic residual features; they are targeted, relation-aware attack signals",
        "a no-communication token lets the actor explicitly ignore communication when local policy is enough",
        "the fusion location is narrowed from the whole actor hidden state to the attack action subspace only",
        "this yields a much cleaner interpretation: communication participates in focus-fire decisions, not every movement reflex",
    ], PALETTE["white"], accent_bar=PALETTE["accent3"], dashed=True)

    pill(canvas, 1090, 115, 280, 42, "final_attack = local_attack + scale * gate * delta", PALETTE["warning"], stroke="#d0b25c")
    canvas.save(OUTPUT_DIR / "ppt_v4_targeted_fusion_structure.svg")


def draw_v5() -> None:
    canvas = SvgCanvas(1680, 1040, "Dual-stream communication")
    canvas.add(text(70, 62, "V5 Dual-Stream Targeted Fusion", size=34, weight="700"))
    canvas.add(text(70, 96, "Attack and movement are treated as two different communication problems.", size=20, fill=PALETTE["muted"]))

    node(canvas, 70, 180, 240, 120, "Shared Backbone", ["obs_i -> FC -> GRU", "policy head -> local logits"], PALETTE["actor"], accent_bar=PALETTE["accent"])
    node(canvas, 360, 180, 220, 120, "Logit Split", ["move logits", "attack logits"], PALETTE["actor"], accent_bar=PALETTE["accent"])

    node(canvas, 650, 80, 370, 210, "Attack Stream", [
        "carrier: attack_probs + can_attack + top1_mass",
        "relation-aware routing with top-1",
        "optional no-comm token",
        "softplus/sigmoid gate + delta_attack",
    ], PALETTE["attack"], accent_bar="#d24b4b")
    node(canvas, 650, 350, 370, 230, "Move Stream", [
        "carrier: move_probs + own_state + top1_mass",
        "separate query/key/value projections",
        "move_topk can differ from attack_topk",
        "softplus move gate + delta_move",
        "optional distance penalty",
    ], PALETTE["move"], accent_bar=PALETTE["accent"])

    node(canvas, 1080, 80, 250, 120, "Attack Fusion", ["correct attack logits only", "attack_fusion_scale * gate * delta"], PALETTE["attack"], accent_bar="#d24b4b")
    node(canvas, 1080, 410, 250, 120, "Move Fusion", ["correct move logits only", "move_fusion_scale * gate * delta"], PALETTE["move"], accent_bar=PALETTE["accent"])
    node(canvas, 1390, 215, 200, 130, "Final Output", ["concat corrected move logits", "+ corrected attack logits"], PALETTE["actor"], accent_bar=PALETTE["accent"])

    canvas.add(line(310, 240, 360, 240))
    canvas.add(line(580, 240, 650, 170))
    canvas.add(line(580, 240, 650, 465))
    canvas.add(line(1020, 185, 1080, 140))
    canvas.add(line(1020, 465, 1080, 470))
    canvas.add(line(1330, 140, 1390, 260))
    canvas.add(line(1330, 470, 1390, 300))

    node(canvas, 70, 670, 1520, 270, "Current diagnostic lesson from V5", [
        "the attack stream already becomes very sharp: attack attention entropy is near zero in successful runs",
        "the move stream is the hard part: top-2 routing often stays close to a symmetric average over two neighbors",
        "therefore the next improvement should not simply increase move gain; it should break move top-2 symmetry",
        "this is why V5 is valuable even when it does not dominate all metrics: it reveals that attack and movement require different routing biases",
    ], PALETTE["white"], accent_bar=PALETTE["accent2"], dashed=True)

    pill(canvas, 1170, 588, 300, 42, "Current bottleneck: move stream still averages", PALETTE["warning"], stroke="#d0b25c")
    canvas.save(OUTPUT_DIR / "ppt_v5_dualstream_structure.svg")


def main() -> None:
    draw_backbone()
    draw_timeline()
    draw_v1_v3_family()
    draw_v4()
    draw_v5()
    print(f"Generated SVG assets in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
