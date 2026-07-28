"""Prompt overlays applied by the local adapter without editing official code.

Keep paths/hardware in settings.json. Put prompt experiments here so each profile
is versioned, readable and usable as an ablation.
"""
from __future__ import annotations

from typing import Dict

DOTA_CLASSES = (
    "plane, ship, storage tank, baseball diamond, tennis court, basketball court, "
    "ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, "
    "roundabout, soccer ball field, swimming pool"
)

DOTA_ALIASES = """
Canonical DOTA mapping rules:
- bus, truck, lorry, trailer, tractor-trailer, heavy road vehicle -> large vehicle
- car, sedan, SUV, pickup, compact van -> small vehicle
- airplane, aircraft, jet -> plane
- boat, vessel -> ship
The final class must be exactly one canonical DOTA label.
""".strip()

DOTA_VISUAL_CUES = """
Compact DOTA visual ontology (use only together with image evidence):
- storage tank: isolated or grouped circular industrial tanks
- baseball diamond: fan/diamond-shaped infield
- tennis court: narrow rectangular court with a central net and service markings
- basketball court: rectangular court with basketball markings/hoops when visible
- ground track field: oval running track surrounding an infield
- soccer ball field: large rectangular pitch with field markings/goals when visible
- swimming pool: rectangular blue water basin
- roundabout: circular road intersection with a central island
- harbor: waterfront docking/berthing region; bridge: narrow structure spanning a gap
Object size words in DOTA are dataset categories: map truck/bus/trailer to large vehicle
and car/sedan/SUV/pickup to small vehicle. These cues do not prove a class by themselves.
""".strip()

PROFILES: Dict[str, Dict[str, str]] = {
    "official_general": {"reasoner": "", "perceiver": "", "verifier": ""},

    # Kept unchanged as the v4.2 baseline prompt for controlled comparisons.
    "obb_grounding_v1": {
        "reasoner": f"""
[Project-specific OBB extension]
The task may request a canonical DOTA class and a rotated bounding box. The canonical labels are: {DOTA_CLASSES}.
{DOTA_ALIASES}
For grounding, use an evidence sequence rather than guessing immediately:
1. identify the referred target and its canonical DOTA class;
2. establish its image region and distinguishing anchors;
3. request a coarse axis-aligned envelope;
4. request orientation and four OBB corners only after identity is stable;
5. verify class, image bounds, target coverage and point order.
It is allowed to ask the Perceiver for approximate coordinates when the query explicitly asks for coordinates. Ask only one atomic fact per round. Prefer normalized [0,1000] coordinates when the query states that convention.
""".strip(),
        "perceiver": f"""
[Project-specific OBB extension]
Answer only the current atomic visual question. The canonical DOTA labels are: {DOTA_CLASSES}.
{DOTA_ALIASES}
When asked for localization after the target is unambiguous, estimate either bbox_2d=[x1,y1,x2,y2] or obb_8=[x1,y1,x2,y2,x3,y3,x4,y4] using the coordinate convention stated in the question. Do not generate a full self-dialogue. State uncertainty instead of inventing a different target.
""".strip(),
        "verifier": f"""
[Project-specific verifier extension]
{DOTA_ALIASES}
Accept a natural subtype only if it maps to the correct canonical DOTA class. For grounding, reject a different object, class or region. Do not require exact coordinate equality during trajectory synthesis when GT teacher forcing is enabled, but require evidence that the intended target was identified.
""".strip(),
    },

    # Recommended v4.2.2 prompt. It preserves the official Plan–Integrate–Decide
    # loop while making the atomic questions suitable for DOTA OBB grounding.
    "obb_grounding_v2": {
        "reasoner": f"""
[DOTA OBB evidence-seeking extension v2]
The canonical DOTA labels are: {DOTA_CLASSES}.
{DOTA_ALIASES}
Preserve every numeric convention and output constraint from the original query.
For a grounding task, follow this order and avoid repeating equivalent questions:
1. identify one referred target, its visible subtype and canonical DOTA class;
2. confirm the target region and the single most useful distinguishing anchor;
3. ask for one coarse HBB of that target;
4. ask for its orientation and four OBB corners;
5. perform one concise consistency check, then finalize.
When asking for coordinates, repeat the requested coordinate range and top-left origin in the atomic question. Ask only about the single referred target, never coordinates of all similar objects. Do not spend multiple rounds re-confirming the same category. A final answer must use exactly the class and coordinate format requested by the original query.
""".strip(),
        "perceiver": f"""
[DOTA OBB perception extension v2]
Answer only the current atomic visual question in at most three short sentences or one requested coordinate structure. The canonical labels are: {DOTA_CLASSES}.
{DOTA_ALIASES}
Normalized coordinates are defined by the image frame; no physical map scale is needed. Never refuse a coordinate estimate merely because the image has no numeric scale. When a target is unambiguous, estimate the requested bbox_2d or clockwise obb_8 in the stated range. Do not create a self-Q&A, do not solve unrelated parts of the original task, and do not switch to another object when uncertain.
""".strip(),
        "verifier": f"""
[DOTA OBB verifier extension v2]
{DOTA_ALIASES}
Judge target identity, canonical class and image region first. Reject a different object, category or cluster. During trajectory synthesis, approximate coordinates may differ from GT when teacher forcing is enabled, but the evidence must consistently refer to the intended target. Do not reject only because a natural subtype was used when it maps correctly to the canonical DOTA label.
""".strip(),
    },


    "obb_grounding_v3": {
        "reasoner": f"""
[DOTA OBB evidence-seeking extension v3]
The canonical DOTA labels are: {DOTA_CLASSES}.
{DOTA_ALIASES}
{DOTA_VISUAL_CUES}
Every response must begin with exactly one <thinking>...</thinking> block and then contain exactly one <question>...</question> OR one [Final Answer]: line. Never omit the tags.
Preserve every numeric convention and output constraint from the original query.
These DOTA tasks already provide a target reference or focus ROI. Skip the generic
whole-scene survey: start with the exact pointed target and keep every later
question bound to that same single object.
For grounding, use this non-repeating sequence: target identity and region -> one distinguishing anchor -> one coarse HBB -> orientation and clockwise OBB -> one consistency check -> final answer. Do not ask the same yes/no category question twice.
For classification, every visual question must be class-neutral. Never include a
canonical label or alias, never ask "what category/class/label is it", and never
ask the Perceiver to choose among candidates. Ask about the single target's
visible shape, parts, orientation, local support/surface, and scale. Do not treat
an airport, marina, parking lot, road, building, field, or other surrounding
scene region as the target object.
When asking for coordinates, repeat: normalized integer coordinates in [0,1000], top-left origin, single referred target only.
If evidence is uncertain, ask one discriminative question rather than switching to another object. On the final allowed round, finalize using the exact requested format.
""".strip(),
        "perceiver": f"""
[DOTA OBB perception extension v3]
Answer only the current atomic visual question in at most three concise sentences or one coordinate structure. The canonical labels are: {DOTA_CLASSES}.
{DOTA_ALIASES}
{DOTA_VISUAL_CUES}
Use the image frame for coordinates; no physical map scale is required. For coordinate requests, use normalized integer coordinates in [0,1000] with top-left origin. Return bbox_2d=[xmin,ymin,xmax,ymax] or clockwise obb_8=[x1,y1,x2,y2,x3,y3,x4,y4].
The user message includes an Original target context block and a Current atomic
visual question. Use the original block only to resolve the target/focus ROI,
then answer only the atomic question. Do not answer a class-neutral question by
guessing a canonical label. Do not switch to another object, describe a
surrounding scene as though it were the target, or repeat the full original
task. If uncertain, state one concrete uncertainty.
""".strip(),
        "verifier": f"""
[DOTA OBB verifier extension v3]
{DOTA_ALIASES}
Judge target identity, canonical class, region, and internal consistency. Do not demand exact numeric equality when GT teacher forcing and the deterministic geometry gate are enabled. A different object, unresolved contradiction, missing target, wrong class, or wrong image region must be rejected. Approximate coordinates may be accepted only when they refer to the correct target region.
""".strip(),
    },

    "obb_grounding_no_alias": {
        "reasoner": f"""
[OBB ablation without ontology hints]
The task may request a DOTA class and rotated box. Allowed final labels are: {DOTA_CLASSES}.
Use the sequence target identity -> region/anchors -> coarse HBB -> orientation/OBB -> geometry check.
Do not use any explicit mapping from natural subtypes to canonical labels.
""".strip(),
        "perceiver": "Answer only the current atomic visual question. Estimate coordinates only after the target is unambiguous.",
        "verifier": "Require an exact canonical DOTA label; do not apply subtype aliases.",
    },
    "single_glance_ablation": {
        "reasoner": "Ask at most one broad visual question, then answer. This is an intentional single-glance ablation.",
        "perceiver": "Answer the one visual question concisely.",
        "verifier": "Judge normally.",
    },
    "no_bbox_questions_ablation": {
        "reasoner": "Do not ask for numeric coordinates. Use only semantic and spatial evidence before the final answer.",
        "perceiver": "Do not return numeric coordinates.",
        "verifier": "Judge normally.",
    },
}


def get_overlay(profile: str, role: str) -> str:
    if profile not in PROFILES:
        raise KeyError(f"Unknown prompt profile: {profile}. Available: {sorted(PROFILES)}")
    return PROFILES[profile].get(role, "").strip()


def apply_overlay(system_prompt: str | None, model_name: str, profile: str) -> str | None:
    base = system_prompt or ""
    low = base.lower()
    if "instruction rewriter" in low:
        return system_prompt
    if model_name == "gpt-5-mini":
        role = "reasoner"
    elif model_name == "gemini-2.5-flash":
        role = "perceiver"
    elif model_name == "doubao-seed-1-6-thinking-250715":
        role = "verifier"
    else:
        return system_prompt
    overlay = get_overlay(profile, role)
    return base if not overlay else f"{base.rstrip()}\n\n{overlay}\n"
