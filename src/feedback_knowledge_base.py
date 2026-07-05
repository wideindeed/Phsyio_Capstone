"""
Lightweight retrieval-augmented grounding layer for get_rep_feedback().

Curated, clinically-phrased corrective snippets keyed by (exercise, detected
issue). get_rep_feedback() looks up the relevant snippet(s) for whatever the
per-exercise FSM/BiLSTM flagged and injects them into the Groq prompt as
grounding context, instead of sending the raw issue label into a free-form
prompt. This is the lightweight version of the retrieval step described in
UbiPhysio (ACM IMWUT 2024, DOI 10.1145/3643552): a dict/lookup table keyed by
issue type rather than a vector DB, since the issue vocabulary here is small
and fixed (produced by the FSM logic in engine.py / *_analyzer.py, not
free text).

Each entry is deliberately written the way a physiotherapist would phrase a
correction cue: names the likely compensation mechanism and gives one
concrete instruction, not just a restatement of the issue label.
"""

_GENERIC_COMPENSATION = (
    "A generic compensatory-motion score below the form threshold usually "
    "means the primary target muscle was not doing enough of the work -- "
    "the patient likely substituted momentum, a different joint, or reduced "
    "range of motion to complete the rep. Cue: slow the rep down and "
    "isolate the target joint before adding more reps."
)

KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    "Deep Squat": {
        "CRITICAL LEAN": (
            "Excessive forward trunk lean during the descent shifts load off "
            "the quadriceps/glutes onto the lower back and often signals "
            "limited ankle dorsiflexion or core bracing. Cue: keep the chest "
            "tall and drive the knees forward over the toes instead of "
            "hinging at the hips."
        ),
        "Chest Up": (
            "A mild anterior trunk lean beyond the height-adjusted tolerance "
            "is an early sign of the same compensation as critical lean, "
            "just less severe. Cue: lift the chest and brace the core before "
            "descending further."
        ),
        "BACK ROUNDING": (
            "Thoracic/lumbar rounding under load increases shear stress on "
            "the spine and usually means the core is not braced against the "
            "descent. Cue: take a breath, brace the abdomen, and keep the "
            "spine in a neutral line throughout the squat."
        ),
        "Compensatory Motion Detected": _GENERIC_COMPENSATION,
    },
    "Sit to Stand": {
        "Compensatory Motion Detected": (
            "In a sit-to-stand transfer, a low form score usually reflects "
            "using arm-push or momentum to rise instead of controlled hip "
            "and knee extension. Cue: lean the chest forward over the knees "
            "and stand up slowly using the legs, not a rocking motion."
        ),
    },
    "Push-Up": {
        "Hip Sag": (
            "The hips dropping below the shoulder-ankle line means the core "
            "and glutes are not holding the plank position under load. Cue: "
            "squeeze the glutes and brace the abdomen to keep the body in a "
            "straight line."
        ),
        "Hip Pike": (
            "The hips rising above the shoulder-ankle line usually means the "
            "patient is unloading the shoulders by pushing the hips up "
            "rather than lowering the chest with control. Cue: lower the "
            "hips back into a straight plank line."
        ),
        "Head Down": (
            "Dropping the head/neck out of alignment with the torso adds "
            "cervical strain and often accompanies fatigue. Cue: keep the "
            "gaze slightly ahead of the hands, neck in line with the spine."
        ),
        "Elbow Flare": (
            "Elbows splaying wide of the wrists at the bottom of the rep "
            "increases shoulder-joint stress. Cue: keep the elbows tracking "
            "closer to the body, roughly a 45-degree angle from the torso."
        ),
        "Compensatory Motion Detected": _GENERIC_COMPENSATION,
    },
    "Bicep Curl": {
        "Drag Cheat (Elbow Shift)": (
            "The elbow drifting forward/backward during the curl means the "
            "shoulder is contributing lift instead of the elbow flexors "
            "working in isolation. Cue: pin the elbow to your side for the "
            "whole rep."
        ),
        "Half Rep (Incomplete ROM)": (
            "The rep did not reach full range of motion, which reduces "
            "training effect and can mask weakness at end-range. Cue: curl "
            "all the way up and lower all the way down each rep."
        ),
        "Heave Cheat (Back Momentum)": (
            "A backward trunk lean or torso heave at the start of the curl "
            "means momentum, not the biceps, is initiating the lift. Cue: "
            "keep the torso still and curl using only the arm."
        ),
        "Swing Cheat (Shoulder Leverage)": (
            "Swinging the upper arm forward from the shoulder turns the "
            "movement into a shoulder-driven swing rather than an elbow "
            "curl. Cue: keep the upper arm fixed and only move at the elbow."
        ),
    },
    "Lateral Raise": {
        "Momentum Cheat (Swinging / Too Fast)": (
            "Raising the arms too quickly relies on momentum rather than "
            "deltoid control, reducing time-under-tension. Cue: raise and "
            "lower the arms at a slow, controlled tempo."
        ),
        "Half Rep (Incomplete ROM)": (
            "Stopping before shoulder height under-trains the deltoid "
            "through its full range. Cue: raise the arms to shoulder "
            "height before lowering."
        ),
        "Raise Arms Slightly Higher": (
            "The peak height reached was just under the target range -- a "
            "small, correctable shortfall rather than a compensation "
            "pattern. Cue: lift a little higher, to shoulder level."
        ),
        "Asymmetric Raise (Uneven Arms)": (
            "A meaningful height difference between the two arms suggests "
            "unilateral weakness or a stability compensation on one side. "
            "Cue: focus on raising both arms evenly and at the same speed."
        ),
        "Shrugging (Trapezius Compensation)": (
            "The shoulders elevating toward the ears during the raise means "
            "the upper trapezius is compensating for deltoid weakness. Cue: "
            "keep the shoulders down and away from the ears as you lift."
        ),
    },
    "Knee Extension": {
        "Compensatory Motion Detected": (
            "In a standing/seated knee extension, a low form score usually "
            "means trunk sway or hip movement is substituting for isolated, "
            "controlled knee extension. Cue: keep the trunk still and "
            "extend the knee slowly through its full range."
        ),
    },
    "Wall Push-Up": {
        "Compensatory Motion Detected": (
            "In a wall push-up, a low form score usually reflects hip "
            "sag/pike or incomplete elbow bend rather than a controlled "
            "plank-to-wall press. Cue: keep the body in a straight line and "
            "bend the elbows fully before pressing back."
        ),
    },
    "Hip March": {
        "Compensatory Motion Detected": (
            "In a standing hip march, a low form score usually means trunk "
            "lean or hip hiking is substituting for controlled hip flexion. "
            "Cue: keep the trunk upright and lift the knee with control, "
            "without leaning to one side."
        ),
    },
    "Shoulder Extension": {
        "Compensatory Motion Detected": (
            "In a standing shoulder extension, a low form score usually "
            "means trunk lean or momentum is substituting for controlled "
            "arm movement behind the body. Cue: keep the torso still and "
            "move only the arm through a slow, controlled range."
        ),
    },
    "Shoulder Scaption": {
        "Compensatory Motion Detected": (
            "In a shoulder scaption raise, a low form score usually means "
            "shrugging or trunk lean is substituting for controlled "
            "deltoid-driven lift along the scapular plane. Cue: keep the "
            "shoulders down and raise the arm slowly along a diagonal line."
        ),
    },
}


def get_grounding_snippet(exercise: str, issue: str) -> str | None:
    """Look up the clinical grounding snippet for one detected issue."""
    exercise_kb = KNOWLEDGE_BASE.get(exercise)
    if not exercise_kb:
        return None
    return exercise_kb.get(issue)


def get_grounding_context(exercise: str, issues: list[str]) -> str:
    """
    Build the grounding context block for the Groq prompt from a list of
    detected issues. Returns an empty string if there's nothing to ground
    (e.g. no issues, or an exercise/issue not in the knowledge base).
    """
    snippets = []
    for issue in issues:
        snippet = get_grounding_snippet(exercise, issue)
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    return "\n".join(f"- {s}" for s in snippets)
