// The widths below which the output column stops being able to answer honestly.
//
// Measured live at a 556 px viewport with the sessions, chat and jobs panels
// open: the output panel rendered 0 px wide and was pushed off-screen at
// left: 634, and `.maplibregl-map` measured 0 px with it. The three side panels
// are all fixed-width and `flexShrink: 0` (232 + 380 + 308 = 920 px), so the
// output panel is the ONLY child that can absorb a narrow viewport -- and with
// `minWidth: 0` it absorbed all of it.
//
// These are floors, not sizes. Above them the panel flexes as it always did.

// The scrub track's own floor. Below this an interactive control cannot
// honestly answer: 260 px over a 49-stop axis is 5.3 px per stop, which is the
// density the track was measured at (284 px) and already at the edge of
// usable -- the smallest empty run real TEMPO data produces is two stops, and
// at this width that is still ~11 px of hatching.
export const SCRUB_TRACK_MIN_WIDTH = 260

// What the output panel pads its content by, either side.
export const PANEL_PADDING_X = 22

// DERIVED, deliberately. The panel exists to contain the track, so its floor is
// the track's floor plus the padding around it rather than a second number
// chosen to look compatible. Two independently-chosen widths is how the inner
// one quietly stops fitting inside the outer one -- the same "one measurement,
// two homes" failure this codebase keeps meeting, in pixels instead of prose.
export const PANEL_MIN_WIDTH = SCRUB_TRACK_MIN_WIDTH + PANEL_PADDING_X * 2
