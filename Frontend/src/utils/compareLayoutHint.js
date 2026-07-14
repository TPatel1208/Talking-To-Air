// Compare mode's grid competes with three side panels for width. Auto-
// collapsing them used to happen automatically but that made the layout jump
// around outside the user's control (App.jsx), so collapsing is manual-only
// now -- this just decides when a one-click nudge to do so is worth showing:
// compare is active and every side panel is still taking up room. It hides
// itself the moment any one of them is collapsed, whether via the hint's own
// action or a plain manual toggle elsewhere.
export function shouldShowCollapseHint({ compareMode, sessionsCollapsed, chatCollapsed, rightPanelCollapsed }) {
  return compareMode === 'active' && !sessionsCollapsed && !chatCollapsed && !rightPanelCollapsed
}
