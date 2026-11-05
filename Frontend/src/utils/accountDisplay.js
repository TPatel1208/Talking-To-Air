/**
 * What the sidebar's account footer and popup show for the signed-in
 * Supabase user. Pure, so it is unit-testable under plain `node --test`.
 */

const NAME_KEYS = ['full_name', 'name', 'display_name']

function nameFromEmail(email) {
  const local = String(email).split('@')[0]
  return local
    .split(/[._-]+/)
    .filter(Boolean)
    .map(word => word[0].toUpperCase() + word.slice(1))
    .join(' ')
}

function initialsOf(name) {
  const words = name.split(/\s+/).filter(Boolean)
  return words.slice(0, 2).map(word => word[0].toUpperCase()).join('') || '?'
}

function calendarDate(value) {
  if (!value) return null
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })
}

export function accountSummary(user) {
  const metadata = user?.user_metadata || {}
  const profileName = NAME_KEYS.map(key => metadata[key]).find(v => typeof v === 'string' && v.trim())
  const email = user?.email || null
  const name = profileName?.trim() || (email ? nameFromEmail(email) : '') || 'Signed in'
  return {
    name,
    initials: name === 'Signed in' ? '?' : initialsOf(name),
    email,
    memberSince: calendarDate(user?.created_at),
    lastSignIn: calendarDate(user?.last_sign_in_at),
  }
}
