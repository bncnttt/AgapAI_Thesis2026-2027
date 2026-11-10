export const DISASTER_OPTIONS = [
  ["all", "All Disasters"],
  ["flood", "Flood"],
  ["fire", "Fire"],
  ["landslide", "Landslide"],
  ["typhoon", "Typhoon"],
  ["earthquake", "Earthquake"],
];

export const ROLE_OPTIONS = [
  ["all", "All Roles"],
  ["victim", "Victim"],
  ["volunteer", "Volunteer"],
  ["general", "General Info"],
];

const ROLE_PATTERNS = {
  victim: /victim/,
  volunteer: /volunteer/,
  general: /general/,
};

export const DEFAULT_FILTERS = { disasterType: "all", roleFilter: "all", authorHandleQuery: "" };

export function normalizeHandle(value) {
  return String(value || "").trim().replace(/^@+/, "").toLowerCase();
}

export function matchesHandle(handle, query) {
  const q = normalizeHandle(query);
  return !q || normalizeHandle(handle).includes(q);
}

export function matchesDisaster(type, filters) {
  return filters.disasterType === "all" || String(type || "").toLowerCase() === filters.disasterType;
}

export function matchesRole(role, filters) {
  if (filters.roleFilter === "all") return true;
  return ROLE_PATTERNS[filters.roleFilter].test(String(role || "").toLowerCase());
}

export function postMatches(post, filters) {
  return (
    matchesDisaster(post.disaster_type, filters)
    && matchesRole(post.role, filters)
    && matchesHandle(post.author_handle, filters.authorHandleQuery)
  );
}

export function filterEvents(events, filters) {
  return events
    .filter((event) => matchesDisaster(event.disaster_type, filters))
    .map((event) => {
      const posts = (event.posts || []).filter(
        (post) => matchesRole(post.role, filters) && matchesHandle(post.author_handle, filters.authorHandleQuery)
      );
      const visiblePostIds = new Set(posts.map((post) => post._id));
      const locations = (event.locations || []).map((location) => {
        if (!Array.isArray(location.post_ids)) return location;
        const post_ids = location.post_ids.filter((postId) => visiblePostIds.has(postId));
        return { ...location, post_ids, post_count: post_ids.length };
      }).filter((location) => !Array.isArray(location.post_ids) || location.post_ids.length > 0);
      return { ...event, posts, locations, post_count: posts.length };
    })
    .filter((event) => event.posts.length > 0);
}
