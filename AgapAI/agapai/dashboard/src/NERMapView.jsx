import { useEffect, useMemo } from "react";
import { filterEvents } from "./filters";
import L from "leaflet";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const CEBU_CENTER = [10.3157, 123.8854];

const DISASTER_PIN_TYPES = [
  {
    color: "#3B82F6",
    label: "Flood",
    iconPath: '<path d="M12 2C9 6 6 9.7 6 13a6 6 0 0 0 12 0c0-3.3-3-7-6-11z"/><path d="M8 15c1.2 1 2.5 1.5 4 1.5s2.8-.5 4-1.5"/>',
  },
  {
    color: "#8B4513",
    label: "Landslide",
    iconPath: '<path d="m3 20 7-14 4 8 2-4 5 10H3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
  },
  {
    color: "#EF4444",
    label: "Fire",
    iconPath: '<path d="M8.5 14.5A4.5 4.5 0 0 0 17 12c0-4-4-7-4-10-2 2-3 4-3 6-1.7-1-2.5-2.5-2.5-4.5C5.5 6 4 8.5 4 12a8 8 0 0 0 16 0"/><path d="M12 22a3 3 0 0 0 3-3c0-1.5-1.5-2.5-3-4-1.5 1.5-3 2.5-3 4a3 3 0 0 0 3 3z"/>',
  },
  {
    color: "#F97316",
    label: "Earthquake",
    iconPath: '<path d="M22 12h-4l-3 8-6-16-3 8H2"/><path d="m13 2-2 7h4l-2 7"/>',
  },
  {
    color: "#9333EA",
    label: "Typhoon",
    iconPath: '<path d="M17.7 7.7A7.7 7.7 0 0 0 4 12"/><path d="M6.3 16.3A7.7 7.7 0 0 0 20 12"/><path d="M2 12h12"/><path d="M10 8h10"/><path d="M6 16h12"/>',
  },
];

function getDisasterConfig(disasterType) {
  const normalizedType = String(disasterType || "").toLowerCase();
  return DISASTER_PIN_TYPES.find((type) => type.label.toLowerCase() === normalizedType);
}

function getPinSvg(config, size = 22) {
  return `
    <svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"
      stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"
      aria-hidden="true">
      ${config.iconPath}
    </svg>
  `;
}

function getDisasterIcon(disasterType) {
  const config = getDisasterConfig(disasterType);

  return L.divIcon({
    className: "agapai-disaster-marker",
    iconSize: [38, 46],
    iconAnchor: [19, 44],
    popupAnchor: [0, -40],
    html: `
      <div style="
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        background: ${config.color};
        border: 3px solid white;
        border-radius: 50% 50% 50% 0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.35);
        transform: rotate(-45deg);
      ">
        <span style="display: grid; place-items: center; transform: rotate(45deg);">
          ${getPinSvg(config)}
        </span>
      </div>
    `,
  });
}

function MapLegend() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        alignItems: "stretch",
        padding: "10px 12px",
        background: "#ffffff",
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        boxShadow: "0 8px 24px rgba(15, 23, 42, 0.18)",
        position: "absolute",
        top: 16,
        right: 16,
        zIndex: 1000,
      }}
    >
      {DISASTER_PIN_TYPES.map((type) => (
        <div
          key={type.label}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            color: "#111827",
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          <span
            style={{
              width: 24,
              height: 24,
              display: "grid",
              placeItems: "center",
              background: type.color,
              borderRadius: "50%",
            }}
            dangerouslySetInnerHTML={{ __html: getPinSvg(type, 15) }}
          />
          <span>{type.label}</span>
        </div>
      ))}
    </div>
  );
}

function EventBounds({ markers, isActive }) {
  const map = useMap();
  useEffect(() => {
    if (!isActive) return;
    let frame;
    const resize = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        map.invalidateSize({ pan: false });
        if (markers.length) {
          map.fitBounds(markers.map((marker) => [marker.lat, marker.lon]), {
            paddingTopLeft: [50, 50], paddingBottomRight: [190, 50], maxZoom: 14,
          });
        }
      });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(map.getContainer());
    resize();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [markers, isActive, map]);
  return null;
}

export default function NERMapContent({ allEvents, loading, onRefresh, hasLoadedPosts, onViewPosts, filters, isActive }) {
  const events = useMemo(() => filterEvents(allEvents, filters).filter((event) =>
    getDisasterConfig(event.disaster_type)
  ), [allEvents, filters]);
  const markers = useMemo(() => events.flatMap((event) => {
    const locations = event.locations?.length ? event.locations : [{
      id: event.event_id,
      place: event.place,
      lat: event.lat,
      lon: event.lon,
      post_ids: (event.posts || []).map((post) => post._id),
      post_count: event.post_count,
    }];
    return locations
      .filter((location) => Number.isFinite(location.lat) && Number.isFinite(location.lon))
      .map((location, index) => {
        const postIds = location.post_ids || (event.posts || []).map((post) => post._id);
        return {
          ...location,
          key: `${event.event_id}:${location.id || location.psgc_code || index}`,
          event,
          post_ids: postIds,
          post_count: location.post_count ?? postIds.length,
        };
      });
  }), [events]);

  return (
    <>
      <section className="toolbar" style={{ marginTop: -4 }}>
        <button type="button" onClick={onRefresh} disabled={loading || !hasLoadedPosts}>
          Refresh Spatiotemporal Info
        </button>
      </section>

      <div style={{ position: "relative" }}>
        <MapLegend />
        <MapContainer center={CEBU_CENTER} zoom={10} style={{ height: "600px", width: "100%" }}>
          <EventBounds markers={markers} isActive={isActive} />
          <TileLayer
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          {markers.map((marker) => {
            const { event } = marker;
            const disasterConfig = getDisasterConfig(event.disaster_type);

            return (
              <Marker
                key={marker.key}
                position={[marker.lat, marker.lon]}
                icon={getDisasterIcon(event.disaster_type)}
              >
                <Popup>
                  <div>
                    <div
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "6px 10px",
                        borderRadius: 999,
                        color: "white",
                        background: disasterConfig.color,
                        fontWeight: 700,
                        marginBottom: 8,
                      }}
                    >
                      <span
                        dangerouslySetInnerHTML={{ __html: getPinSvg(disasterConfig, 18) }}
                      />
                      <span>{event.disaster_type || disasterConfig.label}</span>
                    </div>
                    <p>Location: {marker.place || event.place || "Unknown"}</p>
                    <p>Date: {event.event_date || "Unknown"}</p>
                    <p>Time (PHT): {event.event_time || "Unknown"}</p>
                    <p>{marker.post_count} post{marker.post_count === 1 ? "" : "s"} mentioning this location</p>
                    {marker.post_count > 0 && (
                      <button
                        type="button"
                        onClick={() => onViewPosts?.({
                          eventId: event.event_id,
                          postIds: marker.post_ids,
                        })}
                        style={{ fontSize: 12, padding: "4px 10px" }}
                      >
                        View Posts
                      </button>
                    )}
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
    </>
  );
}
