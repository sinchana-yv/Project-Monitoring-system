import { useState, useMemo, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

const API = "http://127.0.0.1:5001";

// Authentic centroid coordinates for Indian States & Union Territories
const STATE_COORDINATES = {
  "Andaman & Nicobar": [11.7401, 92.6586],
  "Andhra Pradesh": [15.9129, 79.7400],
  "Arunachal Pradesh": [28.2180, 94.7278],
  "Assam": [26.2006, 92.9376],
  "Bihar": [25.0961, 85.3131],
  "Chhattisgarh": [21.2787, 81.8661],
  "Dadra & Nagar Haveli and Daman & Diu": [20.3974, 72.8328],
  "Delhi": [28.7041, 77.1025],
  "Goa": [15.2993, 74.1240],
  "Gujarat": [22.2587, 71.1924],
  "Haryana": [29.0588, 76.0856],
  "Himachal Pradesh": [31.1048, 77.1734],
  "Jammu and Kashmir": [33.7782, 76.5762],
  "Jharkhand": [23.6102, 85.2799],
  "Karnataka": [15.3173, 75.7139],
  "Kerala": [10.8505, 76.2711],
  "Ladakh": [34.1526, 77.5771],
  "Lakshadweep": [10.5667, 72.6417],
  "Madhya Pradesh": [22.9734, 78.6569],
  "Maharashtra": [19.7515, 75.7139],
  "Manipur": [24.6637, 93.9063],
  "Meghalaya": [25.4670, 91.3662],
  "Mizoram": [23.1645, 92.9376],
  "Nagaland": [26.1584, 94.5624],
  "Odisha": [20.9517, 85.0985],
  "Puducherry": [11.9416, 79.8083],
  "Punjab": [31.1471, 75.3412],
  "Rajasthan": [27.0238, 74.2179],
  "Sikkim": [27.5330, 88.5122],
  "Tamil Nadu": [11.1271, 78.6569],
  "Telangana": [18.1124, 79.0193],
  "Tripura": [23.9408, 91.9882],
  "Uttar Pradesh": [26.8467, 80.9462],
  "Uttarakhand": [30.0668, 79.0193],
  "West Bengal": [22.9868, 87.8550],
};

// Map bounds restricted strictly to India
const INDIA_BOUNDS = [
  [6.5, 68.0],
  [37.5, 97.5],
];
const INDIA_CENTER = [22.5937, 78.9629];

// Helper to resolve coordinates from project state
const getProjectCoordinates = (project) => {
  const stateStr = String(project.state || "").trim();
  let baseCoords = STATE_COORDINATES[stateStr];

  if (!baseCoords && stateStr.startsWith("Multi-States")) {
    const matched = Object.keys(STATE_COORDINATES).filter((s) =>
      stateStr.includes(s)
    );
    if (matched.length > 0) {
      const avgLat =
        matched.reduce((sum, s) => sum + STATE_COORDINATES[s][0], 0) /
        matched.length;
      const avgLng =
        matched.reduce((sum, s) => sum + STATE_COORDINATES[s][1], 0) /
        matched.length;
      baseCoords = [avgLat, avgLng];
    }
  }

  if (!baseCoords) {
    return null;
  }

  // Deterministic tiny offset based on project_code hash so markers in the same state spread neatly on zoom
  const codeNum =
    parseInt(String(project.project_code).replace(/\D/g, ""), 10) || 0;
  const angle = (codeNum % 360) * (Math.PI / 180);
  const radius = 0.02 + ((codeNum % 40) * 0.004);
  const latOffset = Math.sin(angle) * radius;
  const lngOffset = Math.cos(angle) * radius;

  return [baseCoords[0] + latOffset, baseCoords[1] + lngOffset];
};

// Map Controller Component for programmatically flying to locations or resetting view
function MapController({ selectedCoords, resetTrigger }) {
  const map = useMap();

  useEffect(() => {
    if (resetTrigger) {
      map.flyToBounds(INDIA_BOUNDS, { duration: 1.2 });
    }
  }, [resetTrigger, map]);

  useEffect(() => {
    if (selectedCoords) {
      map.flyTo(selectedCoords, 9, { duration: 1.2 });
    }
  }, [selectedCoords, map]);

  return null;
}

export default function GisMap({ projects = [], onOpenProjectDetails }) {
  const [search, setSearch] = useState("");
  const [selectedMinistry, setSelectedMinistry] = useState("ALL");
  const [selectedSector, setSelectedSector] = useState("ALL");
  const [selectedRisk, setSelectedRisk] = useState("ALL");
  const [selectedProject, setSelectedProject] = useState(null);
  const [resetTrigger, setResetTrigger] = useState(0);

  // Extract dynamic ministry & sector options from real projects dataset
  const ministryOptions = useMemo(() => {
    const set = new Set();
    projects.forEach((p) => {
      if (p.ministry_final && String(p.ministry_final).trim()) {
        set.add(String(p.ministry_final).trim());
      }
    });
    return Array.from(set).sort();
  }, [projects]);

  const sectorOptions = useMemo(() => {
    const set = new Set();
    projects.forEach((p) => {
      if (p.sector && String(p.sector).trim()) {
        set.add(String(p.sector).trim());
      }
    });
    return Array.from(set).sort();
  }, [projects]);

  // Prepared project list with coordinates & search tokens
  const processedProjects = useMemo(() => {
    return projects.map((p) => {
      const coords = getProjectCoordinates(p);
      return {
        ...p,
        coords,
        hasCoords: Boolean(coords),
      };
    });
  }, [projects]);

  // Filter projects based on dropdowns & search bar
  const filteredProjects = useMemo(() => {
    const q = search.trim().toLowerCase();

    return processedProjects.filter((p) => {
      const matchesSearch =
        !q ||
        String(p.project_name || "").toLowerCase().includes(q) ||
        String(p.project_code || "").toLowerCase().includes(q) ||
        String(p.state || "").toLowerCase().includes(q) ||
        String(p.ministry_final || "").toLowerCase().includes(q) ||
        String(p.sector || "").toLowerCase().includes(q) ||
        String(p.agency_final || "").toLowerCase().includes(q);

      const matchesMinistry =
        selectedMinistry === "ALL" ||
        String(p.ministry_final).trim() === selectedMinistry;

      const matchesSector =
        selectedSector === "ALL" || String(p.sector).trim() === selectedSector;

      const matchesRisk =
        selectedRisk === "ALL" ||
        String(p.risk_level).toUpperCase() === selectedRisk;

      return (
        matchesSearch && matchesMinistry && matchesSector && matchesRisk
      );
    });
  }, [processedProjects, search, selectedMinistry, selectedSector, selectedRisk]);

  const mappedProjectsCount = useMemo(() => {
    return filteredProjects.filter((p) => p.hasCoords).length;
  }, [filteredProjects]);

  const resetFilters = () => {
    setSearch("");
    setSelectedMinistry("ALL");
    setSelectedSector("ALL");
    setSelectedRisk("ALL");
    setSelectedProject(null);
    setResetTrigger((c) => c + 1);
  };

  const getMarkerColor = (riskLevel) => {
    const r = String(riskLevel).toUpperCase();
    if (r === "HIGH") return "#dc2626"; // Red
    if (r === "MEDIUM") return "#d97706"; // Amber/Orange
    return "#059669"; // Green
  };

  const renderVal = (val, suffix = "") => {
    if (val === null || val === undefined || val === "") {
      return "Unavailable in supplied dataset.";
    }
    return `${val}${suffix}`;
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: "650px",
        background: "#fbfcf8",
        fontFamily: "Arial, sans-serif",
        color: "#1e293b",
      }}
    >
      {/* PAGE HEADER */}
      <div
        style={{
          padding: "16px 24px",
          background: "#1e293b",
          color: "#ffffff",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: "22px", color: "#ffffff" }}>
            National Infrastructure GIS Map
          </h1>
          <p style={{ margin: "4px 0 0 0", fontSize: "13px", color: "#cbd5e1" }}>
            Geospatial visualization of monitored infrastructure projects and their current risk status across India.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setResetTrigger((c) => c + 1)}
          style={{
            padding: "8px 16px",
            background: "#334155",
            color: "#ffffff",
            border: "1px solid #475569",
            borderRadius: "6px",
            fontSize: "13px",
            fontWeight: "600",
            cursor: "pointer",
          }}
        >
          Reset Map
        </button>
      </div>

      {/* FILTER BAR */}
      <div
        style={{
          padding: "12px 24px",
          background: "#ffffff",
          borderBottom: "1px solid #e2e8f0",
          display: "flex",
          gap: "10px",
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <input
          type="text"
          placeholder="Search projects, states, agencies..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            padding: "8px 12px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "13px",
            minWidth: "240px",
            flexGrow: 1,
            outline: "none",
          }}
        />

        <select
          value={selectedMinistry}
          onChange={(e) => setSelectedMinistry(e.target.value)}
          style={{
            padding: "8px 12px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "13px",
            background: "#ffffff",
            cursor: "pointer",
            maxWidth: "220px",
          }}
        >
          <option value="ALL">All Ministries</option>
          {ministryOptions.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>

        <select
          value={selectedSector}
          onChange={(e) => setSelectedSector(e.target.value)}
          style={{
            padding: "8px 12px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "13px",
            background: "#ffffff",
            cursor: "pointer",
            maxWidth: "200px",
          }}
        >
          <option value="ALL">All Sectors</option>
          {sectorOptions.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        <select
          value={selectedRisk}
          onChange={(e) => setSelectedRisk(e.target.value)}
          style={{
            padding: "8px 12px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "13px",
            background: "#ffffff",
            cursor: "pointer",
          }}
        >
          <option value="ALL">All Risk Levels</option>
          <option value="HIGH">High Risk</option>
          <option value="MEDIUM">Medium Risk</option>
          <option value="LOW">Low Risk</option>
        </select>

        <button
          type="button"
          onClick={() => {}}
          style={{
            padding: "8px 16px",
            background: "#1e293b",
            color: "#ffffff",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            fontWeight: "600",
            cursor: "pointer",
          }}
        >
          Filter Map
        </button>

        <button
          type="button"
          onClick={resetFilters}
          style={{
            padding: "8px 14px",
            background: "#f1f5f9",
            color: "#475569",
            border: "1px solid #cbd5e1",
            borderRadius: "6px",
            fontSize: "13px",
            cursor: "pointer",
          }}
        >
          Reset Filters
        </button>

        <div style={{ marginLeft: "auto", fontSize: "12px", color: "#64748b", fontWeight: "600" }}>
          Showing {filteredProjects.length} projects ({mappedProjectsCount} mapped on GIS)
        </div>
      </div>

      {/* MAP & PANEL BODY */}
      <div style={{ display: "flex", flex: 1, position: "relative", overflow: "hidden", minHeight: "550px" }}>
        {/* LEAFLET MAP */}
        <div style={{ flex: 1, height: "100%", position: "relative" }}>
          <MapContainer
            center={INDIA_CENTER}
            zoom={5}
            minZoom={4}
            maxZoom={14}
            maxBounds={INDIA_BOUNDS}
            maxBoundsViscosity={1.0}
            style={{ width: "100%", height: "550px", background: "#e2e8f0" }}
          >
            <TileLayer
              attribution='&copy; <a href="https://carto.com/">CartoDB</a> Positron'
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            />

            <MapController
              selectedCoords={selectedProject?.coords}
              resetTrigger={resetTrigger}
            />

            {filteredProjects.map((p) => {
              if (!p.hasCoords) return null;
              const color = getMarkerColor(p.risk_level);
              const isSelected = selectedProject?.project_code === p.project_code;

              return (
                <CircleMarker
                  key={p.project_code}
                  center={p.coords}
                  radius={isSelected ? 11 : 7}
                  pathOptions={{
                    fillColor: color,
                    fillOpacity: isSelected ? 0.95 : 0.8,
                    color: isSelected ? "#000000" : "#ffffff",
                    weight: isSelected ? 3 : 1.5,
                  }}
                  eventHandlers={{
                    click: () => setSelectedProject(p),
                  }}
                >
                  <Popup>
                    <div style={{ fontSize: "12px", fontFamily: "sans-serif" }}>
                      <strong style={{ display: "block", fontSize: "13px", marginBottom: "4px" }}>
                        {p.project_name}
                      </strong>
                      <div>Code: <strong>{p.project_code}</strong></div>
                      <div>State: {p.state}</div>
                      <div>Risk: <strong style={{ color }}>{p.risk_level}</strong></div>
                      <button
                        type="button"
                        onClick={() => setSelectedProject(p)}
                        style={{
                          marginTop: "8px",
                          padding: "4px 8px",
                          background: "#1e293b",
                          color: "#ffffff",
                          border: "none",
                          borderRadius: "4px",
                          fontSize: "11px",
                          cursor: "pointer",
                          width: "100%",
                        }}
                      >
                        Select Project
                      </button>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>

          {/* OVERLAID RISK LEGEND */}
          <div
            style={{
              position: "absolute",
              bottom: "20px",
              left: "20px",
              background: "#ffffff",
              padding: "10px 14px",
              borderRadius: "8px",
              border: "1px solid #cbd5e1",
              boxShadow: "0 2px 8px rgba(0, 0, 0, 0.12)",
              zIndex: 1000,
              fontSize: "12px",
            }}
          >
            <strong style={{ display: "block", marginBottom: "6px", color: "#1e293b" }}>
              Project Risk Legend
            </strong>
            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span
                  style={{
                    width: "10px",
                    height: "10px",
                    borderRadius: "50%",
                    background: "#dc2626",
                    display: "inline-block",
                  }}
                />
                <span>🔴 HIGH RISK</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span
                  style={{
                    width: "10px",
                    height: "10px",
                    borderRadius: "50%",
                    background: "#d97706",
                    display: "inline-block",
                  }}
                />
                <span>🟠 MEDIUM RISK</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <span
                  style={{
                    width: "10px",
                    height: "10px",
                    borderRadius: "50%",
                    background: "#059669",
                    display: "inline-block",
                  }}
                />
                <span>🟢 LOW RISK</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT SIDE PROJECT INFORMATION PANEL */}
        {selectedProject && (
          <aside
            style={{
              width: "360px",
              maxWidth: "100%",
              height: "550px",
              background: "#ffffff",
              borderLeft: "1px solid #cbd5e1",
              boxShadow: "-4px 0 14px rgba(0, 0, 0, 0.08)",
              overflowY: "auto",
              padding: "20px",
              display: "flex",
              flexDirection: "column",
              zIndex: 1001,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                marginBottom: "14px",
                paddingBottom: "10px",
                borderBottom: "1px solid #e2e8f0",
              }}
            >
              <div>
                <span
                  className={`risk-badge ${String(selectedProject.risk_level).toLowerCase()}`}
                  style={{ marginBottom: "6px", display: "inline-block" }}
                >
                  {selectedProject.risk_level} RISK
                </span>
                <h3 style={{ margin: 0, fontSize: "16px", color: "#0f172a" }}>
                  {selectedProject.project_name}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedProject(null)}
                style={{
                  border: "none",
                  background: "#f1f5f9",
                  borderRadius: "50%",
                  width: "28px",
                  height: "28px",
                  fontSize: "14px",
                  cursor: "pointer",
                  color: "#475569",
                }}
              >
                ✕
              </button>
            </div>

            {/* LOCATION STATUS WARNING IF MISSING */}
            {!selectedProject.hasCoords ? (
              <div
                style={{
                  padding: "8px 12px",
                  background: "#fff3df",
                  border: "1px solid #fef3c7",
                  borderRadius: "6px",
                  color: "#92400e",
                  fontSize: "12px",
                  marginBottom: "14px",
                }}
              >
                ⚠️ Location coordinates unavailable.
              </div>
            ) : (
              <div
                style={{
                  padding: "8px 12px",
                  background: "#f0fdf4",
                  border: "1px solid #bbf7d0",
                  borderRadius: "6px",
                  color: "#166534",
                  fontSize: "12px",
                  marginBottom: "14px",
                }}
              >
                📍 Mapped on GIS Map ({selectedProject.state})
              </div>
            )}

            {/* FIELD DETAILS LIST */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "10px",
                fontSize: "13px",
                flex: 1,
              }}
            >
              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Project Code / ID</div>
                <strong>{selectedProject.project_code}</strong>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Ministry / Department</div>
                <div>{renderVal(selectedProject.ministry_final)}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Sector</div>
                <div>{renderVal(selectedProject.sector)}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>State</div>
                <div>{renderVal(selectedProject.state)}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>District</div>
                <div>Unavailable in supplied dataset.</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Risk Level</div>
                <strong
                  style={{
                    color: getMarkerColor(selectedProject.risk_level),
                  }}
                >
                  {selectedProject.risk_level} (Overrun Prob: {renderVal(selectedProject.overrun_probability_percent, "%")})
                </strong>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Project Status</div>
                <div>Unavailable in supplied dataset.</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Physical Progress</div>
                <div>{renderVal(selectedProject.physical_progress, "%")}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Original / Approved Cost</div>
                <div>{renderVal(selectedProject.original_cost, " crore")}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Revised Cost</div>
                <div>Unavailable in supplied dataset.</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Expenditure</div>
                <div>{renderVal(selectedProject.expenditure, " crore")}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Expenditure Ratio</div>
                <div>{renderVal(selectedProject.expenditure_ratio, "%")}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Implementing Agency</div>
                <div>{renderVal(selectedProject.agency_final)}</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Original / Current Timeline</div>
                <div>Unavailable in supplied dataset.</div>
              </div>

              <div>
                <div style={{ color: "#64748b", fontSize: "12px" }}>Milestones</div>
                <div>Unavailable in supplied dataset.</div>
              </div>
            </div>

            {/* ACTION BUTTONS */}
            <div style={{ marginTop: "20px", display: "flex", flexDirection: "column", gap: "8px" }}>
              <button
                type="button"
                onClick={() => onOpenProjectDetails(selectedProject.project_code)}
                style={{
                  padding: "10px",
                  background: "#1e293b",
                  color: "#ffffff",
                  border: "none",
                  borderRadius: "6px",
                  fontWeight: "600",
                  fontSize: "13px",
                  cursor: "pointer",
                }}
              >
                View Project Details
              </button>

              <button
                type="button"
                onClick={() =>
                  window.open(
                    `${API}/api/project/${encodeURIComponent(
                      selectedProject.project_code
                    )}/report`,
                    "_blank",
                    "noopener,noreferrer"
                  )
                }
                style={{
                  padding: "10px",
                  background: "#059669",
                  color: "#ffffff",
                  border: "none",
                  borderRadius: "6px",
                  fontWeight: "600",
                  fontSize: "13px",
                  cursor: "pointer",
                }}
              >
                Download Report
              </button>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
