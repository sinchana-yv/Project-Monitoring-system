
import { useEffect, useState } from "react";
import axios from "axios";
import GisMap from "./components/GisMap";

const API = "http://127.0.0.1:5001";
axios.defaults.withCredentials = true;

const getRiskLevel = (probability) => {
  if (probability >= 0.7) {
    return "HIGH";
  }
  if (probability >= 0.4) {
    return "MEDIUM";
  }
  return "LOW";
};

function HeaderNav({ user, currentRoute, onNavigate, onLogout }) {
  return (
    <header
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "14px 24px",
        marginBottom: "20px",
        border: "1px solid #cbd5e1",
        borderRadius: "8px",
        background: "#ffffff",
        boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        flexWrap: "wrap",
        gap: "12px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "24px" }}>
        <div>
          <strong style={{ fontSize: "17px", color: "#0f172a", display: "block" }}>
            AVLOKAN Platform
          </strong>
          <span style={{ fontSize: "12px", color: "#64748b" }}>
            SIH26103 Infrastructure Risk Monitoring
          </span>
        </div>

        <nav style={{ display: "flex", gap: "8px" }}>
          <button
            type="button"
            onClick={() => onNavigate("/dashboard")}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border:
                "1px solid " +
                (currentRoute === "/dashboard" ? "#1e293b" : "#cbd5e1"),
              background: currentRoute === "/dashboard" ? "#1e293b" : "#ffffff",
              color: currentRoute === "/dashboard" ? "#ffffff" : "#334155",
              fontWeight: "600",
              fontSize: "13px",
              cursor: "pointer",
            }}
          >
            📊 Dashboard
          </button>
          <button
            type="button"
            onClick={() => onNavigate("/gis-map")}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border:
                "1px solid " +
                (currentRoute === "/gis-map" ? "#1e293b" : "#cbd5e1"),
              background: currentRoute === "/gis-map" ? "#1e293b" : "#ffffff",
              color: currentRoute === "/gis-map" ? "#ffffff" : "#334155",
              fontWeight: "600",
              fontSize: "13px",
              cursor: "pointer",
            }}
          >
            🗺️ National GIS Map
          </button>
        </nav>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
        <div style={{ textAlign: "right" }}>
          <strong style={{ fontSize: "14px", color: "#0f172a" }}>
            {user?.first_name} {user?.last_name}
          </strong>
          <div style={{ color: "#64748b", fontSize: "12px" }}>
            {user?.email} | {user?.ministry || "Ministry"}
          </div>
        </div>

        <button
          type="button"
          onClick={onLogout}
          style={{
            padding: "8px 14px",
            cursor: "pointer",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            background: "#f8fafc",
            color: "#334155",
            fontWeight: "500",
            fontSize: "13px",
          }}
        >
          Logout
        </button>
      </div>
    </header>
  );
}

function Dashboard({
  user,
  onLogout,
  onNavigate,
  initialProjectCode,
  clearInitialProjectCode,
}) {
  const [summary, setSummary] = useState(null);
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [timeDelay, setTimeDelay] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [timeDelayExplanation, setTimeDelayExplanation] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [notificationsLoading, setNotificationsLoading] = useState(true);
  const [notificationsError, setNotificationsError] = useState("");
  const [projectNotifications, setProjectNotifications] = useState([]);
  const [riskHistory, setRiskHistory] = useState(null);
  const [riskHistoryLoading, setRiskHistoryLoading] = useState(false);
  const [riskHistoryError, setRiskHistoryError] = useState("");
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [timeDelayLoading, setTimeDelayLoading] = useState(false);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [timeDelayExplanationLoading, setTimeDelayExplanationLoading] =
    useState(false);
  const [detailsError, setDetailsError] = useState("");
  const [timeDelayError, setTimeDelayError] = useState("");
  const [explanationError, setExplanationError] = useState("");
  const [timeDelayExplanationError, setTimeDelayExplanationError] =
    useState("");
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");

  // Alert filters & pagination
  const [alertSearch, setAlertSearch] = useState("");
  const [alertSeverity, setAlertSeverity] = useState("ALL");
  const [alertEscalation, setAlertEscalation] = useState("ALL");
  const [alertReadState, setAlertReadState] = useState("ALL");
  const [alertPage, setAlertPage] = useState(1);
  const ALERT_PAGE_SIZE = 6;

  const refreshNotifications = async () => {
    setNotificationsLoading(true);
    setNotificationsError("");
    try {
      const response = await axios.get(`${API}/api/notifications`);
      setNotifications(response.data.notifications || []);
      setUnreadCount(Number(response.data.unread_count || 0));
    } catch (requestError) {
      console.error("Notifications error:", requestError);
      setNotificationsError("Unable to load risk escalation alerts.");
    } finally {
      setNotificationsLoading(false);
    }
  };

  useEffect(() => {
    axios
      .get(`${API}/api/summary`)
      .then((response) => {
        setSummary(response.data);
      })
      .catch((error) => {
        console.error("Summary error:", error);
        setError(
          "Unable to load dashboard data. Please start the backend and try again."
        );
      });

    void Promise.resolve().then(refreshNotifications);

    axios
      .get(`${API}/api/projects`)
      .then((response) => {
        if (!Array.isArray(response.data)) {
          throw new Error("Projects API returned an invalid response.");
        }

        setProjects(response.data);
      })
      .catch((error) => {
        console.error("Projects error:", error);
        setError(
          "Unable to load project data. Please start the backend and try again."
        );
      });
  }, []);

  // Fetch details of selected project
  const openProject = async (projectCode) => {
    const codeStr = String(projectCode).trim();
    setDetailsLoading(true);
    setTimeDelayLoading(true);
    setExplanationLoading(true);
    setTimeDelayExplanationLoading(true);
    setDetailsError("");
    setTimeDelayError("");
    setExplanationError("");
    setTimeDelayExplanationError("");
    setSelectedProject(null);
    setPrediction(null);
    setTimeDelay(null);
    setExplanation(null);
    setTimeDelayExplanation(null);
    setProjectNotifications([]);
    setRiskHistory(null);
    setRiskHistoryLoading(true);
    setRiskHistoryError("");

    const [
      projectResult,
      predictionResult,
      timeDelayResult,
      timeDelayExplanationResult,
      projectNotificationsResult,
      riskHistoryResult,
      explanationResult,
    ] = await Promise.allSettled([
      axios.get(`${API}/api/projects/${codeStr}`),
      axios.get(`${API}/api/projects/${codeStr}/prediction`),
      axios.get(`${API}/api/projects/${codeStr}/time-delay`),
      axios.get(`${API}/api/projects/${codeStr}/time-delay-explanation`),
      axios.get(`${API}/api/notifications/project/${codeStr}`),
      axios.get(`${API}/api/projects/${codeStr}/risk-history`),
      axios.get(`${API}/api/projects/${codeStr}/explanation`),
    ]);

    if (projectResult.status === "fulfilled") {
      setSelectedProject(projectResult.value.data);
    } else {
      console.error("Project details error:", projectResult.reason);
      const status = projectResult.reason.response?.status;
      setDetailsError(
        status === 404
          ? "This project could not be found."
          : "Unable to load the project details."
      );
    }

    if (predictionResult.status === "fulfilled") {
      const predictionData = predictionResult.value.data;
      const probability = Number(predictionData.overrun_probability);
      setPrediction({
        ...predictionData,
        risk_level: getRiskLevel(probability),
      });
    } else {
      console.error("Project prediction error:", predictionResult.reason);
      const status = predictionResult.reason.response?.status;
      setDetailsError(
        status === 404
          ? "This project could not be found."
          : "Unable to load the project risk assessment."
      );
    }
    setDetailsLoading(false);

    if (timeDelayResult.status === "fulfilled") {
      const timeDelayData = timeDelayResult.value.data;
      const probability = Number(timeDelayData.time_delay_probability);
      setTimeDelay({
        ...timeDelayData,
        risk_level: getRiskLevel(probability),
      });
    } else {
      console.error("Time Delay prediction error:", timeDelayResult.reason);
      const status = timeDelayResult.reason.response?.status;
      setTimeDelayError(
        status === 404
          ? "Time Delay prediction is unavailable for this project."
          : "Unable to load the Time Delay assessment."
      );
    }
    setTimeDelayLoading(false);

    if (timeDelayExplanationResult.status === "fulfilled") {
      setTimeDelayExplanation(timeDelayExplanationResult.value.data);
    } else {
      console.error(
        "Time Delay explanation error:",
        timeDelayExplanationResult.reason
      );
      const status = timeDelayExplanationResult.reason.response?.status;
      setTimeDelayExplanationError(
        status === 404
          ? "Time Delay explanation is unavailable for this project."
          : "Time Delay explanation is currently unavailable."
      );
    }
    setTimeDelayExplanationLoading(false);

    if (projectNotificationsResult.status === "fulfilled") {
      setProjectNotifications(
        projectNotificationsResult.value.data.notifications || []
      );
    } else {
      console.error(
        "Project notifications error:",
        projectNotificationsResult.reason
      );
    }

    if (riskHistoryResult.status === "fulfilled") {
      setRiskHistory(riskHistoryResult.value.data);
    } else {
      console.error("Risk history error:", riskHistoryResult.reason);
      setRiskHistoryError("Unable to load risk history for this project.");
    }
    setRiskHistoryLoading(false);

    if (explanationResult && explanationResult.status === "fulfilled") {
      setExplanation(explanationResult.value.data);
    } else if (explanationResult) {
      console.error("Project explanation error:", explanationResult.reason);
      const status = explanationResult.reason.response?.status;
      setExplanationError(
        status === 404
          ? "SHAP explanation is unavailable for this project."
          : "SHAP explanation is currently unavailable."
      );
    }
    setExplanationLoading(false);
  };

  useEffect(() => {
    if (initialProjectCode) {
      const targetCode = initialProjectCode;
      if (clearInitialProjectCode) clearInitialProjectCode();
      const timer = setTimeout(() => {
        void openProject(targetCode);
        const detailsEl = document.getElementById("project-details-section");
        if (detailsEl) {
          detailsEl.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 50);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [initialProjectCode, clearInitialProjectCode]);

  const markNotificationRead = async (notification) => {
    try {
      await axios.patch(`${API}/api/notifications/${notification.id}/read`);
      setNotifications((current) =>
        current.map((item) =>
          item.id === notification.id ? { ...item, is_read: true } : item
        )
      );
      setUnreadCount((current) =>
        Math.max(0, current - (notification.is_read ? 0 : 1))
      );
    } catch (requestError) {
      console.error("Mark notification read error:", requestError);
      setNotificationsError("Unable to update this risk alert.");
    }
  };

  const markAllNotificationsRead = async () => {
    try {
      await axios.patch(`${API}/api/notifications/mark-all-read`);
      setNotifications((current) =>
        current.map((notification) => ({ ...notification, is_read: true }))
      );
      setUnreadCount(0);
    } catch (requestError) {
      console.error("Mark all notifications read error:", requestError);
      setNotificationsError("Unable to update risk alerts.");
    }
  };

  const openNotificationProject = async (notification) => {
    if (!notification.is_read) {
      await markNotificationRead(notification);
    }
    await openProject(notification.project_code);
    setTimeout(() => {
      const detailsEl = document.getElementById("project-details-section");
      if (detailsEl) {
        detailsEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 100);
  };

  if (error) {
    return (
      <main className="error">
        <h1>Dashboard unavailable</h1>
        <p>{error}</p>
      </main>
    );
  }

  if (!summary) {
    return (
      <main className="loading">
        <h1>Loading...</h1>
      </main>
    );
  }

  const filteredProjects = projects.filter((project) => {
    const name = String(project.project_name || "").toLowerCase();
    const code = String(project.project_code || "");
    const searchText = search.toLowerCase();

    const matchesSearch =
      name.includes(searchText) || code.includes(searchText);

    const matchesRisk =
      riskFilter === "ALL" ||
      String(project.risk_level).toUpperCase() === riskFilter;

    return matchesSearch && matchesRisk;
  });

  const criticalEscalations = notifications.filter(
    (notification) =>
      notification.previous_risk === "MEDIUM" &&
      notification.current_risk === "HIGH"
  ).length;

  // Filter & sort risk alert notifications
  const filteredNotifications = notifications.filter((notification) => {
    const codeStr = String(notification.project_code || "").toLowerCase();
    const nameStr = String(notification.project_name || "").toLowerCase();
    const msgStr = String(notification.message || "").toLowerCase();
    const q = alertSearch.trim().toLowerCase();
    const matchesSearch =
      !q || codeStr.includes(q) || nameStr.includes(q) || msgStr.includes(q);

    const matchesSeverity =
      alertSeverity === "ALL" ||
      String(notification.current_risk).toUpperCase() === alertSeverity ||
      String(notification.severity).toUpperCase() === alertSeverity;

    const transitionStr = `${notification.previous_risk} -> ${notification.current_risk}`;
    const matchesEscalation =
      alertEscalation === "ALL" || transitionStr === alertEscalation;

    const matchesRead =
      alertReadState === "ALL" ||
      (alertReadState === "UNREAD" && !notification.is_read) ||
      (alertReadState === "READ" && notification.is_read);

    return matchesSearch && matchesSeverity && matchesEscalation && matchesRead;
  });

  const sortedNotifications = [...filteredNotifications].sort((a, b) => {
    const dateA = new Date(
      a.current_report_month || a.report_month || a.created_at || 0
    ).getTime();
    const dateB = new Date(
      b.current_report_month || b.report_month || b.created_at || 0
    ).getTime();
    return dateB - dateA;
  });

  const totalAlertPages = Math.max(
    1,
    Math.ceil(sortedNotifications.length / ALERT_PAGE_SIZE)
  );
  const currentPageAlerts = sortedNotifications.slice(
    (alertPage - 1) * ALERT_PAGE_SIZE,
    alertPage * ALERT_PAGE_SIZE
  );

  const latestDateInDataset = notifications.reduce((latest, item) => {
    const d = item.current_report_month || item.report_month;
    if (!d) return latest;
    return !latest || d > latest ? d : latest;
  }, "");

  // Overall risk calculator
  const getOverallRiskLevel = () => {
    if (!prediction && !timeDelay) return "LOW";
    const pRisk = prediction?.risk_level || "LOW";
    const tRisk = timeDelay?.risk_level || "LOW";
    if (pRisk === "HIGH" || tRisk === "HIGH") return "HIGH";
    if (pRisk === "MEDIUM" || tRisk === "MEDIUM") return "MEDIUM";
    return "LOW";
  };

  const renderCellVal = (val, suffix = "") => {
    if (val === null || val === undefined || val === "") {
      return "Unavailable in supplied dataset";
    }
    return `${val}${suffix}`;
  };

  return (
    <div style={{ padding: "25px 35px", fontFamily: "Arial, sans-serif" }}>
      <HeaderNav
        user={user}
        currentRoute="/dashboard"
        onNavigate={onNavigate}
        onLogout={onLogout}
      />

      <h1 style={{ margin: "0 0 10px 0", fontSize: "26px", color: "#102a43" }}>
        AVLOKAN AI Project Monitoring
      </h1>

      {/* SUMMARY KPI CARDS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
          gap: "14px",
          marginBottom: "25px",
        }}
      >
        <div
          style={{
            padding: "14px 18px",
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: "8px",
          }}
        >
          <div style={{ fontSize: "13px", color: "#64748b" }}>Total Projects</div>
          <strong style={{ fontSize: "24px", color: "#1e293b" }}>
            {summary.total_projects}
          </strong>
        </div>
        <div
          style={{
            padding: "14px 18px",
            background: "#fff",
            border: "1px solid #fee2e2",
            borderLeft: "4px solid #ef4444",
            borderRadius: "8px",
          }}
        >
          <div style={{ fontSize: "13px", color: "#991b1b" }}>High Risk</div>
          <strong style={{ fontSize: "24px", color: "#dc2626" }}>
            {summary.high_risk}
          </strong>
        </div>
        <div
          style={{
            padding: "14px 18px",
            background: "#fff",
            border: "1px solid #fef3c7",
            borderLeft: "4px solid #f59e0b",
            borderRadius: "8px",
          }}
        >
          <div style={{ fontSize: "13px", color: "#92400e" }}>Medium Risk</div>
          <strong style={{ fontSize: "24px", color: "#d97706" }}>
            {summary.medium_risk}
          </strong>
        </div>
        <div
          style={{
            padding: "14px 18px",
            background: "#fff",
            border: "1px solid #dcfce7",
            borderLeft: "4px solid #10b981",
            borderRadius: "8px",
          }}
        >
          <div style={{ fontSize: "13px", color: "#166534" }}>Low Risk</div>
          <strong style={{ fontSize: "24px", color: "#059669" }}>
            {summary.low_risk}
          </strong>
        </div>
        <div
          style={{
            padding: "14px 18px",
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: "8px",
          }}
        >
          <div style={{ fontSize: "13px", color: "#64748b" }}>Average Risk</div>
          <strong style={{ fontSize: "24px", color: "#1e293b" }}>
            {summary.average_risk_probability}%
          </strong>
        </div>
      </div>

      {/* RISK ALERTS SECTION */}
      <section
        aria-labelledby="risk-alerts-heading"
        style={{
          marginBottom: "30px",
          padding: "20px",
          border: "1px solid #e5c9a8",
          borderRadius: "10px",
          background: "#fffaf3",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "12px",
            flexWrap: "wrap",
            marginBottom: "14px",
          }}
        >
          <div>
            <h2 id="risk-alerts-heading" style={{ margin: 0, fontSize: "20px" }}>
              Risk Alerts
            </h2>
            <p style={{ margin: "4px 0 0 0", color: "#64748b", fontSize: "13px" }}>
              Early warning alerts based on project risk transitions in the dataset.
            </p>
          </div>
          <button
            type="button"
            onClick={markAllNotificationsRead}
            disabled={!unreadCount}
            style={{
              padding: "8px 14px",
              cursor: unreadCount ? "pointer" : "default",
              borderRadius: "6px",
              border: "1px solid #d97706",
              background: unreadCount ? "#fef3c7" : "#f1f5f9",
              color: unreadCount ? "#92400e" : "#94a3b8",
              fontWeight: "600",
              fontSize: "13px",
            }}
          >
            Mark All as Read
          </button>
        </div>

        {notificationsLoading && <p>Loading risk escalation alerts...</p>}
        {notificationsError && (
          <p role="alert" style={{ color: "#c53030" }}>
            {notificationsError}
          </p>
        )}

        {!notificationsLoading && !notificationsError && (
          <>
            {/* ALERT COUNTS BAR */}
            <div
              style={{
                display: "flex",
                gap: "14px",
                flexWrap: "wrap",
                alignItems: "center",
                fontSize: "13px",
                marginBottom: "14px",
                padding: "8px 12px",
                background: "#fef9c3",
                borderRadius: "6px",
                border: "1px solid #fef08a",
              }}
            >
              <span>
                <strong>Total:</strong> {notifications.length} Alerts
              </span>
              <span>·</span>
              <span>
                <strong>Unread:</strong> {unreadCount}
              </span>
              <span>·</span>
              <span>
                <strong>Critical Escalations (MEDIUM → HIGH):</strong>{" "}
                {criticalEscalations}
              </span>
              {latestDateInDataset && (
                <>
                  <span>·</span>
                  <span>
                    <strong>Latest Observation Date:</strong> {latestDateInDataset}
                  </span>
                </>
              )}
            </div>

            {/* ALERT CONTROLS / FILTERS */}
            <div
              style={{
                display: "flex",
                gap: "10px",
                flexWrap: "wrap",
                marginBottom: "16px",
                alignItems: "center",
              }}
            >
              <input
                type="text"
                placeholder="Search alert by project code or name..."
                value={alertSearch}
                onChange={(e) => {
                  setAlertSearch(e.target.value);
                  setAlertPage(1);
                }}
                style={{
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "13px",
                  width: "250px",
                  flexGrow: 1,
                }}
              />
              <select
                value={alertSeverity}
                onChange={(e) => {
                  setAlertSeverity(e.target.value);
                  setAlertPage(1);
                }}
                style={{
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "13px",
                  background: "#fff",
                }}
              >
                <option value="ALL">All Severities</option>
                <option value="HIGH">High Severity</option>
                <option value="MEDIUM">Medium Severity</option>
                <option value="LOW">Low Severity</option>
              </select>
              <select
                value={alertEscalation}
                onChange={(e) => {
                  setAlertEscalation(e.target.value);
                  setAlertPage(1);
                }}
                style={{
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "13px",
                  background: "#fff",
                }}
              >
                <option value="ALL">All Escalations</option>
                <option value="LOW -> MEDIUM">LOW → MEDIUM</option>
                <option value="LOW -> HIGH">LOW → HIGH</option>
                <option value="MEDIUM -> HIGH">MEDIUM → HIGH</option>
              </select>
              <select
                value={alertReadState}
                onChange={(e) => {
                  setAlertReadState(e.target.value);
                  setAlertPage(1);
                }}
                style={{
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "13px",
                  background: "#fff",
                }}
              >
                <option value="ALL">All Statuses</option>
                <option value="UNREAD">Unread Only</option>
                <option value="READ">Read Only</option>
              </select>
              {(alertSearch ||
                alertSeverity !== "ALL" ||
                alertEscalation !== "ALL" ||
                alertReadState !== "ALL") && (
                <button
                  type="button"
                  onClick={() => {
                    setAlertSearch("");
                    setAlertSeverity("ALL");
                    setAlertEscalation("ALL");
                    setAlertReadState("ALL");
                    setAlertPage(1);
                  }}
                  style={{
                    padding: "8px 12px",
                    borderRadius: "6px",
                    border: "1px solid #cbd5e1",
                    background: "#f1f5f9",
                    fontSize: "13px",
                    cursor: "pointer",
                  }}
                >
                  Reset Filters
                </button>
              )}
            </div>

            {/* ALERT LIST */}
            {sortedNotifications.length === 0 ? (
              <p style={{ marginTop: "12px", color: "#64748b" }}>
                No risk escalation alerts match the selected criteria.
              </p>
            ) : (
              <div>
                <div style={{ fontSize: "13px", color: "#64748b", marginBottom: "8px" }}>
                  Showing {currentPageAlerts.length} of {sortedNotifications.length} matching alerts (Page {alertPage} of {totalAlertPages})
                </div>

                {currentPageAlerts.map((notification) => (
                  <article
                    key={notification.id}
                    style={{
                      marginTop: "8px",
                      padding: "12px 14px",
                      border: "1px solid #eadfce",
                      borderRadius: "8px",
                      background: notification.is_read ? "#fff" : "#fff3df",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "baseline",
                        flexWrap: "wrap",
                        gap: "6px",
                      }}
                    >
                      <div>
                        <span
                          className={`risk-badge ${(
                            notification.current_risk || "HIGH"
                          ).toLowerCase()}`}
                          style={{ marginRight: "8px" }}
                        >
                          {notification.current_risk} Risk
                        </span>
                        <strong>
                          Project {notification.project_code}
                          {notification.project_name ? `: ${notification.project_name}` : ""}
                        </strong>
                      </div>
                      <span style={{ fontSize: "12px", color: "#64748b" }}>
                        Report Month: {notification.current_report_month || notification.report_month}
                      </span>
                    </div>

                    <p style={{ marginTop: "6px", marginBottom: "8px", fontSize: "14px", color: "#334155" }}>
                      {notification.message}
                    </p>

                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        onClick={() => openNotificationProject(notification)}
                        style={{
                          padding: "6px 12px",
                          cursor: "pointer",
                          borderRadius: "5px",
                          border: "none",
                          background: "#2563eb",
                          color: "#fff",
                          fontWeight: "600",
                          fontSize: "13px",
                        }}
                      >
                        View Project
                      </button>
                      {!notification.is_read && (
                        <button
                          type="button"
                          onClick={() => markNotificationRead(notification)}
                          style={{
                            padding: "6px 12px",
                            cursor: "pointer",
                            borderRadius: "5px",
                            border: "1px solid #cbd5e1",
                            background: "#fff",
                            fontSize: "13px",
                          }}
                        >
                          Mark as Read
                        </button>
                      )}
                    </div>
                  </article>
                ))}

                {/* PAGINATION CONTROLS */}
                {totalAlertPages > 1 && (
                  <div
                    style={{
                      display: "flex",
                      gap: "10px",
                      alignItems: "center",
                      justifyContent: "center",
                      marginTop: "16px",
                    }}
                  >
                    <button
                      type="button"
                      disabled={alertPage <= 1}
                      onClick={() => setAlertPage((p) => Math.max(1, p - 1))}
                      style={{
                        padding: "6px 14px",
                        cursor: alertPage > 1 ? "pointer" : "default",
                        borderRadius: "5px",
                        border: "1px solid #cbd5e1",
                        background: "#fff",
                      }}
                    >
                      Previous Page
                    </button>
                    <span style={{ fontSize: "13px", color: "#475569" }}>
                      Page {alertPage} of {totalAlertPages}
                    </span>
                    <button
                      type="button"
                      disabled={alertPage >= totalAlertPages}
                      onClick={() =>
                        setAlertPage((p) => Math.min(totalAlertPages, p + 1))
                      }
                      style={{
                        padding: "6px 14px",
                        cursor: alertPage < totalAlertPages ? "pointer" : "default",
                        borderRadius: "5px",
                        border: "1px solid #cbd5e1",
                        background: "#fff",
                      }}
                    >
                      Next Page
                    </button>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </section>

      {/* PROJECT SEARCH & TABLE */}
      <h2 style={{ fontSize: "20px", color: "#102a43", marginBottom: "12px" }}>
        Project Risk Monitoring
      </h2>

      <div style={{ display: "flex", gap: "10px", marginBottom: "14px", flexWrap: "wrap" }}>
        <input
          type="text"
          placeholder="Search project name or code..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            padding: "9px 13px",
            width: "320px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "14px",
          }}
        />

        <select
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value)}
          style={{
            padding: "9px 13px",
            borderRadius: "6px",
            border: "1px solid #cbd5e1",
            fontSize: "14px",
            background: "#fff",
          }}
        >
          <option value="ALL">All Risk Levels</option>
          <option value="HIGH">High Risk</option>
          <option value="MEDIUM">Medium Risk</option>
          <option value="LOW">Low Risk</option>
        </select>
      </div>

      <p style={{ fontSize: "13px", color: "#64748b", marginBottom: "12px" }}>
        Showing {filteredProjects.length} projects
      </p>

      {/* PROJECTS TABLE */}
      <div style={{ overflowX: "auto", marginBottom: "25px" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: "8px",
          }}
        >
          <thead>
            <tr style={{ background: "#f8fafc", borderBottom: "2px solid #e2e8f0" }}>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>Rank</th>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>Project Code</th>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>Project Name</th>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>State</th>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>Sector</th>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>Cost Overrun Prob</th>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>Risk Level</th>
              <th style={{ padding: "10px 12px", textAlign: "left", fontSize: "13px" }}>Report</th>
            </tr>
          </thead>
          <tbody>
            {filteredProjects.slice(0, 15).map((project, index) => (
              <tr
                key={project.project_code}
                style={{ borderBottom: "1px solid #f1f5f9" }}
              >
                <td style={{ padding: "10px 12px", fontSize: "13px", color: "#64748b" }}>
                  #{index + 1}
                </td>
                <td style={{ padding: "10px 12px", fontSize: "13px", fontWeight: "600" }}>
                  {project.project_code}
                </td>
                <td style={{ padding: "10px 12px", fontSize: "14px" }}>
                  <button
                    onClick={() => openProject(project.project_code)}
                    style={{
                      background: "none",
                      border: "none",
                      padding: 0,
                      color: "#2563eb",
                      cursor: "pointer",
                      textAlign: "left",
                      fontSize: "14px",
                      fontWeight: "500",
                    }}
                  >
                    {project.project_name}
                  </button>
                </td>
                <td style={{ padding: "10px 12px", fontSize: "13px" }}>{project.state}</td>
                <td style={{ padding: "10px 12px", fontSize: "13px" }}>{project.sector}</td>
                <td style={{ padding: "10px 12px", fontSize: "13px" }}>
                  {Number(project.overrun_probability_percent).toFixed(2)}%
                </td>
                <td style={{ padding: "10px 12px" }}>
                  <span className={`risk-badge ${String(project.risk_level).toLowerCase()}`}>
                    {project.risk_level}
                  </span>
                </td>
                <td style={{ padding: "10px 12px" }}>
                  <button
                    type="button"
                    onClick={() =>
                      window.open(
                        `${API}/api/project/${encodeURIComponent(
                          String(project.project_code)
                        )}/report`,
                        "_blank",
                        "noopener,noreferrer"
                      )
                    }
                    style={{
                      padding: "5px 10px",
                      cursor: "pointer",
                      borderRadius: "5px",
                      border: "none",
                      background: "#059669",
                      color: "#fff",
                      fontWeight: "600",
                      fontSize: "12px",
                      whiteSpace: "nowrap",
                    }}
                  >
                    ⬇ PDF
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* PROJECT DETAILS MODAL / SECTION */}
      {detailsLoading && (
        <div
          id="project-details-section"
          style={{
            marginTop: "20px",
            padding: "20px",
            border: "1px solid #cbd5e1",
            borderRadius: "8px",
            background: "#fff",
          }}
        >
          Loading project details and AI assessment...
        </div>
      )}

      {selectedProject && (
        <div
          id="project-details-section"
          style={{
            marginTop: "20px",
            padding: "24px",
            border: "1px solid #cbd5e1",
            borderRadius: "10px",
            background: "#fff",
            boxShadow: "0 4px 16px rgba(0, 0, 0, 0.05)",
          }}
        >
          {/* TITLE & ACTIONS */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "10px",
              marginBottom: "16px",
              paddingBottom: "12px",
              borderBottom: "1px solid #e2e8f0",
            }}
          >
            <div>
              <h2 style={{ margin: 0, fontSize: "20px", color: "#0f172a" }}>
                {selectedProject.project_name}
              </h2>
              <p style={{ margin: "4px 0 0 0", fontSize: "13px", color: "#64748b" }}>
                Project Code: <strong>{selectedProject.project_code}</strong> · {selectedProject.state} · {selectedProject.sector}
              </p>
            </div>
            <div style={{ display: "flex", gap: "8px" }}>
              <button
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
                  padding: "8px 14px",
                  cursor: "pointer",
                  borderRadius: "6px",
                  border: "none",
                  background: "#059669",
                  color: "#fff",
                  fontWeight: "600",
                  fontSize: "13px",
                }}
              >
                Download PDF Report
              </button>
              <button
                onClick={() => setSelectedProject(null)}
                style={{
                  padding: "8px 14px",
                  cursor: "pointer",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  background: "#f8fafc",
                  fontSize: "13px",
                }}
              >
                Close
              </button>
            </div>
          </div>

          {/* BASIC METRICS & FINANCIALS */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "12px",
              marginBottom: "20px",
              padding: "14px",
              background: "#f8fafc",
              borderRadius: "8px",
              border: "1px solid #f1f5f9",
            }}
          >
            <div>
              <div style={{ fontSize: "12px", color: "#64748b" }}>Ministry / Department</div>
              <div style={{ fontWeight: "600", fontSize: "13px", color: "#1e293b" }}>
                {selectedProject.ministry_final || selectedProject.ministry || "N/A"}
              </div>
            </div>
            <div>
              <div style={{ fontSize: "12px", color: "#64748b" }}>Agency</div>
              <div style={{ fontWeight: "600", fontSize: "13px", color: "#1e293b" }}>
                {selectedProject.agency_final || selectedProject.agency || "N/A"}
              </div>
            </div>
            <div>
              <div style={{ fontSize: "12px", color: "#64748b" }}>Original Cost</div>
              <div style={{ fontWeight: "600", fontSize: "13px", color: "#1e293b" }}>
                {renderCellVal(selectedProject.original_cost, " crore")}
              </div>
            </div>
            <div>
              <div style={{ fontSize: "12px", color: "#64748b" }}>Expenditure</div>
              <div style={{ fontWeight: "600", fontSize: "13px", color: "#1e293b" }}>
                {renderCellVal(selectedProject.expenditure, " crore")}
              </div>
            </div>
            <div>
              <div style={{ fontSize: "12px", color: "#64748b" }}>Physical Progress</div>
              <div style={{ fontWeight: "600", fontSize: "13px", color: "#1e293b" }}>
                {renderCellVal(selectedProject.physical_progress, "%")}
              </div>
            </div>
            <div>
              <div style={{ fontSize: "12px", color: "#64748b" }}>Expenditure Ratio</div>
              <div style={{ fontWeight: "600", fontSize: "13px", color: "#1e293b" }}>
                {renderCellVal(selectedProject.expenditure_ratio, "%")}
              </div>
            </div>
          </div>

          {/* COMPACT SUMMARY CARDS */}
          <div style={{ marginBottom: "24px" }}>
            <h3 style={{ fontSize: "16px", color: "#0f172a", marginBottom: "12px" }}>
              Risk Assessment Summary
            </h3>
            {detailsError && (
              <p style={{ color: "#ef4444", fontSize: "13px", marginBottom: "8px" }}>
                {detailsError}
              </p>
            )}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
                gap: "14px",
              }}
            >
              {/* CARD 1: COST OVERRUN RISK */}
              <div
                style={{
                  padding: "16px",
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: "8px",
                  borderLeft: `4px solid ${
                    prediction?.risk_level === "HIGH"
                      ? "#ef4444"
                      : prediction?.risk_level === "MEDIUM"
                      ? "#f59e0b"
                      : "#10b981"
                  }`,
                }}
              >
                <div style={{ fontSize: "12px", color: "#64748b" }}>Cost Overrun Risk</div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", margin: "6px 0" }}>
                  <span
                    className={`risk-badge ${(
                      prediction?.risk_level || "LOW"
                    ).toLowerCase()}`}
                  >
                    {prediction?.risk_level || "UNKNOWN"}
                  </span>
                  <strong style={{ fontSize: "18px", color: "#0f172a" }}>
                    {prediction
                      ? `${prediction.overrun_probability_percent}%`
                      : "Unavailable"}
                  </strong>
                </div>
                <div style={{ fontSize: "12px", color: "#475569" }}>
                  Prediction: <strong>{prediction?.prediction || "N/A"}</strong>
                </div>
              </div>

              {/* CARD 2: SCHEDULE DELAY RISK */}
              <div
                style={{
                  padding: "16px",
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: "8px",
                  borderLeft: `4px solid ${
                    timeDelay?.risk_level === "HIGH"
                      ? "#ef4444"
                      : timeDelay?.risk_level === "MEDIUM"
                      ? "#f59e0b"
                      : "#10b981"
                  }`,
                }}
              >
                <div style={{ fontSize: "12px", color: "#64748b" }}>Schedule Delay Risk</div>
                {timeDelayLoading ? (
                  <p style={{ fontSize: "13px", color: "#64748b", margin: "6px 0" }}>
                    Loading assessment...
                  </p>
                ) : (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", margin: "6px 0" }}>
                      <span
                        className={`risk-badge ${(
                          timeDelay?.risk_level || "LOW"
                        ).toLowerCase()}`}
                      >
                        {timeDelay?.risk_level || "UNKNOWN"}
                      </span>
                      <strong style={{ fontSize: "18px", color: "#0f172a" }}>
                        {timeDelay
                          ? `${timeDelay.time_delay_probability_percent}%`
                          : "Unavailable"}
                      </strong>
                    </div>
                    {timeDelayError ? (
                      <p style={{ color: "#a56b00", fontSize: "12px", margin: 0 }}>
                        {timeDelayError}
                      </p>
                    ) : (
                      <div style={{ fontSize: "12px", color: "#475569" }}>
                        Prediction: <strong>{timeDelay?.prediction || "N/A"}</strong>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* CARD 3: OVERALL RISK */}
              <div
                style={{
                  padding: "16px",
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: "8px",
                  borderLeft: `4px solid ${
                    getOverallRiskLevel() === "HIGH"
                      ? "#ef4444"
                      : getOverallRiskLevel() === "MEDIUM"
                      ? "#f59e0b"
                      : "#10b981"
                  }`,
                }}
              >
                <div style={{ fontSize: "12px", color: "#64748b" }}>Current Overall Risk</div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", margin: "6px 0" }}>
                  <span className={`risk-badge ${getOverallRiskLevel().toLowerCase()}`}>
                    {getOverallRiskLevel()}
                  </span>
                </div>
                <div style={{ fontSize: "12px", color: "#475569" }}>
                  Combined assessment across Cost Overrun & Schedule Slippage metrics.
                </div>
              </div>
            </div>

            {/* PROJECT NOTIFICATIONS BLOCK */}
            {projectNotifications && projectNotifications.length > 0 && (
              <div style={{ marginTop: "14px" }}>
                <div style={{ fontSize: "13px", fontWeight: "600", color: "#92400e", marginBottom: "6px" }}>
                  Project Escalation History Alerts ({projectNotifications.length})
                </div>
                {projectNotifications.map((notif) => (
                  <div
                    key={notif.id}
                    style={{
                      fontSize: "12px",
                      padding: "8px 12px",
                      background: "#fff3df",
                      border: "1px solid #fef3c7",
                      borderRadius: "6px",
                      marginBottom: "6px",
                    }}
                  >
                    <strong>{notif.previous_risk} → {notif.current_risk}</strong>: {notif.message} ({notif.current_report_month})
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* RISK HISTORY SECTION */}
          <div style={{ marginBottom: "24px" }}>
            <h3 style={{ fontSize: "16px", color: "#0f172a", marginBottom: "12px" }}>
              Project Risk History
            </h3>

            {riskHistoryLoading && <p style={{ fontSize: "14px" }}>Loading risk history...</p>}

            {riskHistoryError && (
              <p style={{ color: "#a56b00", fontSize: "14px" }}>{riskHistoryError}</p>
            )}

            {!riskHistoryLoading && !riskHistoryError && riskHistory && (
              <>
                {riskHistory.message && (
                  <div
                    style={{
                      padding: "10px 14px",
                      background: "#f0f9ff",
                      border: "1px solid #bae6fd",
                      borderRadius: "6px",
                      color: "#0369a1",
                      fontSize: "13px",
                      marginBottom: "12px",
                    }}
                  >
                    {riskHistory.message}
                  </div>
                )}

                {riskHistory.records?.length > 0 && (
                  <div style={{ overflowX: "auto" }}>
                    <table
                      style={{
                        width: "100%",
                        borderCollapse: "collapse",
                        minWidth: "700px",
                        border: "1px solid #e2e8f0",
                        fontSize: "13px",
                      }}
                    >
                      <thead>
                        <tr style={{ background: "#f8fafc", borderBottom: "2px solid #e2e8f0" }}>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Report Month</th>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Cost Risk</th>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Time-Delay Risk</th>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Cost Probability</th>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Time-Delay Probability</th>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Physical Progress</th>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Expenditure Ratio</th>
                          <th style={{ padding: "9px 12px", textAlign: "left" }}>Escalation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {riskHistory.records.map((record, rIdx) => {
                          const isEscalation =
                            record.cost_risk_change === "LOW -> MEDIUM" ||
                            record.cost_risk_change === "LOW -> HIGH" ||
                            record.cost_risk_change === "MEDIUM -> HIGH" ||
                            record.time_delay_risk_change === "LOW -> MEDIUM" ||
                            record.time_delay_risk_change === "LOW -> HIGH" ||
                            record.time_delay_risk_change === "MEDIUM -> HIGH";

                          return (
                            <tr
                              key={`${record.project_code}-${record.report_month}-${rIdx}`}
                              style={{
                                background: isEscalation ? "#fff3df" : rIdx % 2 === 0 ? "#fff" : "#f8fafc",
                                borderBottom: "1px solid #f1f5f9",
                              }}
                            >
                              <td style={{ padding: "9px 12px", fontWeight: "600" }}>
                                {record.report_month}
                              </td>
                              <td style={{ padding: "9px 12px" }}>
                                {record.cost_risk_level ? (
                                  <span
                                    className={`risk-badge ${record.cost_risk_level.toLowerCase()}`}
                                  >
                                    {record.cost_risk_level}
                                  </span>
                                ) : (
                                  "Unavailable in supplied dataset"
                                )}
                              </td>
                              <td style={{ padding: "9px 12px" }}>
                                {record.time_delay_risk_level ? (
                                  <span
                                    className={`risk-badge ${record.time_delay_risk_level.toLowerCase()}`}
                                  >
                                    {record.time_delay_risk_level}
                                  </span>
                                ) : (
                                  "Unavailable in supplied dataset"
                                )}
                              </td>
                              <td style={{ padding: "9px 12px" }}>
                                {record.cost_overrun_probability == null
                                  ? "Unavailable in supplied dataset"
                                  : `${(record.cost_overrun_probability * 100).toFixed(2)}%`}
                              </td>
                              <td style={{ padding: "9px 12px" }}>
                                {record.time_delay_probability == null
                                  ? "Unavailable in supplied dataset"
                                  : `${(record.time_delay_probability * 100).toFixed(2)}%`}
                              </td>
                              <td style={{ padding: "9px 12px" }}>
                                {record.physical_progress == null
                                  ? "Unavailable in supplied dataset"
                                  : `${Number(record.physical_progress).toFixed(2)}%`}
                              </td>
                              <td style={{ padding: "9px 12px" }}>
                                {record.expenditure_ratio == null
                                  ? "Unavailable in supplied dataset"
                                  : `${Number(record.expenditure_ratio).toFixed(2)}%`}
                              </td>
                              <td style={{ padding: "9px 12px" }}>
                                {isEscalation ? (
                                  <strong style={{ color: "#d97706" }}>
                                    {record.escalation || record.cost_risk_change || record.time_delay_risk_change}
                                  </strong>
                                ) : (
                                  <span style={{ color: "#94a3b8" }}>No escalation</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}

            {!riskHistoryLoading &&
              !riskHistoryError &&
              riskHistory &&
              !riskHistory.records?.length && (
                <p style={{ color: "#64748b", fontSize: "14px" }}>
                  No historical records are available for this project.
                </p>
              )}
          </div>

          {/* AI EXPLANATIONS */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
              gap: "16px",
              marginTop: "20px",
              paddingTop: "16px",
              borderTop: "1px solid #e2e8f0",
            }}
          >
            {/* SCHEDULE SLIPPAGE FACTORS */}
            <div style={{ padding: "14px", background: "#f8fafc", borderRadius: "8px" }}>
              <h4 style={{ margin: "0 0 8px 0", fontSize: "14px", color: "#0f172a" }}>
                Schedule Slippage Key Factors
              </h4>
              {timeDelayExplanationLoading && (
                <p style={{ fontSize: "13px" }}>Loading explanation...</p>
              )}
              {timeDelayExplanationError && (
                <p style={{ color: "#a56b00", fontSize: "13px" }}>
                  {timeDelayExplanationError}
                </p>
              )}
              {timeDelayExplanation && (
                <div style={{ fontSize: "13px" }}>
                  <div style={{ fontWeight: "600", color: "#991b1b", marginBottom: "4px" }}>
                    Top Risk Factors:
                  </div>
                  {timeDelayExplanation.risk_factors?.length ? (
                    <ul style={{ paddingLeft: "18px", margin: "0 0 10px 0" }}>
                      {timeDelayExplanation.risk_factors.map((f, i) => (
                        <li key={i}>
                          <strong>{f.feature}</strong> ({Number(f.shap_value).toFixed(3)})
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p style={{ margin: 0, color: "#64748b" }}>None identified</p>
                  )}
                </div>
              )}
            </div>

            {/* COST OVERRUN FACTORS */}
            <div style={{ padding: "14px", background: "#f8fafc", borderRadius: "8px" }}>
              <h4 style={{ margin: "0 0 8px 0", fontSize: "14px", color: "#0f172a" }}>
                Cost Overrun Key Factors
              </h4>
              {explanationLoading && <p style={{ fontSize: "13px" }}>Loading explanation...</p>}
              {explanationError && (
                <p style={{ color: "#a56b00", fontSize: "13px" }}>{explanationError}</p>
              )}
              {explanation && (
                <div style={{ fontSize: "13px" }}>
                  <div style={{ fontWeight: "600", color: "#991b1b", marginBottom: "4px" }}>
                    Top Risk Factors:
                  </div>
                  {explanation.top_risk_factors?.length ? (
                    <ul style={{ paddingLeft: "18px", margin: "0 0 10px 0" }}>
                      {explanation.top_risk_factors.map((f, i) => (
                        <li key={i}>
                          <strong>{f.feature}</strong> ({Number(f.shap_value).toFixed(3)})
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p style={{ margin: 0, color: "#64748b" }}>None identified</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const authCardStyle = {
  width: "min(460px, calc(100% - 32px))",
  margin: "40px auto",
  padding: "28px",
  background: "white",
  border: "1px solid #d9e2ec",
  borderRadius: "10px",
  boxShadow: "0 4px 18px rgba(16, 42, 67, 0.08)",
};

const authInputStyle = {
  width: "100%",
  padding: "10px 12px",
  marginTop: "6px",
  border: "1px solid #bcccdc",
  borderRadius: "5px",
  fontSize: "15px",
};

const getApiError = (error, fallback) => {
  if (!error.response) {
    return "Unable to connect to the authentication server. Please try again.";
  }
  return error.response.data?.error || fallback;
};

function AuthShell({ title, subtitle, children }) {
  return (
    <main style={{ minHeight: "100vh", padding: "1px 0", background: "#f4f7fb", color: "#172033" }}>
      <section style={authCardStyle}>
        <h1 style={{ marginTop: 0, color: "#102a43" }}>AVLOKAN</h1>
        <p style={{ color: "#52606d" }}>SIH26103 | Intelligent Infrastructure Project Monitoring</p>
        <h2>{title}</h2>
        {subtitle && <p style={{ color: "#52606d" }}>{subtitle}</p>}
        {children}
        <p style={{ marginBottom: 0, fontSize: "13px", color: "#52606d" }}>
          AVLOKAN prototype authentication. Use only an account you are authorized to access.
        </p>
      </section>
    </main>
  );
}

function PasswordField({ label, value, onChange, visible, onToggle, error, name, autoComplete }) {
  return (
    <label style={{ display: "block", marginBottom: "14px" }}>
      {label}
      <div style={{ display: "flex", gap: "6px" }}>
        <input
          name={name}
          type={visible ? "text" : "password"}
          value={value}
          onChange={onChange}
          autoComplete={autoComplete}
          style={authInputStyle}
          aria-invalid={Boolean(error)}
        />
        <button type="button" onClick={onToggle} aria-label={`Show or hide ${label.toLowerCase()}`}>
          {visible ? "Hide" : "Show"}
        </button>
      </div>
      {error && <span style={{ display: "block", color: "#c53030", fontSize: "13px" }}>{error}</span>}
    </label>
  );
}

function SignIn({ onNavigate, onLogin, initialEmail = "" }) {
  const [form, setForm] = useState({ email: initialEmail, password: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    const nextErrors = {};
    if (!form.email) nextErrors.email = "Email Address is required.";
    if (!form.password) nextErrors.password = "Password is required.";
    setErrors(nextErrors);
    setApiError("");
    if (Object.keys(nextErrors).length) return;
    setSubmitting(true);
    try {
      const response = await axios.post(`${API}/api/auth/login`, form);
      onLogin(response.data.user);
      onNavigate("/dashboard");
    } catch (error) {
      setApiError(getApiError(error, "Invalid email or password. Please check your credentials and try again."));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell title="Sign In" onNavigate={onNavigate}>
      <form onSubmit={submit} noValidate>
        <label style={{ display: "block", marginBottom: "14px" }}>
          Email Address
          <input type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} style={authInputStyle} autoComplete="email" aria-invalid={Boolean(errors.email)} />
          {errors.email && <span style={{ display: "block", color: "#c53030", fontSize: "13px" }}>{errors.email}</span>}
        </label>
        <PasswordField label="Password" name="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} visible={showPassword} onToggle={() => setShowPassword(!showPassword)} error={errors.password} autoComplete="current-password" />
        {apiError && <p role="alert" style={{ color: "#c53030" }}>{apiError}</p>}
        <button type="submit" disabled={submitting} style={{ width: "100%", padding: "11px", cursor: submitting ? "wait" : "pointer" }}>
          {submitting ? "Signing In..." : "Sign In"}
        </button>
      </form>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: "18px" }}>
        <button type="button" onClick={() => onNavigate("/forgot-password")} style={{ border: 0, background: "none", color: "#1f6f8b", cursor: "pointer" }}>Forgot Password</button>
        <button type="button" onClick={() => onNavigate("/signup")} style={{ border: 0, background: "none", color: "#1f6f8b", cursor: "pointer" }}>Create Account</button>
      </div>
    </AuthShell>
  );
}

const signUpInitial = {
  first_name: "", last_name: "", contact_number: "", email: "", ministry: "", organization: "", state: "", password: "", confirm_password: "",
};

function passwordMessage(value) {
  const missing = [];
  if (value.length < 8) missing.push("8+ characters");
  if (!/[A-Z]/.test(value)) missing.push("uppercase");
  if (!/[a-z]/.test(value)) missing.push("lowercase");
  if (!/\d/.test(value)) missing.push("number");
  if (!/[^A-Za-z0-9]/.test(value)) missing.push("special character");
  return missing.length ? `Needs: ${missing.join(", ")}` : "Strong password";
}

function SignUp({ onNavigate }) {
  const [form, setForm] = useState(signUpInitial);
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState("");
  const [success, setSuccess] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const update = (name, value) => setForm((current) => ({ ...current, [name]: value }));
  const submit = async (event) => {
    event.preventDefault();
    const nextErrors = {};
    ["first_name", "last_name", "contact_number", "email", "ministry", "organization", "state", "password", "confirm_password"].forEach((field) => {
      if (!form[field].trim()) nextErrors[field] = "This field is required.";
    });
    if (form.password && passwordMessage(form.password) !== "Strong password") nextErrors.password = "Password must be at least 8 characters with uppercase, lowercase, number, and special character.";
    if (form.confirm_password && form.password !== form.confirm_password) nextErrors.confirm_password = "Passwords do not match.";
    if (form.email && !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(form.email)) nextErrors.email = "Enter a valid email address.";
    if (form.contact_number && !/(?:\+91[- ]?)?[6-9]\d{9}/.test(form.contact_number)) nextErrors.contact_number = "Enter a valid Indian mobile number.";
    setErrors(nextErrors);
    setApiError("");
    if (Object.keys(nextErrors).length) return;
    setSubmitting(true);
    try {
      const response = await axios.post(`${API}/api/auth/register`, form);
      setSuccess(response.data.message);
      setForm({ ...signUpInitial, email: form.email.toLowerCase() });
    } catch (error) {
      setErrors(error.response?.data?.errors || {});
      setApiError(getApiError(error, "Registration failed. Please try again."));
    } finally {
      setSubmitting(false);
    }
  };

  const fields = [
    ["first_name", "First Name", "text"], ["last_name", "Last Name", "text"],
    ["contact_number", "Contact Number", "tel"], ["email", "Email Address", "email"],
    ["ministry", "Ministry/Department", "text"], ["organization", "Organization", "text"], ["state", "State", "text"],
  ];
  return (
    <AuthShell title="Create Account" subtitle="Register an AVLOKAN user account." onNavigate={onNavigate}>
      {success ? (
        <div role="status">
          <p style={{ color: "#276749" }}>{success}</p>
          <button type="button" onClick={() => onNavigate("/signin")}>Go to Sign In</button>
        </div>
      ) : (
        <form onSubmit={submit} noValidate>
          {fields.map(([name, label, type]) => (
            <label key={name} style={{ display: "block", marginBottom: "12px" }}>
              {label}
              <input type={type} value={form[name]} onChange={(e) => update(name, e.target.value)} style={authInputStyle} autoComplete={name === "email" ? "email" : "off"} aria-invalid={Boolean(errors[name])} />
              {errors[name] && <span style={{ display: "block", color: "#c53030", fontSize: "13px" }}>{errors[name]}</span>}
            </label>
          ))}
          <PasswordField label="Create Password" name="password" value={form.password} onChange={(e) => update("password", e.target.value)} visible={showPassword} onToggle={() => setShowPassword(!showPassword)} error={errors.password} autoComplete="new-password" />
          <p style={{ marginTop: "-8px", fontSize: "13px", color: passwordMessage(form.password) === "Strong password" ? "#276749" : "#7b341e" }}>{passwordMessage(form.password)}</p>
          <PasswordField label="Confirm Password" name="confirm_password" value={form.confirm_password} onChange={(e) => update("confirm_password", e.target.value)} visible={showPassword} onToggle={() => setShowPassword(!showPassword)} error={errors.confirm_password} autoComplete="new-password" />
          {apiError && <p role="alert" style={{ color: "#c53030" }}>{apiError}</p>}
          <button type="submit" disabled={submitting} style={{ width: "100%", padding: "11px", cursor: submitting ? "wait" : "pointer" }}>{submitting ? "Creating Account..." : "Submit Registration"}</button>
        </form>
      )}
      <p style={{ textAlign: "center" }}><button type="button" onClick={() => onNavigate("/signin")} style={{ border: 0, background: "none", color: "#1f6f8b", cursor: "pointer" }}>Sign In</button></p>
    </AuthShell>
  );
}

function ForgotPassword({ onNavigate }) {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const submit = async (event) => {
    event.preventDefault();
    setError("");
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) { setError("Enter a valid email address."); return; }
    setSubmitting(true);
    try { const response = await axios.post(`${API}/api/auth/forgot-password`, { email }); setMessage(response.data.message); }
    catch (requestError) { setError(getApiError(requestError, "Unable to send the password reset email.")); }
    finally { setSubmitting(false); }
  };
  return <AuthShell title="Forgot Password" subtitle="Enter your registered email address and we will send you a password reset link." onNavigate={onNavigate}>
    <form onSubmit={submit} noValidate><label style={{ display: "block", marginBottom: "14px" }}>Email Address<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} style={authInputStyle} autoComplete="email" />{error && <span style={{ display: "block", color: "#c53030", fontSize: "13px" }}>{error}</span>}</label>{message && <p role="status" style={{ color: "#276749" }}>{message}</p>}<button type="submit" disabled={submitting} style={{ width: "100%", padding: "11px" }}>{submitting ? "Sending..." : "Send Reset Link"}</button></form>
    <button type="button" onClick={() => onNavigate("/signin")} style={{ marginTop: "16px", border: 0, background: "none", color: "#1f6f8b", cursor: "pointer" }}>Back to Sign In</button>
  </AuthShell>;
}

function ResetPassword({ onNavigate }) {
  const token = new URLSearchParams(window.location.search).get("token") || "";
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const submit = async (event) => {
    event.preventDefault(); setError("");
    if (!token) { setError("This password reset link is invalid or has expired. Please request a new reset link."); return; }
    if (passwordMessage(password) !== "Strong password") { setError("Password must be at least 8 characters with uppercase, lowercase, number, and special character."); return; }
    if (password !== confirm) { setError("Passwords do not match."); return; }
    setSubmitting(true);
    try { const response = await axios.post(`${API}/api/auth/reset-password`, { token, password }); setSuccess(response.data.message); }
    catch (requestError) { setError(getApiError(requestError, "This password reset link is invalid or has expired. Please request a new reset link.")); }
    finally { setSubmitting(false); }
  };
  return <AuthShell title="Reset Password" onNavigate={onNavigate}>
    {success ? <div role="status"><p style={{ color: "#276749" }}>{success}</p><button type="button" onClick={() => onNavigate("/signin")}>Go to Sign In</button></div> : <form onSubmit={submit} noValidate><PasswordField label="New Password" name="password" value={password} onChange={(e) => setPassword(e.target.value)} visible={showPassword} onToggle={() => setShowPassword(!showPassword)} error="" autoComplete="new-password" /><p style={{ fontSize: "13px" }}>{passwordMessage(password)}</p><PasswordField label="Confirm New Password" name="confirm" value={confirm} onChange={(e) => setConfirm(e.target.value)} visible={showPassword} onToggle={() => setShowPassword(!showPassword)} error="" autoComplete="new-password" />{error && <p role="alert" style={{ color: "#c53030" }}>{error}</p>}<button type="submit" disabled={submitting} style={{ width: "100%", padding: "11px" }}>{submitting ? "Resetting..." : "Reset Password"}</button></form>}
  </AuthShell>;
}

function GisMapPage({ user, onLogout, onNavigate, onOpenProjectDetails }) {
  return (
    <div style={{ padding: "25px 35px", background: "#f8fafc", minHeight: "100vh" }}>
      <HeaderNav user={user} currentRoute="/gis-map" onNavigate={onNavigate} onLogout={onLogout} />
      <GisMap
        onOpenProjectDetails={(projectCode) => {
          onOpenProjectDetails(projectCode);
        }}
      />
    </div>
  );
}

function App() {
  const [route, setRoute] = useState(window.location.pathname || "/signin");
  const [user, setUser] = useState(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const [registeredEmail, setRegisteredEmail] = useState("");
  const [initialProjectCode, setInitialProjectCode] = useState("");

  const navigate = (path) => {
    window.history.pushState({}, "", path);
    setRoute(path);
  };

  useEffect(() => {
    const onPopState = () => setRoute(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    axios.get(`${API}/api/auth/me`).then((response) => setUser(response.data.user)).catch(() => {}).finally(() => setCheckingSession(false));
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (!checkingSession && (route === "/dashboard" || route === "/gis-map") && !user) {
      const redirect = window.setTimeout(() => {
        window.history.pushState({}, "", "/signin");
        setRoute("/signin");
      }, 0);
      return () => window.clearTimeout(redirect);
    }
    return undefined;
  }, [checkingSession, route, user]);

  const logout = async () => {
    await axios.post(`${API}/api/auth/logout`).catch(() => {});
    setUser(null);
    navigate("/signin");
  };

  const handleOpenProjectDetails = (projectCode) => {
    setInitialProjectCode(projectCode);
    navigate("/dashboard");
  };

  if (checkingSession) return <main className="loading"><h1>Loading...</h1></main>;
  if (route === "/reset-password") return <ResetPassword onNavigate={navigate} />;
  if (route === "/signup") return <SignUp onNavigate={(path) => { if (path === "/signin") setRegisteredEmail(""); navigate(path); }} />;
  if (route === "/forgot-password") return <ForgotPassword onNavigate={navigate} />;
  if (route === "/gis-map" && user) return <GisMapPage user={user} onLogout={logout} onNavigate={navigate} onOpenProjectDetails={handleOpenProjectDetails} />;
  if (route === "/gis-map" && !user) return <main className="loading"><h1>Redirecting...</h1></main>;
  if (route === "/dashboard" && user) return <Dashboard user={user} onLogout={logout} onNavigate={navigate} initialProjectCode={initialProjectCode} clearInitialProjectCode={() => setInitialProjectCode("")} />;
  if (route === "/dashboard" && !user) return <main className="loading"><h1>Redirecting...</h1></main>;
  return <SignIn initialEmail={registeredEmail} onNavigate={navigate} onLogin={setUser} />;
}

export default App;
