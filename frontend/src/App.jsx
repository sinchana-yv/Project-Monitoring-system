
import { useEffect, useState } from "react";
import axios from "axios";

const API = "http://127.0.0.1:5001";

const getRiskLevel = (probability) => {
  if (probability >= 0.7) {
    return "HIGH";
  }
  if (probability >= 0.4) {
    return "MEDIUM";
  }
  return "LOW";
};

function App() {
  const [summary, setSummary] = useState(null);
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [timeDelay, setTimeDelay] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [timeDelayExplanation, setTimeDelayExplanation] = useState(null);
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

    const [
      projectResult,
      predictionResult,
      timeDelayResult,
      timeDelayExplanationResult,
    ] =
      await Promise.allSettled([
        axios.get(`${API}/api/projects/${projectCode}`),
        axios.get(`${API}/api/projects/${projectCode}/prediction`),
        axios.get(`${API}/api/projects/${projectCode}/time-delay`),
        axios.get(
          `${API}/api/projects/${projectCode}/time-delay-explanation`
        ),
      ]);

    if (projectResult.status === "fulfilled") {
      setSelectedProject(projectResult.value.data);
    } else {
      console.error("Project details error:", projectResult.reason);
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
    if (projectResult.status === "rejected") {
      const status = projectResult.reason.response?.status;
      setDetailsError(
        status === 404
          ? "This project could not be found."
          : "Unable to load the project details."
      );
    }

    if (timeDelayResult.status === "fulfilled") {
      const timeDelayData = timeDelayResult.value.data;
      const probability = Number(
          timeDelayData.time_delay_probability
      );
      setTimeDelay({
          ...timeDelayData,
          risk_level: getRiskLevel(probability),
      });
    } else {
      console.error(
          "Time Delay prediction error:",
          timeDelayResult.reason
      );
      const status = timeDelayResult.reason.response?.status;
      setTimeDelayError(
          status === 404
            ? "Time Delay prediction is unavailable for this project."
            : "Unable to load the Time Delay assessment."
      );
    }
    setDetailsLoading(false);
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

    try {
      const response = await axios.get(
        `${API}/api/projects/${projectCode}/explanation`
      );
      setExplanation(response.data);
    } catch (requestError) {
      console.error("Project explanation error:", requestError);
      const status = requestError.response?.status;
      setExplanationError(
        status === 404
          ? "SHAP explanation is unavailable for this project."
          : "SHAP explanation is currently unavailable."
      );
    } finally {
      setExplanationLoading(false);
    }
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
      name.includes(searchText) ||
      code.includes(searchText);

    const matchesRisk =
      riskFilter === "ALL" ||
      String(project.risk_level).toUpperCase() === riskFilter;

    return matchesSearch && matchesRisk;
  });

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>

      <h1>PAIMANA AI Project Monitoring</h1>

      <hr />

      {/* SUMMARY */}

      <h2>Total Projects: {summary.total_projects}</h2>

      <h2>High Risk: {summary.high_risk}</h2>

      <h2>Medium Risk: {summary.medium_risk}</h2>

      <h2>Low Risk: {summary.low_risk}</h2>

      <h2>
        Average Risk: {summary.average_risk_probability}%
      </h2>

      <hr />

      {/* SEARCH */}

      <h2>Project Risk Monitoring</h2>

      <input
        type="text"
        placeholder="Search project name or code..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{
          padding: "10px",
          width: "350px",
          marginRight: "10px",
        }}
      />

      <select
        value={riskFilter}
        onChange={(e) => setRiskFilter(e.target.value)}
        style={{
          padding: "10px",
        }}
      >
        <option value="ALL">All Risk Levels</option>
        <option value="HIGH">High Risk</option>
        <option value="MEDIUM">Medium Risk</option>
        <option value="LOW">Low Risk</option>
      </select>

      <p>
        Showing {filteredProjects.length} projects
      </p>

      {/* TABLE */}

      <table
        border="1"
        cellPadding="10"
        style={{
          borderCollapse: "collapse",
          width: "100%",
        }}
      >

        <thead>
          <tr>
            <th>Rank</th>
            <th>Project</th>
            <th>State</th>
            <th>Sector</th>
            <th>Probability</th>
            <th>Risk</th>
          </tr>
        </thead>

        <tbody>

          {filteredProjects.slice(0, 10).map(
            (project, index) => (

              <tr key={project.project_code}>

                <td>
                  #{index + 1}
                </td>

                {/* CLICKABLE PROJECT */}

                <td>
                  <button
                    onClick={() =>
                      openProject(project.project_code)
                    }
                    style={{
                      background: "none",
                      border: "none",
                      padding: 0,
                      color: "blue",
                      cursor: "pointer",
                      textAlign: "left",
                      fontSize: "15px",
                    }}
                  >
                    {project.project_name}
                  </button>
                </td>

                <td>
                  {project.state}
                </td>

                <td>
                  {project.sector}
                </td>

                <td>
                  {Number(
                    project.overrun_probability_percent
                  ).toFixed(2)}
                  %
                </td>

                <td>
                  {project.risk_level}
                </td>

              </tr>
            )
          )}

        </tbody>

      </table>

      {/* PROJECT DETAILS */}

      {detailsLoading && (
        <div
          style={{
            marginTop: "30px",
            padding: "25px",
            border: "1px solid #ccc",
            borderRadius: "10px",
            background: "#f8f9fa",
          }}
        >
          Loading project details and AI assessment...
        </div>
      )}

      {selectedProject && (
        <div
          style={{
            marginTop: "30px",
            padding: "25px",
            border: "1px solid #ccc",
            borderRadius: "10px",
            background: "#f8f9fa",
          }}
        >

          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >

            <h2>
              Project Details
            </h2>

            <button
              onClick={() => setSelectedProject(null)}
              style={{
                padding: "8px 15px",
                cursor: "pointer",
              }}
            >
              Close
            </button>

          </div>

          <hr />

          <h3>
            {selectedProject.project_name}
          </h3>

          <p>
            <strong>Project Code:</strong>{" "}
            {selectedProject.project_code}
          </p>

          <p>
            <strong>State:</strong>{" "}
            {selectedProject.state}
          </p>

          <p>
            <strong>Sector:</strong>{" "}
            {selectedProject.sector}
          </p>

          <p>
            <strong>Ministry:</strong>{" "}
            {selectedProject.ministry_final}
          </p>

          <p>
            <strong>Agency:</strong>{" "}
            {selectedProject.agency_final}
          </p>

          <hr />

          <h3>Financial Information</h3>

          <p>
            <strong>Original Cost:</strong> ₹
            {Number(selectedProject.original_cost).toFixed(2)} crore
          </p>

          <p>
            <strong>Expenditure:</strong> ₹
            {Number(selectedProject.expenditure).toFixed(2)} crore
          </p>

          <p>
            <strong>Expenditure Ratio:</strong>{" "}
            {Number(selectedProject.expenditure_ratio).toFixed(2)}%
          </p>

          <hr />

          <h3>Project Progress</h3>

          <p>
            <strong>Physical Progress:</strong>{" "}
            {Number(selectedProject.physical_progress).toFixed(2)}%
          </p>

          <p>
            <strong>Previous Progress:</strong>{" "}
            {Number(selectedProject.previous_progress).toFixed(2)}%
          </p>

          <p>
            <strong>Progress Change:</strong>{" "}
            {Number(selectedProject.progress_change).toFixed(2)}%
          </p>

          <p>
            <strong>Reports Observed:</strong>{" "}
            {selectedProject.months_observed}
          </p>

          <hr />

          <h3>🤖 AI Risk Assessment</h3>

          {detailsLoading && <p>Loading AI risk assessment...</p>}

          {detailsError && <p style={{ color: "#c53030" }}>{detailsError}</p>}

          {prediction && (
            <div>
              <p>
                <strong>Cost Overrun Probability:</strong>{" "}
                {Number(prediction.overrun_probability * 100).toFixed(2)}%
              </p>

              <p>
                <strong>Prediction:</strong> {prediction.prediction}
              </p>

              <p>
                <strong>AI Risk Level:</strong>{" "}
                <span
                  className={`risk-badge ${prediction.risk_level.toLowerCase()}`}
                >
                  {prediction.risk_level}
                </span>
              </p>
            </div>
          )}

          <hr />

          <h3>Current Schedule Slippage Risk</h3>

          {timeDelayLoading && <p>Loading Time Delay assessment...</p>}

          {timeDelayError && (
            <p style={{ color: "#c53030" }}>{timeDelayError}</p>
          )}

          {timeDelay && (
            <div>
              <p>
                <strong>Time Delay Probability:</strong>{" "}
                {Number(
                  timeDelay.time_delay_probability * 100
                ).toFixed(2)}
                %
              </p>

              <p>
                <strong>Prediction:</strong> {timeDelay.prediction}
              </p>

              <p>
                <strong>Risk Level:</strong>{" "}
                <span
                  className={`risk-badge ${timeDelay.risk_level.toLowerCase()}`}
                >
                  {timeDelay.risk_level}
                </span>
              </p>
            </div>
          )}

          <hr />

          <h3>Why is this project at schedule slippage risk?</h3>

          <p>
            These factors contributed most to the project's schedule slippage
            risk.
          </p>

          {timeDelayExplanationLoading && (
            <p>Loading Time Delay explanation...</p>
          )}

          {timeDelayExplanationError && (
            <p style={{ color: "#a56b00" }}>
              {timeDelayExplanationError}
            </p>
          )}

          {timeDelayExplanation && (
            <div>
              <h4>Top risk factors</h4>
              {timeDelayExplanation.risk_factors?.length ? (
                <ul>
                  {timeDelayExplanation.risk_factors.map((factor) => (
                    <li key={`${factor.feature}-${factor.shap_value}`}>
                      <strong>{factor.feature}</strong>:{" "}
                      {Number(factor.shap_value).toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No schedule risk factors were returned.</p>
              )}

              <h4>Protective factors</h4>
              {timeDelayExplanation.protective_factors?.length ? (
                <ul>
                  {timeDelayExplanation.protective_factors.map((factor) => (
                    <li key={`${factor.feature}-${factor.shap_value}`}>
                      <strong>{factor.feature}</strong>:{" "}
                      {Number(factor.shap_value).toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No protective factors were returned.</p>
              )}
            </div>
          )}

          <hr />

          <h3>Why is this project at risk?</h3>

          <p>
            These factors contributed most to the model's risk prediction.
          </p>

          {explanationLoading && <p>Loading SHAP explanation...</p>}

          {explanationError && (
            <p style={{ color: "#a56b00" }}>{explanationError}</p>
          )}

          {explanation && (
            <div>
              <h4>Risk-increasing factors</h4>
              {explanation.top_risk_factors?.length ? (
                <ul>
                  {explanation.top_risk_factors.map((factor) => (
                    <li key={`${factor.feature}-${factor.shap_value}`}>
                      <strong>{factor.feature}</strong>:{" "}
                      {Number(factor.shap_value).toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No risk-increasing factors were returned.</p>
              )}

              <h4>Protective factors</h4>
              {explanation.top_protective_factors?.length ? (
                <ul>
                  {explanation.top_protective_factors.map((factor) => (
                    <li key={`${factor.feature}-${factor.shap_value}`}>
                      <strong>{factor.feature}</strong>:{" "}
                      {Number(factor.shap_value).toFixed(4)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No protective factors were returned.</p>
              )}
            </div>
          )}

        </div>
      )}

    </div>
  );
}

export default App;
