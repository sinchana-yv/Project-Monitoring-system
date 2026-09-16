
import { useEffect, useState } from "react";
import axios from "axios";

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

function Dashboard({ user, onLogout }) {
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

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
          padding: "12px 16px",
          border: "1px solid #d9e2ec",
          borderRadius: "8px",
          background: "#f8f9fa",
        }}
      >
        <div>
          <strong>{user.first_name} {user.last_name}</strong>
          <div style={{ color: "#52606d", fontSize: "13px" }}>
            {user.email} | {user.ministry} | {user.organization} | {user.state}
          </div>
        </div>
        <button onClick={onLogout} style={{ padding: "8px 14px", cursor: "pointer" }}>
          Logout
        </button>
      </div>

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
                padding: "8px 15px",
                cursor: "pointer",
              }}
            >
              Download Report
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
                      {factor.explanation && <span> ({factor.explanation})</span>}
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
                      {factor.explanation && <span> ({factor.explanation})</span>}
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
                      {factor.explanation && <span> ({factor.explanation})</span>}
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
                      {factor.explanation && <span> ({factor.explanation})</span>}
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
        <h1 style={{ marginTop: 0, color: "#102a43" }}>NIRIKSHAN</h1>
        <p style={{ color: "#52606d" }}>SIH26103 | Intelligent Infrastructure Project Monitoring</p>
        <h2>{title}</h2>
        {subtitle && <p style={{ color: "#52606d" }}>{subtitle}</p>}
        {children}
        <p style={{ marginBottom: 0, fontSize: "13px", color: "#52606d" }}>
          NIRIKSHAN prototype authentication. Use only an account you are authorized to access.
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
    <AuthShell title="Create Account" subtitle="Register a NIRIKSHAN user account." onNavigate={onNavigate}>
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

function App() {
  const [route, setRoute] = useState(window.location.pathname || "/signin");
  const [user, setUser] = useState(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const [registeredEmail, setRegisteredEmail] = useState("");

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
    if (!checkingSession && route === "/dashboard" && !user) {
      navigate("/signin");
    }
  }, [checkingSession, route, user]);

  const logout = async () => {
    await axios.post(`${API}/api/auth/logout`).catch(() => {});
    setUser(null);
    navigate("/signin");
  };

  if (checkingSession) return <main className="loading"><h1>Loading...</h1></main>;
  if (route === "/reset-password") return <ResetPassword onNavigate={navigate} />;
  if (route === "/signup") return <SignUp onNavigate={(path) => { if (path === "/signin") setRegisteredEmail(""); navigate(path); }} />;
  if (route === "/forgot-password") return <ForgotPassword onNavigate={navigate} />;
  if (route === "/dashboard" && user) return <Dashboard user={user} onLogout={logout} />;
  if (route === "/dashboard" && !user) return <main className="loading"><h1>Redirecting...</h1></main>;
  return <SignIn initialEmail={registeredEmail} onNavigate={navigate} onLogin={setUser} />;
}

export default App;
