# AVLOKAN

## AI-Powered Infrastructure Project Monitoring and Risk Prediction Platform

AVLOKAN is a web-based infrastructure project-monitoring platform designed to help officials monitor large-scale infrastructure projects, identify high-risk projects, and understand possible cost overruns and schedule slippages.

The platform uses historical project data, machine learning, interactive dashboards, GIS visualization, and explainable AI to support data-driven project monitoring.

---

## Problem Statement

Large infrastructure projects may experience:

- Cost overruns
- Delays in project execution
- Slow physical progress
- Increasing expenditure
- Differences between planned and actual progress
- Difficulty identifying high-risk projects at an early stage

AVLOKAN helps officials identify these risks through a centralized monitoring and analytics platform.

---

## Key Features

### 1. Infrastructure Project Dashboard

- Overall project statistics
- Total number of monitored projects
- Project progress overview
- Cost and expenditure information
- Risk-level summary
- Project monitoring insights

### 2. Project Listing and Search

- View infrastructure projects in a centralized table
- Search projects using project code or other details
- Filter projects based on available categories
- View project-specific information
- Identify projects requiring attention

### 3. AI-Based Cost Overrun Prediction

The platform uses a machine learning model to estimate the probability of future cost-overrun risk.

The model analyzes available project information such as:

- Original project cost
- Revised project cost
- Expenditure
- Expenditure ratio
- Physical progress
- Cost growth
- Previous project progress
- Historical project information

Risk levels are displayed as:

- LOW
- MEDIUM
- HIGH

### 4. Current Schedule Slippage Risk Prediction

AVLOKAN identifies projects that are currently showing signs of schedule slippage.

The model considers project progress-related information such as:

- Planned progress
- Actual physical progress
- Progress gap
- Previous progress
- Progress change
- Progress velocity
- Project observation history

This feature indicates current schedule-slippage risk. It does not claim to predict the exact final completion date.

### 5. Explainable AI

AVLOKAN provides explanations for model predictions using SHAP-based analysis.

Officials can understand:

- Why a project is marked as high risk
- Which factors increase the risk
- Which factors reduce the risk
- How individual project features influence the prediction

This improves transparency and helps officials interpret AI results.

### 6. GIS Map Visualization

The GIS Map feature provides a geographical view of monitored infrastructure projects.

It supports:

- Project location visualization
- Project information through map popups
- Project risk information
- Filtering based on available categories
- Access to project details
- PDF report download from the map popup

### 7. Downloadable Project Reports

Officials can download a PDF report for individual projects from:

- Project table report column
- Project details panel
- GIS Map popup

Each report is generated using the selected project’s project code and contains relevant project and risk information.

### 8. Project Details

Officials can view detailed information for individual projects, including:

- Project code
- Project name
- Ministry or department
- Agency
- Sector
- State
- Original cost
- Revised cost
- Expenditure
- Physical progress
- Project risk information
- AI prediction results
- Risk explanations

### 9. Risk-Based Monitoring

Projects can be examined according to risk levels:

- High-risk projects
- Medium-risk projects
- Low-risk projects

This helps officials focus attention on projects that require further monitoring.

---

## Technology Stack

### Frontend

- React.js
- Vite
- JavaScript
- HTML
- CSS
- Axios

### Backend

- Python
- Flask
- REST APIs
- ReportLab

### Machine Learning

- LightGBM
- Scikit-learn
- Pandas
- NumPy
- SHAP

### Data Visualization

- Interactive dashboard charts
- GIS map visualization
- Project risk summaries
- Project-level analytics

### Data Sources

- Infrastructure project monitoring datasets
- Historical project records
- Project cost and expenditure data
- Physical progress information
- Project monitoring reports

---

## System Architecture

```text
Project Monitoring Data
          |
          v
Data Cleaning and Preprocessing
          |
          v
Feature Engineering
          |
          v
Machine Learning Models
          |
          +--------------------------+
          |                          |
          v                          v
Cost Overrun Model        Schedule Slippage Model
          |                          |
          +------------+-------------+
                       |
                       v
              Explainable AI
                 using SHAP
                       |
                       v
                Flask Backend
                       |
                       v
                React Frontend
                       |
          +------------+-------------+
          |            |             |
          v            v             v
       Dashboard    GIS Map      PDF Reports



AVLOKAN/
│
├── backend/
│   ├── app.py
│   ├── report_generator.py
│   ├── notification_service.py
│   └── model and preprocessing files
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       └── GisMap.jsx
│   ├── package.json
│   └── vite.config.js
│
├── datasets/
├── machine learning scripts/
├── model files/
├── README.md
└── .gitignore
