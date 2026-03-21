import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("Dashboard crash:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, color: "#ff4b4b", background: "#0a0c10", minHeight: "100vh", fontFamily: "monospace", fontSize: 14, whiteSpace: "pre-wrap" }}>
          <h1 style={{ color: "#e10600", marginBottom: 20 }}>Dashboard Error</h1>
          <p style={{ color: "#ccc" }}>{String(this.state.error)}</p>
          <p style={{ color: "#888", marginTop: 10 }}>Check browser console (F12) for full stack trace.</p>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
