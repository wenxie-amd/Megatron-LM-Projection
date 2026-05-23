import { useState } from "react";

import { ExplanationPanel, ExplanationProvider } from "./components/ExplanationPanel";
import { StepNav } from "./components/StepNav";
import { ProjectionProvider, useProjection } from "./state/context";
import { Step1Model } from "./steps/Step1Model";
import { Step2Machine } from "./steps/Step2Machine";
import { Step3Training } from "./steps/Step3Training";
import { Step4Memory } from "./steps/Step4Memory";
import { Step5Script } from "./steps/Step5Script";
import "./App.css";

const STEP_TITLES = ["Model", "Machine", "Training", "Memory", "Script"];

function WizardInner() {
  const { state } = useProjection();
  const [current, setCurrent] = useState(0);

  if (state.bridgeStatus === "loading") {
    return (
      <div className="bridge-status">
        <h1>Megatron-LM-Projection</h1>
        <p>Loading Pyodide and the projection package…</p>
        <p className="hint">First load is ~10 MB; subsequent loads are cached by the browser.</p>
      </div>
    );
  }
  if (state.bridgeStatus === "error") {
    return (
      <div className="bridge-status">
        <h1>Megatron-LM-Projection</h1>
        <pre className="error">Failed to load Pyodide: {state.bridgeError}</pre>
      </div>
    );
  }

  return (
    <div className="wizard">
      <header className="wizard-header">
        <h1>Megatron-LM-Projection</h1>
        <p className="subtitle">A static, in-browser memory projector for Megatron-LM training runs.</p>
      </header>
      <StepNav current={current} steps={STEP_TITLES} onSelect={setCurrent} />
      <div className="step-body">
        {current === 0 && <Step1Model />}
        {current === 1 && <Step2Machine />}
        {current === 2 && <Step3Training />}
        {current === 3 && <Step4Memory />}
        {current === 4 && <Step5Script />}
      </div>
      <ExplanationPanel />
    </div>
  );
}

function App() {
  return (
    <ExplanationProvider>
      <ProjectionProvider>
        <WizardInner />
      </ProjectionProvider>
    </ExplanationProvider>
  );
}

export default App;
