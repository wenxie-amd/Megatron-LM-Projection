import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";

export interface Explanation {
  title: string;
  body: ReactNode;
}

interface Ctx {
  current: Explanation | null;
  open: (e: Explanation) => void;
  close: () => void;
}

const ExplanationCtx = createContext<Ctx | null>(null);

export function ExplanationProvider({ children }: { children: ReactNode }) {
  const [current, setCurrent] = useState<Explanation | null>(null);
  const ctx: Ctx = {
    current,
    open: setCurrent,
    close: () => setCurrent(null),
  };
  return <ExplanationCtx.Provider value={ctx}>{children}</ExplanationCtx.Provider>;
}

export function useExplanation(): Ctx {
  const ctx = useContext(ExplanationCtx);
  if (!ctx) throw new Error("useExplanation must be used inside <ExplanationProvider>");
  return ctx;
}

export function ExplanationPanel() {
  const { current, close } = useExplanation();
  if (!current) return null;
  return (
    <aside className="explanation-panel" role="complementary" aria-labelledby="explanation-title">
      <header>
        <h3 id="explanation-title">{current.title}</h3>
        <button type="button" onClick={close} aria-label="Close explanation">
          ×
        </button>
      </header>
      <div className="explanation-body">{current.body}</div>
    </aside>
  );
}

interface InfoIconProps {
  title: string;
  body: ReactNode;
}

export function InfoIcon({ title, body }: InfoIconProps) {
  const { open } = useExplanation();
  return (
    <button
      type="button"
      className="info-icon"
      aria-label={`Explain ${title}`}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        open({ title, body });
      }}
    >
      i
    </button>
  );
}
