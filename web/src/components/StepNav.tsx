interface Props {
  current: number;
  steps: string[];
  onSelect: (idx: number) => void;
}

export function StepNav({ current, steps, onSelect }: Props) {
  return (
    <nav className="step-nav">
      {steps.map((label, idx) => (
        <button
          key={idx}
          type="button"
          className={"step-tab" + (idx === current ? " active" : "")}
          onClick={() => onSelect(idx)}
        >
          <span className="step-num">{idx + 1}</span>
          <span className="step-label">{label}</span>
        </button>
      ))}
    </nav>
  );
}
