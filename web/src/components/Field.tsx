import type { ChangeEvent, ReactNode } from "react";

interface NumberFieldProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: ReactNode;
  disabled?: boolean;
}

export function NumberField({ label, value, onChange, min, max, step, hint, disabled }: NumberFieldProps) {
  const handle = (e: ChangeEvent<HTMLInputElement>) => {
    const n = Number(e.target.value);
    if (Number.isFinite(n)) onChange(n);
  };
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input type="number" value={value} min={min} max={max} step={step} disabled={disabled} onChange={handle} />
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

interface TextFieldProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  hint?: ReactNode;
}

export function TextField({ label, value, onChange, placeholder, hint }: TextFieldProps) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

interface SelectFieldProps<T extends string> {
  label: string;
  value: T;
  options: { value: T; label: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  hint?: ReactNode;
}

export function SelectField<T extends string>({ label, value, options, onChange, hint }: SelectFieldProps<T>) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={o.disabled}>
            {o.label}
          </option>
        ))}
      </select>
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: ReactNode;
  disabled?: boolean;
}

export function CheckboxField({ label, checked, onChange, hint, disabled }: CheckboxFieldProps) {
  return (
    <label className="field field-checkbox">
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="field-label">{label}</span>
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}
