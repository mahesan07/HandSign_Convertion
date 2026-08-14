/** Small shared presentation pieces. */

import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Card({
  title,
  action,
  className = "",
  children,
}: {
  title?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`card raised ${className}`}>
      {(title || action) && (
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "0.75rem",
          }}
        >
          {title && <h2 className="card__title">{title}</h2>}
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "danger";
  hint?: string;
};

export function Button({
  variant = "default",
  hint,
  children,
  className = "",
  ...rest
}: ButtonProps) {
  const variantClass =
    variant === "primary"
      ? "button--primary"
      : variant === "danger"
        ? "button--danger"
        : "";
  return (
    <button type="button" className={`button ${variantClass} ${className}`} {...rest}>
      {children}
      {hint && <span className="button__hint">{hint}</span>}
    </button>
  );
}

export function Notice({
  tone = "info",
  title,
  detail,
  icon,
  action,
}: {
  tone?: "info" | "warn" | "danger";
  title: string;
  detail?: ReactNode;
  icon?: string;
  action?: ReactNode;
}) {
  const toneClass = tone === "info" ? "" : `notice--${tone}`;
  return (
    <div className={`notice ${toneClass}`} role={tone === "danger" ? "alert" : "status"}>
      <span className="notice__icon" aria-hidden="true">
        {icon ?? (tone === "danger" ? "!" : tone === "warn" ? "!" : "i")}
      </span>
      <div className="notice__body">
        <p className="notice__title">{title}</p>
        {detail && <p className="notice__detail">{detail}</p>}
      </div>
      {action}
    </div>
  );
}

export function Badge({
  children,
  muted = false,
  tone,
}: {
  children: ReactNode;
  muted?: boolean;
  tone?: string;
}) {
  return (
    <span
      className={`badge ${muted ? "badge--muted" : ""}`}
      style={tone ? { color: tone } : undefined}
    >
      {children}
    </span>
  );
}
