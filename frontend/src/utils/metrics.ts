/**
 * Shared metric validation and formatting utilities.
 * Single frontend source for display consistency.
 */

export function safeNum(value: unknown, fallback = 0): number {
  if (value === null || value === undefined) return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return n;
}

export function formatPercent(value: unknown, decimals = 1, fallback = 0): string {
  return `${safeNum(value, fallback).toFixed(decimals)}%`;
}

export function formatMs(value: unknown, decimals = 1): string {
  return `${safeNum(value).toFixed(decimals)} ms`;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export type WarningLevel = 'ok' | 'warn' | 'critical';

export interface ThresholdResult {
  level: WarningLevel;
  color: string;
}

export function getUsageWarning(value: number, warnAt = 75, criticalAt = 90): ThresholdResult {
  if (value >= criticalAt) return { level: 'critical', color: '#ff2a9d' };
  if (value >= warnAt) return { level: 'warn', color: '#ffea00' };
  return { level: 'ok', color: '#00ff9d' };
}

export function getFpsWarning(fps: number, warnBelow = 30, criticalBelow = 20): ThresholdResult {
  if (fps < criticalBelow) return { level: 'critical', color: '#ff2a9d' };
  if (fps < warnBelow) return { level: 'warn', color: '#ffea00' };
  return { level: 'ok', color: '#00ff9d' };
}

export function getLatencyWarning(ms: number, warnAbove = 50, criticalAbove = 100): ThresholdResult {
  if (ms >= criticalAbove) return { level: 'critical', color: '#ff2a9d' };
  if (ms >= warnAbove) return { level: 'warn', color: '#ffea00' };
  return { level: 'ok', color: '#00ff9d' };
}

/** Model validation accuracy — NOT prediction confidence */
export function getModelAccuracy(modelStats: Record<string, unknown> | null | undefined): number {
  return safeNum(modelStats?.accuracy);
}

/** Session-average confidence from backend session analytics */
export function getSessionConfidence(session: Record<string, unknown> | null | undefined): number {
  return safeNum(session?.avg_confidence);
}
