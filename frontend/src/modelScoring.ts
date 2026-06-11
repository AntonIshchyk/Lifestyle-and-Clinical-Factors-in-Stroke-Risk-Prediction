export const STROKE_SCORE_WEIGHTS = {
  auc: 0.35,
  f1: 0.3,
  recall: 0.25,
  precision: 0.1,
} as const

export type ScoreMetrics = {
  auc: number
  f1: number
  recall: number
  precision: number
}

export function strokeRiskScore(model: ScoreMetrics) {
  return (
    model.auc * STROKE_SCORE_WEIGHTS.auc +
    model.f1 * STROKE_SCORE_WEIGHTS.f1 +
    model.recall * STROKE_SCORE_WEIGHTS.recall +
    model.precision * STROKE_SCORE_WEIGHTS.precision
  )
}
