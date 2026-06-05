import { fetchJson } from './api'
import {
  ALGORITHM_LABELS,
  FEATURE_SET_LABELS,
  UNCERTAINTY_VARIANT_LABELS,
  type Algorithm,
  type FeatureSet,
  type UncertaintyVariant,
} from './modelMetadata'

export type ClassMetrics = {
  precision: number
  recall: number
  'f1-score': number
  support: number
}

export type ClassificationReport = {
  classes: Record<string, ClassMetrics>
  macro_avg: ClassMetrics
  weighted_avg: ClassMetrics
  accuracy: number
}

export type ConfusionMatrix = {
  tn: number
  fp: number
  fn: number
  tp: number
}

export type FeatureImportance = {
  feature: string
  importance: number
}

export type RocCurvePoint = {
  fpr: number
  tpr: number
}

export type RocCurve = {
  fpr: number[]
  tpr: number[]
} | RocCurvePoint[]

export type ModelDetail = {
  id: string
  algorithm: Algorithm
  featureSet: FeatureSet
  uncertaintyVariant: UncertaintyVariant
  auc: number
  classificationReport: ClassificationReport
  confusionMatrix: ConfusionMatrix
  featureImportances: FeatureImportance[]
  rocCurve: RocCurve
  featureColumns: string[]
}

export async function fetchModelDetail(modelId: string): Promise<ModelDetail> {
  return fetchJson<ModelDetail>(`/api/models/${modelId}`)
}

export const fmt3 = (v: number) => v.toFixed(3)
export const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export function modelLabel(model: Pick<ModelDetail, 'algorithm' | 'featureSet' | 'uncertaintyVariant'>) {
  return `${ALGORITHM_LABELS[model.algorithm]} - ${FEATURE_SET_LABELS[model.featureSet]} - ${UNCERTAINTY_VARIANT_LABELS[model.uncertaintyVariant]}`
}

export function confusionTotal(cm: ConfusionMatrix) {
  return cm.tn + cm.fp + cm.fn + cm.tp
}

function sanitizeRocPoints(points: RocCurvePoint[]): RocCurvePoint[] {
  return points
    .filter((point) => Number.isFinite(point.fpr) && Number.isFinite(point.tpr))
    .sort((left, right) => left.fpr - right.fpr)
}

export function toRocPoints(rocCurve: RocCurve | null | undefined): RocCurvePoint[] {
  if (!rocCurve) return []

  if (Array.isArray(rocCurve)) return sanitizeRocPoints(rocCurve)

  const len = Math.min(rocCurve.fpr.length, rocCurve.tpr.length)
  return sanitizeRocPoints(Array.from({ length: len }, (_, index) => ({ fpr: rocCurve.fpr[index], tpr: rocCurve.tpr[index] })))
}
