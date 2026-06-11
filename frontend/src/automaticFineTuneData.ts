import { fetchJson, postJson } from './api'
import { pct } from './modelData'
import { strokeRiskScore } from './modelScoring'
import type {
  Algorithm,
  BalancingMethod,
  FeatureSet,
  UncertaintyVariant,
} from './modelMetadata'

export type AutomaticTrainingRequest = {
  modelId: string
}

export type AutomaticTrainingResult = {
  modelId: string
  algorithm: Algorithm
  datasetId: string
  featureSet: FeatureSet
  uncertaintyVariant: UncertaintyVariant
  balancingMethod: BalancingMethod
  targetRatio: number
  classificationThreshold: number
  reusedExistingModel: boolean
  reusedBalancedData: boolean
  removedFeatures: string[]
  hyperparameters: Record<string, number>
  metrics: {
    auc: number
    accuracy: number
    f1: number
    precision: number
    recall: number
    classificationThreshold: number
  }
  confusionMatrix?: {
    tn: number
    fp: number
    fn: number
    tp: number
  }
  classificationReport?: unknown
}

export type AutomaticTrainingJob = {
  id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  message: string
  createdAt: string
  startedAt: string | null
  finishedAt: string | null
  request: {
    automatic?: boolean
    baseModelId?: string
    models?: unknown[]
  }
  result: {
    models: AutomaticTrainingResult[]
    total: number
    trained: number
    reused: number
  } | null
  error: string | null
}

export type AutomaticTrainingRunSummary = {
  id: string
  status: AutomaticTrainingJob['status']
  message: string
  createdAt: string
  startedAt: string | null
  finishedAt: string | null
  baseModelId: string | null
  total: number | null
  trained: number | null
  reused: number | null
  bestModelId: string | null
  bestScore: number | null
  error: string | null
}

export type AutomaticResultRow = AutomaticTrainingResult & {
  id: string
  score: number
  rank: number
}

export const AUTOMATIC_METRIC_LABELS = [
  { key: 'auc', label: 'AUC-ROC', format: (value: number) => value.toFixed(3) },
  { key: 'accuracy', label: 'Accuracy', format: pct },
  { key: 'f1', label: 'F1', format: (value: number) => value.toFixed(3) },
  { key: 'precision', label: 'Precision', format: pct },
  { key: 'recall', label: 'Recall', format: pct },
] as const

export async function startAutomaticTrainingJob(request: AutomaticTrainingRequest): Promise<AutomaticTrainingJob> {
  return postJson<AutomaticTrainingJob>('/api/training/automatic-jobs', request)
}

export async function fetchTrainingJob(jobId: string): Promise<AutomaticTrainingJob> {
  return fetchJson<AutomaticTrainingJob>(`/api/training/jobs/${jobId}`)
}

export async function fetchAutomaticTrainingRuns(): Promise<AutomaticTrainingRunSummary[]> {
  return fetchJson<AutomaticTrainingRunSummary[]>('/api/training/automatic-runs')
}

export async function fetchAutomaticTrainingRun(jobId: string): Promise<AutomaticTrainingJob> {
  return fetchJson<AutomaticTrainingJob>(`/api/training/automatic-runs/${jobId}`)
}

export function automaticTrainingRunExportUrl(jobId: string) {
  return `/api/training/automatic-runs/${encodeURIComponent(jobId)}/export`
}

export function automaticResultRows(results: AutomaticTrainingResult[] = []): AutomaticResultRow[] {
  return results
    .map((result) => ({
      ...result,
      id: result.modelId,
      score: strokeRiskScore(result.metrics),
      rank: 0,
    }))
    .sort((left, right) => (
      right.score - left.score ||
      right.metrics.auc - left.metrics.auc ||
      right.metrics.recall - left.metrics.recall ||
      right.metrics.f1 - left.metrics.f1
    ))
    .map((result, index) => ({ ...result, rank: index + 1 }))
}

export function paramLabel(key: string) {
  return key
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}
