import { useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { ALGO_LABEL } from './ModelComparisonPage'
import type { Algorithm, FeatureSet } from './ModelComparisonPage'

type ClassMetrics = {
  precision: number
  recall: number
  f1: number
  support: number
}

type ClassificationReport = {
  classes: Record<string, ClassMetrics>
  macro_avg: ClassMetrics
  weighted_avg: ClassMetrics
  accuracy: number
}

type ConfusionMatrix = {
  tn: number
  fp: number
  fn: number
  tp: number
}

type FeatureImportance = {
  feature: string
  importance: number
}

export type ModelDetail = {
  id: string
  algorithm: Algorithm
  featureSet: FeatureSet
  auc: number
  classificationReport: ClassificationReport
  confusionMatrix: ConfusionMatrix
  featureImportances: FeatureImportance[]
}

function fmt3(v: number) {
  return v.toFixed(3)
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Paper
      elevation={0}
      sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}
    >
      <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
      </Box>
      <Box sx={{ p: 2 }}>
        {children}
      </Box>
    </Paper>
  )
}

function ClassificationReportTable({ report }: { report: ClassificationReport | null }) {
  if (!report) {
    return <Typography variant="body2" color="text.secondary">No classification report yet.</Typography>
  }

  const rows = [
    ...Object.entries(report.classes).map(([name, m]) => ({ name, ...m, isAvg: false })),
    { name: 'macro avg',    ...report.macro_avg,    isAvg: true },
    { name: 'weighted avg', ...report.weighted_avg, isAvg: true },
  ]

  const thSx = {
    fontSize: '0.7rem',
    fontWeight: 700,
    color: 'text.secondary',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    py: 0.75,
    px: 1,
    borderBottom: '1px solid',
    borderColor: 'divider',
    bgcolor: 'grey.50',
    whiteSpace: 'nowrap' as const,
  }

  const tdSx = {
    fontSize: '0.8125rem',
    py: 0.75,
    px: 1,
    borderBottom: '1px solid',
    borderColor: 'divider',
  }

  return (
    <Box sx={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {['Class', 'Precision', 'Recall', 'F1-score', 'Support'].map((h) => (
              <Box
                component="th"
                key={h}
                sx={{ ...thSx, textAlign: h === 'Class' ? 'left' : 'right' }}
              >
                {h}
              </Box>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <Box
              component="tr"
              key={row.name}
              sx={{
                bgcolor: row.isAvg ? 'grey.50' : 'transparent',
                '&:last-child td': { borderBottom: 0 },
              }}
            >
              <Box
                component="td"
                sx={{
                  ...tdSx,
                  fontWeight: row.isAvg ? 600 : 400,
                  color: row.isAvg ? 'text.secondary' : 'text.primary',
                  whiteSpace: 'nowrap',
                }}
              >
                {row.name}
              </Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right' }}>{fmt3(row.precision)}</Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right' }}>{fmt3(row.recall)}</Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right' }}>{fmt3(row.f1)}</Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right', color: 'text.secondary' }}>
                {row.support.toLocaleString()}
              </Box>
            </Box>
          ))}

          <Box component="tr" sx={{ bgcolor: 'grey.50' }}>
            <Box component="td" sx={{ ...tdSx, fontWeight: 600, color: 'text.secondary', borderBottom: 0 }}>
              accuracy
            </Box>
            <Box component="td" colSpan={3} sx={{ ...tdSx, borderBottom: 0 }} />
            <Box component="td" sx={{ ...tdSx, textAlign: 'right', fontWeight: 400, borderBottom: 0 }}>
              {pct(report.accuracy)}
            </Box>
          </Box>
        </tbody>
      </table>
    </Box>
  )
}

function ConfusionMatrixGrid({ cm }: { cm: ConfusionMatrix | null }) {
  if (!cm) {
    return <Typography variant="body2" color="text.secondary">No confusion matrix yet.</Typography>
  }

  const total = cm.tn + cm.fp + cm.fn + cm.tp

  const cells = [
    { label: 'True Negative',  value: cm.tn, sub: 'Actual: No stroke · Predicted: No stroke', positive: true },
    { label: 'False Positive', value: cm.fp, sub: 'Actual: No stroke · Predicted: Stroke',    positive: false },
    { label: 'False Negative', value: cm.fn, sub: 'Actual: Stroke · Predicted: No stroke',    positive: false },
    { label: 'True Positive',  value: cm.tp, sub: 'Actual: Stroke · Predicted: Stroke',       positive: true },
  ]

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
      {cells.map((c) => (
        <Box
          key={c.label}
          sx={{
            bgcolor: c.positive ? 'success.50' : 'error.50',
            borderRadius: 2,
            px: 1.5,
            py: 1.5,
          }}
        >
          <Typography
            variant="h5"
            sx={{ fontWeight: 700, color: c.positive ? 'success.dark' : 'error.dark', lineHeight: 1 }}
          >
            {c.value.toLocaleString()}
          </Typography>
          <Typography
            variant="caption"
            sx={{ fontWeight: 700, color: c.positive ? 'success.dark' : 'error.dark', display: 'block', mt: 0.5 }}
          >
            {c.label}
          </Typography>
          <Typography
            variant="caption"
            sx={{ color: c.positive ? 'success.dark' : 'error.dark', opacity: 0.75 }}
          >
            {pct(c.value / total)} of total
          </Typography>
        </Box>
      ))}
    </Box>
  )
}

function FeatureImportanceList({ importances }: { importances: FeatureImportance[] | null }) {
  if (!importances?.length) {
    return <Typography variant="body2" color="text.secondary">No feature importances yet.</Typography>
  }

  const sorted = useMemo(
    () => [...importances].sort((a, b) => b.importance - a.importance),
    [importances],
  )
  const max = sorted[0]?.importance ?? 1

  return (
    <Box sx={{ maxHeight: 380, overflowY: 'auto' }}>
      {sorted.map((fi, idx) => (
        <Box
          key={fi.feature}
          sx={{
            display: 'grid',
            gridTemplateColumns: '28px 1fr 120px 48px',
            alignItems: 'center',
            gap: 1,
            py: 0.6,
            borderBottom: '1px solid',
            borderColor: 'divider',
            '&:last-child': { borderBottom: 0 },
          }}
        >
          <Typography variant="caption" color="text.disabled" sx={{ textAlign: 'right' }}>
            {idx + 1}
          </Typography>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
            {fi.feature}
          </Typography>
          <Box sx={{ bgcolor: 'grey.100', borderRadius: 1, height: 6, overflow: 'hidden' }}>
            <Box
              sx={{
                height: '100%',
                width: `${(fi.importance / max) * 100}%`,
                bgcolor: 'primary.main',
                borderRadius: 1,
              }}
            />
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'right' }}>
            {(fi.importance * 100).toFixed(1)}%
          </Typography>
        </Box>
      ))}
    </Box>
  )
}

function ModelDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const model = null as ModelDetail | null

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Box>
          <Button
            startIcon={<ArrowBackIcon />}
            onClick={() => navigate('/models')}
            size="small"
            sx={{ mb: 1.5, color: 'text.secondary' }}
          >
            All models
          </Button>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              {model ? `${ALGO_LABEL[model.algorithm]} - ${model.featureSet}` : `Model details${id ? `: ${id}` : ''}`}
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '3fr 2fr' }, gap: 2.5, alignItems: 'start' }}>
          <SectionCard title="Classification report">
            <ClassificationReportTable report={model?.classificationReport ?? null} />
          </SectionCard>

          <SectionCard title="Confusion matrix">
            <ConfusionMatrixGrid cm={model?.confusionMatrix ?? null} />
          </SectionCard>
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2.5, alignItems: 'start' }}>
          <SectionCard title={`Feature importances (${model?.featureImportances.length ?? 0})`}>
            <FeatureImportanceList importances={model?.featureImportances ?? null} />
          </SectionCard>

          <SectionCard title="ROC curve">
            <Box
              sx={{
                height: 280,
                bgcolor: 'grey.50',
                borderRadius: 2,
                border: '1px dashed',
                borderColor: 'divider',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 0.5,
              }}
            >
              <Typography variant="body2" color="text.secondary">ROC curve</Typography>
              <Typography variant="caption" color="text.disabled">
                AUC = {model ? fmt3(model.auc) : 'N/A'}
              </Typography>
            </Box>
          </SectionCard>
        </Box>
      </Box>
    </main>
  )
}

export default ModelDetailPage