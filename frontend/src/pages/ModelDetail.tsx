import { useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Typography from '@mui/material/Typography'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { LineChart } from '@mui/x-charts/LineChart'
import { useQuery } from '@tanstack/react-query'
import SectionCard from '../components/SectionCard'
import {
  confusionTotal,
  fetchModelDetail,
  fmt3,
  modelLabel,
  pct,
  toRocPoints,
  type ClassificationReport,
  type ConfusionMatrix,
  type FeatureImportance,
} from '../modelData'

function ClassificationReportTable({ report }: { report: ClassificationReport | null }) {
  if (!report) return <Typography variant="body2">No classification report yet.</Typography>

  const rows = [
    ...Object.entries(report.classes).map(([name, m]) => ({ name, ...m })),
    { name: 'macro avg', ...report.macro_avg },
    { name: 'weighted avg', ...report.weighted_avg },
  ]

  const thSx = {
    fontSize: '0.7rem', fontWeight: 700,
    textTransform: 'uppercase' as const, letterSpacing: '0.08em',
    py: 0.75, px: 1.25, borderBottom: '1px solid', borderColor: 'divider',
    bgcolor: 'transparent', whiteSpace: 'nowrap' as const,
  }
  const tdSx = {
    fontSize: '0.875rem',
    py: 0.75,
    px: 1.25,
    borderBottom: '1px solid',
    borderColor: 'divider',
    fontVariantNumeric: 'tabular-nums' as const,
  }

  return (
    <Box sx={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {['Class', 'Precision', 'Recall', 'F1-score', 'Support'].map((h) => (
              <Box component="th" key={h} sx={{ ...thSx, textAlign: h === 'Class' ? 'left' : 'right' }}>{h}</Box>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <Box component="tr" key={row.name} sx={{ bgcolor: 'transparent' }}>
                <Box component="td" sx={{ ...tdSx, fontWeight: 400, color: '#0f172a', whiteSpace: 'nowrap' }}>
                {row.name}
              </Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right', color: 'text.primary' }}>{fmt3(row.precision)}</Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right', color: 'text.primary' }}>{fmt3(row.recall)}</Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right', color: 'text.primary' }}>{fmt3(row['f1-score'])}</Box>
              <Box component="td" sx={{ ...tdSx, textAlign: 'right', color: 'text.primary' }}>
                {Math.round(row.support).toLocaleString()}
              </Box>
            </Box>
          ))}
          <Box component="tr" sx={{ bgcolor: 'transparent' }}>
            <Box component="td" sx={{ ...tdSx, fontWeight: 400, color: '#0f172a' }}>accuracy</Box>
            <Box component="td" colSpan={3} sx={{ ...tdSx }} />
            <Box component="td" sx={{ ...tdSx, textAlign: 'right', color: 'text.primary' }}>{pct(report.accuracy)}</Box>
          </Box>
        </tbody>
      </table>
    </Box>
  )
}

function ConfusionMatrixGrid({ cm }: { cm: ConfusionMatrix | null }) {
  if (!cm) return <Typography variant="body2">No confusion matrix yet.</Typography>

  const total = cm.tn + cm.fp + cm.fn + cm.tp
  const formatShare = (value: number) => (total > 0 ? pct(value / total) : '0.0%')
  const cells = [
    { label: 'True Negative',  value: cm.tn, positive: true  },
    { label: 'False Positive', value: cm.fp, positive: false },
    { label: 'False Negative', value: cm.fn, positive: false },
    { label: 'True Positive',  value: cm.tp, positive: true  },
  ]

  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
      {cells.map((c) => (
        <Box key={c.label} sx={{ bgcolor: c.positive ? 'success.50' : 'error.50', borderRadius: 2, px: 1.5, py: 1.5 }}>
          <Typography variant="h5" sx={{ fontWeight: 700, color: c.positive ? 'success.dark' : 'error.dark', lineHeight: 1 }}>
            {c.value.toLocaleString()}
          </Typography>
          <Typography variant="caption" sx={{ fontWeight: 700, color: c.positive ? 'success.dark' : 'error.dark', display: 'block', mt: 0.5 }}>
            {c.label}
          </Typography>
          <Typography variant="caption" sx={{ color: c.positive ? 'success.dark' : 'error.dark' }}>
            {formatShare(c.value)} of total
          </Typography>
        </Box>
      ))}
    </Box>
  )
}

function FeatureImportanceList({ importances }: { importances: FeatureImportance[] | null }) {
  const sorted = useMemo(() => [...(importances ?? [])].sort((a, b) => b.importance - a.importance), [importances])
  if (!sorted.length) return <Typography variant="body2" color="text.secondary">No feature importances yet.</Typography>

  const max = sorted[0].importance
  const barWidth = (importance: number) => (max > 0 ? `${(importance / max) * 100}%` : '0%')

  return (
    <Box sx={{ maxHeight: 380, overflowY: 'auto' }}>
      {sorted.map((fi, idx) => (
        <Box key={fi.feature} sx={{ display: 'grid', gridTemplateColumns: '28px 1fr 120px 48px', alignItems: 'center', gap: 1, py: 0.6, borderBottom: '1px solid', borderColor: 'divider', '&:last-child': { borderBottom: 0 } }}>
          <Typography variant="caption" color="text.disabled" sx={{ textAlign: 'right' }}>{idx + 1}</Typography>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>{fi.feature}</Typography>
          <Box sx={{ borderRadius: 1, height: 6, overflow: 'hidden' }}>
            <Box sx={{ height: '100%', width: barWidth(fi.importance), bgcolor: 'primary.main', borderRadius: 1 }} />
          </Box>
          <Typography variant="caption" sx={{ textAlign: 'right' }}>
            {(fi.importance * 100).toFixed(1)}%
          </Typography>
        </Box>
      ))}
    </Box>
  )
}

function ModelDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const query = useQuery({
    queryKey: ['model', id],
    queryFn: () => fetchModelDetail(id!),
    enabled: !!id,
    staleTime: 30_000,
  })

  const model = query.data ?? null
  const rocPoints = useMemo(() => toRocPoints(model?.rocCurve), [model?.rocCurve])
  const modelConfusionTotal = model
    ? confusionTotal(model.confusionMatrix)
    : null

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Box>
          <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/models')} size="small" sx={{ mb: 1.5, color: 'text.secondary' }}>
            All models
          </Button>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            {model ? modelLabel(model) : `Model details${id ? `: ${id}` : ''}`}
          </Typography>
          {query.isError   && <Typography variant="body2" color="error"          sx={{ mt: 1 }}>Could not load model details from the backend.</Typography>}
          {query.isLoading && <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Loading model details...</Typography>}
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '3fr 2fr' }, gap: 2.5, alignItems: 'start' }}>
          <SectionCard title="Classification report">
            <ClassificationReportTable report={model?.classificationReport ?? null} />
          </SectionCard>
          <SectionCard title={`Confusion matrix (${modelConfusionTotal ?? '-'})`}>
            <ConfusionMatrixGrid cm={model?.confusionMatrix ?? null} />
          </SectionCard>
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2.5, alignItems: 'start' }}>
          <SectionCard title={`Feature importances (${model?.featureImportances.length ?? 0})`}>
            <FeatureImportanceList importances={model?.featureImportances ?? null} />
          </SectionCard>
          <SectionCard title="ROC curve">
            {rocPoints.length ? (
              <LineChart
                height={380}
                xAxis={[{ data: rocPoints.map((point) => point.fpr), label: 'False Positive Rate' }]}
                yAxis={[{ min: 0, max: 1, label: 'True Positive Rate' }]}
                series={[
                  {
                    data: rocPoints.map((point) => point.fpr),
                    label: 'Chance',
                    color: '#cbd5e1',
                    showMark: false,
                  },
                  {
                    data: rocPoints.map((point) => point.tpr),
                    label: `ROC curve (AUC = ${model ? fmt3(model.auc) : 'N/A'})`,
                    color: '#1976d2',
                    showMark: false,
                  },
                ]}
                margin={{ top: 16, right: 16, bottom: 28, left: 52 }}
              />
            ) : (
              <Box sx={{ height: 380, borderRadius: 2, border: '1px dashed', borderColor: 'divider', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                <Typography variant="body2" color="text.secondary">ROC curve</Typography>
                <Typography variant="caption" color="text.disabled">AUC = {model ? fmt3(model.auc) : 'N/A'}</Typography>
              </Box>
            )}
          </SectionCard>
        </Box>
      </Box>
    </main>
  )
}

export default ModelDetail
