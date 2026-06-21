import { useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import TableSortLabel from '@mui/material/TableSortLabel'
import Typography from '@mui/material/Typography'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { LineChart } from '@mui/x-charts/LineChart'
import { useQueries } from '@tanstack/react-query'
import SectionCard from '../components/SectionCard'
import {
  confusionTotal,
  fetchModelDetail,
  fmt3,
  modelLabel,
  pct,
  toRocPoints,
  type ModelDetail,
  type RocCurvePoint,
} from '../modelData'

type MetricRow = {
  label: string
  getValue: (model: ModelDetail) => string
}

const chartColors = ['#1976d2', '#2e7d32', '#9c27b0', '#ed6c02', '#00838f', '#c2185b']

const metricRows: MetricRow[] = [
  { label: 'AUC-ROC', getValue: (model) => fmt3(model.auc) },
  { label: 'Accuracy', getValue: (model) => pct(model.classificationReport.accuracy) },
  { label: 'F1', getValue: (model) => fmt3(model.classificationReport.macro_avg['f1-score']) },
  { label: 'Precision', getValue: (model) => pct(model.classificationReport.macro_avg.precision) },
  { label: 'Recall', getValue: (model) => pct(model.classificationReport.macro_avg.recall) },
  { label: 'Threshold', getValue: (model) => model.classificationThreshold.toFixed(2) },
]

function interpolateTpr(points: RocCurvePoint[], fpr: number) {
  if (!points.length) return null
  if (fpr <= points[0].fpr) return points[0].tpr

  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1]
    const current = points[index]

    if (fpr <= current.fpr) {
      const span = current.fpr - previous.fpr
      if (span <= 0) return current.tpr
      const share = (fpr - previous.fpr) / span
      return previous.tpr + share * (current.tpr - previous.tpr)
    }
  }

  return points[points.length - 1].tpr
}

function PerformanceTable({ models }: { models: ModelDetail[] }) {
  const thSx = {
    fontSize: '0.7rem',
    fontWeight: 700,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    py: 0.75,
    px: 1.25,
    borderBottom: '1px solid',
    borderColor: 'divider',
    whiteSpace: 'nowrap' as const,
  }
  const tdSx = {
    fontSize: '0.875rem',
    py: 0.85,
    px: 1.25,
    borderBottom: '1px solid',
    borderColor: 'divider',
    fontVariantNumeric: 'tabular-nums' as const,
    whiteSpace: 'nowrap' as const,
  }

  return (
    <SectionCard title="Performance summary" stretch>
      <Box sx={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <Box component="th" sx={{ ...thSx, textAlign: 'left', minWidth: 150 }}>Measure</Box>
              {models.map((model) => (
                <Box component="th" key={model.id} sx={{ ...thSx, textAlign: 'right', minWidth: 170 }}>
                  {modelLabel(model)}
                </Box>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricRows.map((row) => (
              <Box component="tr" key={row.label}>
                <Box component="td" sx={{ ...tdSx, color: '#0f172a' }}>{row.label}</Box>
                {models.map((model) => (
                  <Box component="td" key={model.id} sx={{ ...tdSx, textAlign: 'right' }}>
                    {row.getValue(model)}
                  </Box>
                ))}
              </Box>
            ))}
          </tbody>
        </table>
      </Box>
    </SectionCard>
  )
}

function RocComparison({ models }: { models: ModelDetail[] }) {
  const fprScale = useMemo(
    () => Array.from({ length: 101 }, (_, index) => Number((index / 100).toFixed(2))),
    [],
  )

  const series = useMemo(() => {
    return models.map((model, index) => {
      const points = toRocPoints(model.rocCurve)

      return {
        data: fprScale.map((fpr) => interpolateTpr(points, fpr)),
        label: `${modelLabel(model)} (${fmt3(model.auc)})`,
        color: chartColors[index % chartColors.length],
        showMark: false,
        connectNulls: true,
      }
    })
  }, [fprScale, models])

  return (
    <SectionCard title="ROC curves" stretch>
      <LineChart
        height={360}
        xAxis={[{ data: fprScale, label: 'False Positive Rate', min: 0, max: 1 }]}
        yAxis={[{ min: 0, max: 1, label: 'True Positive Rate' }]}
        series={[
          {
            data: fprScale,
            label: 'Chance',
            color: '#cbd5e1',
            showMark: false,
          },
          ...series,
        ]}
        margin={{ top: 16, right: 16, bottom: 28, left: 52 }}
      />
    </SectionCard>
  )
}

function ConfusionCell({ label, value, total, good }: { label: string; value: number; total: number; good: boolean }) {
  return (
    <Box sx={{ bgcolor: good ? 'success.50' : 'error.50', borderRadius: 2, px: 1.25, py: 1.25 }}>
      <Typography variant="h6" sx={{ fontWeight: 700, color: good ? 'success.dark' : 'error.dark', lineHeight: 1 }}>
        {value.toLocaleString()}
      </Typography>
      <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: good ? 'success.dark' : 'error.dark' }}>
        {label}
      </Typography>
      <Typography variant="caption" sx={{ color: good ? 'success.dark' : 'error.dark' }}>
        {total > 0 ? pct(value / total) : '0.0%'}
      </Typography>
    </Box>
  )
}

function ConfusionMatrixComparison({ models }: { models: ModelDetail[] }) {
  const columns = models.length >= 5
    ? { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))', xl: 'repeat(3, minmax(0, 1fr))' }
    : { xs: '1fr', lg: 'repeat(2, minmax(0, 1fr))' }

  return (
    <SectionCard title="Confusion matrices" stretch>
      <Box sx={{ display: 'grid', gridTemplateColumns: columns, gridAutoRows: '1fr', gap: 2, height: '100%' }}>
        {models.map((model) => {
          const cm = model.confusionMatrix
          const total = confusionTotal(cm)

          return (
            <Box key={model.id} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <Box sx={{ px: 1.5, py: 1.25, borderBottom: '1px solid', borderColor: 'divider' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 400 }}>{modelLabel(model)}</Typography>
                <Typography variant="caption" color="text.secondary">{total.toLocaleString()} evaluated records</Typography>
              </Box>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gridAutoRows: '1fr', gap: 1, p: 1.25, flex: 1 }}>
                <ConfusionCell label="True negative" value={cm.tn} total={total} good />
                <ConfusionCell label="False positive" value={cm.fp} total={total} good={false} />
                <ConfusionCell label="False negative" value={cm.fn} total={total} good={false} />
                <ConfusionCell label="True positive" value={cm.tp} total={total} good />
              </Box>
            </Box>
          )
        })}
      </Box>
    </SectionCard>
  )
}

function FeatureComparison({ models }: { models: ModelDetail[] }) {
  const [sortKey, setSortKey] = useState('feature')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  const importanceByModel = useMemo(() => {
    return new Map(models.map((model) => [
      model.id,
      new Map(model.featureImportances.map((item) => [item.feature, item.importance])),
    ]))
  }, [models])

  const features = useMemo(() => {
    const totals = new Map<string, number>()

    models.forEach((model) => {
      model.featureImportances.forEach((item) => {
        totals.set(item.feature, (totals.get(item.feature) ?? 0) + item.importance)
      })
    })

    const valueFor = (feature: string) => {
      return importanceByModel.get(sortKey)?.get(feature) ?? 0
    }

    return [...totals.keys()].sort((left, right) => {
      if (sortKey === 'feature') {
        const diff = left.localeCompare(right)
        return sortDirection === 'asc' ? diff : -diff
      }

      const diff = valueFor(left) - valueFor(right)
      if (diff !== 0) return sortDirection === 'asc' ? diff : -diff
      return left.localeCompare(right)
    })
  }, [importanceByModel, models, sortDirection, sortKey])

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDirection((direction) => (direction === 'desc' ? 'asc' : 'desc'))
      return
    }

    setSortKey(key)
    setSortDirection(key === 'feature' ? 'asc' : 'desc')
  }

  return (
    <SectionCard title="Feature importances" stretch>
      <Box sx={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <Box component="th" sx={{ py: 0.75, px: 1.25, borderBottom: '1px solid', borderColor: 'divider', textAlign: 'left', fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                <TableSortLabel
                  active={sortKey === 'feature'}
                  direction={sortKey === 'feature' ? sortDirection : 'asc'}
                  onClick={() => handleSort('feature')}
                >
                  Feature
                </TableSortLabel>
              </Box>
              {models.map((model) => (
                <Box component="th" key={model.id} sx={{ py: 0.75, px: 1.25, borderBottom: '1px solid', borderColor: 'divider', textAlign: 'right', fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', whiteSpace: 'nowrap' }}>
                  <TableSortLabel
                    active={sortKey === model.id}
                    direction={sortKey === model.id ? sortDirection : 'desc'}
                    onClick={() => handleSort(model.id)}
                  >
                    {modelLabel(model)}
                  </TableSortLabel>
                </Box>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((feature) => (
              <Box component="tr" key={feature}>
                <Box component="td" sx={{ py: 0.75, px: 1.25, borderBottom: '1px solid', borderColor: 'divider', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                  {feature}
                </Box>
                {models.map((model) => {
                  const value = importanceByModel.get(model.id)?.get(feature) ?? 0
                  return (
                    <Box component="td" key={model.id} sx={{ py: 0.75, px: 1.25, borderBottom: '1px solid', borderColor: 'divider', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                      {(value * 100).toFixed(1)}%
                    </Box>
                  )
                })}
              </Box>
            ))}
          </tbody>
        </table>
      </Box>
    </SectionCard>
  )
}

function ModelCompare() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const ids = useMemo(
    () => [...new Set((searchParams.get('ids') ?? '').split(',').map((id) => id.trim()).filter(Boolean))],
    [searchParams],
  )

  const queries = useQueries({
    queries: ids.map((id) => ({
      queryKey: ['model', id],
      queryFn: () => fetchModelDetail(id),
      staleTime: 30_000,
      enabled: ids.length >= 2,
    })),
  })

  const loading = queries.some((query) => query.isLoading || query.isFetching)
  const error = queries.some((query) => query.isError)
  const models = queries.map((query) => query.data).filter((model): model is ModelDetail => !!model)
  const stackedDiagnostics = models.length >= 5

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Box>
          <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/models')} size="small" sx={{ mb: 1.5, color: 'text.secondary' }}>
            All models
          </Button>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            Compare models
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {ids.length >= 2 ? `${ids.length} selected models` : 'Select two or more models from the model list.'}
          </Typography>
        </Box>

        {ids.length < 2 ? (
          <Alert severity="info">Choose at least two models to compare.</Alert>
        ) : error ? (
          <Alert severity="error">Could not load one or more selected models.</Alert>
        ) : loading && models.length === 0 ? (
          <Typography variant="body2" color="text.secondary">Loading comparison data...</Typography>
        ) : (
          <>
            <PerformanceTable models={models} />
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: stackedDiagnostics
                  ? '1fr'
                  : { xs: '1fr', xl: 'minmax(0, 1.1fr) minmax(420px, 0.9fr)' },
                gap: 2.5,
                alignItems: 'stretch',
              }}
            >
              <RocComparison models={models} />
              <ConfusionMatrixComparison models={models} />
            </Box>
            <FeatureComparison models={models} />
          </>
        )}
      </Box>
    </main>
  )
}

export default ModelCompare
