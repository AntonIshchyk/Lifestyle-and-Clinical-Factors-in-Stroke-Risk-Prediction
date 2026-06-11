import { useEffect, useMemo, useState } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import LinearProgress from '@mui/material/LinearProgress'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import DownloadIcon from '@mui/icons-material/Download'
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents'
import HistoryIcon from '@mui/icons-material/History'
import VisibilityIcon from '@mui/icons-material/Visibility'
import { DataGrid, type GridColDef, type GridRenderCellParams, type GridRowParams } from '@mui/x-data-grid'
import { useQuery } from '@tanstack/react-query'
import { Link as RouterLink, useNavigate, useParams } from 'react-router-dom'
import {
  AUTOMATIC_METRIC_LABELS,
  automaticResultRows,
  automaticTrainingRunExportUrl,
  automaticTrainingRunsExportUrl,
  fetchAutomaticTrainingRun,
  fetchAutomaticTrainingRuns,
  paramLabel,
  type AutomaticResultRow,
  type AutomaticTrainingRunSummary,
} from '../automaticFineTuneData'
import { pct } from '../modelData'

function formatRunTime(value: string | null) {
  if (!value) return '-'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function statusColor(status: AutomaticTrainingRunSummary['status']) {
  if (status === 'succeeded') return 'success'
  if (status === 'failed') return 'error'
  if (status === 'running') return 'primary'
  return 'default'
}

function AutomaticFineTuneRuns() {
  const navigate = useNavigate()
  const { id } = useParams()
  const [selectedRunId, setSelectedRunId] = useState('')
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([])

  const runsQuery = useQuery({
    queryKey: ['automatic-fine-tuning-runs'],
    queryFn: fetchAutomaticTrainingRuns,
    refetchInterval: (query) => {
      const hasActiveRun = query.state.data?.some((run) => run.status === 'queued' || run.status === 'running')
      return hasActiveRun ? 3000 : false
    },
  })

  const runs = runsQuery.data ?? []
  const activeRunId = id ?? ''
  const allRunIds = useMemo(() => runs.map((savedRun) => savedRun.id), [runs])
  const selectedRunIdSet = useMemo(() => new Set(selectedRunIds), [selectedRunIds])
  const selectedRunsExportUrl = selectedRunIds.length ? automaticTrainingRunsExportUrl(selectedRunIds) : ''

  useEffect(() => {
    if (activeRunId || !runs[0]) return
    navigate(`/fine-tune/automatic/runs/${runs[0].id}`, { replace: true })
  }, [activeRunId, navigate, runs])

  const runQuery = useQuery({
    queryKey: ['automatic-fine-tuning-run', activeRunId],
    queryFn: () => fetchAutomaticTrainingRun(activeRunId),
    enabled: Boolean(activeRunId),
    refetchInterval: (query) => {
      const run = query.state.data
      return run?.status === 'queued' || run?.status === 'running' ? 2000 : false
    },
  })

  const run = runQuery.data ?? null
  const rows = useMemo<AutomaticResultRow[]>(
    () => automaticResultRows(run?.status === 'succeeded' ? run.result?.models ?? [] : []),
    [run?.result?.models, run?.status],
  )
  const exportUrl = run ? automaticTrainingRunExportUrl(run.id) : ''
  const bestRun = rows[0] ?? null
  const selectedResult = rows.find((row) => row.id === selectedRunId) ?? bestRun

  useEffect(() => {
    setSelectedRunId('')
  }, [activeRunId])

  useEffect(() => {
    setSelectedRunIds((current) => {
      const knownRunIds = new Set(allRunIds)
      const filtered = current.filter((runId) => knownRunIds.has(runId))
      return filtered.length === current.length ? current : filtered
    })
  }, [allRunIds])

  function toggleRunSelection(runId: string) {
    setSelectedRunIds((current) => (
      current.includes(runId)
        ? current.filter((selectedId) => selectedId !== runId)
        : [...current, runId]
    ))
  }

  function selectAllRuns() {
    setSelectedRunIds(allRunIds)
  }

  function clearSelectedRuns() {
    setSelectedRunIds([])
  }

  const columns = useMemo<GridColDef<AutomaticResultRow>[]>(() => [
    {
      field: 'rank',
      headerName: '#',
      width: 70,
      renderCell: ({ row }) => (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          <Typography variant="body2" sx={{ fontWeight: 800 }}>{row.rank}</Typography>
          {row.id === bestRun?.id && <EmojiEventsIcon color="warning" fontSize="small" />}
        </Box>
      ),
    },
    {
      field: 'score',
      headerName: 'Score',
      flex: 0.8,
      minWidth: 110,
      valueFormatter: (value) => (value as number).toFixed(3),
    },
    {
      field: 'auc',
      headerName: 'AUC-ROC',
      flex: 0.8,
      minWidth: 110,
      valueGetter: (_, row) => row.metrics.auc,
      valueFormatter: (value) => (value as number).toFixed(3),
    },
    {
      field: 'accuracy',
      headerName: 'Accuracy',
      flex: 0.8,
      minWidth: 110,
      valueGetter: (_, row) => row.metrics.accuracy,
      valueFormatter: (value) => pct(value as number),
    },
    {
      field: 'f1',
      headerName: 'F1',
      flex: 0.7,
      minWidth: 90,
      valueGetter: (_, row) => row.metrics.f1,
      valueFormatter: (value) => (value as number).toFixed(3),
    },
    {
      field: 'precision',
      headerName: 'Precision',
      flex: 0.8,
      minWidth: 110,
      valueGetter: (_, row) => row.metrics.precision,
      valueFormatter: (value) => pct(value as number),
    },
    {
      field: 'recall',
      headerName: 'Recall',
      flex: 0.8,
      minWidth: 110,
      valueGetter: (_, row) => row.metrics.recall,
      valueFormatter: (value) => pct(value as number),
    },
    {
      field: 'classificationThreshold',
      headerName: 'Threshold',
      flex: 0.8,
      minWidth: 115,
      valueFormatter: (value) => (value as number).toFixed(2),
    },
    {
      field: 'hyperparameters',
      headerName: 'Parameters',
      flex: 1.4,
      minWidth: 220,
      sortable: false,
      renderCell: ({ row }: GridRenderCellParams<AutomaticResultRow>) => (
        <Typography variant="body2" noWrap>
          {Object.entries(row.hyperparameters).map(([key, value]) => `${key}: ${value}`).join(', ')}
        </Typography>
      ),
    },
  ], [bestRun?.id])

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, overflow: 'hidden' }}>
          <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
              <HistoryIcon color="primary" />
              <Box>
                <Typography variant="h5" sx={{ fontSize: '1.35rem', fontWeight: 800 }}>Automatic run history</Typography>
                <Typography variant="body2" color="text.secondary">Saved automatic fine-tuning runs and their full ranked parameter lists.</Typography>
              </Box>
            </Box>
            <Button component={RouterLink} to="/fine-tune/automatic" variant="outlined" startIcon={<ArrowBackIcon />}>
              Automatic
            </Button>
          </Box>

          {(runsQuery.isFetching || runQuery.isFetching) && <LinearProgress />}

          <Box sx={{ p: 2.5, display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '320px minmax(0, 1fr)' }, gap: 2 }}>
            <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, overflow: 'hidden', alignSelf: 'start' }}>
              <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 1 }}>
                  <Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Runs</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {runs.length} saved - {selectedRunIds.length} selected
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    <Button size="small" variant="text" onClick={selectAllRuns} disabled={!runs.length || selectedRunIds.length === runs.length}>
                      All
                    </Button>
                    <Button size="small" variant="text" onClick={clearSelectedRuns} disabled={!selectedRunIds.length}>
                      Clear
                    </Button>
                  </Box>
                </Box>
                <Button
                  component="a"
                  href={selectedRunsExportUrl || undefined}
                  download
                  size="small"
                  variant="contained"
                  startIcon={<DownloadIcon />}
                  disabled={!selectedRunIds.length}
                  sx={{ alignSelf: 'stretch' }}
                >
                  Export selected to Excel
                </Button>
              </Box>
              <Box sx={{ maxHeight: { xs: 360, xl: 'calc(100vh - 245px)' }, overflow: 'auto' }}>
                {runs.map((savedRun) => {
                  const active = savedRun.id === activeRunId
                  const checked = selectedRunIdSet.has(savedRun.id)
                  return (
                    <Box
                      key={savedRun.id}
                      sx={{
                        width: '100%',
                        borderBottom: '1px solid',
                        borderColor: 'divider',
                        bgcolor: active ? 'primary.50' : 'background.paper',
                        color: 'text.primary',
                        display: 'flex',
                        alignItems: 'flex-start',
                        '&:hover': { bgcolor: active ? 'primary.50' : 'grey.50' },
                      }}
                    >
                      <Checkbox
                        checked={checked}
                        onChange={() => toggleRunSelection(savedRun.id)}
                        slotProps={{ input: { 'aria-label': `Select automatic run ${savedRun.baseModelId || savedRun.id}` } }}
                        sx={{ mt: 0.5, ml: 0.5 }}
                      />
                      <Box
                        component="button"
                        type="button"
                        onClick={() => navigate(`/fine-tune/automatic/runs/${savedRun.id}`)}
                        sx={{
                          minWidth: 0,
                          flex: 1,
                          border: 0,
                          bgcolor: 'transparent',
                          color: 'text.primary',
                          cursor: 'pointer',
                          display: 'block',
                          p: 1.5,
                          pl: 0.5,
                          textAlign: 'left',
                        }}
                      >
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" sx={{ fontWeight: 800 }} noWrap>
                            {savedRun.baseModelId || savedRun.id}
                          </Typography>
                          <Chip size="small" color={statusColor(savedRun.status)} label={savedRun.status} sx={{ borderRadius: 1, textTransform: 'capitalize' }} />
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                          {formatRunTime(savedRun.createdAt)}
                        </Typography>
                        <Box sx={{ mt: 1, display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
                          <Chip size="small" variant="outlined" label={`${savedRun.total ?? '-'} combos`} sx={{ borderRadius: 1 }} />
                          <Chip size="small" variant="outlined" label={`Best ${savedRun.bestScore === null ? '-' : savedRun.bestScore.toFixed(3)}`} sx={{ borderRadius: 1 }} />
                        </Box>
                      </Box>
                    </Box>
                  )
                })}
                {!runs.length && (
                  <Box sx={{ p: 2 }}>
                    <Typography variant="body2" color="text.secondary">No automatic runs saved yet.</Typography>
                  </Box>
                )}
              </Box>
            </Paper>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {runsQuery.isError && <Alert severity="error">Could not load automatic run history.</Alert>}
              {runQuery.isError && <Alert severity="error">Could not load the selected automatic run.</Alert>}
              {run?.status === 'failed' && <Alert severity="error">{run.error || run.message}</Alert>}
              {(run?.status === 'queued' || run?.status === 'running') && <Alert severity="info">{run.message}</Alert>}

              <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, minWidth: 0 }}>
                    <AutoFixHighIcon color="primary" />
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
                        {run?.request.baseModelId ?? 'Select a run'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {run ? `${formatRunTime(run.createdAt)} - ${run.message}` : 'Choose a saved run from the list.'}
                      </Typography>
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                    {run?.result?.models.length ? (
                      <Button
                        component="a"
                        href={exportUrl}
                        download
                        size="small"
                        variant="outlined"
                        startIcon={<DownloadIcon />}
                      >
                        Excel
                      </Button>
                    ) : null}
                    {run && <Chip color={statusColor(run.status)} label={run.status} sx={{ borderRadius: 1, textTransform: 'capitalize' }} />}
                  </Box>
                </Box>
                <Divider sx={{ my: 1.5 }} />
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', md: 'repeat(4, minmax(0, 1fr))' }, gap: 1 }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Combinations</Typography>
                    <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{run?.result?.total ?? run?.request.models?.length ?? '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Trained</Typography>
                    <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{run?.result?.trained ?? '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Reused</Typography>
                    <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{run?.result?.reused ?? '-'}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">Best score</Typography>
                    <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{bestRun ? bestRun.score.toFixed(3) : '-'}</Typography>
                  </Box>
                </Box>
              </Paper>

              {run?.status === 'succeeded' && (
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: 'minmax(0, 1.7fr) minmax(320px, 0.8fr)' }, gap: 2 }}>
                  <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, overflow: 'hidden' }}>
                    <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap' }}>
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Automatic results</Typography>
                        <Typography variant="caption" color="text.secondary">Full ranked list from this saved run.</Typography>
                      </Box>
                      {bestRun && <Chip color="primary" icon={<EmojiEventsIcon />} label={`Best: ${bestRun.score.toFixed(3)}`} sx={{ borderRadius: 1 }} />}
                    </Box>
                    <DataGrid
                      rows={rows}
                      columns={columns}
                      density="compact"
                      hideFooter
                      disableRowSelectionOnClick
                      onRowClick={(params: GridRowParams<AutomaticResultRow>) => setSelectedRunId(params.row.id)}
                      getRowClassName={(params) => (params.row.id === bestRun?.id ? 'best-run' : '')}
                      sx={{
                        border: 0,
                        minHeight: 540,
                        cursor: 'pointer',
                        '& .MuiDataGrid-columnHeaders': { bgcolor: 'grey.50' },
                        '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 800 },
                        '& .best-run': {
                          bgcolor: 'rgba(255, 193, 7, 0.16)',
                          '&:hover': { bgcolor: 'rgba(255, 193, 7, 0.24)' },
                        },
                      }}
                    />
                  </Paper>

                  <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, p: 2, alignSelf: 'start' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap' }}>
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Selected result</Typography>
                        <Typography variant="caption" color="text.secondary">{selectedResult ? `Rank ${selectedResult.rank}` : 'No result selected'}</Typography>
                      </Box>
                      {selectedResult && (
                        <Button
                          component={RouterLink}
                          to={`/models/${selectedResult.modelId}`}
                          size="small"
                          variant="outlined"
                          startIcon={<VisibilityIcon />}
                        >
                          View model
                        </Button>
                      )}
                    </Box>

                    {selectedResult ? (
                      <>
                        <Box sx={{ mt: 2, display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 1 }}>
                          {AUTOMATIC_METRIC_LABELS.map((metric) => (
                            <Box key={metric.key} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1.5, p: 1.25 }}>
                              <Typography variant="caption" color="text.secondary">{metric.label}</Typography>
                              <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>
                                {metric.format(selectedResult.metrics[metric.key])}
                              </Typography>
                            </Box>
                          ))}
                          <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1.5, p: 1.25 }}>
                            <Typography variant="caption" color="text.secondary">Score</Typography>
                            <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{selectedResult.score.toFixed(3)}</Typography>
                          </Box>
                        </Box>

                        <Divider sx={{ my: 2 }} />

                        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>Parameters</Typography>
                        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr', gap: 1 }}>
                          {Object.entries(selectedResult.hyperparameters).map(([key, value]) => (
                            <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, borderBottom: '1px solid', borderColor: 'divider', pb: 0.75 }}>
                              <Typography variant="body2" color="text.secondary">{paramLabel(key)}</Typography>
                              <Typography variant="body2" sx={{ fontWeight: 800 }}>{value}</Typography>
                            </Box>
                          ))}
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, borderBottom: '1px solid', borderColor: 'divider', pb: 0.75 }}>
                            <Typography variant="body2" color="text.secondary">Classification threshold</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 800 }}>{selectedResult.classificationThreshold.toFixed(2)}</Typography>
                          </Box>
                        </Box>
                      </>
                    ) : (
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>Select a result to inspect it.</Typography>
                    )}
                  </Paper>
                </Box>
              )}
            </Box>
          </Box>
        </Paper>
      </Box>
    </main>
  )
}

export default AutomaticFineTuneRuns
