import { useEffect, useMemo, useState } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Select from '@mui/material/Select'
import Typography from '@mui/material/Typography'
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh'
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents'
import HistoryIcon from '@mui/icons-material/History'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import VisibilityIcon from '@mui/icons-material/Visibility'
import { DataGrid, type GridColDef, type GridRenderCellParams, type GridRowParams } from '@mui/x-data-grid'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link as RouterLink, useNavigate } from 'react-router-dom'
import { fetchJson } from '../api'
import {
  AUTOMATIC_METRIC_LABELS,
  automaticResultRows,
  fetchTrainingJob,
  paramLabel,
  startAutomaticTrainingJob,
  type AutomaticResultRow,
} from '../automaticFineTuneData'
import {
  ALGORITHM_LABELS,
  BALANCING_METHOD_LABELS,
  FEATURE_SET_LABELS,
  UNCERTAINTY_VARIANT_LABELS,
} from '../modelMetadata'
import { modelLabel, pct } from '../modelData'
import type { ModelRow } from './ModelComparison'

async function fetchModels(): Promise<ModelRow[]> {
  return fetchJson<ModelRow[]>('/api/models')
}

function AutomaticFineTune() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [selectedModelId, setSelectedModelId] = useState('')
  const [jobId, setJobId] = useState('')
  const [selectedRunId, setSelectedRunId] = useState('')

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 30_000,
  })

  const normalModels = useMemo(
    () => (modelsQuery.data ?? []).filter((model) => !model.isTuned),
    [modelsQuery.data],
  )
  const selectedModel = normalModels.find((model) => model.id === selectedModelId) ?? normalModels[0] ?? null

  const startMutation = useMutation({
    mutationFn: startAutomaticTrainingJob,
    onSuccess: (job) => {
      setJobId(job.id)
      setSelectedRunId('')
    },
  })

  const jobQuery = useQuery({
    queryKey: ['automatic-fine-tuning-job', jobId],
    queryFn: () => fetchTrainingJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const job = query.state.data
      return job?.status === 'queued' || job?.status === 'running' ? 2000 : false
    },
  })

  const job = jobQuery.data ?? (startMutation.data?.id === jobId ? startMutation.data : null)
  const active = startMutation.isPending || job?.status === 'queued' || job?.status === 'running'
  const rows = useMemo<AutomaticResultRow[]>(() => {
    const results = job?.status === 'succeeded' ? job.result?.models ?? [] : []
    return automaticResultRows(results)
  }, [job?.result?.models, job?.status])
  const bestRun = rows[0] ?? null
  const selectedRun = rows.find((row) => row.id === selectedRunId) ?? bestRun

  useEffect(() => {
    if (job?.status !== 'succeeded') return
    queryClient.invalidateQueries({ queryKey: ['models'] })
    if (!selectedRunId && rows[0]) {
      setSelectedRunId(rows[0].id)
    }
  }, [job?.status, queryClient, rows, selectedRunId])

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

  const canStart = Boolean(selectedModel) && !active

  const handleStart = () => {
    if (!selectedModel || !canStart) return
    startMutation.mutate({ modelId: selectedModel.id })
  }

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, overflow: 'hidden' }}>
          <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
              <AutoFixHighIcon color="primary" />
              <Box>
                <Typography variant="h5" sx={{ fontSize: '1.35rem', fontWeight: 800 }}>Automatic fine-tuning</Typography>
                <Typography variant="body2" color="text.secondary">Select a normal model and run the full logical parameter grid.</Typography>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Button
                component={RouterLink}
                to="/fine-tune/automatic/runs"
                variant="outlined"
                startIcon={<HistoryIcon />}
                sx={{ minWidth: 118 }}
              >
                History
              </Button>
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                disabled={!canStart}
                onClick={handleStart}
                sx={{ minWidth: 180 }}
              >
                Start fine-tuning
              </Button>
            </Box>
          </Box>

          {active && <LinearProgress />}

          <Box sx={{ p: 2.5, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '1.1fr 0.9fr' }, gap: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <FormControl size="small" fullWidth disabled={modelsQuery.isLoading || active}>
                <InputLabel id="automatic-model-label">Normal model</InputLabel>
                <Select
                  labelId="automatic-model-label"
                  label="Normal model"
                  value={selectedModel?.id ?? ''}
                  renderValue={(value) => {
                    const model = normalModels.find((item) => item.id === value)
                    return model ? modelLabel(model) : ''
                  }}
                  onChange={(event) => {
                    setSelectedModelId(event.target.value)
                    setJobId('')
                    setSelectedRunId('')
                    startMutation.reset()
                  }}
                >
                  {normalModels.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, width: '100%' }}>
                        <Chip size="small" variant="outlined" label="Normal" sx={{ borderRadius: 1, minWidth: 68 }} />
                        <Typography variant="body2" noWrap>{modelLabel(model)}</Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedModel ? (
                <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 2 }}>
                  <Typography variant="caption" color="text.secondary">Selected baseline</Typography>
                  <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800, mt: 0.25 }}>{modelLabel(selectedModel)}</Typography>
                  <Box sx={{ mt: 1.5, display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                    <Chip size="small" variant="outlined" label={ALGORITHM_LABELS[selectedModel.algorithm]} />
                    <Chip size="small" variant="outlined" label={FEATURE_SET_LABELS[selectedModel.featureSet]} />
                    <Chip size="small" variant="outlined" label={UNCERTAINTY_VARIANT_LABELS[selectedModel.uncertaintyVariant]} />
                    <Chip size="small" variant="outlined" label={BALANCING_METHOD_LABELS[selectedModel.balancingMethod]} />
                  </Box>
                </Paper>
              ) : (
                <Alert severity="warning">No normal models are available.</Alert>
              )}
            </Box>

            <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Run status</Typography>
              <Box sx={{ mt: 1.5, display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 1 }}>
                <Box>
                  <Typography variant="caption" color="text.secondary">Combinations</Typography>
                  <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>
                    {job?.result?.total ?? job?.request.models?.length ?? '-'}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">Trained</Typography>
                  <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{job?.result?.trained ?? '-'}</Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">Best score</Typography>
                  <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{bestRun ? bestRun.score.toFixed(3) : '-'}</Typography>
                </Box>
              </Box>
              <Divider sx={{ my: 1.5 }} />
              <Typography variant="body2" color="text.secondary">
                {job?.message ?? 'Ready'}
              </Typography>
            </Paper>
          </Box>
        </Paper>

        {modelsQuery.isError && <Alert severity="error">Could not load models from the backend.</Alert>}
        {startMutation.isError && <Alert severity="error">{startMutation.error.message}</Alert>}
        {job?.status === 'failed' && <Alert severity="error">{job.error || job.message}</Alert>}
        {(job?.status === 'queued' || job?.status === 'running') && <Alert severity="info">{job.message}</Alert>}

        {job?.status === 'succeeded' && (
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: 'minmax(0, 1.7fr) minmax(320px, 0.8fr)' }, gap: 2 }}>
            <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, overflow: 'hidden' }}>
              <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap' }}>
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Automatic results</Typography>
                  <Typography variant="caption" color="text.secondary">Ranked with the model comparison score.</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                  {bestRun && <Chip color="primary" icon={<EmojiEventsIcon />} label={`Best: ${bestRun.score.toFixed(3)}`} sx={{ borderRadius: 1 }} />}
                  {job?.id && (
                    <Button
                      component={RouterLink}
                      to={`/fine-tune/automatic/runs/${job.id}`}
                      size="small"
                      variant="outlined"
                      startIcon={<HistoryIcon />}
                    >
                      Saved run
                    </Button>
                  )}
                </Box>
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
                  minHeight: 460,
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
                  <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Selected run</Typography>
                  <Typography variant="caption" color="text.secondary">{selectedRun ? `Rank ${selectedRun.rank}` : 'No run selected'}</Typography>
                </Box>
                {selectedRun && (
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<VisibilityIcon />}
                    onClick={() => navigate(`/models/${selectedRun.modelId}`)}
                  >
                    View model
                  </Button>
                )}
              </Box>

              {selectedRun ? (
                <>
                  <Box sx={{ mt: 2, display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 1 }}>
                    {AUTOMATIC_METRIC_LABELS.map((metric) => (
                      <Box key={metric.key} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1.5, p: 1.25 }}>
                        <Typography variant="caption" color="text.secondary">{metric.label}</Typography>
                        <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>
                          {metric.format(selectedRun.metrics[metric.key])}
                        </Typography>
                      </Box>
                    ))}
                    <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1.5, p: 1.25 }}>
                      <Typography variant="caption" color="text.secondary">Score</Typography>
                      <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{selectedRun.score.toFixed(3)}</Typography>
                    </Box>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>Parameters</Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: '1fr', gap: 1 }}>
                    {Object.entries(selectedRun.hyperparameters).map(([key, value]) => (
                      <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, borderBottom: '1px solid', borderColor: 'divider', pb: 0.75 }}>
                        <Typography variant="body2" color="text.secondary">{paramLabel(key)}</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 800 }}>{value}</Typography>
                      </Box>
                    ))}
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, borderBottom: '1px solid', borderColor: 'divider', pb: 0.75 }}>
                      <Typography variant="body2" color="text.secondary">Classification threshold</Typography>
                      <Typography variant="body2" sx={{ fontWeight: 800 }}>{selectedRun.classificationThreshold.toFixed(2)}</Typography>
                    </Box>
                  </Box>
                </>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>Select a trained run to inspect it.</Typography>
              )}
            </Paper>
          </Box>
        )}
      </Box>
    </main>
  )
}

export default AutomaticFineTune
