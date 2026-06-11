import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import Chip from '@mui/material/Chip'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import FormControl from '@mui/material/FormControl'
import FormControlLabel from '@mui/material/FormControlLabel'
import IconButton from '@mui/material/IconButton'
import InputLabel from '@mui/material/InputLabel'
import LinearProgress from '@mui/material/LinearProgress'
import ListItemText from '@mui/material/ListItemText'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Select from '@mui/material/Select'
import Switch from '@mui/material/Switch'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import DeleteOutlinedIcon from '@mui/icons-material/DeleteOutlined'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import ScienceIcon from '@mui/icons-material/Science'
import { DataGrid, type GridColDef, type GridRenderCellParams, type GridRowParams, type GridRowSelectionModel } from '@mui/x-data-grid'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteJson, fetchJson, postJson } from '../api'
import {
  ALGORITHM_LABELS,
  BALANCING_METHOD_LABELS,
  FEATURE_SET_LABELS,
  UNCERTAINTY_VARIANT_LABELS,
  type Algorithm,
  type BalancingMethod,
  type FeatureSet,
  type UncertaintyVariant,
} from '../modelMetadata'
import { pct } from '../modelData'

export type ModelRow = {
  id: string
  algorithm: Algorithm
  featureSet: FeatureSet
  uncertaintyVariant: UncertaintyVariant
  balancingMethod: BalancingMethod
  auc: number
  accuracy: number
  f1: number
  precision: number
  recall: number
  classificationThreshold: number
  isTuned: boolean
}

type TrainingDatasetOption = {
  id: string
  label: string
  featureSet: FeatureSet
  uncertaintyVariant: UncertaintyVariant
}

type TrainingOptions = {
  algorithms: Array<{ id: Algorithm; label: string }>
  datasets: TrainingDatasetOption[]
  balancingMethods: Array<{ id: BalancingMethod }>
  defaults: TrainingRequest
}

type TrainingRequest = {
  algorithms: Algorithm[]
  datasetIds: string[]
  balancingMethods: BalancingMethod[]
  targetRatio: number
  forceRetrain: boolean
  useGpu: boolean
  models?: TrainingSpec[]
}

type TrainingSpec = {
  id?: string
  algorithm: Algorithm
  datasetId: string
  featureSet?: FeatureSet
  uncertaintyVariant?: UncertaintyVariant
  balancingMethod: BalancingMethod
}

type TrainingModelResult = {
  modelId: string
  reusedExistingModel: boolean
  metrics: {
    auc: number
    accuracy: number
    f1: number
    precision: number
    recall: number
    classificationThreshold: number
  }
}

type TrainingJob = {
  id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  message: string
  createdAt: string
  startedAt: string | null
  finishedAt: string | null
  request: TrainingRequest
  result: {
    models: TrainingModelResult[]
    total: number
    trained: number
    reused: number
  } | null
  error: string | null
}

type TrainingCoverage = {
  totalExpected: number
  availableCount: number
  missingCount: number
  available: TrainingSpec[]
  missing: TrainingSpec[]
  missingByDataset: Record<string, number>
  missingByAlgorithm: Partial<Record<Algorithm, number>>
  missingByBalancingMethod: Partial<Record<BalancingMethod, number>>
}

type ModelComparisonProps = {
  embedded?: boolean
  mode?: 'compare' | 'select'
  selectedModelId?: string
  onModelSelect?: (model: ModelRow) => void
}

const STROKE_SCORE_WEIGHTS = {
  auc: 0.35,
  f1: 0.3,
  recall: 0.25,
  precision: 0.1,
} as const

function strokeRiskScore(model: Pick<ModelRow, 'auc' | 'f1' | 'recall' | 'precision'>) {
  return (
    model.auc * STROKE_SCORE_WEIGHTS.auc +
    model.f1 * STROKE_SCORE_WEIGHTS.f1 +
    model.recall * STROKE_SCORE_WEIGHTS.recall +
    model.precision * STROKE_SCORE_WEIGHTS.precision
  )
}

async function fetchModels(): Promise<ModelRow[]> {
  return fetchJson<ModelRow[]>('/api/models')
}

type DeleteModelResult = {
  ok: boolean
  modelId: string
  fileDeleted: boolean
}

async function deleteModel(modelId: string): Promise<DeleteModelResult> {
  return deleteJson(`/api/models/${encodeURIComponent(modelId)}`)
}

async function deleteModels(modelIds: string[]): Promise<DeleteModelResult[]> {
  return Promise.all(modelIds.map(deleteModel))
}

async function fetchTrainingOptions(): Promise<TrainingOptions> {
  return fetchJson<TrainingOptions>('/api/training/options')
}

async function startTrainingJob(request: TrainingRequest): Promise<TrainingJob> {
  return postJson<TrainingJob>('/api/training/jobs', request)
}

async function fetchTrainingJob(jobId: string): Promise<TrainingJob> {
  return fetchJson<TrainingJob>(`/api/training/jobs/${jobId}`)
}

async function fetchTrainingCoverage(): Promise<TrainingCoverage> {
  return fetchJson<TrainingCoverage>('/api/training/coverage')
}

function useColumns({
  allowDelete = false,
  deletingModelId = '',
  onDelete,
}: {
  allowDelete?: boolean
  deletingModelId?: string
  onDelete?: (model: ModelRow) => void
} = {}): GridColDef[] {
  return useMemo<GridColDef[]>(() => [
    {
      field: 'algorithm',
      headerName: 'Algorithm',
      flex: 1.4,
      minWidth: 150,
      sortable: true,
      renderCell: ({ value }) => <Typography variant="body2">{ALGORITHM_LABELS[value as Algorithm]}</Typography>,
    },
    {
      field: 'featureSet',
      headerName: 'Feature set',
      flex: 1,
      minWidth: 120,
      sortable: true,
      renderCell: ({ value }) => <Typography variant="body2">{FEATURE_SET_LABELS[value as FeatureSet]}</Typography>,
    },
    {
      field: 'uncertaintyVariant',
      headerName: 'Uncertainty',
      flex: 1.2,
      minWidth: 180,
      sortable: true,
      renderCell: ({ value }) => <Typography variant="body2">{UNCERTAINTY_VARIANT_LABELS[value as UncertaintyVariant]}</Typography>,
    },
    {
      field: 'balancingMethod',
      headerName: 'Balance',
      flex: 1.15,
      minWidth: 155,
      sortable: true,
      renderCell: ({ value }) => (
        <Chip
          label={BALANCING_METHOD_LABELS[value as BalancingMethod]}
          size="small"
          variant="outlined"
          sx={{ borderRadius: 1.5, height: 24 }}
        />
      ),
    },
    {
      field: 'auc',
      headerName: 'AUC-ROC',
      flex: 1,
      minWidth: 110,
      sortable: true,
      align: 'left',
      headerAlign: 'left',
      valueFormatter: (value) => (value as number).toFixed(3),
    },
    {
      field: 'accuracy',
      headerName: 'Accuracy',
      flex: 1,
      minWidth: 100,
      sortable: true,
      align: 'left',
      headerAlign: 'left',
      valueFormatter: (value) => pct(value as number),
    },
    {
      field: 'f1',
      headerName: 'F1',
      flex: 0.8,
      minWidth: 90,
      sortable: true,
      align: 'left',
      headerAlign: 'left',
      valueFormatter: (value) => (value as number).toFixed(3),
    },
    {
      field: 'precision',
      headerName: 'Precision',
      flex: 1,
      minWidth: 100,
      sortable: true,
      align: 'left',
      headerAlign: 'left',
      valueFormatter: (value) => pct(value as number),
    },
    {
      field: 'recall',
      headerName: 'Recall',
      flex: 1,
      minWidth: 100,
      sortable: true,
      align: 'left',
      headerAlign: 'left',
      valueFormatter: (value) => pct(value as number),
    },
    ...(allowDelete ? [{
      field: 'actions',
      headerName: '',
      width: 64,
      sortable: false,
      filterable: false,
      disableColumnMenu: true,
      align: 'center' as const,
      headerAlign: 'center' as const,
      renderCell: ({ row }: GridRenderCellParams<ModelRow>) => {
        const model = row as ModelRow
        if (!model.isTuned) return null

        return (
          <Tooltip title="Delete fine-tuned model">
            <span>
              <IconButton
                size="small"
                color="error"
                disabled={deletingModelId === model.id}
                onClick={(event) => {
                  event.stopPropagation()
                  onDelete?.(model)
                }}
                aria-label={`Delete ${model.id}`}
              >
                <DeleteOutlinedIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
        )
      },
    }] : []),
  ], [allowDelete, deletingModelId, onDelete])
}

function ModelOverview({ models }: { models: ModelRow[] }) {
  const topModels = useMemo(() => {
    return models
      .map((model) => ({
        model,
        score: strokeRiskScore(model),
      }))
      .sort((left, right) => (
        right.score - left.score ||
        right.model.auc - left.model.auc ||
        right.model.recall - left.model.recall ||
        right.model.f1 - left.model.f1
      ))
      .slice(0, 3)
  }, [models])

  return (
    <Paper elevation={0} sx={{ borderRadius: 2, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
      <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Top 3 overall models</Typography>
          <Typography variant="caption" color="text.secondary">Ranked for stroke risk prediction</Typography>
        </Box>
        <Chip
          label="Score = 0.35 AUC + 0.30 F1 + 0.25 recall + 0.10 precision"
          size="small"
          variant="outlined"
          sx={{ borderRadius: 1, maxWidth: '100%', '& .MuiChip-label': { whiteSpace: 'normal' } }}
        />
      </Box>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(3, minmax(0, 1fr))' }, gap: 1.5, bgcolor: 'grey.50', p: 1.5 }}>
        {topModels.length > 0 ? topModels.map(({ model, score }, index) => (
          <Box key={model.id} sx={{ p: 2, bgcolor: 'background.paper', border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 1.5, mb: 1.5 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">Rank {index + 1}</Typography>
                <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 700, lineHeight: 1.25 }}>
                  {ALGORITHM_LABELS[model.algorithm]}
                </Typography>
              </Box>
              <Chip label={`Score ${score.toFixed(3)}`} size="small" color={index === 0 ? 'primary' : 'default'} sx={{ borderRadius: 1 }} />
            </Box>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mb: 1.75 }}>
              <Chip label={FEATURE_SET_LABELS[model.featureSet]} size="small" variant="outlined" sx={{ borderRadius: 1 }} />
              <Chip label={UNCERTAINTY_VARIANT_LABELS[model.uncertaintyVariant]} size="small" variant="outlined" sx={{ borderRadius: 1 }} />
              <Chip label={BALANCING_METHOD_LABELS[model.balancingMethod]} size="small" variant="outlined" sx={{ borderRadius: 1 }} />
            </Box>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 1.25 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">AUC-ROC</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>{model.auc.toFixed(3)}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">F1</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>{model.f1.toFixed(3)}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Recall</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>{pct(model.recall)}</Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Precision</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>{pct(model.precision)}</Typography>
              </Box>
            </Box>
          </Box>
        )) : (
          <Box sx={{ p: 2, bgcolor: 'background.paper', border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
            <Typography variant="body2" color="text.secondary">No trained models available yet.</Typography>
          </Box>
        )}
      </Box>
    </Paper>
  )
}

function TrainingPanel() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [jobId, setJobId] = useState('')
  const [form, setForm] = useState<TrainingRequest>({
    algorithms: [],
    datasetIds: [],
    balancingMethods: [],
    targetRatio: 1,
    forceRetrain: false,
    useGpu: true,
  })

  const optionsQuery = useQuery({
    queryKey: ['training-options'],
    queryFn: fetchTrainingOptions,
    staleTime: 60_000,
  })

  useEffect(() => {
    if (!optionsQuery.data || form.algorithms.length || form.datasetIds.length || form.balancingMethods.length) return
    setForm(optionsQuery.data.defaults)
  }, [form.algorithms.length, form.balancingMethods.length, form.datasetIds.length, optionsQuery.data])

  const startMutation = useMutation({
    mutationFn: startTrainingJob,
    onSuccess: (job) => setJobId(job.id),
  })

  const jobQuery = useQuery({
    queryKey: ['training-job', jobId],
    queryFn: () => fetchTrainingJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const job = query.state.data
      return job?.status === 'queued' || job?.status === 'running' ? 2000 : false
    },
  })

  const job = jobQuery.data ?? (startMutation.data?.id === jobId ? startMutation.data : null)
  const active = startMutation.isPending || job?.status === 'queued' || job?.status === 'running'
  const selectionCount = form.algorithms.length * form.datasetIds.length * form.balancingMethods.length
  const canStart = selectionCount > 0 && !active
  const selectedDatasets = optionsQuery.data?.datasets.filter((dataset) => form.datasetIds.includes(dataset.id)) ?? []
  const firstTrainedModelId = job?.result?.models[0]?.modelId

  useEffect(() => {
    if (job?.status !== 'succeeded') return
    queryClient.invalidateQueries({ queryKey: ['models'] })
    queryClient.invalidateQueries({ queryKey: ['training-coverage'] })
  }, [job?.status, queryClient])

  const handleStart = () => {
    if (!canStart) return
    startMutation.mutate({
      ...form,
      targetRatio: 1,
      useGpu: form.useGpu,
    })
  }

  const algorithmLabel = (algorithm: Algorithm) => ALGORITHM_LABELS[algorithm]
  const datasetLabel = (datasetId: string) => {
    const dataset = optionsQuery.data?.datasets.find((option) => option.id === datasetId)
    if (!dataset) return datasetId
    return `${FEATURE_SET_LABELS[dataset.featureSet]} / ${UNCERTAINTY_VARIANT_LABELS[dataset.uncertaintyVariant]}`
  }
  const balancingLabel = (method: BalancingMethod) => BALANCING_METHOD_LABELS[method]
  const multiValue = <T extends string>(value: T[] | string): T[] => (
    typeof value === 'string' ? value.split(',').filter(Boolean) as T[] : value
  )

  return (
    <Paper elevation={0} sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 2,
          flexWrap: 'wrap',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
          <ScienceIcon color="primary" fontSize="small" />
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Train a model</Typography>
        </Box>
        {selectedDatasets.length > 0 && (
          <Typography variant="caption" color="text.secondary">
            {selectionCount.toLocaleString()} model{selectionCount === 1 ? '' : 's'} selected
          </Typography>
        )}
      </Box>

      <Box sx={{ p: 2, display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1.35fr 1fr auto' }, gap: 1.5, alignItems: 'center' }}>
        <FormControl size="small" fullWidth disabled={optionsQuery.isLoading || active}>
          <InputLabel id="training-algorithm-label">Algorithms</InputLabel>
          <Select
            multiple
            labelId="training-algorithm-label"
            label="Algorithms"
            value={form.algorithms}
            renderValue={(selected) => selected.map(algorithmLabel).join(', ')}
            onChange={(event) => {
              const algorithms = multiValue<Algorithm>(event.target.value)
              setForm((current) => ({
                ...current,
                algorithms,
              }))
            }}
          >
            {(optionsQuery.data?.algorithms ?? []).map((algorithm) => (
              <MenuItem key={algorithm.id} value={algorithm.id}>
                <Checkbox checked={form.algorithms.includes(algorithm.id)} size="small" />
                <ListItemText primary={algorithm.label} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" fullWidth disabled={optionsQuery.isLoading || active}>
          <InputLabel id="training-dataset-label">Training data</InputLabel>
          <Select
            multiple
            labelId="training-dataset-label"
            label="Training data"
            value={form.datasetIds}
            renderValue={(selected) => selected.map(datasetLabel).join(', ')}
            onChange={(event) => setForm((current) => ({ ...current, datasetIds: multiValue<string>(event.target.value) }))}
          >
            {(optionsQuery.data?.datasets ?? []).map((dataset) => (
              <MenuItem key={dataset.id} value={dataset.id}>
                <Checkbox checked={form.datasetIds.includes(dataset.id)} size="small" />
                <ListItemText primary={`${FEATURE_SET_LABELS[dataset.featureSet]} / ${UNCERTAINTY_VARIANT_LABELS[dataset.uncertaintyVariant]}`} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" fullWidth disabled={optionsQuery.isLoading || active}>
          <InputLabel id="training-balance-label">Balancing</InputLabel>
          <Select
            multiple
            labelId="training-balance-label"
            label="Balancing"
            value={form.balancingMethods}
            renderValue={(selected) => selected.map(balancingLabel).join(', ')}
            onChange={(event) => setForm((current) => ({ ...current, balancingMethods: multiValue<BalancingMethod>(event.target.value) }))}
          >
            {(optionsQuery.data?.balancingMethods ?? []).map((method) => (
              <MenuItem key={method.id} value={method.id}>
                <Checkbox checked={form.balancingMethods.includes(method.id)} size="small" />
                <ListItemText primary={BALANCING_METHOD_LABELS[method.id]} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button
          variant="contained"
          startIcon={<PlayArrowIcon />}
          disabled={!canStart}
          onClick={handleStart}
          sx={{ minWidth: 132, height: 40 }}
        >
          Start ({selectionCount})
        </Button>
      </Box>

      <Box sx={{ px: 2, pb: 2, display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={form.forceRetrain}
              disabled={active}
              onChange={(event) => setForm((current) => ({ ...current, forceRetrain: event.target.checked }))}
            />
          }
          label={<Typography variant="body2">Retrain even if saved model exists</Typography>}
        />
        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={form.useGpu}
              disabled={active}
              onChange={(event) => setForm((current) => ({ ...current, useGpu: event.target.checked }))}
            />
          }
          label={<Typography variant="body2">Use GPU when available</Typography>}
        />
      </Box>

      {(active || job || startMutation.isError || optionsQuery.isError) && (
        <Box sx={{ px: 2, pb: 2 }}>
          {active && <LinearProgress sx={{ mb: 1.5, borderRadius: 1 }} />}
          {optionsQuery.isError && <Alert severity="error">Could not load training options from the backend.</Alert>}
          {startMutation.isError && <Alert severity="error">{startMutation.error.message}</Alert>}
          {job?.status === 'failed' && <Alert severity="error">{job.error || job.message}</Alert>}
          {(job?.status === 'queued' || job?.status === 'running') && (
            <Alert severity="info">{job.message}</Alert>
          )}
          {job?.status === 'succeeded' && job.result && (
            <Alert
              severity="success"
              action={
                firstTrainedModelId ? <Button color="inherit" size="small" onClick={() => navigate(`/models/${firstTrainedModelId}`)}>
                  View
                </Button> : undefined
              }
            >
              {job.message} {job.result.trained} trained, {job.result.reused} reused.
            </Alert>
          )}
        </Box>
      )}
    </Paper>
  )
}

function MissingStatisticsPanel() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [jobId, setJobId] = useState('')
  const [selectedMissingIds, setSelectedMissingIds] = useState<Set<string>>(new Set())

  const coverageQuery = useQuery({
    queryKey: ['training-coverage'],
    queryFn: fetchTrainingCoverage,
    staleTime: 30_000,
  })

  const startMutation = useMutation({
    mutationFn: startTrainingJob,
    onSuccess: (job) => setJobId(job.id),
  })

  const jobQuery = useQuery({
    queryKey: ['missing-training-job', jobId],
    queryFn: () => fetchTrainingJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const job = query.state.data
      return job?.status === 'queued' || job?.status === 'running' ? 2000 : false
    },
  })

  const coverage = coverageQuery.data
  const missing = coverage?.missing ?? []
  const job = jobQuery.data ?? (startMutation.data?.id === jobId ? startMutation.data : null)
  const active = startMutation.isPending || job?.status === 'queued' || job?.status === 'running'
  const firstTrainedModelId = job?.result?.models[0]?.modelId
  const completionPct = coverage && coverage.totalExpected > 0
    ? Math.round((coverage.availableCount / coverage.totalExpected) * 100)
    : 0
  const selectedMissing = missing.filter((spec) => spec.id && selectedMissingIds.has(spec.id))
  const allMissingSelected = missing.length > 0 && selectedMissing.length === missing.length
  const someMissingSelected = selectedMissing.length > 0 && selectedMissing.length < missing.length

  useEffect(() => {
    if (job?.status !== 'succeeded') return
    queryClient.invalidateQueries({ queryKey: ['models'] })
    queryClient.invalidateQueries({ queryKey: ['training-coverage'] })
    setSelectedMissingIds(new Set())
  }, [job?.status, queryClient])

  useEffect(() => {
    setSelectedMissingIds((current) => {
      const validIds = new Set(missing.map((spec) => spec.id).filter(Boolean) as string[])
      const next = new Set([...current].filter((id) => validIds.has(id)))
      return next.size === current.size ? current : next
    })
  }, [missing])

  const trainMissing = (models: TrainingSpec[]) => {
    if (!models.length || active) return
    startMutation.mutate({
      algorithms: [],
      datasetIds: [],
      balancingMethods: [],
      targetRatio: 1,
      forceRetrain: false,
      useGpu: true,
      models,
    })
  }

  const toggleMissing = (spec: TrainingSpec) => {
    if (!spec.id || active) return
    setSelectedMissingIds((current) => {
      const next = new Set(current)
      if (next.has(spec.id!)) {
        next.delete(spec.id!)
      } else {
        next.add(spec.id!)
      }
      return next
    })
  }

  const toggleAllMissing = () => {
    if (active) return
    setSelectedMissingIds((current) => {
      if (missing.length > 0 && current.size === missing.length) return new Set()
      return new Set(missing.map((spec) => spec.id).filter(Boolean) as string[])
    })
  }

  const specLabel = (spec: TrainingSpec) => {
    const feature = spec.featureSet ? FEATURE_SET_LABELS[spec.featureSet] : spec.datasetId
    const uncertainty = spec.uncertaintyVariant ? UNCERTAINTY_VARIANT_LABELS[spec.uncertaintyVariant] : ''
    return uncertainty ? `${feature} / ${uncertainty}` : feature
  }

  const topMissingAlgorithms = Object.entries(coverage?.missingByAlgorithm ?? {})
  const topMissingBalancing = Object.entries(coverage?.missingByBalancingMethod ?? {})

  return (
    <Paper elevation={0} sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 2,
          flexWrap: 'wrap',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
          <ScienceIcon color="primary" fontSize="small" />
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Missing statistics</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            size="small"
            startIcon={<PlayArrowIcon />}
            disabled={!selectedMissing.length || active}
            onClick={() => trainMissing(selectedMissing)}
          >
            Train selected ({selectedMissing.length})
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<PlayArrowIcon />}
            disabled={!missing.length || active}
            onClick={() => trainMissing(missing)}
          >
            Train all missing ({missing.length})
          </Button>
        </Box>
      </Box>

      {coverageQuery.isError ? (
        <Box sx={{ px: 2, py: 2 }}>
          <Alert severity="error">Could not load model coverage from the backend.</Alert>
        </Box>
      ) : (
        <>
          <Box sx={{ p: 2, display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, minmax(0, 1fr))' }, gap: 1.5 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">Stats coverage</Typography>
              <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 700 }}>
                {coverageQuery.isLoading ? '-' : `${coverage?.availableCount ?? 0}/${coverage?.totalExpected ?? 0}`}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Complete</Typography>
              <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 700 }}>
                {coverageQuery.isLoading ? '-' : `${completionPct}%`}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Missing combinations</Typography>
              <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 700 }}>
                {coverageQuery.isLoading ? '-' : (coverage?.missingCount ?? 0).toLocaleString()}
              </Typography>
            </Box>
          </Box>

          {missing.length > 0 && (
            <Box sx={{ px: 2, pb: 1.5, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {topMissingAlgorithms.map(([algorithm, count]) => (
                <Chip key={algorithm} size="small" variant="outlined" label={`${ALGORITHM_LABELS[algorithm as Algorithm]}: ${count}`} />
              ))}
              {topMissingBalancing.map(([method, count]) => (
                <Chip key={method} size="small" variant="outlined" label={`${BALANCING_METHOD_LABELS[method as BalancingMethod]}: ${count}`} />
              ))}
            </Box>
          )}

          {missing.length === 0 && !coverageQuery.isLoading ? (
            <Box sx={{ px: 2, pb: 2 }}>
              <Alert severity="success">All expected model statistics are present.</Alert>
            </Box>
          ) : (
            <Box sx={{ borderTop: '1px solid', borderColor: 'divider', maxHeight: 340, overflowY: 'auto' }}>
              {missing.length > 0 && (
                <Box
                  sx={{
                    px: 2,
                    py: 0.75,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    bgcolor: 'grey.50',
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                  }}
                >
                  <Checkbox
                    size="small"
                    checked={allMissingSelected}
                    indeterminate={someMissingSelected}
                    disabled={active}
                    onChange={toggleAllMissing}
                  />
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>
                    {selectedMissing.length ? `${selectedMissing.length} selected` : 'Select missing combinations'}
                  </Typography>
                </Box>
              )}
              {missing.map((spec) => (
                <Box
                  key={spec.id}
                  sx={{
                    px: 2,
                    py: 1,
                    display: 'grid',
                    gridTemplateColumns: { xs: 'auto 1fr', md: 'auto 1fr 1fr 1fr auto' },
                    gap: 1,
                    alignItems: 'center',
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                    '&:last-child': { borderBottom: 0 },
                  }}
                >
                  <Checkbox
                    size="small"
                    checked={Boolean(spec.id && selectedMissingIds.has(spec.id))}
                    disabled={active}
                    onChange={() => toggleMissing(spec)}
                    sx={{ p: 0.5 }}
                  />
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>
                    {ALGORITHM_LABELS[spec.algorithm]}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {specLabel(spec)}
                  </Typography>
                  <Chip
                    label={BALANCING_METHOD_LABELS[spec.balancingMethod]}
                    size="small"
                    variant="outlined"
                    sx={{ borderRadius: 1.5, justifySelf: { xs: 'start', md: 'start' } }}
                  />
                  <Button
                    size="small"
                    startIcon={<PlayArrowIcon />}
                    disabled={active}
                    onClick={() => trainMissing([spec])}
                    sx={{ justifySelf: { xs: 'start', md: 'end' } }}
                  >
                    Train
                  </Button>
                </Box>
              ))}
            </Box>
          )}

          {(active || job || startMutation.isError) && (
            <Box sx={{ px: 2, py: 2, borderTop: '1px solid', borderColor: 'divider' }}>
              {active && <LinearProgress sx={{ mb: 1.5, borderRadius: 1 }} />}
              {startMutation.isError && <Alert severity="error">{startMutation.error.message}</Alert>}
              {job?.status === 'failed' && <Alert severity="error">{job.error || job.message}</Alert>}
              {(job?.status === 'queued' || job?.status === 'running') && (
                <Alert severity="info">{job.message}</Alert>
              )}
              {job?.status === 'succeeded' && job.result && (
                <Alert
                  severity="success"
                  action={
                    firstTrainedModelId ? <Button color="inherit" size="small" onClick={() => navigate(`/models/${firstTrainedModelId}`)}>
                      View
                    </Button> : undefined
                  }
                >
                  {job.message} {job.result.trained} trained, {job.result.reused} reused.
                </Alert>
              )}
            </Box>
          )}
        </>
      )}
    </Paper>
  )
}

function ModelListSection({
  title,
  subtitle,
  rows,
  columns,
  loading,
  selectMode,
  rowSelectionModel,
  onSelectionChange,
  onRowClick,
  actions,
}: {
  title: string
  subtitle: string
  rows: ModelRow[]
  columns: GridColDef[]
  loading: boolean
  selectMode: boolean
  rowSelectionModel: GridRowSelectionModel
  onSelectionChange: (rows: ModelRow[], model: GridRowSelectionModel) => void
  onRowClick: (params: GridRowParams) => void
  actions?: ReactNode
}) {
  return (
    <Paper elevation={0} sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 1.5,
          flexWrap: 'wrap',
        }}
      >
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>{title}</Typography>
          <Typography variant="caption" color="text.secondary">{subtitle}</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          {actions}
          <Chip label={`${rows.length} model${rows.length === 1 ? '' : 's'}`} size="small" variant="outlined" sx={{ borderRadius: 1 }} />
        </Box>
      </Box>

      {rows.length === 0 && !loading ? (
        <Box sx={{ px: 2, py: 4 }}>
          <Typography variant="body2" color="text.secondary">No models in this group yet.</Typography>
        </Box>
      ) : (
        <DataGrid
          rows={rows}
          columns={columns}
          loading={loading}
          density="compact"
          autoHeight
          hideFooter
          checkboxSelection={!selectMode}
          disableRowSelectionOnClick={!selectMode}
          disableRowSelectionExcludeModel
          rowSelectionModel={rowSelectionModel}
          onRowSelectionModelChange={(model) => onSelectionChange(rows, model)}
          onRowClick={onRowClick}
          sx={{
            border: 0,
            cursor: 'pointer',
            '& .MuiDataGrid-columnHeaders': {
              bgcolor: 'grey.50',
              borderBottom: '1px solid',
              borderColor: 'divider',
            },
            '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 700, textAlign: 'left' },
            '& .MuiDataGrid-row:hover': { bgcolor: 'action.hover' },
            '& .MuiDataGrid-cell': { py: 0.75 },
            '& .MuiDataGrid-cellContent': { justifyContent: 'flex-start' },
          }}
        />
      )}
    </Paper>
  )
}

function ModelComparison({
  embedded = false,
  mode = 'compare',
  selectedModelId = '',
  onModelSelect,
}: ModelComparisonProps) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [selectionModel, setSelectionModel] = useState<GridRowSelectionModel>({ type: 'include', ids: new Set() })
  const [deleteTargets, setDeleteTargets] = useState<ModelRow[]>([])

  const query = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 30_000,
  })

  const models = query.data ?? []
  const normalModels = useMemo(() => models.filter((model) => !model.isTuned), [models])
  const tunedModels = useMemo(() => models.filter((model) => model.isTuned), [models])
  const loading = query.isLoading || query.isFetching
  const error = query.isError ? 'Could not load models from the backend.' : ''
  const selectedIds = useMemo(
    () => (selectionModel.type === 'include' ? [...selectionModel.ids].map(String) : []),
    [selectionModel],
  )
  const selectedTunedModels = useMemo(
    () => tunedModels.filter((model) => selectedIds.includes(model.id)),
    [selectedIds, tunedModels],
  )
  const canCompare = selectedIds.length >= 2
  const selectMode = mode === 'select'
  const deleteMutation = useMutation({
    mutationFn: deleteModels,
    onSuccess: (results) => {
      const deletedIds = new Set(results.map((result) => result.modelId))
      setSelectionModel((current) => {
        if (current.type !== 'include') return current
        const ids = new Set(current.ids)
        deletedIds.forEach((id) => ids.delete(id))
        return { type: 'include', ids }
      })
      setDeleteTargets([])
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({ queryKey: ['training-coverage'] })
    },
  })
  const deletingModelIds = useMemo(
    () => new Set(deleteMutation.variables ?? []),
    [deleteMutation.variables],
  )
  const columns = useColumns({
    allowDelete: !selectMode,
    deletingModelId: [...deletingModelIds][0] ?? '',
    onDelete: (model) => setDeleteTargets([model]),
  })
  const rowSelectionModel: GridRowSelectionModel = selectMode
    ? selectedModelId
      ? { type: 'include', ids: new Set([selectedModelId]) }
      : { type: 'include', ids: new Set() }
    : selectionModel

  const handleCompare = () => {
    if (!canCompare) return
    navigate(`/models/compare?ids=${encodeURIComponent(selectedIds.join(','))}`)
  }

  const handleRowClick = (params: GridRowParams) => {
    if (selectMode) {
      onModelSelect?.(params.row as ModelRow)
      return
    }
    navigate(`/models/${params.row.id}`)
  }

  const handleSectionSelectionChange = (rows: ModelRow[], nextModel: GridRowSelectionModel) => {
    if (selectMode) return
    const rowIds = new Set(rows.map((row) => row.id))
    const nextIds = nextModel.type === 'include' ? new Set([...nextModel.ids].map(String)) : new Set<string>()

    setSelectionModel((current) => {
      const ids = current.type === 'include' ? new Set([...current.ids].map(String)) : new Set<string>()
      rowIds.forEach((id) => ids.delete(id))
      nextIds.forEach((id) => ids.add(id))
      return { type: 'include', ids }
    })
  }

  const confirmDelete = () => {
    if (!deleteTargets.length || deleteMutation.isPending) return
    deleteMutation.mutate(deleteTargets.map((model) => model.id))
  }

  const content = (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Box
        sx={{
          px: 2,
          py: 1.5,
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 3,
          bgcolor: 'background.paper',
          display: 'flex',
          gap: 1.5,
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
        }}
      >
        <Typography variant="subtitle2" sx={{ fontWeight: 400 }}>
          {selectMode ? 'Click a row to choose the prediction model' : 'Select models to compare, or click a row to view details'}
        </Typography>
        {!selectMode && (
          <Button
            variant="contained"
            size="small"
            startIcon={<CompareArrowsIcon />}
            disabled={!canCompare}
            onClick={handleCompare}
          >
            Compare selected ({selectedIds.length})
          </Button>
        )}
      </Box>

      {error ? (
        <Paper elevation={0} sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', px: 2, py: 4 }}>
          <Typography variant="body2" color="error">{error}</Typography>
        </Paper>
      ) : (
        <>
          <ModelListSection
            title="Normal models"
            subtitle="Baseline trained models saved in the regular model folder."
            rows={normalModels}
            columns={columns}
            loading={loading}
            selectMode={selectMode}
            rowSelectionModel={rowSelectionModel}
            onSelectionChange={handleSectionSelectionChange}
            onRowClick={handleRowClick}
          />
          <ModelListSection
            title="Fine-tuned models"
            subtitle="Tuned runs saved separately in the tuned-models folder."
            rows={tunedModels}
            columns={columns}
            loading={loading}
            selectMode={selectMode}
            rowSelectionModel={rowSelectionModel}
            onSelectionChange={handleSectionSelectionChange}
            onRowClick={handleRowClick}
            actions={!selectMode && (
              <Button
                size="small"
                color="error"
                variant="outlined"
                startIcon={<DeleteOutlinedIcon />}
                disabled={!selectedTunedModels.length || deleteMutation.isPending}
                onClick={() => setDeleteTargets(selectedTunedModels)}
              >
                Delete selected ({selectedTunedModels.length})
              </Button>
            )}
          />
        </>
      )}

      {deleteMutation.isError && (
        <Alert severity="error">{deleteMutation.error.message}</Alert>
      )}

      <Dialog open={deleteTargets.length > 0} onClose={() => !deleteMutation.isPending && setDeleteTargets([])} maxWidth="xs" fullWidth>
        <DialogTitle>{deleteTargets.length === 1 ? 'Delete model?' : 'Delete models?'}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary">
            This removes the fine-tuned model file{deleteTargets.length === 1 ? '' : 's'} and database records. This cannot be undone.
          </Typography>
          {deleteTargets.length > 0 && (
            <Typography variant="body2" sx={{ mt: 1.5, fontWeight: 700, wordBreak: 'break-word' }}>
              {deleteTargets.length === 1
                ? deleteTargets[0].id
                : `${deleteTargets.length} fine-tuned models selected`}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button disabled={deleteMutation.isPending} onClick={() => setDeleteTargets([])}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            startIcon={<DeleteOutlinedIcon />}
            disabled={deleteMutation.isPending}
            onClick={confirmDelete}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )

  if (embedded) return content

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <TrainingPanel />
        <MissingStatisticsPanel />
        <ModelOverview models={models} />
        {content}
      </Box>
    </main>
  )
}

export default ModelComparison
