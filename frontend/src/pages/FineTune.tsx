import { useEffect, useMemo, useState } from 'react'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import LinearProgress from '@mui/material/LinearProgress'
import ListSubheader from '@mui/material/ListSubheader'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Select from '@mui/material/Select'
import Slider from '@mui/material/Slider'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import AutoGraphIcon from '@mui/icons-material/AutoGraph'
import MemoryIcon from '@mui/icons-material/Memory'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import TuneIcon from '@mui/icons-material/Tune'
import { DataGrid, type GridColDef, type GridRenderCellParams, type GridRowSelectionModel } from '@mui/x-data-grid'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchJson, postJson } from '../api'
import {
  ALGORITHM_LABELS,
  BALANCING_METHOD_LABELS,
  FEATURE_SET_LABELS,
  UNCERTAINTY_VARIANT_LABELS,
  type Algorithm,
  type BalancingMethod,
} from '../modelMetadata'
import { fetchModelDetail, modelLabel, pct, type ConfusionMatrix, type ModelDetail } from '../modelData'

type ModelRow = {
  id: string
  algorithm: Algorithm
  featureSet: 'lifestyle' | 'clinical' | 'combined'
  uncertaintyVariant: 'with_uncertain' | 'without_uncertain'
  balancingMethod: BalancingMethod
  auc: number
  accuracy: number
  f1: number
  precision: number
  recall: number
  classificationThreshold: number
  isTuned: boolean
}

type TrainingRequest = {
  algorithms: Algorithm[]
  datasetIds: string[]
  balancingMethods: BalancingMethod[]
  targetRatio: number
  classificationThreshold: number
  forceRetrain: boolean
  useGpu: boolean
  removedFeatures: string[]
  hyperparameters: Record<string, number>
}

type TrainingJob = {
  id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  message: string
  result: {
    models: Array<{
      modelId: string
      reusedExistingModel: boolean
      metrics: Pick<ModelRow, 'auc' | 'accuracy' | 'f1' | 'precision' | 'recall'>
    }>
    total: number
    trained: number
    reused: number
  } | null
  error: string | null
}

type TuneForm = {
  algorithm: Algorithm
  balancingMethod: BalancingMethod
  targetRatio: number
  classificationThreshold: number
  useGpu: boolean
  nEstimators: number
  maxDepth: number
  learningRate: number
  minSamplesLeaf: number
  subsample: number
  numLeaves: number
}

type FeatureImportanceComparisonRow = {
  id: string
  feature: string
  baselineImportance: number
  tunedImportance: number | null
  delta: number | null
}

const METRICS = [
  { key: 'auc', label: 'AUC-ROC', format: (value: number) => value.toFixed(3) },
  { key: 'accuracy', label: 'Accuracy', format: pct },
  { key: 'f1', label: 'F1', format: (value: number) => value.toFixed(3) },
  { key: 'precision', label: 'Precision', format: pct },
  { key: 'recall', label: 'Recall', format: pct },
] as const

const defaultForm = (model: ModelRow | null): TuneForm => ({
  algorithm: model?.algorithm ?? 'xgboost',
  balancingMethod: model?.balancingMethod ?? 'random_oversampling',
  targetRatio: model?.balancingMethod === 'weighted' ? 1 : 1,
  classificationThreshold: model?.classificationThreshold ?? 0.5,
  useGpu: model?.algorithm === 'random_forest' ? false : true,
  nEstimators: model?.algorithm === 'random_forest' ? 200 : 100,
  maxDepth: model?.algorithm === 'random_forest' ? 0 : model?.algorithm === 'lightgbm' ? -1 : 6,
  learningRate: model?.algorithm === 'xgboost' ? 0.3 : 0.1,
  minSamplesLeaf: 1,
  subsample: 1,
  numLeaves: 31,
})

async function fetchModels(): Promise<ModelRow[]> {
  return fetchJson<ModelRow[]>('/api/models')
}

async function startTrainingJob(request: TrainingRequest): Promise<TrainingJob> {
  return postJson<TrainingJob>('/api/training/jobs', request)
}

async function fetchTrainingJob(jobId: string): Promise<TrainingJob> {
  return fetchJson<TrainingJob>(`/api/training/jobs/${jobId}`)
}

function datasetIdFor(model: Pick<ModelRow, 'featureSet' | 'uncertaintyVariant'>) {
  return `${model.featureSet}_${model.uncertaintyVariant}`
}

function modelTypeLabel(model: Pick<ModelRow, 'isTuned'>) {
  return model.isTuned ? 'Fine-tuned model' : 'Normal model'
}

function metricValue(model: Pick<ModelRow, 'auc' | 'accuracy' | 'f1' | 'precision' | 'recall'>, key: typeof METRICS[number]['key']) {
  return model[key]
}

function MetricComparison({
  baseline,
  tuned,
}: {
  baseline: ModelRow
  tuned: Pick<ModelRow, 'auc' | 'accuracy' | 'f1' | 'precision' | 'recall'> | null
}) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(5, minmax(0, 1fr))' }, gap: 1.25 }}>
      {METRICS.map((metric) => {
        const before = metricValue(baseline, metric.key)
        const after = tuned ? metricValue(tuned, metric.key) : null
        const delta = after === null ? null : after - before
        const positive = (delta ?? 0) >= 0

        return (
          <Box key={metric.key} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1.5, p: 1.5, bgcolor: 'background.paper' }}>
            <Typography variant="caption" color="text.secondary">{metric.label}</Typography>
            <Typography variant="h6" sx={{ fontSize: '1.15rem', fontWeight: 800, mt: 0.25 }}>
              {after === null ? metric.format(before) : metric.format(after)}
            </Typography>
            <Typography variant="caption" sx={{ color: delta === null ? 'text.secondary' : positive ? 'success.dark' : 'error.dark', fontWeight: 700 }}>
              {delta === null ? 'Baseline' : `${positive ? '+' : ''}${(delta * 100).toFixed(2)} pts`}
            </Typography>
          </Box>
        )
      })}
    </Box>
  )
}

function matrixTotal(matrix: ConfusionMatrix) {
  return matrix.tn + matrix.fp + matrix.fn + matrix.tp
}

function ConfusionMatrixPanel({
  title,
  matrix,
}: {
  title: string
  matrix: ConfusionMatrix | null
}) {
  const total = matrix ? matrixTotal(matrix) : 0
  const cells = matrix
    ? [
        { label: 'True negative', value: matrix.tn, color: 'success.dark', bgcolor: 'success.50' },
        { label: 'False positive', value: matrix.fp, color: 'error.dark', bgcolor: 'error.50' },
        { label: 'False negative', value: matrix.fn, color: 'error.dark', bgcolor: 'error.50' },
        { label: 'True positive', value: matrix.tp, color: 'success.dark', bgcolor: 'success.50' },
      ]
    : []

  return (
    <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1.5, overflow: 'hidden', bgcolor: 'background.paper' }}>
      <Box sx={{ px: 1.5, py: 1.25, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', justifyContent: 'space-between', gap: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>{title}</Typography>
        <Typography variant="caption" color="text.secondary">{matrix ? `${total.toLocaleString()} cases` : 'Not available'}</Typography>
      </Box>
      {matrix ? (
        <Box sx={{ p: 1.25, display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 1 }}>
          {cells.map((cell) => (
            <Box key={cell.label} sx={{ bgcolor: cell.bgcolor, borderRadius: 1.25, p: 1.25, minHeight: 88 }}>
              <Typography variant="h6" sx={{ color: cell.color, fontSize: '1.1rem', fontWeight: 900, lineHeight: 1 }}>
                {cell.value.toLocaleString()}
              </Typography>
              <Typography variant="caption" sx={{ color: cell.color, fontWeight: 800, display: 'block', mt: 0.75 }}>
                {cell.label}
              </Typography>
              <Typography variant="caption" sx={{ color: cell.color }}>
                {total > 0 ? pct(cell.value / total) : '0.0%'}
              </Typography>
            </Box>
          ))}
        </Box>
      ) : (
        <Box sx={{ p: 2, minHeight: 198, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="body2" color="text.secondary">Train a tuned run to show this matrix.</Typography>
        </Box>
      )}
    </Box>
  )
}

function normalizedImportances(detail: ModelDetail | null) {
  const raw = detail?.featureImportances ?? []
  const total = raw.reduce((sum, item) => sum + Math.max(0, item.importance), 0)
  return new Map(raw.map((item) => [
    item.feature,
    total > 0 ? Math.max(0, item.importance) / total : 0,
  ]))
}

function FeatureImportanceComparison({
  baseline,
  tuned,
}: {
  baseline: ModelDetail | null
  tuned: ModelDetail | null
}) {
  const rows = useMemo<FeatureImportanceComparisonRow[]>(() => {
    const baselineMap = normalizedImportances(baseline)
    const tunedMap = normalizedImportances(tuned)
    const features = new Set([...baselineMap.keys(), ...tunedMap.keys()])

    return [...features]
      .map((feature) => {
        const baselineImportance = baselineMap.get(feature) ?? 0
        const tunedImportance = tuned ? tunedMap.get(feature) ?? 0 : null
        return {
          id: feature,
          feature,
          baselineImportance,
          tunedImportance,
          delta: tunedImportance === null ? null : tunedImportance - baselineImportance,
        }
      })
      .sort((left, right) => {
        const leftScore = Math.max(left.baselineImportance, left.tunedImportance ?? 0)
        const rightScore = Math.max(right.baselineImportance, right.tunedImportance ?? 0)
        return rightScore - leftScore || left.feature.localeCompare(right.feature)
      })
      .slice(0, 30)
  }, [baseline, tuned])

  const columns = useMemo<GridColDef<FeatureImportanceComparisonRow>[]>(() => [
    { field: 'feature', headerName: 'Feature', flex: 1.6, minWidth: 180 },
    {
      field: 'baselineImportance',
      headerName: 'Before',
      flex: 0.8,
      minWidth: 110,
      valueFormatter: (value) => pct(value as number),
    },
    {
      field: 'tunedImportance',
      headerName: 'After',
      flex: 0.8,
      minWidth: 110,
      valueFormatter: (value) => (typeof value === 'number' ? pct(value) : '-'),
    },
    {
      field: 'delta',
      headerName: 'Change',
      flex: 0.8,
      minWidth: 110,
      renderCell: ({ value }: GridRenderCellParams<FeatureImportanceComparisonRow, number | null>) => {
        if (typeof value !== 'number') {
          return <Typography variant="body2" color="text.secondary">-</Typography>
        }
        const positive = value >= 0
        return (
          <Typography variant="body2" sx={{ color: positive ? 'success.dark' : 'error.dark', fontWeight: 800 }}>
            {positive ? '+' : ''}{(value * 100).toFixed(2)} pts
          </Typography>
        )
      },
    },
  ], [])

  return (
    <Box sx={{ mt: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1.5, overflow: 'hidden', bgcolor: 'background.paper' }}>
      <Box sx={{ px: 1.5, py: 1.25, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Feature importance before and after</Typography>
          <Typography variant="caption" color="text.secondary">
            {tuned ? 'Top 30 features by importance across both runs.' : 'Train a tuned run to fill the after column.'}
          </Typography>
        </Box>
        <Chip size="small" variant="outlined" label={`${rows.length} shown`} sx={{ borderRadius: 1 }} />
      </Box>
      <DataGrid
        rows={rows}
        columns={columns}
        loading={!baseline}
        density="compact"
        hideFooter
        disableRowSelectionOnClick
        sx={{
          border: 0,
          height: 420,
          '& .MuiDataGrid-columnHeaders': { bgcolor: 'grey.50' },
          '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 800 },
        }}
      />
    </Box>
  )
}

function SelectedModelSummary({ model, detail }: { model: ModelRow; detail: ModelDetail | null }) {
  return (
    <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap' }}>
        <Box>
          <Typography variant="caption" color="text.secondary">Selected baseline</Typography>
          <Typography variant="h6" sx={{ fontSize: '1.05rem', fontWeight: 800 }}>{modelLabel(model)}</Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <Chip
            size="small"
            color={model.isTuned ? 'primary' : 'default'}
            variant={model.isTuned ? 'filled' : 'outlined'}
            label={modelTypeLabel(model)}
            sx={{ borderRadius: 1 }}
          />
          <Chip size="small" label={`${detail?.featureColumns.length ?? '-'} features`} sx={{ borderRadius: 1 }} />
        </Box>
      </Box>
      <Box sx={{ mt: 1.5, display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
        <Chip size="small" variant="outlined" label={ALGORITHM_LABELS[model.algorithm]} />
        <Chip size="small" variant="outlined" label={FEATURE_SET_LABELS[model.featureSet]} />
        <Chip size="small" variant="outlined" label={UNCERTAINTY_VARIANT_LABELS[model.uncertaintyVariant]} />
        <Chip size="small" variant="outlined" label={BALANCING_METHOD_LABELS[model.balancingMethod]} />
      </Box>
    </Paper>
  )
}

function FineTune() {
  const queryClient = useQueryClient()
  const [selectedModelId, setSelectedModelId] = useState('')
  const [form, setForm] = useState<TuneForm>(defaultForm(null))
  const [formModelId, setFormModelId] = useState('')
  const [featureSelection, setFeatureSelection] = useState<GridRowSelectionModel>({ type: 'include', ids: new Set() })
  const [jobId, setJobId] = useState('')

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 30_000,
  })

  const models = useMemo(() => modelsQuery.data ?? [], [modelsQuery.data])
  const normalModels = useMemo(() => models.filter((model) => !model.isTuned), [models])
  const fineTunedModels = useMemo(() => models.filter((model) => model.isTuned), [models])
  const selectedModel = models.find((model) => model.id === selectedModelId) ?? models[0] ?? null
  const activeForm = useMemo(
    () => (selectedModel && formModelId !== selectedModel.id ? defaultForm(selectedModel) : form),
    [form, formModelId, selectedModel],
  )

  const updateForm = (updater: (current: TuneForm) => TuneForm) => {
    setForm(updater(activeForm))
    setFormModelId(selectedModel?.id ?? '')
  }

  const handleModelChange = (modelId: string) => {
    const nextModel = models.find((model) => model.id === modelId) ?? null
    setSelectedModelId(modelId)
    setForm(defaultForm(nextModel))
    setFormModelId(modelId)
    setFeatureSelection({ type: 'include', ids: new Set() })
    setJobId('')
  }

  const handleReset = () => {
    setForm(defaultForm(selectedModel))
    setFormModelId(selectedModel?.id ?? '')
    setFeatureSelection({ type: 'include', ids: new Set() })
    setJobId('')
    startMutation.reset()
  }

  const detailQuery = useQuery({
    queryKey: ['model', selectedModel?.id],
    queryFn: () => fetchModelDetail(selectedModel!.id),
    enabled: Boolean(selectedModel?.id),
    staleTime: 30_000,
  })

  const startMutation = useMutation({
    mutationFn: startTrainingJob,
    onSuccess: (job) => setJobId(job.id),
  })

  const jobQuery = useQuery({
    queryKey: ['fine-tuning-job', jobId],
    queryFn: () => fetchTrainingJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const job = query.state.data
      return job?.status === 'queued' || job?.status === 'running' ? 2000 : false
    },
  })

  const job = jobQuery.data ?? (startMutation.data?.id === jobId ? startMutation.data : null)
  const active = startMutation.isPending || job?.status === 'queued' || job?.status === 'running'
  const tunedResult = job?.status === 'succeeded' ? job.result?.models[0] ?? null : null
  const tunedModelId = tunedResult?.modelId ?? ''

  const tunedDetailQuery = useQuery({
    queryKey: ['model', tunedModelId],
    queryFn: () => fetchModelDetail(tunedModelId),
    enabled: Boolean(tunedModelId),
    staleTime: 0,
  })

  useEffect(() => {
    if (job?.status !== 'succeeded') return
    queryClient.invalidateQueries({ queryKey: ['models'] })
  }, [job?.status, queryClient])

  const removedFeatures = useMemo(
    () => (featureSelection.type === 'include' ? [...featureSelection.ids].map(String) : []),
    [featureSelection],
  )

  const baselineDetail = detailQuery.data ?? null
  const featureRows = useMemo(() => {
    const importanceByFeature = new Map((baselineDetail?.featureImportances ?? []).map((item) => [item.feature, item.importance]))
    const totalImportance = [...importanceByFeature.values()].reduce((sum, value) => sum + Math.max(0, value), 0)
    return (baselineDetail?.featureColumns ?? [])
      .map((feature) => ({
        id: feature,
        feature,
        importance: totalImportance > 0 ? (importanceByFeature.get(feature) ?? 0) / totalImportance : 0,
      }))
      .sort((left, right) => right.importance - left.importance || left.feature.localeCompare(right.feature))
  }, [baselineDetail])

  const featureColumns = useMemo<GridColDef[]>(() => [
    { field: 'feature', headerName: 'Feature', flex: 1.4, minWidth: 160 },
    {
      field: 'importance',
      headerName: 'Importance',
      flex: 0.7,
      minWidth: 120,
      valueFormatter: (value) => `${((value as number) * 100).toFixed(2)}%`,
    },
  ], [])

  const hyperparameters = useMemo<Record<string, number>>(() => {
    if (activeForm.algorithm === 'random_forest') {
      const params: Record<string, number> = {
        n_estimators: activeForm.nEstimators,
        max_depth: activeForm.maxDepth,
        min_samples_leaf: activeForm.minSamplesLeaf,
      }
      return params
    }
    if (activeForm.algorithm === 'lightgbm') {
      const params: Record<string, number> = {
        n_estimators: activeForm.nEstimators,
        max_depth: activeForm.maxDepth,
        learning_rate: activeForm.learningRate,
        num_leaves: activeForm.numLeaves,
      }
      return params
    }
    const params: Record<string, number> = {
      n_estimators: activeForm.nEstimators,
      max_depth: Math.max(1, activeForm.maxDepth),
      learning_rate: activeForm.learningRate,
      subsample: activeForm.subsample,
    }
    return params
  }, [activeForm])

  const canTrain = Boolean(selectedModel && baselineDetail && featureRows.length - removedFeatures.length > 0 && !active)

  const handleTrain = () => {
    if (!selectedModel || !canTrain) return
    startMutation.mutate({
      algorithms: [activeForm.algorithm],
      datasetIds: [datasetIdFor(selectedModel)],
      balancingMethods: [activeForm.balancingMethod],
      targetRatio: activeForm.balancingMethod === 'weighted' ? 1 : activeForm.targetRatio,
      classificationThreshold: activeForm.classificationThreshold,
      forceRetrain: true,
      useGpu: activeForm.algorithm === 'random_forest' ? false : activeForm.useGpu,
      removedFeatures,
      hyperparameters,
    })
  }

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, overflow: 'hidden' }}>
          <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
              <TuneIcon color="primary" />
              <Box>
                <Typography variant="h5" sx={{ fontSize: '1.35rem', fontWeight: 800 }}>Model fine-tuning</Typography>
                <Typography variant="body2" color="text.secondary">Adjust training settings, remove features, and compare the new run with the selected baseline.</Typography>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Button
                variant="outlined"
                startIcon={<RestartAltIcon />}
                disabled={active || !selectedModel}
                onClick={handleReset}
                sx={{ minWidth: 112 }}
              >
                Reset
              </Button>
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                disabled={!canTrain}
                onClick={handleTrain}
                sx={{ minWidth: 160 }}
              >
                Train model
              </Button>
            </Box>
          </Box>

          {active && <LinearProgress />}

          <Box sx={{ p: 2.5, display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '1fr 1.25fr' }, gap: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl size="small" fullWidth disabled={modelsQuery.isLoading || active}>
                <InputLabel id="fine-tune-model-label">Model baseline</InputLabel>
                <Select
                  labelId="fine-tune-model-label"
                  label="Model baseline"
                  value={selectedModel?.id ?? ''}
                  renderValue={(value) => {
                    const model = models.find((item) => item.id === value)
                    return model ? `${modelTypeLabel(model)} - ${modelLabel(model)}` : ''
                  }}
                  onChange={(event) => handleModelChange(event.target.value)}
                >
                  <ListSubheader>Normal models ({normalModels.length})</ListSubheader>
                  {normalModels.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, width: '100%' }}>
                        <Chip size="small" variant="outlined" label="Normal" sx={{ borderRadius: 1, minWidth: 68 }} />
                        <Typography variant="body2" noWrap>{modelLabel(model)}</Typography>
                      </Box>
                    </MenuItem>
                  ))}
                  <ListSubheader>Fine-tuned models ({fineTunedModels.length})</ListSubheader>
                  {fineTunedModels.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, width: '100%' }}>
                        <Chip size="small" color="primary" label="Fine-tuned" sx={{ borderRadius: 1, minWidth: 92 }} />
                        <Typography variant="body2" noWrap>{modelLabel(model)}</Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mt: -1 }}>
                <Chip size="small" variant="outlined" label={`${normalModels.length} normal`} sx={{ borderRadius: 1 }} />
                <Chip size="small" color="primary" variant="outlined" label={`${fineTunedModels.length} fine-tuned`} sx={{ borderRadius: 1 }} />
                {selectedModel && (
                  <Chip
                    size="small"
                    color={selectedModel.isTuned ? 'primary' : 'default'}
                    label={`Selected: ${modelTypeLabel(selectedModel)}`}
                    sx={{ borderRadius: 1 }}
                  />
                )}
              </Box>

              {selectedModel && <SelectedModelSummary model={selectedModel} detail={baselineDetail} />}

              <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                  <MemoryIcon color="primary" fontSize="small" />
                  <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Training settings</Typography>
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 1.5 }}>
                  <FormControl size="small">
                    <InputLabel id="fine-tune-algorithm-label">Algorithm</InputLabel>
                    <Select
                      labelId="fine-tune-algorithm-label"
                      label="Algorithm"
                      value={activeForm.algorithm}
                      disabled={active}
                      onChange={(event) => {
                        const algorithm = event.target.value as Algorithm
                        updateForm((current) => ({
                          ...current,
                          algorithm,
                          maxDepth: algorithm === 'random_forest' ? 0 : algorithm === 'lightgbm' ? -1 : current.maxDepth < 1 ? 6 : current.maxDepth,
                          learningRate: algorithm === 'xgboost' ? 0.3 : algorithm === 'lightgbm' ? 0.1 : current.learningRate,
                          nEstimators: algorithm === 'random_forest' ? 200 : current.nEstimators,
                          useGpu: algorithm === 'random_forest' ? false : current.useGpu,
                        }))
                      }}
                    >
                      {Object.entries(ALGORITHM_LABELS).map(([value, label]) => (
                        <MenuItem key={value} value={value}>{label}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <FormControl size="small">
                    <InputLabel id="fine-tune-balance-label">Balancing</InputLabel>
                    <Select
                      labelId="fine-tune-balance-label"
                      label="Balancing"
                      value={activeForm.balancingMethod}
                      disabled={active}
                      onChange={(event) => updateForm((current) => ({ ...current, balancingMethod: event.target.value as BalancingMethod }))}
                    >
                      {Object.entries(BALANCING_METHOD_LABELS).map(([value, label]) => (
                        <MenuItem key={value} value={value}>{label}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <TextField
                    size="small"
                    label="Trees / estimators"
                    type="number"
                    value={activeForm.nEstimators}
                    disabled={active}
                    onChange={(event) => updateForm((current) => ({ ...current, nEstimators: Math.max(10, Number(event.target.value) || 10) }))}
                    slotProps={{ htmlInput: { min: 10, step: 10 } }}
                  />
                  <TextField
                    size="small"
                    label="Max depth"
                    type="number"
                    value={activeForm.maxDepth}
                    disabled={active}
                    helperText={
                      activeForm.algorithm === 'random_forest'
                        ? '0 keeps Random Forest unrestricted'
                        : activeForm.algorithm === 'lightgbm'
                          ? '-1 keeps LightGBM unrestricted'
                          : undefined
                    }
                    onChange={(event) => updateForm((current) => ({ ...current, maxDepth: Number(event.target.value) }))}
                  />
                  {activeForm.algorithm !== 'random_forest' && (
                    <TextField
                      size="small"
                      label="Learning rate"
                      type="number"
                      value={activeForm.learningRate}
                      disabled={active}
                      onChange={(event) => updateForm((current) => ({ ...current, learningRate: Math.max(0.01, Number(event.target.value) || 0.01) }))}
                      slotProps={{ htmlInput: { min: 0.01, max: 1, step: 0.01 } }}
                    />
                  )}
                  {activeForm.algorithm === 'random_forest' && (
                    <TextField
                      size="small"
                      label="Min samples leaf"
                      type="number"
                      value={activeForm.minSamplesLeaf}
                      disabled={active}
                      onChange={(event) => updateForm((current) => ({ ...current, minSamplesLeaf: Math.max(1, Number(event.target.value) || 1) }))}
                      slotProps={{ htmlInput: { min: 1, step: 1 } }}
                    />
                  )}
                  {activeForm.algorithm === 'xgboost' && (
                    <TextField
                      size="small"
                      label="Subsample"
                      type="number"
                      value={activeForm.subsample}
                      disabled={active}
                      onChange={(event) => updateForm((current) => ({ ...current, subsample: Math.min(1, Math.max(0.1, Number(event.target.value) || 0.1)) }))}
                      slotProps={{ htmlInput: { min: 0.1, max: 1, step: 0.05 } }}
                    />
                  )}
                  {activeForm.algorithm === 'lightgbm' && (
                    <TextField
                      size="small"
                      label="Num leaves"
                      type="number"
                      value={activeForm.numLeaves}
                      disabled={active}
                      onChange={(event) => updateForm((current) => ({ ...current, numLeaves: Math.max(2, Number(event.target.value) || 2) }))}
                      slotProps={{ htmlInput: { min: 2, step: 1 } }}
                    />
                  )}
                </Box>

                <Divider sx={{ my: 2 }} />

                <Typography variant="caption" color="text.secondary">Target minority/majority ratio</Typography>
                <Slider
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={activeForm.balancingMethod === 'weighted' ? 1 : activeForm.targetRatio}
                  disabled={active || activeForm.balancingMethod === 'weighted'}
                  valueLabelDisplay="auto"
                  onChange={(_, value) => updateForm((current) => ({ ...current, targetRatio: value as number }))}
                />
                <TextField
                  size="small"
                  label="Classification threshold"
                  type="number"
                  value={activeForm.classificationThreshold}
                  disabled={active}
                  onChange={(event) => updateForm((current) => ({
                    ...current,
                    classificationThreshold: Math.min(0.99, Math.max(0.01, Number(event.target.value) || 0.5)),
                  }))}
                  slotProps={{ htmlInput: { min: 0.01, max: 0.99, step: 0.01 } }}
                  fullWidth
                  sx={{ mb: 1.5 }}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
                  <Typography variant="body2">Use GPU when available</Typography>
                  <Switch
                    checked={activeForm.algorithm === 'random_forest' ? false : activeForm.useGpu}
                    disabled={active || activeForm.algorithm === 'random_forest'}
                    onChange={(event) => updateForm((current) => ({ ...current, useGpu: event.target.checked }))}
                  />
                </Box>
              </Paper>
            </Box>

            <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, overflow: 'hidden' }}>
              <Box sx={{ px: 2, py: 1.5, display: 'flex', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap', borderBottom: '1px solid', borderColor: 'divider' }}>
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>Feature removal</Typography>
                  <Typography variant="caption" color="text.secondary">{removedFeatures.length} removed, {Math.max(0, featureRows.length - removedFeatures.length)} retained</Typography>
                </Box>
                <Button
                  size="small"
                  startIcon={<RestartAltIcon />}
                  disabled={!removedFeatures.length || active}
                  onClick={() => setFeatureSelection({ type: 'include', ids: new Set() })}
                >
                  Reset
                </Button>
              </Box>
              <DataGrid
                rows={featureRows}
                columns={featureColumns}
                loading={detailQuery.isLoading}
                density="compact"
                hideFooter
                checkboxSelection
                disableRowSelectionExcludeModel
                rowSelectionModel={featureSelection}
                onRowSelectionModelChange={setFeatureSelection}
                sx={{
                  border: 0,
                  height: 510,
                  '& .MuiDataGrid-columnHeaders': { bgcolor: 'grey.50' },
                  '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 800 },
                }}
              />
            </Paper>
          </Box>
        </Paper>

        {modelsQuery.isError && <Alert severity="error">Could not load models from the backend.</Alert>}
        {detailQuery.isError && <Alert severity="error">Could not load the selected model details.</Alert>}
        {startMutation.isError && <Alert severity="error">{startMutation.error.message}</Alert>}
        {job?.status === 'failed' && <Alert severity="error">{job.error || job.message}</Alert>}
        {(job?.status === 'queued' || job?.status === 'running') && <Alert severity="info">{job.message}</Alert>}

        {selectedModel && (
          <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, p: 2.5 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, mb: 2 }}>
              <AutoGraphIcon color="primary" />
              <Box>
                <Typography variant="h6" sx={{ fontSize: '1.1rem', fontWeight: 800 }}>Updated statistics</Typography>
                <Typography variant="body2" color="text.secondary">
                  {tunedResult ? `Tuned model: ${tunedResult.modelId}` : 'Train a tuned run to compare against the baseline.'}
                </Typography>
              </Box>
            </Box>

            <MetricComparison baseline={selectedModel} tuned={tunedResult?.metrics ?? null} />

            <Box sx={{ mt: 2, display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 1.5 }}>
              <ConfusionMatrixPanel title="Baseline confusion matrix" matrix={baselineDetail?.confusionMatrix ?? null} />
              <ConfusionMatrixPanel title="Tuned confusion matrix" matrix={tunedDetailQuery.data?.confusionMatrix ?? null} />
            </Box>

            <FeatureImportanceComparison baseline={baselineDetail} tuned={tunedDetailQuery.data ?? null} />
          </Paper>
        )}
      </Box>
    </main>
  )
}

export default FineTune
