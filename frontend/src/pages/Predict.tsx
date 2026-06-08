import { useMemo, useState } from 'react'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Step from '@mui/material/Step'
import StepButton from '@mui/material/StepButton'
import Stepper from '@mui/material/Stepper'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import ScienceIcon from '@mui/icons-material/Science'
import { DataGrid, type GridColDef, type GridRowSelectionModel } from '@mui/x-data-grid'
import { useQuery } from '@tanstack/react-query'
import SectionCard from '../components/SectionCard'
import { fetchJson, postJson } from '../api'
import { fetchModelDetail, modelLabel, pct, type ModelDetail } from '../modelData'
import ModelComparison, { type ModelRow } from './ModelComparison'
import Patients, { type PatientSelection, type RegistryItem } from './Patients'

type PredictionResponse = {
  prediction: number
  probability: number
  label: string
  explanation?: ShapExplanation | null
  explanationError?: string
}

type ShapFeatureContribution = {
  feature: string
  value: number | null
  contribution: number
  absContribution: number
  direction: 'increases' | 'decreases'
}

type ShapExplanation = {
  modelOutput: 'probability'
  classLabel: string
  baseValue: number
  outputValue: number
  sumValue: number
  backgroundRows: number | null
  features: ShapFeatureContribution[]
}

type PredictionLogPayload = {
  modelId: string
  patient: {
    datasetId: string
    datasetLabel: string
    rowId: string
    rowIndex: number
    absoluteIndex: number
    page: number
    perPage: number
    row: Record<string, string>
  }
  selectedFeatures: { key: string; label: string }[]
  baselineFeatures: Record<string, number | null>
  scenarioFeatures: Record<string, number | null>
  baseline: PredictionResponse
  scenario: PredictionResponse
  changedFeatures: {
    key: string
    label: string
    before: number | null
    after: number | null
  }[]
}

type LabFeature = {
  key: string
  label: string
  helper: string
  kind: 'select' | 'number'
  options?: { value: number; label: string }[]
  toDisplay?: (value: number | null) => string
  toModel?: (value: string) => number | null
}

type LabResult = {
  baseline: PredictionResponse
  scenario: PredictionResponse
  deltas: {
    key: string
    label: string
    before: number | null
    after: number | null
    probability: number
    reduction: number
    relativeReduction: number
    globalImportance: number
  }[]
}

const FEATURE_OVERRIDES: Record<string, Partial<LabFeature>> = {
  _SMOKER3: {
    key: '_SMOKER3',
    label: 'Smoking status',
    helper: 'BRFSS _SMOKER3',
    kind: 'select',
    options: [
      { value: 4, label: 'Never smoked' },
      { value: 3, label: 'Former smoker' },
      { value: 2, label: 'Current, some days' },
      { value: 1, label: 'Current, every day' },
      { value: 9, label: 'Unknown' },
    ],
  },
  _TOTINDA: {
    key: '_TOTINDA',
    label: 'Physical activity',
    helper: 'BRFSS _TOTINDA',
    kind: 'select',
    options: [
      { value: 1, label: 'Active in past month' },
      { value: 2, label: 'No activity' },
      { value: 9, label: 'Unknown' },
    ],
  },
  _DRNKWK3: {
    key: '_DRNKWK3',
    label: 'Alcohol drinks/week',
    helper: 'BRFSS _DRNKWK3, displayed as weekly drinks',
    kind: 'number',
    toDisplay: (value) => {
      if (value === null || value >= 99900) return ''
      const drinks = Math.max(0, value)
      return Number.isInteger(drinks) ? String(drinks) : drinks.toFixed(2)
    },
    toModel: (value) => {
      if (value.trim() === '') return null
      return Math.max(0, Number(value))
    },
  },
  _BMI5: {
    key: '_BMI5',
    label: 'BMI',
    helper: 'BRFSS _BMI5, displayed as BMI',
    kind: 'number',
    toDisplay: (value) => (value === null ? '' : (value).toFixed(1)),
    toModel: (value) => {
      if (value.trim() === '') return null
      return Math.max(0, Number(value))
    },
  },
}

function makeFeature(key: string): LabFeature {
  const override = FEATURE_OVERRIDES[key] ?? {}
  return {
    key,
    label: override.label ?? key,
    helper: override.helper ?? `Model feature ${key}`,
    kind: override.kind ?? (override.options ? 'select' : 'number'),
    options: override.options,
    toDisplay: override.toDisplay,
    toModel: override.toModel,
  }
}

function parseModelValue(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === '-') return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function displayValue(feature: LabFeature, value: number | null) {
  if (feature.toDisplay) return feature.toDisplay(value)
  return value === null ? '' : String(value)
}

function modelValue(feature: LabFeature, value: string) {
  if (value.trim() === '') return null
  const converted = feature.toModel ? feature.toModel(value) : Number(value)
  return Number.isFinite(converted) ? converted : null
}

function sameModelValue(left: number | null, right: number | null) {
  if (left === null || right === null) return left === right
  return Math.abs(left - right) < 1e-6
}

function featureText(feature: LabFeature, value: number | null) {
  if (value === null) return 'Missing'
  const option = feature.options?.find((item) => item.value === value)
  return option?.label ?? displayValue(feature, value)
}

function formatDelta(value: number) {
  if (Math.abs(value) < 1e-6) return '±0.00 pp'
  const arrow = value > 0 ? '↓' : '↑'
  return `${arrow} ${Math.abs(value * 100).toFixed(2)} pp`
}

function formatRisk(value: number) {
  if (value > 0 && value < 0.0001) return '<0.01%'
  return `${(value * 100).toFixed(2)}%`
}

function formatSignedProbabilityPoints(value: number) {
  const sign = value >= 0 ? '+' : '-'
  return `${sign}${Math.abs(value * 100).toFixed(2)} pp`
}

function buildFeatures(row: Record<string, string>, model: ModelDetail) {
  return Object.fromEntries(
    model.featureColumns.map((column) => [column, parseModelValue(row[column])]),
  )
}

function ResultPanel({ title, result }: { title: string; result: PredictionResponse | null }) {
  const probability = result?.probability ?? 0

  return (
    <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 1.5 }}>
      <Typography variant="caption" color="text.secondary">{title}</Typography>
      <Typography variant="h4" sx={{ mt: 0.5, fontWeight: 700, lineHeight: 1 }}>
        {result ? formatRisk(probability) : '-'}
      </Typography>
      <Box sx={{ mt: 1 }}>
        <LinearProgress variant="determinate" value={Math.min(100, probability * 100)} sx={{ height: 7, borderRadius: 1 }} />
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.75, display: 'block' }}>
        {result ? 'Predicted probability of stroke' : 'Run prediction'}
      </Typography>
    </Box>
  )
}

function ShapContributionRows({
  contributions,
  featureByKey,
  maxContribution,
}: {
  contributions: ShapFeatureContribution[]
  featureByKey: Map<string, LabFeature>
  maxContribution: number
}) {
  if (!contributions.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No matching SHAP drivers in this direction.
      </Typography>
    )
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {contributions.map((item) => {
        const feature = featureByKey.get(item.feature) ?? makeFeature(item.feature)
        const width = `${Math.max(4, Math.min(100, item.absContribution / maxContribution * 100))}%`
        const isIncrease = item.contribution >= 0

        return (
          <Box key={item.feature} sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'minmax(160px, 1fr) 1fr 76px' }, gap: 1, alignItems: 'center' }}>
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, lineHeight: 1.15 }}>{feature.label}</Typography>
              <Typography variant="caption" color="text.secondary">
                {featureText(feature, item.value)}
              </Typography>
            </Box>
            <Box sx={{ height: 8, borderRadius: 1, bgcolor: 'grey.100', overflow: 'hidden' }}>
              <Box sx={{ height: '100%', width, bgcolor: isIncrease ? 'error.main' : 'success.main' }} />
            </Box>
            <Typography
              variant="caption"
              color={isIncrease ? 'error.dark' : 'success.dark'}
              sx={{ fontWeight: 700, textAlign: { xs: 'left', sm: 'right' }, fontVariantNumeric: 'tabular-nums' }}
            >
              {formatSignedProbabilityPoints(item.contribution)}
            </Typography>
          </Box>
        )
      })}
    </Box>
  )
}

function ShapExplanationPanel({
  title,
  result,
  featureByKey,
}: {
  title: string
  result: PredictionResponse | null
  featureByKey: Map<string, LabFeature>
}) {
  const explanation = result?.explanation ?? null

  if (!result) {
    return (
      <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 1.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{title}</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75 }}>
          Run prediction to calculate local SHAP values.
        </Typography>
      </Box>
    )
  }

  if (!explanation) {
    return (
      <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 1.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{title}</Typography>
        <Typography variant="body2" color="error" sx={{ mt: 0.75 }}>
          {result.explanationError ?? 'No SHAP explanation was returned for this prediction.'}
        </Typography>
      </Box>
    )
  }

  const increasing = explanation.features
    .filter((feature) => feature.contribution > 0)
    .slice(0, 5)
  const decreasing = explanation.features
    .filter((feature) => feature.contribution < 0)
    .slice(0, 5)
  const maxContribution = Math.max(
    ...[...increasing, ...decreasing].map((feature) => feature.absContribution),
    0.0001,
  )

  return (
    <Box sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 2, p: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap', mb: 1.25 }}>
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{title}</Typography>
          <Typography variant="caption" color="text.secondary">
            Baseline {formatRisk(explanation.baseValue)} to prediction {formatRisk(explanation.outputValue)}
          </Typography>
        </Box>
        <Chip
          size="small"
          label={explanation.backgroundRows ? `${explanation.backgroundRows} background rows` : 'Tree path SHAP'}
          sx={{ alignSelf: 'flex-start' }}
        />
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' }, gap: 2 }}>
        <Box>
          <Typography variant="caption" color="error.dark" sx={{ display: 'block', mb: 0.75, fontWeight: 700, textTransform: 'uppercase' }}>
            Pushes risk up
          </Typography>
          <ShapContributionRows contributions={increasing} featureByKey={featureByKey} maxContribution={maxContribution} />
        </Box>
        <Box>
          <Typography variant="caption" color="success.dark" sx={{ display: 'block', mb: 0.75, fontWeight: 700, textTransform: 'uppercase' }}>
            Pulls risk down
          </Typography>
          <ShapContributionRows contributions={decreasing} featureByKey={featureByKey} maxContribution={maxContribution} />
        </Box>
      </Box>
    </Box>
  )
}

const gridSx = {
  border: 0,
  '& .MuiDataGrid-columnHeaders': {
    position: 'sticky',
    top: 0,
    zIndex: 1,
    bgcolor: 'grey.50',
    borderBottom: '1px solid',
    borderColor: 'divider',
  },
  '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 700 },
  '& .MuiDataGrid-row:hover': { bgcolor: 'action.hover' },
  '& .MuiDataGrid-cell': { whiteSpace: 'nowrap', py: 0.5 },
  '& .MuiDataGrid-columnHeader': { px: 1 },
}

function Predict() {
  const [activeStep, setActiveStep] = useState(0)
  const [selectedDataset, setSelectedDataset] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [selectedPatient, setSelectedPatient] = useState<PatientSelection | null>(null)
  const [selectedFeatureKeys, setSelectedFeatureKeys] = useState<string[]>([])
  const [scenarioValues, setScenarioValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<LabResult | null>(null)
  const [predictError, setPredictError] = useState('')
  const [isPredicting, setIsPredicting] = useState(false)

  const registryQuery = useQuery({
    queryKey: ['registry'],
    queryFn: () => fetchJson<RegistryItem[]>('/api/registry'),
  })

  const datasetLabels = useMemo(
    () => Object.fromEntries(
      registryQuery.data
        ?.filter((item) => item.type === 'dataset')
        .map((item) => [item.id, item.label]) ?? [],
    ),
    [registryQuery.data],
  )

  const modelQuery = useQuery({
    queryKey: ['model', selectedModelId],
    queryFn: () => fetchModelDetail(selectedModelId),
    enabled: !!selectedModelId,
    staleTime: 30_000,
  })

  const activePatient = selectedPatient?.row ?? null
  const activeModel = modelQuery.data ?? null
  const selectedModelSummary = activeModel
  const allFeatures = useMemo(
    () => activeModel?.featureColumns.map(makeFeature) ?? [],
    [activeModel?.featureColumns],
  )
  const featureByKey = useMemo(
    () => new Map(allFeatures.map((feature) => [feature.key, feature] as const)),
    [allFeatures],
  )
  const visibleFeatureKeys = useMemo(() => {
    return selectedFeatureKeys.filter((key) => activeModel?.featureColumns.includes(key))
  }, [activeModel?.featureColumns, selectedFeatureKeys])

  const activeFeatures = useMemo(
    () => visibleFeatureKeys.map(makeFeature),
    [visibleFeatureKeys],
  )

  const baselineValues = useMemo(() => {
    if (!activePatient) return {}
    return Object.fromEntries((activeModel?.featureColumns ?? []).map((key) => [key, parseModelValue(activePatient[key])]))
  }, [activeModel?.featureColumns, activePatient])

  const importanceByFeature = useMemo(() => {
    const total = activeModel?.featureImportances.reduce((sum, feature) => sum + feature.importance, 0) ?? 0
    return Object.fromEntries(
      activeModel?.featureImportances.map((feature) => [
        feature.feature,
        total > 0 ? feature.importance / total : 0,
      ]) ?? [],
    )
  }, [activeModel?.featureImportances])

  const handleScenarioChange = (feature: LabFeature, value: string) => {
    setScenarioValues((current) => ({ ...current, [feature.key]: value }))
    setResult(null)
  }

  const resetComparison = () => {
    setScenarioValues({})
    setResult(null)
    setPredictError('')
  }

  const handlePredict = async () => {
    if (!activePatient || !activeModel) return

    setIsPredicting(true)
    setPredictError('')

    try {
      const baselineFeatures = buildFeatures(activePatient, activeModel)
      const scenarioFeatures = { ...baselineFeatures }

      for (const feature of activeFeatures) {
        const enteredValue = scenarioValues[feature.key] ?? ''
        if (enteredValue.trim() !== '') {
          scenarioFeatures[feature.key] = modelValue(feature, enteredValue)
        }
      }

      const changedFeatures = activeFeatures.filter((feature) => !sameModelValue(baselineFeatures[feature.key], scenarioFeatures[feature.key]))
      const predictUrl = `/api/models/${activeModel.id}/predict`
      const predictionResults = await Promise.allSettled([
        postJson<PredictionResponse>(predictUrl, { features: baselineFeatures, explain: true }),
        postJson<PredictionResponse>(predictUrl, { features: scenarioFeatures, explain: true }),
        ...changedFeatures.map((feature) => postJson<PredictionResponse>(predictUrl, {
          features: { ...baselineFeatures, [feature.key]: scenarioFeatures[feature.key] },
        })),
      ])

      const [baselineResult, scenarioResult, ...singleResults] = predictionResults
      if (baselineResult.status === 'rejected' || scenarioResult.status === 'rejected') {
        throw new Error('Core prediction failed')
      }

      const baseline = baselineResult.value
      const scenario = scenarioResult.value

      const deltas = changedFeatures
        .map((feature, index) => {
          const singleResult = singleResults[index]
          if (singleResult?.status !== 'fulfilled') return null

          const probability = singleResult.value.probability
          const reduction = baseline.probability - probability
          return {
            key: feature.key,
            label: feature.label,
            before: baselineFeatures[feature.key],
            after: scenarioFeatures[feature.key],
            probability,
            reduction,
            relativeReduction: baseline.probability > 0 ? reduction / baseline.probability : 0,
            globalImportance: importanceByFeature[feature.key] ?? 0,
          }
        })
        .filter((delta): delta is NonNullable<typeof delta> => delta !== null)
        .sort((left, right) => Math.abs(right.reduction) - Math.abs(left.reduction))

      setResult({ baseline, scenario, deltas })

      if (selectedPatient) {
        const logPayload: PredictionLogPayload = {
          modelId: activeModel.id,
          patient: {
            datasetId: selectedPatient.datasetId,
            datasetLabel: selectedPatient.datasetLabel,
            rowId: selectedPatient.rowId,
            rowIndex: selectedPatient.rowIndex,
            absoluteIndex: selectedPatient.absoluteIndex,
            page: selectedPatient.page,
            perPage: selectedPatient.perPage,
            row: selectedPatient.row,
          },
          selectedFeatures: activeFeatures.map((feature) => ({ key: feature.key, label: feature.label })),
          baselineFeatures,
          scenarioFeatures,
          baseline,
          scenario,
          changedFeatures: deltas.map((delta) => ({
            key: delta.key,
            label: delta.label,
            before: delta.before,
            after: delta.after,
          })),
        }

        void postJson<{ ok: boolean }>('/api/predictions/log', logPayload).catch(() => undefined)
      }
    } catch {
      setPredictError('Prediction failed. Check that the backend is running and the entered values are valid.')
    } finally {
      setIsPredicting(false)
    }
  }

  const totalReduction = result ? result.baseline.probability - result.scenario.probability : 0
  const strongestChange = result?.deltas[0] ?? null
  const selectedPatientNumber = selectedPatient?.absoluteIndex ?? null
  const loading = modelQuery.isLoading || registryQuery.isLoading || isPredicting
  const error = modelQuery.isError || registryQuery.isError || !!predictError
  const steps = ['Model', 'Patient', 'Features', 'Summary']
  const featureSelectionModel: GridRowSelectionModel = {
    type: 'include',
    ids: new Set(visibleFeatureKeys),
  }
  const featureGridRows = useMemo(
    () => allFeatures.map((feature) => ({
      id: feature.key,
      feature: feature.label,
      key: feature.key,
      initialValue: featureText(feature, baselineValues[feature.key] ?? null),
      importance: importanceByFeature[feature.key] ?? 0,
    })),
    [allFeatures, baselineValues, importanceByFeature],
  )
  const featureColumns = useMemo<GridColDef[]>(() => [
    { field: 'feature', headerName: 'Feature', flex: 1.3, minWidth: 170 },
    { field: 'key', headerName: 'Code', flex: 1, minWidth: 130 },
    { field: 'initialValue', headerName: 'Initial value', flex: 1, minWidth: 140 },
    {
      field: 'importance',
      headerName: 'Importance',
      flex: 0.8,
      minWidth: 120,
      valueFormatter: (value) => pct(value as number),
    },
  ], [])

  const selectModel = (model: ModelRow) => {
    setSelectedModelId(model.id)
    setSelectedDataset('')
    setSelectedPatient(null)
    setSelectedFeatureKeys([])
    setScenarioValues({})
    setResult(null)
    setPredictError('')
  }

  const modelTable = (
    <ModelComparison embedded mode="select" selectedModelId={selectedModelId} onModelSelect={selectModel} />
  )

  const patientTable = (
    <Patients
      embedded
      selectable
      selectedRowId={selectedPatient?.rowId ?? null}
      onSelectionChange={(selection) => {
        setSelectedPatient(selection)
        setSelectedDataset(selection ? selection.datasetId : '')
        resetComparison()
      }}
    />
  )

  const featureTable = (
    <SectionCard title="Select Features">
      <DataGrid
        rows={featureGridRows}
        columns={featureColumns}
        density="compact"
        autoHeight
        hideFooter
        checkboxSelection
        rowSelectionModel={featureSelectionModel}
        onRowSelectionModelChange={(selection) => {
          const nextKeys = Array.from(selection.ids).map(String)
          setSelectedFeatureKeys(nextKeys)
          resetComparison()
        }}
        sx={gridSx}
      />
    </SectionCard>
  )

  const summary = (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '1.2fr 0.8fr' }, gap: 2.5, alignItems: 'start' }}>
      <SectionCard title="Summary And Scenario">
        <Box sx={{ mb: 2, display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr 1fr' }, gap: 1.5 }}>
          <Box>
            <Typography variant="caption" color="text.secondary">Model</Typography>
            <Typography variant="body2" sx={{ fontWeight: 700 }}>{selectedModelSummary ? modelLabel(selectedModelSummary) : '-'}</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">Dataset</Typography>
            <Typography variant="body2" sx={{ fontWeight: 700 }}>{selectedDataset ? datasetLabels[selectedDataset] ?? selectedDataset : '-'}</Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">Patient</Typography>
            <Typography variant="body2" sx={{ fontWeight: 700 }}>{selectedPatientNumber ? `Patient ${selectedPatientNumber}` : '-'}</Typography>
          </Box>
        </Box>
        <Stack direction="row" spacing={1} useFlexGap sx={{ mb: 2, flexWrap: 'wrap' }}>
          {activeFeatures.map((feature) => (
            <Chip key={feature.key} label={feature.label} size="small" />
          ))}
        </Stack>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Prediction sends all {activeModel?.featureColumns.length ?? 0} model features. Empty changed values keep the selected patient's original value.
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2 }}>
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 1.25, fontWeight: 700 }}>Initial values</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
              {activeFeatures.map((feature) => (
                <TextField key={feature.key} label={feature.label} value={featureText(feature, baselineValues[feature.key] ?? null)} size="small" slotProps={{ input: { readOnly: true } }} helperText={feature.helper} />
              ))}
            </Box>
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 1.25, fontWeight: 700 }}>Changed values</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
              {activeFeatures.map((feature) => (
                feature.kind === 'select' ? (
                  <TextField key={feature.key} select label={feature.label} value={scenarioValues[feature.key] ?? ''} onChange={(event) => handleScenarioChange(feature, event.target.value)} size="small" helperText={feature.helper}>
                    <MenuItem value="">
                      <em>No change</em>
                    </MenuItem>
                    {feature.options?.map((option) => (
                      <MenuItem key={option.value} value={String(option.value)}>{option.label}</MenuItem>
                    ))}
                  </TextField>
                ) : (
                  <TextField key={feature.key} label={feature.label} type="number" value={scenarioValues[feature.key] ?? ''} onChange={(event) => handleScenarioChange(feature, event.target.value)} size="small" helperText={feature.helper} slotProps={{ htmlInput: { step: feature.key === '_BMI5' ? 0.1 : 1 } }} />
                )
              ))}
            </Box>
          </Box>
        </Box>
      </SectionCard>

      <SectionCard title="Predicted Stroke Risk">
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          <Button variant="contained" startIcon={<ScienceIcon />} disabled={!activePatient || !activeModel || isPredicting} onClick={handlePredict}>Predict</Button>
          <ResultPanel title="Initial patient" result={result?.baseline ?? null} />
          <ResultPanel title="Changed scenario" result={result?.scenario ?? null} />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CompareArrowsIcon color={totalReduction >= 0 ? 'success' : 'error'} />
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                {result ? `${formatDelta(totalReduction)} total change` : 'No comparison yet'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {strongestChange
                  ? `${strongestChange.label} ${strongestChange.reduction >= 0 ? 'reduced' : 'increased'} predicted stroke risk the most.`
                  : 'Run prediction to identify the strongest feature change.'}
              </Typography>
            </Box>
          </Box>
        </Box>
      </SectionCard>
    </Box>
  )

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, p: 2 }}>
          <Stepper activeStep={activeStep} alternativeLabel>
            {steps.map((step, index) => (
              <Step key={step}>
                <StepButton onClick={() => setActiveStep(index)}>
                  {step}
                </StepButton>
              </Step>
            ))}
          </Stepper>
          <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1 }}>
            <Chip size="small" label={selectedModelSummary ? modelLabel(selectedModelSummary) : 'No model selected'} />
            <Chip size="small" label={activePatient ? `Patient ${selectedPatientNumber}` : 'No patient selected'} />
            <Chip size="small" label={`${activeFeatures.length} features selected`} />
          </Box>
          {loading && <LinearProgress sx={{ mt: 1.5, borderRadius: 1 }} />}
          {error && (
            <Typography variant="body2" color="error" sx={{ mt: 1.25 }}>
              {predictError || 'Could not load Prediction Lab data from the backend.'}
            </Typography>
          )}
        </Paper>

        {activeStep === 0 && modelTable}
        {activeStep === 1 && patientTable}
        {activeStep === 2 && featureTable}
        {activeStep === 3 && summary}

        {activeStep === 3 && (
          <SectionCard title="SHAP Local Explanation">
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '1fr 1fr' }, gap: 2 }}>
              <ShapExplanationPanel title="Initial patient" result={result?.baseline ?? null} featureByKey={featureByKey} />
              <ShapExplanationPanel title="Changed scenario" result={result?.scenario ?? null} featureByKey={featureByKey} />
            </Box>
          </SectionCard>
        )}

        {activeStep === 3 && (
          <SectionCard title="Feature-Level Risk Changes">
            {result?.deltas.length ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {result.deltas.map((delta) => {
                  const maxReduction = Math.max(...result.deltas.map((item) => Math.abs(item.reduction)), 0.0001)
                  const width = `${Math.min(100, Math.abs(delta.reduction) / maxReduction * 100)}%`
                  return (
                    <Box key={delta.key} sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.1fr 1fr 0.9fr' }, gap: 1.5, alignItems: 'center', borderBottom: '1px solid', borderColor: 'divider', py: 1, '&:last-child': { borderBottom: 0 } }}>
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{delta.label}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {featureText(featureByKey.get(delta.key) ?? makeFeature(delta.key), delta.before)} → {featureText(featureByKey.get(delta.key) ?? makeFeature(delta.key), delta.after)}
                        </Typography>
                      </Box>
                      <Box>
                        <Box sx={{ height: 8, borderRadius: 1, bgcolor: 'grey.100', overflow: 'hidden' }}>
                          <Box sx={{ height: '100%', width, bgcolor: delta.reduction >= 0 ? 'success.main' : 'error.main' }} />
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          Single-change risk: {formatRisk(delta.probability)}
                        </Typography>
                      </Box>
                      <Box sx={{ textAlign: { xs: 'left', md: 'right' } }}>
                        <Typography variant="subtitle2" color={delta.reduction >= 0 ? 'success.dark' : 'error.dark'} sx={{ fontWeight: 700 }}>
                          {formatDelta(delta.reduction)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {result.baseline.probability > 0
                            ? `${pct(Math.abs(delta.relativeReduction))} relative ${delta.reduction >= 0 ? 'decrease' : 'increase'}`
                            : 'N/A relative'}, {pct(delta.globalImportance)} model importance
                        </Typography>
                      </Box>
                    </Box>
                  )
                })}
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Change at least one selected feature and run prediction to see exact per-feature risk changes.
              </Typography>
            )}
          </SectionCard>
        )}

        <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 3, p: 1.5, display: 'flex', justifyContent: 'space-between', gap: 1 }}>
          <Button variant="outlined" disabled={activeStep === 0} onClick={() => setActiveStep((step) => Math.max(0, step - 1))}>Back</Button>
          <Button
            variant="contained"
            disabled={(activeStep === 0 && !selectedModelId) || (activeStep === 1 && !activePatient) || (activeStep === 2 && activeFeatures.length === 0) || activeStep === 3}
            onClick={() => setActiveStep((step) => Math.min(3, step + 1))}
          >
            Next
          </Button>
        </Paper>
      </Box>
    </main>
  )
}

export default Predict
