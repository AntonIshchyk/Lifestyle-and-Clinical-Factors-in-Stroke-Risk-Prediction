import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import { DataGrid, type GridColDef, type GridRowParams, type GridRowSelectionModel } from '@mui/x-data-grid'
import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api'
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

type ModelComparisonProps = {
  embedded?: boolean
  mode?: 'compare' | 'select'
  selectedModelId?: string
  onModelSelect?: (model: ModelRow) => void
}

const EMPTY_MODELS: ModelRow[] = []

async function fetchModels(): Promise<ModelRow[]> {
  return fetchJson<ModelRow[]>('/api/models')
}

function useColumns(): GridColDef[] {
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
      renderCell: ({ value }) => <Typography variant="body2">{BALANCING_METHOD_LABELS[value as BalancingMethod]}</Typography>
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
    {
      field: 'classificationThreshold',
      headerName: 'Threshold',
      flex: 0.8,
      minWidth: 90,
      sortable: true,
      align: 'left',
      headerAlign: 'left',
      valueFormatter: (value) => (value as number).toFixed(2),
    },
  ], [])
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
}: {
  title: string
  subtitle?: string
  rows: ModelRow[]
  columns: GridColDef[]
  loading: boolean
  selectMode: boolean
  rowSelectionModel: GridRowSelectionModel
  onSelectionChange: (rows: ModelRow[], model: GridRowSelectionModel) => void
  onRowClick: (params: GridRowParams) => void
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
          {subtitle && <Typography variant="caption" color="text.secondary">{subtitle}</Typography>}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
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
  const [selectionModel, setSelectionModel] = useState<GridRowSelectionModel>({ type: 'include', ids: new Set() })

  const query = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 30_000,
  })

  const models = query.data ?? EMPTY_MODELS
  const normalModels = useMemo(() => models.filter((model) => !model.isTuned), [models])
  const loading = query.isLoading || query.isFetching
  const error = query.isError ? 'Could not load models from the backend.' : ''
  const selectedIds = useMemo(
    () => (selectionModel.type === 'include' ? [...selectionModel.ids].map(String) : []),
    [selectionModel],
  )
  const canCompare = selectedIds.length >= 2
  const selectMode = mode === 'select'
  const columns = useColumns()
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
        <ModelListSection
          title="Models"
          rows={normalModels}
          columns={columns}
          loading={loading}
          selectMode={selectMode}
          rowSelectionModel={rowSelectionModel}
          onSelectionChange={handleSectionSelectionChange}
          onRowClick={handleRowClick}
        />
      )}
    </Box>
  )

  if (embedded) return content

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        {content}
      </Box>
    </main>
  )
}

export default ModelComparison
