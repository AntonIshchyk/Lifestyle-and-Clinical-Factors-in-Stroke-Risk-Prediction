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
}

type ModelComparisonProps = {
  embedded?: boolean
  mode?: 'compare' | 'select'
  selectedModelId?: string
  onModelSelect?: (model: ModelRow) => void
}

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
  ], [])
}

function ModelOverview({ models }: { models: ModelRow[] }) {
  const summaries = useMemo(() => {
    const methods = [...new Set(models.map((model) => model.balancingMethod))]
      .sort((left, right) => {
        const order: BalancingMethod[] = ['random_oversampling', 'smote', 'smotenc', 'smote_tomek', 'weighted']
        return order.indexOf(left) - order.indexOf(right)
      })

    return methods.map((method) => {
      const methodModels = models.filter((model) => model.balancingMethod === method)
      const bestAuc = methodModels.reduce<ModelRow | null>(
        (best, model) => (!best || model.auc > best.auc ? model : best),
        null,
      )
      const bestF1 = methodModels.reduce<ModelRow | null>(
        (best, model) => (!best || model.f1 > best.f1 ? model : best),
        null,
      )

      return {
        method,
        label: BALANCING_METHOD_LABELS[method],
        count: methodModels.length,
        bestAuc,
        bestF1,
      }
    })
  }, [models])

  return (
    <Paper elevation={0} sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
      <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Model overview</Typography>
      </Box>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' }, gap: 0, bgcolor: 'grey.50' }}>
        {summaries.map((summary, index) => (
          <Box key={summary.method} sx={{ p: 2, bgcolor: 'background.paper', borderRight: { md: index < summaries.length - 1 ? '1px solid' : 0 }, borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1.5, mb: 1.5 }}>
              <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 700 }}>{summary.label}</Typography>
              <Chip label={`${summary.count} models`} size="small" sx={{ borderRadius: 1.5 }} />
            </Box>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 1.5 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">Best AUC-ROC</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  {summary.bestAuc ? `${summary.bestAuc.auc.toFixed(3)} - ${ALGORITHM_LABELS[summary.bestAuc.algorithm]}` : '-'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">Best F1</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>
                  {summary.bestF1 ? `${summary.bestF1.f1.toFixed(3)} - ${ALGORITHM_LABELS[summary.bestF1.algorithm]}` : '-'}
                </Typography>
              </Box>
            </Box>
          </Box>
        ))}
      </Box>
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

  const models = query.data ?? []
  const loading = query.isLoading || query.isFetching
  const error = query.isError ? 'Could not load models from the backend.' : ''
  const columns = useColumns()
  const selectedIds = useMemo(
    () => (selectionModel.type === 'include' ? [...selectionModel.ids].map(String) : []),
    [selectionModel],
  )
  const canCompare = selectedIds.length >= 2
  const selectMode = mode === 'select'
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

  const content = (
    <Paper elevation={0} sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
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
        <Box sx={{ px: 2, py: 4 }}>
          <Typography variant="body2" color="error">{error}</Typography>
        </Box>
      ) : (
        <DataGrid
          rows={models}
          columns={columns}
          loading={loading}
          density="compact"
          autoHeight
          hideFooter
          checkboxSelection={!selectMode}
          disableRowSelectionOnClick={!selectMode}
          disableRowSelectionExcludeModel
          rowSelectionModel={rowSelectionModel}
          onRowSelectionModelChange={selectMode ? undefined : setSelectionModel}
          onRowClick={handleRowClick}
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

  if (embedded) return content

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <ModelOverview models={models} />
        {content}
      </Box>
    </main>
  )
}

export default ModelComparison
