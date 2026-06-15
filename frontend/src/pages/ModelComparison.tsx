import { useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import Paper from '@mui/material/Paper'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import DeleteOutlinedIcon from '@mui/icons-material/DeleteOutlined'
import { DataGrid, type GridColDef, type GridRenderCellParams, type GridRowParams, type GridRowSelectionModel } from '@mui/x-data-grid'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteJson, fetchJson } from '../api'
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
        {content}
      </Box>
    </main>
  )
}

export default ModelComparison
