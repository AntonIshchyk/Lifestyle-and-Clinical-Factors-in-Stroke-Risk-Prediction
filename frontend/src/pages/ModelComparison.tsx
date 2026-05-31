import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import CompareArrowsIcon from '@mui/icons-material/CompareArrows'
import { DataGrid, type GridColDef, type GridRowParams, type GridRowSelectionModel } from '@mui/x-data-grid'
import { useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api'
import { ALGORITHM_LABELS, FEATURE_SET_LABELS, type Algorithm, type FeatureSet } from '../modelMetadata'

type ModelRow = {
  id: string
  algorithm: Algorithm
  featureSet: FeatureSet
  auc: number
  accuracy: number
  f1: number
  precision: number
  recall: number
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
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

function ModelComparison() {
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

  const handleCompare = () => {
    if (!canCompare) return
    navigate(`/models/compare?ids=${encodeURIComponent(selectedIds.join(','))}`)
  }

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Paper
          elevation={0}
          sx={{ borderRadius: 3, border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}
        >
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
              Select models to compare, or click a row to view details
            </Typography>
            <Button
              variant="contained"
              size="small"
              startIcon={<CompareArrowsIcon />}
              disabled={!canCompare}
              onClick={handleCompare}
            >
              Compare selected ({selectedIds.length})
            </Button>
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
              checkboxSelection
              disableRowSelectionOnClick
              disableRowSelectionExcludeModel
              rowSelectionModel={selectionModel}
              onRowSelectionModelChange={setSelectionModel}
              onRowClick={(params: GridRowParams) => navigate(`/models/${params.row.id}`)}
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
      </Box>
    </main>
  )
}

export default ModelComparison
