import { useMemo, useState, type ChangeEvent } from 'react'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid, type GridColDef, type GridRowId, type GridRowParams, type GridRowSelectionModel } from '@mui/x-data-grid'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { fetchJson } from '../api'

export type PatientsResponse = {
  name: string
  columns: string[]
  rows: Record<string, string>[]
  page: number
  per_page: number
  total: number
}

export type RegistryItem = {
  id: string
  type: string
  label: string
  reference: string
  created_at: string
}

export type PatientSelection = {
  datasetId: string
  datasetLabel: string
  row: Record<string, string>
  rowId: string
  rowIndex: number
  absoluteIndex: number
  page: number
  perPage: number
}

type PatientsProps = {
  embedded?: boolean
  helperText?: string
  selectable?: boolean
  selectedRowId?: GridRowId | null
  onSelectionChange?: (selection: PatientSelection | null) => void
}

function Patients({
  embedded = false,
  helperText = 'Use the menu in each column header to filter and sort the data',
  selectable = false,
  selectedRowId = null,
  onSelectionChange,
}: PatientsProps) {
  const [selectedDataset, setSelectedDataset] = useState('')
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(10)

  const registryQuery = useQuery({
    queryKey: ['registry'],
    queryFn: () => fetchJson<RegistryItem[]>('/api/registry'),
  })

  const datasets = useMemo(
    () => registryQuery.data?.filter((item) => item.type === 'dataset') ?? [],
    [registryQuery.data],
  )
  const activeDataset = datasets.some((dataset) => dataset.id === selectedDataset)
    ? selectedDataset
    : datasets[0]?.id ?? ''
  const activeDatasetLabel = datasets.find((dataset) => dataset.id === activeDataset)?.label ?? activeDataset

  const query = useQuery({
    queryKey: ['data', activeDataset, page, perPage],
    queryFn: () => {
      const params = new URLSearchParams({
        page: String(page),
        per_page: String(perPage),
      })

      return fetchJson<PatientsResponse>(`/api/data/${activeDataset}?${params.toString()}`)
    },
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    enabled: !!activeDataset,
  })

  const data = query.data ?? null
  const loading = query.isLoading || query.isFetching
  const error = query.isError ? 'Could not load data from the backend.' : ''
  const columns = useMemo(() => data?.columns ?? [], [data?.columns])
  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.per_page)) : 1

  const gridColumns: GridColDef[] = useMemo(
    () => columns.map((column) => ({
      field: column,
      headerName: column,
      flex: 1,
      minWidth: 160,
      sortable: true,
      type: column.includes('id') ? 'number' as const : 'string' as const,
      align: 'left' as const,
      headerAlign: 'left' as const,
      valueGetter: (_: unknown, row: Record<string, string>) => (column.includes('id') ? Number(row[column]) : row[column] ?? '-'),
    })),
    [columns],
  )

  const rows = useMemo(
    () => data?.rows.map((row, idx) => ({
      id: `${activeDataset}-${page}-${idx}`,
      ...row,
    })) ?? [],
    [activeDataset, data?.rows, page],
  )

  const rowSelectionModel: GridRowSelectionModel = selectedRowId
    ? { type: 'include', ids: new Set([selectedRowId]) }
    : { type: 'include', ids: new Set() }

  const emitSelection = (params: GridRowParams) => {
    const rowIndex = rows.findIndex((row) => row.id === params.id)
    const sourceRow = rowIndex >= 0 ? data?.rows[rowIndex] : null
    if (!sourceRow) return

    onSelectionChange?.({
      datasetId: activeDataset,
      datasetLabel: activeDatasetLabel,
      row: sourceRow,
      rowId: String(params.id),
      rowIndex,
      absoluteIndex: (page - 1) * perPage + rowIndex + 1,
      page,
      perPage,
    })
  }

  const handleRowsChange = (event: ChangeEvent<HTMLInputElement>) => {
    setPage(1)
    setPerPage(Number(event.target.value))
    onSelectionChange?.(null)
  }

  const handleDatasetChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedDataset(event.target.value)
    setPage(1)
    onSelectionChange?.(null)
  }

  const handlePageChange = (nextPage: number) => {
    setPage(nextPage)
    onSelectionChange?.(null)
  }

  const content = (
    <Paper
      elevation={0}
      sx={{
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 3,
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ px: 2, pt: 2, pb: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
          <Typography variant="body2" color="text.secondary">
            {helperText}
          </Typography>

          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'flex-end' }}>
            <TextField
              select
              label="Dataset"
              value={activeDataset}
              onChange={handleDatasetChange}
              size="small"
              sx={{ minWidth: 200 }}
            >
              {datasets.map((option) => (
                <MenuItem key={option.id} value={option.id}>
                  {option.label}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              select
              label="Rows"
              value={perPage}
              onChange={handleRowsChange}
              size="small"
              sx={{ minWidth: 120 }}
            >
              {[10, 25, 50, 100].map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </TextField>
          </Box>
        </Box>
      </Box>

      <Box sx={{ width: '100%' }}>
        {error ? (
          <Box sx={{ px: 2, py: 4 }}>
            <Typography variant="body2" color="error">{error}</Typography>
          </Box>
        ) : loading ? (
          <Box sx={{ px: 2, py: 4 }}>
            <Typography variant="body2" color="text.secondary">Fetching rows...</Typography>
          </Box>
        ) : data && data.rows.length > 0 ? (
          <Box sx={{ width: '100%' }}>
            <DataGrid
              rows={rows}
              columns={gridColumns}
              density="compact"
              disableRowSelectionOnClick={!selectable}
              hideFooter
              autoHeight
              rowSelectionModel={selectable ? rowSelectionModel : undefined}
              onRowClick={selectable ? emitSelection : undefined}
              sx={{
                border: 0,
                '& .MuiDataGrid-columnHeaders': {
                  position: 'sticky', top: 0, zIndex: 1, bgcolor: 'grey.50',
                  borderBottom: '1px solid', borderColor: 'divider',
                },
                '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 700 },
                '& .MuiDataGrid-cell': { whiteSpace: 'nowrap', py: 0.5 },
                '& .MuiDataGrid-columnHeader': { px: 1 },
                ...(selectable ? { '& .MuiDataGrid-row': { cursor: 'pointer' } } : {}),
              }}
            />
          </Box>
        ) : (
          <Box sx={{ px: 2, py: 4 }}>
            <Typography variant="body2" color="text.secondary">No data found.</Typography>
          </Box>
        )}
      </Box>

      <Box sx={{ flexShrink: 0, px: 2, py: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 1.5 }}>
          <Typography variant="body2" color="text.secondary">
            {data ? `Showing page ${data.page} of ${totalPages} - ${columns.length} columns - ${perPage} rows per page` : null}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button variant="outlined" disabled={page <= 1 || loading} onClick={() => handlePageChange(Math.max(1, page - 1))}>Previous</Button>
            <Button variant="outlined" disabled={loading || (data ? page >= totalPages : true)} onClick={() => handlePageChange(page + 1)}>Next</Button>
          </Box>
        </Box>
      </Box>
    </Paper>
  )

  if (embedded) return content

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', minHeight: 0, flex: 1, flexDirection: 'column', gap: 2.5 }}>
        {content}
      </Box>
    </main>
  )
}

export default Patients
