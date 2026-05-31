import { useState, useEffect, type ChangeEvent } from 'react'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { keepPreviousData, useQuery } from '@tanstack/react-query'

type PatientsResponse = {
  name: string
  columns: string[]
  rows: Record<string, string>[]
  page: number
  per_page: number
  total: number
}

type RegistryItem = {
  id: string
  type: string
  label: string
  reference: string
  created_at: string
}

function Patients() {
  const [selectedDataset, setSelectedDataset] = useState('')
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(10)
  
  const registryQuery = useQuery({
    queryKey: ['registry'],
    queryFn: async () => {
      const response = await fetch(`/api/registry`)
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`)
      return (await response.json()) as RegistryItem[]
    },
  })

  // Set the first dataset as default if available and none selected (or default is missing)
  const datasets = registryQuery.data?.filter(item => item.type === 'dataset') ?? []

  useEffect(() => {
    if (!selectedDataset && datasets.length > 0) {
      setSelectedDataset(datasets[0].id)
    }
  }, [datasets, selectedDataset])

  const query = useQuery({
    queryKey: ['data', selectedDataset, page, perPage],
    queryFn: async () => {
      if (!selectedDataset) return null;
      const params = new URLSearchParams({
        page: String(page),
        per_page: String(perPage),
      })

      const response = await fetch(`/api/data/${selectedDataset}?${params.toString()}`)

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      return (await response.json()) as PatientsResponse
    },
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    enabled: !!selectedDataset
  })

  const data = query.data ?? null
  const loading = query.isLoading || query.isFetching
  const error = query.isError ? 'Could not load data from the backend.' : ''

  const columns = data?.columns ?? []
  const gridColumns: GridColDef[] = columns.map((column) => ({
    field: column,
    headerName: column,
    flex: 1,
    minWidth: 160,
    sortable: true,
    type: column.includes('id') ? 'number' : 'string',
    align: 'left',
    headerAlign: 'left',
    valueGetter: (_, row) => (column.includes('id') ? Number(row[column]) : row[column] ?? '—'),
  }))

  const rows = data?.rows.map((row, idx) => ({
    id: idx, // since no primary key is guaranteed
    ...row,
  })) ?? []

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.per_page)) : 1

  const handleRowsChange = (event: ChangeEvent<HTMLInputElement>) => {
    setPage(1)
    setPerPage(Number(event.target.value))
  }

  const handleDatasetChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedDataset(event.target.value)
    setPage(1)
  }

  return (
    <main className="flex min-h-0 flex-1 flex-col px-4 py-4 sm:px-6 lg:px-8">
      <Box sx={{ display: 'flex', minHeight: 0, flex: 1, flexDirection: 'column', gap: 2.5 }}>
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
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, minWidth: 0 }}>
                  <Typography variant="body2" color="text.secondary">
                    Use the menu in each column header to filter and sort the data
                  </Typography>
                </Box>

                <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'flex-end' }}>
                  <TextField
                    select
                    label="Dataset"
                    value={selectedDataset}
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
                  disableRowSelectionOnClick
                  hideFooter
                  autoHeight
                  sx={{
                    border: 0,
                    '& .MuiDataGrid-columnHeaders': {
                      position: 'sticky', top: 0, zIndex: 1, bgcolor: 'grey.50',
                      borderBottom: '1px solid', borderColor: 'divider',
                    },
                    '& .MuiDataGrid-columnHeaderTitle': { fontWeight: 700 },
                    '& .MuiDataGrid-cell': { whiteSpace: 'nowrap', py: 0.5 },
                    '& .MuiDataGrid-columnHeader': { px: 1 },
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
                {data ? `Showing page ${data.page} of ${totalPages} • ${columns.length} columns • ${perPage} rows per page` : null}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button variant="outlined" disabled={page <= 1 || loading} onClick={() => setPage(p => Math.max(1, p - 1))}>Previous</Button>
                <Button variant="outlined" disabled={loading || (data ? page >= totalPages : true)} onClick={() => setPage(p => p + 1)}>Next</Button>
              </Box>
            </Box>
          </Box>
        </Paper>
      </Box>
    </main>
  )
}

export default Patients