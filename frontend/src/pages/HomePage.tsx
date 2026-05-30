import { useState, type ChangeEvent } from 'react'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { keepPreviousData, useQuery } from '@tanstack/react-query'

type PatientsResponse = {
  columns: string[]
  patients: Record<string, string>[]
  page: number
  per_page: number
  total: number
}

function HomePage() {
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(10)
  
  const query = useQuery({
    queryKey: ['patients', page, perPage],
    queryFn: async () => {
      const params = new URLSearchParams({
        page: String(page),
        per_page: String(perPage),
      })

      const response = await fetch(`/api/patients?${params.toString()}`)

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      return (await response.json()) as PatientsResponse
    },
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  })

  const data = query.data ?? null
  const loading = query.isLoading || query.isFetching
  const error = query.isError ? 'Could not load patients from the backend.' : ''

  const columns = data?.columns ?? []
  const gridColumns: GridColDef[] = columns.map((column) => ({
    field: column,
    headerName: column,
    flex: 1,
    minWidth: 160,
    sortable: true,
    type: column === 'patient_id' ? 'number' : 'string',
    align: 'left',
    headerAlign: 'left',
    valueGetter: (_, row) => (column === 'patient_id' ? Number(row[column]) : row[column] ?? '—'),
  }))

  const rows = data?.patients.map((patient) => ({
    id: patient.patient_id,
    ...patient,
  })) ?? []

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.per_page)) : 1

  const handleRowsChange = (event: ChangeEvent<HTMLInputElement>) => {
    setPage(1)
    setPerPage(Number(event.target.value))
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
                  Use the menu in each column header to filter and sort the data.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Study based on:{' '}
                  <Link href="https://www.cdc.gov/brfss/annual_data/annual_2024.html" target="_blank" rel="noreferrer">
                    CDC - 2024 BRFSS Survey Data and Documentation
                  </Link>
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'flex-end' }}>
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
                <Typography variant="body2" color="error">
                  {error}
                </Typography>
              </Box>
            ) : loading ? (
              <Box sx={{ px: 2, py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  Fetching rows from the backend.
                </Typography>
              </Box>
            ) : data && data.patients.length > 0 ? (
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
                      position: 'sticky',
                      top: 0,
                      zIndex: 1,
                      bgcolor: 'grey.50',
                      borderBottom: '1px solid',
                      borderColor: 'divider',
                    },
                    '& .MuiDataGrid-columnHeaderTitle': {
                      fontWeight: 700,
                    },
                    '& .MuiDataGrid-cell': {
                      whiteSpace: 'nowrap',
                      py: 0.5,
                    },
                    '& .MuiDataGrid-columnHeader': {
                      px: 1,
                    },
                  }}
                />
              </Box>
            ) : (
              <Box sx={{ px: 2, py: 4 }}>
                <Typography variant="body2" color="text.secondary">
                  No patients found.
                </Typography>
              </Box>
            )}
          </Box>

          <Box
            sx={{
              flexShrink: 0,
              px: 2,
              py: 1.5,
              borderTop: '1px solid',
              borderColor: 'divider',
            }}
          >
            <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 1.5 }}>
              <Typography variant="body2" color="text.secondary">
                {data ? `Showing page ${data.page} of ${totalPages} • ${perPage} rows per page` : null}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  type="button"
                  disabled={page <= 1 || loading}
                  onClick={() => setPage((currentPage) => Math.max(1, currentPage - 1))}
                >
                  Previous
                </Button>
                <Button
                  variant="outlined"
                  type="button"
                  disabled={loading || (data ? page >= totalPages : true)}
                  onClick={() => setPage((currentPage) => currentPage + 1)}
                >
                  Next
                </Button>
              </Box>
            </Box>
          </Box>
        </Paper>
      </Box>
    </main>
  )
}

export default HomePage