import { useState, type FormEvent } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'

type PatientsResponse = {
  columns: string[]
  patients: Record<string, string>[]
  page: number
  per_page: number
  total: number
}

const visibleColumnLimit = 8

function HomePage() {
  const [searchInput, setSearchInput] = useState('')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const query = useQuery({
    queryKey: ['patients', page, search],
    queryFn: async () => {
      const params = new URLSearchParams({
        page: String(page),
        per_page: '10',
      })

      if (search) {
        params.set('search', search)
      }

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
  const visibleColumns = columns.slice(0, visibleColumnLimit)
  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.per_page)) : 1

  const handleSearchSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setPage(1)
    setSearch(searchInput.trim())
  }

  const clearSearch = () => {
    setSearchInput('')
    setSearch('')
    setPage(1)
  }

  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-8 sm:px-8">
      <section className="space-y-5">
        <div>
          <p className="text-sm font-medium uppercase tracking-[0.25em] text-emerald-600">
            Patients
          </p>
          <h2 className="mt-2 text-3xl font-semibold text-slate-900">Patient explorer</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
            Live cohort preview from the backend.
          </p>
        </div>

        <div className="rounded-3xl border border-emerald-100 bg-emerald-50/40 p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <form className="flex flex-wrap items-center gap-3" onSubmit={handleSearchSubmit}>
              <input
                className="h-11 w-72 rounded-full border border-emerald-200 bg-white px-4 text-sm text-slate-900 outline-none transition focus:border-emerald-400"
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
                placeholder="Search patient ID"
              />
              <button
                className="h-11 rounded-full bg-emerald-500 px-5 text-sm font-medium text-white transition hover:bg-emerald-600"
                type="submit"
              >
                Search
              </button>
              <button
                className="h-11 rounded-full border border-emerald-200 bg-white px-5 text-sm font-medium text-emerald-900 transition hover:bg-emerald-50"
                type="button"
                onClick={clearSearch}
              >
                Reset
              </button>
            </form>

            <div className="text-sm text-slate-600">
                {data ? `${data.total.toLocaleString()} patients` : 'Loading patients...'}
            </div>
          </div>

          <div className="mt-5 overflow-hidden rounded-2xl border border-emerald-100 bg-white">
            <div className="border-b border-emerald-100 px-4 py-3 text-sm font-medium text-slate-900">
              {loading ? 'Loading patients...' : 'Patient table'}
            </div>

            {error ? (
              <div className="px-4 py-8 text-sm text-rose-600">{error}</div>
            ) : loading ? (
              <div className="px-4 py-8 text-sm text-slate-500">Fetching rows from the backend.</div>
            ) : data && data.patients.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-emerald-100 text-left text-sm">
                  <thead className="bg-emerald-50 text-emerald-900">
                    <tr>
                      {visibleColumns.map((column) => (
                        <th key={column} className="px-4 py-3 font-medium">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-emerald-50 bg-white text-slate-700">
                    {data.patients.map((patient) => (
                      <tr key={patient.patient_id} className="hover:bg-emerald-50/40">
                        {visibleColumns.map((column) => (
                          <td key={`${patient.patient_id}-${column}`} className="px-4 py-3">
                            {patient[column] || '—'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="px-4 py-8 text-sm text-slate-500">No patients found.</div>
            )}
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-600">
            <div>
              {data ? (
                <>
                  Showing page {data.page} of {totalPages}
                </>
              ) : null}
            </div>
            <div className="flex gap-2">
              <button
                className="rounded-full border border-emerald-200 bg-white px-4 py-2 font-medium text-emerald-900 disabled:opacity-50"
                type="button"
                disabled={page <= 1 || loading}
                onClick={() => setPage((currentPage) => Math.max(1, currentPage - 1))}
              >
                Previous
              </button>
              <button
                className="rounded-full border border-emerald-200 bg-white px-4 py-2 font-medium text-emerald-900 disabled:opacity-50"
                type="button"
                disabled={loading || (data ? page >= totalPages : true)}
                onClick={() => setPage((currentPage) => currentPage + 1)}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}

export default HomePage