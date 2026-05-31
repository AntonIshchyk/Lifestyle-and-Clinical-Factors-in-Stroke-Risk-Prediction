import Link from '@mui/material/Link'
import { NavLink, Route, Routes } from 'react-router-dom'
import ModelDetail from './pages/ModelDetail'
import ModelComparison from './pages/ModelComparison'
import Predict from './pages/Predict'
import Patients from './pages/Patients'

function App() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    [
      'rounded-full px-5 py-2 text-sm font-medium transition-colors duration-150',
      isActive
        ? 'bg-blue-600 text-white shadow-sm shadow-blue-200'
        : 'text-blue-900 hover:bg-blue-50',
    ].join(' ')

  return (
    <main className="flex min-h-screen flex-col bg-slate-50 text-slate-900">
      <div className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex w-full max-w-5xl items-center px-6 py-5 sm:px-8">
          <div className="flex-1" />
          <nav className="flex flex-wrap justify-center gap-3">
            <NavLink to="/" className={linkClass} end>
              Patients
            </NavLink>
            <NavLink to="/predict" className={linkClass}>
              Prediction Lab
            </NavLink>
            <NavLink to="/models" className={linkClass}>
              Model Comparison
            </NavLink>
          </nav>
          <div className="flex flex-1 justify-end">
            <Link href="https://www.cdc.gov/brfss/annual_data/annual_2024.html" target="_blank" rel="noreferrer" underline="hover" className="text-xs text-slate-500 whitespace-nowrap">
              Study based on: CDC BRFSS 2024
            </Link>
          </div>
        </div>
      </div>

      <Routes>
        <Route path="/" element={<Patients />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/models" element={<ModelComparison />} />
        <Route path="/models/:id" element={<ModelDetail />} />
      </Routes>
    </main>
  )
}

export default App
