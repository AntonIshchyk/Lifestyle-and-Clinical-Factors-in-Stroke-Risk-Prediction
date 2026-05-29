import { NavLink, Route, Routes } from 'react-router-dom'
import HomePage from './pages/HomePage'
import ModelComparisonPage from './pages/ModelComparisonPage'
import PredictPage from './pages/PredictPage'

function App() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    [
      'rounded-full px-5 py-2 text-sm font-medium transition',
      isActive
        ? 'bg-emerald-500 text-white shadow-sm shadow-emerald-200'
        : 'text-emerald-900 hover:bg-emerald-100',
    ].join(' ')

  return (
    <main className="flex min-h-screen flex-col bg-slate-50 text-slate-900">
        <div className="mx-auto flex w-full max-w-5xl flex-col items-center gap-4 px-6 py-5 sm:px-8">
          <nav className="flex flex-wrap justify-center gap-3">
            <NavLink to="/" className={linkClass} end>
              Patients
            </NavLink>
            <NavLink to="/predict" className={linkClass}>
              Prediction Lab
            </NavLink>
            <NavLink to="/models-comparison" className={linkClass}>
              Model Comparison
            </NavLink>
          </nav>
        </div>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/models-comparison" element={<ModelComparisonPage />} />
      </Routes>
    </main>
  )
}

export default App
