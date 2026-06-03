export type Algorithm = 'random_forest' | 'xgboost' | 'lightgbm'
export type FeatureSet = 'lifestyle' | 'clinical' | 'combined'

export const ALGORITHM_LABELS: Record<Algorithm, string> = {
  random_forest: 'Random Forest',
  xgboost: 'XGBoost',
  lightgbm: 'LightGBM',
}

export const FEATURE_SET_LABELS: Record<FeatureSet, string> = {
  lifestyle: 'Lifestyle',
  clinical: 'Clinical',
  combined: 'Combined',
}
