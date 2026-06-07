export type Algorithm = 'random_forest' | 'xgboost' | 'lightgbm'
export type FeatureSet = 'lifestyle' | 'clinical' | 'combined'
export type UncertaintyVariant = 'with_uncertain' | 'without_uncertain'

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

export const UNCERTAINTY_VARIANT_LABELS: Record<UncertaintyVariant, string> = {
  with_uncertain: 'With uncertain features',
  without_uncertain: 'Without uncertain features',
}
