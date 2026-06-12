# Analiza czynników ryzyka choroby Alzheimera

## Dane

**Dataset:** [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) (Kaggle)

- Liczba pacjentów: 2149
- Liczba kolumn oryginalnych: 35
- Po wyczyszczeniu: 32 cechy predykcyjne + zmienna docelowa `Diagnosis`
- Usunięte kolumny: `PatientID` (data leakage) i `DoctorInCharge` (jedna wartość `XXXConfid`, brak wartości predykcyjnej)
- Rozkład klas: 64,7% bez choroby / 35,3% z diagnozą Alzheimera

Dane obejmują informacje demograficzne (wiek, płeć, wykształcenie), styl życia (BMI, aktywność fizyczna, dieta, sen), historię medyczną (choroby współistniejące), wyniki testów kognitywnych (MMSE, FunctionalAssessment, ADL) oraz objawy behawioralne.

## Cel

Klasyfikacja binarna — przewidywanie diagnozy Alzheimera na podstawie danych klinicznych i demograficznych. Projekt weryfikuje cztery hipotezy badawcze:

| Hipoteza | Treść | Wynik |
|----------|-------|-------|
| H1 | Cechy lifestyle i medyczne są lepszym predyktorem niż demograficzne (macro F1 wyższy o >= 0.08) | Obalona |
| H2 | Usunięcie testów kognitywnych (MMSE, ADL, FA) obniży macro F1 o >= 0.10 | Potwierdzona (spadek 0.2794) |
| H3 | Same symptomy behawioralne wystarczą do klasyfikacji z F1 klasy Alzheimer >= 0.75 | Obalona (F1 klasy AD = 0.5828) |
| H4 | XGBoost po RandomizedSearchCV pobije najlepszy model klasyczny o >= 0.05 | Częściowo potwierdzona (różnica 0.0108) |

## Wyniki

Podział: 80% trening / 20% test (stratified, `random_state=67`). Miara: macro F1-score.

| Rank | Model | Cechy | Macro F1 |
|------|-------|-------|----------|
| 1 | XGBoost (RandomizedSearchCV, 50 iter.) | 32 wszystkie | 0.9341 |
| 2 | Random Forest (GridSearchCV) | 32 wszystkie | 0.9232 |
| 3 | Decision Tree (GridSearchCV, symptomy, H3) | 7 symptomów | 0.6785 |
| 4 | XGBoost bez cech kognitywnych (H4b) | 29 bez kogn. | 0.6489 |
| 5 | Random Forest bez cech kognitywnych (H2) | 29 bez kogn. | 0.6438 |
| 6 | Logistic Regression (demografia, H1) | 4 demogr. | 0.5243 |
| 7 | Logistic Regression (lifestyle+med, H1) | 12 life+med | 0.5067 |
| 8 | Random Forest (demografia, H1) | 4 demogr. | 0.4765 |
| 9 | Random Forest (lifestyle+med, H1) | 12 life+med | 0.4313 |

Cechy kognitywne (MMSE, FunctionalAssessment, ADL) odpowiadają łącznie za 52,7% feature importance w Random Forest. Bez nich macro F1 spada z 0.9232 do 0.6438 (spadek 0.2794). XGBoost bez cech kognitywnych (0.6489) osiąga niemal identyczny wynik co RF bez nich (0.6438) — żaden algorytm nie rekompensuje braku tych danych.

## Modele

- **Logistic Regression** — Pipeline (StandardScaler + LR), `class_weight='balanced'`, `max_iter=1000`
- **Decision Tree** — GridSearchCV (criterion, max_depth 2–None, min_samples_split, min_samples_leaf), 5-fold CV; najlepsze params: `criterion=gini`, `max_depth=5`, `min_samples_leaf=2`, `min_samples_split=2`
- **Random Forest** — GridSearchCV (18 kombinacji: n_estimators, max_depth, min_samples_split), 5-fold CV, `class_weight='balanced'`; najlepsze params: `max_depth=10`, `min_samples_split=5`, `n_estimators=200`
- **XGBoost** — RandomizedSearchCV (50 iteracji, 5-fold CV), `scale_pos_weight=1.83`; analiza SHAP; najlepszy CV macro F1 = 0.9467

