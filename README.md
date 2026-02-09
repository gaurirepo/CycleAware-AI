This project explores whether including menstrual cycle phase as a feature improves the prediction of attention and memory scores using machine learning.
The core idea is to compare:
Cycle-aware models → trained with menstrual cycle phase
Cycle-agnostic models → trained without menstrual cycle phase
and evaluate whether cycle awareness provides statistically meaningful improvement in cognitive predictions.
Dataset
File used: menstrual_aware_dataset_with_user_id.csv
Key columns
Category
Columns
Physiological / behavioral inputs
sleep,stress, mood, activity
Cycle information
cycle phase
Targets
Attention score, memory score
Identifier
user_id

Modeling Strategy
1. Two Feature Configurations
Build two parallel learning pipelines to enable direct causal comparison of menstrual-cycle awareness.
With cycle phase
[sleep, stress, mood, activity, cycle_phase]
Without cycle phase
[sleep, stress, mood, activity]
2. Targets Predicted
Attention score
Memory score
Each target has:
Cycle-aware model
Cycle-agnostic model
Total models trained using Random Forest regressors
Why Random Forest?
Handles nonlinear biological patterns
Robust to noise
Works well with mixed feature types
Provides stable baseline for comparison


Train/Test Split
80% training
20% testing
random_state = 42
Split is performed once on indices so that:
All four models evaluate on the same users
Comparisons remain statistically fair
Outputs Generated
1. Prediction Comparison Table
For test users, the script prints: Quick human-readable validation of differences.
Actual attention & memory
Predictions with cycle
Predictions without cycle
Cycle phase for context
