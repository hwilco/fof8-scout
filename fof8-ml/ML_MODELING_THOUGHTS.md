# FOF8 Draft Analyzer: ML Pipeline & Modeling Blueprint

## 1. The Core Challenge: The Power Law & Scarcity
The fundamental challenge of the FOF8 dataset is an extreme class imbalance. Career outcomes follow a strict power law: only ~1% of drafted rookies (roughly 167 players in a 91-year span) achieve a `Career_Cap_Share` of >= 1.0.

Standard regression models will fail here, as they will optimize for the 99% of washouts and career backups to minimize overall error. The pipeline must be engineered to hunt for the elite tail.

## 2. Target Engineering & Model Evaluation
To prevent the model from playing it safe, we move away from standard accuracy and Mean Squared Error (MSE), shifting to asymmetric evaluation.

* **The Target Variable:** `Career_Cap_Share` (Total Career Earnings / Annual Salary Cap). This creates a normalized "Cap Years" metric that rewards both peak dominance and longevity.
* **Evaluation Metrics:**
    * **Precision-Recall (PR) Curves & Top-K Accuracy:** Focus on how many actual stars the model ranks in its top 50. ROC-AUC is misleading due to the massive number of easy-to-predict washouts.
    * **F-Beta Score:** Use a tunable F-measure. Set Beta based on the draft round (e.g., prioritize Precision in Round 1 to avoid costly busts; prioritize Recall in Round 7 to hunt for high-variance upside).
    * **Asymmetric Loss:** Apply cost-matrix weighting or Tweedie loss to heavily penalize False Negatives (missing a superstar) at high-leverage positions.

## 3. Target Paradigm: Value Over Replacement Player (VORP)
Predicting raw Cap Share can overvalue compilers (players who hang around a long time) and undervalue the structural scarcity of certain positions. Integrating VORP centers the model on pure surplus value.

* **Defining Replacement Level:** In a 32-team league, the "replacement" is the freely available street free agent. For a QB, this is roughly the 33rd-ranked player in the league in any given year. For a WR, it might be the 97th (assuming 3 starters per team).
* **Calculating VORP:** * First, calculate the `Replacement_Baseline_Cap_Share` for each position, per year.
    * `VORP = Player_Annual_Cap_Share - Replacement_Baseline_Cap_Share`
    * Sum the annual VORP over the player's career. If a player earns less than the replacement baseline, their VORP for that year is 0 (or negative, representing a sunk-cost bust).
* **ML Application:** By using Career VORP as the ultimate regression target, the model natively learns positional positional leverage. A 0.08 Cap Share season for a Kicker might yield 0 VORP, while a 0.08 Cap Share for a Rookie QB might yield high VORP. The model learns to ignore positions with flat talent distributions.

## 4. Model Architecture: The Hurdle Pipeline
Instead of a single "Generalist" model, the architecture splits the problem into two distinct tasks to isolate the threshold-clearance signal from the elite-intensity signal.

* **Stage 1: The Career-Threshold Classifier**
    * **Goal:** Filter out the noise. Predict the probability (P) that a prospect will "survive" in the league (e.g., `Career_Starts > X` or `Was_Drafted == True`).
    * **Tech:** XGBoost or LightGBM (handles categorical position data well).
* **Stage 2: The Outcome Regressor**
    * **Goal:** Predict the intensity of success (Career VORP or Career Cap Share) for players who cleared the Stage 1 threshold.
    * **Tech:** A regressor utilizing a Log-Normal transformation or Tweedie loss to handle the extreme right-skew of the earnings data.
    * **Positional Handling:** Utilize shared embeddings with feature masking so the model only "pays attention" to passing ratings for QBs and pass-rush ratings for DEs, without atomizing the dataset into 15 separate tiny models.

## 5. Feature Engineering (Polars Pipeline)
Raw physical and scouted data must be contextualized before hitting the models.

* **Relative Athleticism (Z-Scores):** A 4.80 `Dash` is elite for a DT but a failure for a WR. Normalize combine scores using Polars window functions (`.over("Position_Group")`).
* **Scouting Uncertainty:** Extract the delta between the `High_` and `Low_` scouted skill ranges in `draft_personal.csv`. Wider ranges indicate high variance/uncertainty.
* **Expert Adjustment:** Weight the scouted ratings by the Evaluating Staff's `Scouting_Ability` to give more trust to high-quality scouts.

## 6. The Decision Engine: Expected Draft Value (EDV)
The final output is a decision-support metric rooted in Expected Value, decoupling raw talent from positional leverage (if not already fully captured by VORP).

* **Talent Score (Tp):** The model's predicted percentile of success for the player *relative to their position group*. (e.g., 0.90 = 90th percentile Guard).
* **Positional Leverage (W_pos):** If predicting raw Cap Share, multiply by the historical delta between an Elite player's Cap % and a Replacement Level player's Cap % at that specific position. (If predicting VORP, this weight is mathematically built into the target).
* **The Output Metric (EDV):** `Tp * W_pos` (or simply `Predicted_VORP * P(Career Threshold)`).

## 7. The Draft Board Output
When operationalized for a live draft, the pipeline will output a unified dashboard for each prospect featuring:

1.  **Value Rank (EDV/VORP):** The primary sort. Who provides the most surplus value and wins above replacement to the franchise?
2.  **Risk Profile:** Displaying the 25th percentile prediction (The Floor) and the 95th percentile prediction (The Ceiling/Upside).
3.  **Positional Scarcity Alerts:** Flags when the final high-EDV player at a premium position is about to be drafted, signaling when to reach and when to wait.
