# Chicago Beach Weather Sensors Analysis

## Executive Summary

This project analyzes 196,279 hourly weather observations collected between April 2015 and December 2025 from three Chicago beach weather stations. The goal was to identify temporal weather patterns and build predictive models for air temperature using a structured nine-phase data science workflow. After extensive cleaning, wrangling, feature engineering, and model development, three models were evaluated. XGBoost performed best, achieving a test R² of 0.7879, RMSE of 4.67°C, and MAE of 2.86°C, driven largely by the strong predictive importance of Wet Bulb Temperature. The results highlight clear seasonal cycles, stable weekly humidity patterns, and consistent relationships among core meteorological variables.

## Phase-by-Phase Findings

### Phase 1-2: Exploration

The dataset contains 196,279 rows and 18 columns collected from three stations along the Chicago lakefront: 63rd Street, Foster, and Oak Street. Measurements include air temperature, wet bulb temperature, humidity, solar radiation, battery life, and detailed wind, rain, and barometric variables. The data span April 25, 2015 to December 2, 2025.

Key Data Quality Issues:

- Approximately 38.7% of precipitation-related fields (Wet Bulb Temperature, Rain Intensity, Total Rain, Interval Rain, Precipitation Type, and Heading) shared identical missing rows, suggesting coordinated sensor downtime rather than random loss.
- Air Temperature (75 missing values) and Barometric Pressure (146 missing values) showed minor gaps but required imputation to maintain continuity.
- More than 29,000 Solar Radiation values were identified as outliers, many of them negative and physically implausible, indicating sensor noise or calibration drift.

Initial Visualizations:

- The Air Temperature histogram shows a broad but realistic distribution, with most values falling between –5°C and 25°C.
- The time series plot displays clear annual cycles with warm summers and cold winters, emphasizing the importance of temporal structure.

![Figure 1: Distribution and Time Series of Air Temperature](output/q1_visualizations.png)
*Figure 1: Early exploratory plots showing the distribution of air temperature and its behavior over time.*

### Phase 3: Data Cleaning

The cleaning stage addressed missingness, implausible values, outliers, and inconsistent data types. Precipitation-related variables with nearly 39% missingness were set to zero to indicate no rainfall. Negative Solar Radiation values were corrected to zero, reflecting sensor behavior rather than actual environmental conditions. Remaining numeric fields were forward- and backward-filled to preserve continuity. Outliers were capped using the 1.5×IQR rule, timestamps were standardized, and no duplicate rows were found. The dataset retained all 196,279 rows after cleaning.

Key Cleaning Steps:

- Precipitation variables (Rain Intensity, Total Rain, Interval Rain, Precipitation Type, Heading) were imputed with 0 to indicate no rainfall.
- Remaining numeric fields were forward-filled and backward-filled to preserve temporal continuity.
- Outliers across all numeric columns were capped using 1.5×IQR thresholds (e.g., 97 in Air Temperature, 185 in Humidity, >29k in Solar Radiation).
- Negative Solar Radiation values were corrected to 0, reflecting nighttime or sensor dropout rather than true negative radiation.
- Measurement Timestamp was converted to datetime64[ns].
- The dataset remained the same size before and after cleaning (196,279 rows).
- No duplicate rows were found.

### Phase 4: Data Wrangling

This phase prepared the dataset for time-series analysis by standardizing timestamps, organizing observations in chronological order, and creating features that capture daily, weekly, and seasonal patterns. These steps ensured that the data was structured appropriately for both exploration and modeling.

Key Data Wrangling Steps Taken:

- Converted Measurement Timestamp to datetime64[ns] and set it as the DataFrame index.
- Sorted all observations chronologically to preserve time-series order.
- Extracted temporal features including hour, day of week, month, year, day name, and a weekend indicator.
- Ensured the dataset remained in long-format time series (no merging or pivoting needed).
- Verified time continuity to support rolling windows and temporal train–test splitting.

### Phase 5: Feature Engineering

Feature engineering focused on enhancing the dataset with derived variables that capture meaningful physical relationships and temporal structure. These engineered features help the models better learn patterns, interactions, and trends that are not explicitly encoded in the raw data. All engineered features were designed carefully to avoid data leakage, meaning none were derived from the target variable (Air Temperature) in a way that would reveal future information.

Features Created:

- Temperature Difference: captures the contrast between air temperature and related thermal measures.
- Wind Speed Squared: models the nonlinear effect of stronger winds.
- Air Temperature (F): a Fahrenheit conversion retained for interpretability.
- Comfort Index: summarizes perceived comfort using temperature and humidity.
- Temp Ratio: reflects proportional relationships among temperature variables.
- Air Temperature Categories: bins temperature into cold, mild, and warm regimes.
- Wind Speed Categories: groups wind speeds into slow, moderate, and fast conditions.
- 24-hour Rolling Mean of Air Temperature: highlights day-to-day temperature trends.
- 7-hour Rolling Mean of Wind Speed: captures short-term fluctuations in wind behavior.
  
### Phase 6: Pattern Analysis

The pattern analysis phase explored temporal trends, seasonal cycles, and correlations among key meteorological variables. By aggregating measurements across daily, weekly, and monthly time scales, the analysis revealed clear environmental rhythms and relationships that informed both feature engineering and model interpretation. These patterns highlight predictable atmospheric dynamics in Chicago’s coastal environment and validate the usefulness of extracted temporal features.

Key Patterns Identified:

- Seasonal Temperature Patterns:
  - Air temperature followed a strong seasonal cycle, peaking in midsummer (warmest in July 2020) and reaching its lowest point in January 2022. Monthly averages ranged from –5.04°C to 25.25°C, reflecting the large seasonal swings typical of the Chicago lakefront.

- Daily Humidity Cycles:
  - Humidity was relatively stable throughout the week, with only small increases on Wednesdays and Thursdays. Weekly averages stayed tightly grouped between 67.20% and 68.82%, indicating consistently moist conditions across days.

- Correlation Structure:
  - Air Temperature and Wet Bulb Temperature showed a strong positive correlation (r = 0.82), while Wind Speed and Maximum Wind Speed were also closely linked (r = 0.91). Humidity and Solar Radiation had a mild negative correlation (r ≈ –0.03), meaning sunnier periods tended to coincide with slightly drier air.

- Additional Temporal Insights:
  - These patterns reinforced the value of including features such as hour, day of week, and month, and aligned well with expected weather behavior for a Midwestern coastal climate.

![Figure 2: Monthly, hourly, and daily temperature patterns with correlation heatmap](output/q5_patterns.png)

*Figure 2: Pattern analysis showing (1) monthly average air temperature with clear seasonal cycles, (2) hourly temperature variation across the day, (3) daily average humidity fluctuations, and (4) a correlation heatmap illustrating relationships among key meteorological variables.*

### Phase 7: Modeling Preparation

The modeling preparation phase focused on structuring the dataset for supervised learning while preventing data leakage and preserving the temporal integrity of the weather time series. This included defining the prediction target, selecting appropriate features, encoding categorical variables, and performing a chronological train–test split that reflects real-world forecasting conditions.

Steps Performed:

- Defined the target variable as Air Temperature, the value to be predicted in subsequent modeling phases.
- Removed leakage-prone features, including any direct transformations or labels derived from the target (e.g., Air Temperature Categories, Comfort Index, Temp Ratio), as well as non-predictive identifiers such as Measurement Timestamp Label and Measurement ID.
- One-hot encoded categorical variables, including Station Name and Wind Speed Categories, enabling tree-based models to learn from station-specific and condition-specific patterns.
- Converted temporal attributes (hour, day of week, month, year) into numeric predictors to capture seasonal and diurnal cycles.
- Performed an 80/20 temporal train–test split, ensuring the model was trained on earlier time periods and evaluated on later periods to mimic real-world forecasting and avoid future information leaking into the training data.
- Reset index values for consistency and compatibility with scikit-learn modeling pipelines.
- Resulting dataset: 157,023 training rows and 39,256 test rows, preserving full chronological continuity.

### Phase 8: Modeling

The modeling phase involved training and evaluating three predictive models—Linear Regression, Random Forest, and XGBoost—to estimate air temperature using the engineered features and temporally split training data. Each model was fitted on 157,023 training observations and evaluated on 39,256 unseen test observations.

Models Trained:

- Linear Regression: a baseline model capturing only linear relationships.
- Random Forest Regressor: a tree-based ensemble capable of learning nonlinear interactions.
- XGBoost Regressor: a gradient-boosting model optimized for structured datasets.

| Model | R² | RMSE | MAE |
|-------|----|----|----|
| Linear Regression | 0.4948 | 7.20 | 5.09 |
| Random Forest | 0.7386 | 5.18 | 2.96 |
| XGBoost | 0.7897 | 4.65 | 2.84 |

Interpretation:

- Linear Regression served as a useful baseline model, but its strictly linear structure limited its ability to capture the nonlinear relationships inherent in atmospheric processes.
- Random Forest demonstrated a clear improvement, benefiting from its ensemble structure and ability to model complex interactions among predictors such as humidity, wind speed, and thermal measures, which contributed to lower RMSE and MAE values.
- XGBoost achieved the strongest overall performance, explaining approximately 79% of the variance in air temperature (R² = 0.7897) and producing the lowest predictive error (RMSE = 4.65°C). Its robustness and ability to generalize effectively to unseen data make it the most suitable model for this forecasting task.

Feature Importance (XGBoost):

- Wet Bulb Temperature accounted for the majority of model importance (~62%), which is expected given its strong thermodynamic link to air temperature and its role in describing heat–moisture interactions in the atmosphere.
- Battery Life contributed roughly 18% of importance—an unexpectedly large share that may reflect systematic differences across stations or indirect associations with environmental conditions, rather than a true physical relationship.
- Humidity, Barometric Pressure, Wind Direction, Solar Radiation, and various wind-related features each added smaller but still meaningful contributions, capturing secondary atmospheric influences on temperature variability.
- Rain-related variables carried virtually no predictive weight, which aligns with their sparse, near-zero distribution and their limited direct influence on short-term temperature changes in this dataset.

Overall, XGBoost provided the most accurate and stable predictions and was selected as the final model for downstream interpretation and visualization.

![Figure 3: Model performance comparison, predicted vs. actual values, and top 10 feature importances](output/q8_final_visualizations.png)

*Figure 3: Final modeling visualizations summarizing key results. Panel 1 compares model test R² across Linear Regression, Random Forest, and XGBoost. Panel 2 shows predicted vs. actual air temperature for XGBoost, illustrating strong alignment along the 1:1 line. Panel 3 presents the top ten most important features, highlighting Wet Bulb Temperature and Battery Life as the dominant predictors.*

### Phase 9: Results

The analysis revealed clear temporal structure in Chicago’s beach weather, with Air Temperature showing strong seasonal swings—warming into midsummer and cooling in winter—while Humidity remained relatively consistent across the week. After resolving major gaps in precipitation variables, correcting negative solar radiation values, and capping extreme outliers, the dataset remained fully intact and ready for modeling. Among the three models evaluated, XGBoost delivered the best performance (test R² = 0.7879), capturing nonlinear meteorological relationships more effectively than Linear Regression or Random Forest. Feature importance results highlighted Wet Bulb Temperature as the dominant predictor, with smaller but meaningful contributions from Battery Life and Humidity. Overall, the workflow produced accurate temperature predictions and clarified the key environmental factors shaping weather patterns along Chicago’s shoreline.

## Time Series Patterns

The time series analysis revealed a clear and stable temporal structure within the Chicago Beach Weather Sensors dataset. Air Temperature exhibited strong and consistent seasonality, warming into midsummer—with the highest monthly average occurring in July 2020—and cooling sharply in midwinter, reaching its lowest point in January 2022. These patterns reflect a predictable annual rhythm, with temperatures ranging from roughly −5°C to 25°C across the year. Daily moisture patterns were more subtle: Humidity remained relatively stable across the week, with only slight midweek increases, suggesting generally consistent atmospheric moisture levels.

Relationships among variables aligned well with established meteorological behavior. Air Temperature showed a strong positive correlation with Wet Bulb Temperature, while Wind Speed and Maximum Wind Speed were closely related, reflecting expected physical dynamics. Humidity and Solar Radiation displayed a modest inverse association, consistent with sunnier days tending to be slightly drier. No major anomalies or structural breaks were detected, indicating reliable sensor performance and a consistent observation record across the ten-year span of the dataset.

## Limitations & Next Steps

Although the workflow produced strong predictive models and clear temporal insights, several limitations should be acknowledged. The dataset contained substantial missingness in precipitation-related variables (~38.7%), which required imputing zeros—an approach that is meteorologically reasonable but may mask more nuanced precipitation patterns. Outlier capping using the IQR method also introduced some constraints, particularly for Solar Radiation, where extreme but potentially valid values may have been overly restricted. Sensor issues, such as negative Solar Radiation readings prior to correction, further suggest that some measurements were affected by hardware or logging inconsistencies.

From a modeling perspective, the use of conventional machine learning methods—while effective—did not explicitly leverage temporal dependence. Models such as ARIMA, Prophet, or recurrent neural networks (e.g., LSTMs) could capture time dynamics more directly. Feature engineering was intentionally conservative to avoid leakage, but more advanced temporal features (lags, differences, or multi-scale rolling windows) could enhance performance with careful construction. Future work could incorporate additional weather stations for broader spatial coverage, integrate external climate variables such as lake temperature or cloud cover, and evaluate ensemble or hybrid modeling frameworks. Validation on a fully held-out future period would also strengthen confidence in long-term generalization.