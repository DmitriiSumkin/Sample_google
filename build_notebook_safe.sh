
# Turn on exit on error
set -e

# --- Create individual JSON files for each new cell ---

# Cell 1: EDA Markdown
cat << 'EOF' > eda_md.json
{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 7.1. Exploratory Data Visualization\n",
    "Before diving into the model, let's visualize our key variables to better understand the data distribution and identify any potential outliers or patterns."
   ]
}
EOF

# Cell 2: EDA Code
cat << 'EOF' > eda_code.json
{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set_style(\"whitegrid\")\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "sns.histplot(df['Primary_Visits'], bins=50, ax=axes[0], kde=True)\n",
    "axes[0].set_title(\"Distribution of Primary Visits\")\n",
    "sns.histplot(df['Emergency_Visits'], bins=50, ax=axes[1], kde=True, color='salmon')\n",
    "axes[1].set_title(\"Distribution of Emergency Visits\")\n",
    "plt.suptitle(\"Visit Distributions\")\n",
    "plt.show()\n",
    "monthly_visits = df.groupby('Month')[['Primary_Visits', 'Emergency_Visits']].mean()\n",
    "monthly_visits.plot(figsize=(14, 7), subplots=True, layout=(2,1), title=\"Average Visits Per Month\")\n",
    "plt.xlabel(\"Month\")\n",
    "plt.tight_layout(rect=[0, 0, 1, 0.96])\n",
    "plt.show()"
   ]
}
EOF

# Cell 3: Pre-Trend Markdown
cat << 'EOF' > pre_trend_md.json
{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 8.1. Visual Pre-Trend Checks for DiD Model\n",
    "It is crucial to test the parallel trends assumption by plotting the outcome for both groups in the pre-treatment period."
   ]
}
EOF

# Cell 4: Pre-Trend Code
cat << 'EOF' > pre_trend_code.json
{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "pre_treatment_df = df[df['Month'] < df['event_date']].copy()\n",
    "pre_trend_data = pre_treatment_df.groupby(['Month', 'Treat'])[['Primary_Visits', 'Emergency_Visits']].mean().reset_index()\n",
    "plt.figure(figsize=(14, 7))\n",
    "sns.lineplot(data=pre_trend_data, x='Month', y='Primary_Visits', hue='Treat', marker='o')\n",
    "plt.title(\"Pre-Treatment Trends in Primary Visits (Treatment vs. Control)\")\n",
    "plt.legend(title='Group', labels=['Control', 'Treatment'])\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
}
EOF

# Cell 5: Diagnostics Markdown
cat << 'EOF' > diagnostics_md.json
{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 9.1. Formal Diagnostic Tests for Regression Models\n",
    "After estimating the DiD model, we must perform diagnostic tests to validate the assumptions of the underlying linear regression."
   ]
}
EOF

# Cell 6: Diagnostics Code
cat << 'EOF' > diagnostics_code.json
{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "import statsmodels.api as sm\n",
    "import statsmodels.stats.api as sms\n",
    "import scipy.stats as stats\n",
    "fitted_vals = res_did.fittedvalues\n",
    "residuals = res_did.resid\n",
    "sns.residplot(x=fitted_vals, y=residuals, lowess=True, line_kws={'color': 'red'})\n",
    "plt.title(\"Residuals vs. Fitted Values\")\n",
    "plt.show()\n",
    "bp_test = sms.het_breuschpagan(residuals, res_did.model.exog)\n",
    "print(f\"Breusch-Pagan Test p-value: {bp_test[1]:.4f}\")\n",
    "sm.qqplot(residuals, stats.t, fit=True, line='45')\n",
    "plt.title(\"Q-Q Plot of Model Residuals\")\n",
    "plt.show()"
   ]
}
EOF

# Cell 7: Robustness Markdown
cat << 'EOF' > robustness_md.json
{
   "cell_type": "markdown", "metadata": {},
   "source": ["# 10.1. Perform Robustness Checks"]
}
EOF

# Cell 8: Robustness Code (Alt Outcome)
cat << 'EOF' > robustness_alt.json
{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "print(\"--- Robustness Check: DiD Model with 'Emergency_Visits' as Outcome ---\")\n",
    "formula_robust = 'Emergency_Visits ~ Treat * Post + C(Month) + C(zip_code)'\n",
    "res_did_robust = smf.ols(formula_robust, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['zip_code']})\n",
    "print(res_did_robust.summary())"
   ]
}
EOF

# Cell 9: Robustness Code (Placebo)
cat << 'EOF' > robustness_placebo.json
{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "print(\"--- Robustness Check: Placebo Test ---\")\n",
    "df_placebo = df.copy()\n",
    "placebo_event_date = pd.to_datetime(df_placebo['event_date']).median() - pd.DateOffset(years=1)\n",
    "df_placebo['Post_Placebo'] = (pd.to_datetime(df_placebo['Month']) >= placebo_event_date).astype(int)\n",
    "formula_placebo = 'Primary_Visits ~ Treat * Post_Placebo + C(Month) + C(zip_code)'\n",
    "res_did_placebo = smf.ols(formula_placebo, data=df_placebo).fit(cov_type='cluster', cov_kwds={'groups': df_placebo['zip_code']})\n",
    "print(res_did_placebo.summary())"
   ]
}
EOF

# Cell 10: Visualization Markdown
cat << 'EOF' > viz_md.json
{
   "cell_type": "markdown", "metadata": {},
   "source": ["# 11.1. Visualizing the Main Difference-in-Differences Effect"]
}
EOF

# Cell 11: Visualization Code
cat << 'EOF' > viz_code.json
{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "plot_data = df.groupby(['Month', 'Treat'])['Primary_Visits'].mean().reset_index()\n",
    "avg_event_date = pd.to_datetime(df['event_date']).mean()\n",
    "plt.figure(figsize=(14, 8))\n",
    "ax = sns.lineplot(data=plot_data, x='Month', y='Primary_Visits', hue='Treat', style='Treat', markers=True, dashes=False)\n",
    "ax.axvline(x=avg_event_date, color='r', linestyle='--', lw=2, label='Avg. Event Date')\n",
    "ax.set_title('Average Primary Visits Over Time (Treatment vs. Control)')\n",
    "ax.legend(title='Group')\n",
    "plt.show()"
   ]
}
EOF

# --- Assemble the notebook ---

base_notebook_json=$(cat notebooks/OSF_1109_all_sample_3M_1charR.ipynb)
part1_cells=$(echo "$base_notebook_json" | jq '.cells[0:6]')
did_model_cells=$(echo "$base_notebook_json" | jq '.cells[6:]')

final_cells_array=$(jq -n '
    $p1 + [ $eda_md, $eda_code, $pre_trend_md, $pre_trend_code ] +
    $did_model +
    [ $diagnostics_md, $diagnostics_code, $robustness_md, $robustness_alt, $robustness_placebo, $viz_md, $viz_code ]
' \
--argjson p1 "$part1_cells" \
--argjson did_model "$did_model_cells" \
--argjson eda_md "$(cat eda_md.json)" \
--argjson eda_code "$(cat eda_code.json)" \
--argjson pre_trend_md "$(cat pre_trend_md.json)" \
--argjson pre_trend_code "$(cat pre_trend_code.json)" \
--argjson diagnostics_md "$(cat diagnostics_md.json)" \
--argjson diagnostics_code "$(cat diagnostics_code.json)" \
--argjson robustness_md "$(cat robustness_md.json)" \
--argjson robustness_alt "$(cat robustness_alt.json)" \
--argjson robustness_placebo "$(cat robustness_placebo.json)" \
--argjson viz_md "$(cat viz_md.json)" \
--argjson viz_code "$(cat viz_code.json)" \
)

echo "$base_notebook_json" | jq --argjson final_cells "$final_cells_array" '.cells = $final_cells' > notebooks/OSF_1109_all_sample_3M_1charR.ipynb

# --- Clean up temporary files ---
rm *.json

echo "Successfully rebuilt the notebook with all enhancements."
