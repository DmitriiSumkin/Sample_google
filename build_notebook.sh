
# Turn on exit on error
set -e

# Fetch the base notebook content.
base_notebook_json=$(cat notebooks/OSF_1109_all_sample_3M_1charR.ipynb)

# Extract the original cells.
original_cells=$(echo "$base_notebook_json" | jq '.cells')

# --- Define all new cell contents as JSON objects ---

# Cell 1: EDA Markdown
eda_md='{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 7.1. Exploratory Data Visualization\n",
    "\n",
    "Before diving into the model, let's visualize our key variables to better understand the data distribution and identify any potential outliers or patterns.\n",
    "\n",
    "We will look at:\n",
    "1.  **Distribution of Visits**: Histograms for `Primary_Visits` and `Emergency_Visits`.\n",
    "2.  **Visits Over Time**: A time-series plot showing the average number of visits per month."
   ]
}'

# Cell 2: EDA Code
eda_code='{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "sns.set_style(\\"whitegrid\\")\n",
    "\n",
    "# 1. Distribution of Primary and Emergency Visits\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "sns.histplot(df[\\'Primary_Visits\\'], bins=50, ax=axes[0], kde=True)\n",
    "axes[0].set_title(\\"Distribution of Primary Visits\\")\n",
    "sns.histplot(df[\\'Emergency_Visits\\'], bins=50, ax=axes[1], kde=True, color=\\'salmon\\')\n",
    "axes[1].set_title(\\"Distribution of Emergency Visits\\")\n",
    "plt.suptitle(\\"Visit Distributions\\")\n",
    "plt.show()\n",
    "\n",
    "# 2. Average Visits Over Time\n",
    "monthly_visits = df.groupby(\\'Month\\')[[\\'Primary_Visits\\', \\'Emergency_Visits\\']].mean()\n",
    "monthly_visits.plot(figsize=(14, 7), subplots=True, layout=(2,1), title=\\"Average Visits Per Month\\")\n",
    "plt.xlabel(\\"Month\\")\n",
    "plt.tight_layout(rect=[0, 0, 1, 0.96])\n",
    "plt.show()"
   ]
}'

# Cell 3: Pre-Trend Markdown
pre_trend_md='{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 8.1. Visual Pre-Trend Checks for DiD Model\n",
    "\n",
    "Before running the Difference-in-Differences (DiD) model, it is crucial to test the **parallel trends assumption**. This assumption states that, in the absence of the treatment, the average outcomes for the treatment and control groups would have followed parallel paths over time. We can visually inspect this by plotting the trends of the outcome variable for both groups in the **pre-treatment period**."
   ]
}'

# Cell 4: Pre-Trend Code
pre_trend_code='{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "pre_treatment_df = df[df[\\'Month\\'] < df[\\'event_date\\']].copy()\n",
    "pre_trend_data = pre_treatment_df.groupby([\\'Month\\', \\'Treat\\'])[[\\'Primary_Visits\\', \\'Emergency_Visits\\']].mean().reset_index()\n",
    "\n",
    "# Visualize for Primary_Visits\n",
    "plt.figure(figsize=(14, 7))\n",
    "sns.lineplot(data=pre_trend_data, x=\\'Month\\', y=\\'Primary_Visits\\', hue=\\'Treat\\', marker=\\'o\\')\n",
    "plt.title(\\"Pre-Treatment Trends in Primary Visits (Treatment vs. Control)\\")\n",
    "plt.legend(title=\\'Group\\', labels=[\\'Control\\', \\'Treatment\\'])\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# Visualize for Emergency_Visits\n",
    "plt.figure(figsize=(14, 7))\n",
    "sns.lineplot(data=pre_trend_data, x=\\'Month\\', y=\\'Emergency_Visits\\', hue=\\'Treat\\', marker=\\'o\\')\n",
    "plt.title(\\"Pre-Treatment Trends in Emergency Visits (Treatment vs. Control)\\")\n",
    "plt.legend(title=\\'Group\\', labels=[\\'Control\\', \\'Treatment\\'])\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
}'

# Cell 5: Diagnostics Markdown
diagnostics_md='{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 9.1. Formal Diagnostic Tests for Regression Models\n",
    "\n",
    "After estimating the DiD model, we must perform diagnostic tests to validate the assumptions of the underlying linear regression (homoscedasticity, independence, normality of residuals). Violating these assumptions can make the model results unreliable."
   ]
}'

# Cell 6: Diagnostics Code
diagnostics_code='{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "import statsmodels.api as sm\n",
    "import statsmodels.stats.api as sms\n",
    "import scipy.stats as stats\n",
    "\n",
    "fitted_vals = res_did.fittedvalues\n",
    "residuals = res_did.resid\n",
    "\n",
    "# 1. Homoscedasticity Check (Residuals vs. Fitted Plot & Breusch-Pagan Test)\n",
    "print(\\"--- Homoscedasticity Check ---\\")\n",
    "sns.residplot(x=fitted_vals, y=residuals, lowess=True, scatter_kws={\\'alpha\\': 0.5}, line_kws={\\'color\\': \\'red\\', \\'lw\\': 2})\n",
    "plt.title(\\"Residuals vs. Fitted Values\\")\n",
    "plt.show()\n",
    "bp_test = sms.het_breuschpagan(residuals, res_did.model.exog)\n",
    "print(f\\"Breusch-Pagan Test p-value: {bp_test[1]:.4f}\\")\n",
    "\n",
    "# 2. Independence of Residuals (Durbin-Watson Test)\n",
    "print(\\"--- Independence of Residuals Check ---\\")\n",
    "dw_stat = sm.stats.durbin_watson(residuals)\n",
    "print(f\\"Durbin-Watson Statistic: {dw_stat:.4f}\\")\n",
    "\n",
    "# 3. Normality of Residuals (Q-Q Plot & Jarque-Bera Test)\n",
    "print(\\"--- Normality of Residuals Check ---\\")\n",
    "sm.qqplot(residuals, stats.t, fit=True, line=\\'45\\')\n",
    "plt.title(\\'Q-Q Plot of Model Residuals\\')\n",
    "plt.show()\n",
    "jb_stat, jb_pvalue, _, _ = sm.stats.stattools.jarque_bera(residuals)\n",
    "print(f\\"Jarque-Bera Test p-value: {jb_pvalue:.4f}\\")"
   ]
}'

# Cell 7: Robustness Markdown
robustness_md='{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 10.1. Perform Robustness Checks\n",
    "\n",
    "We will now perform robustness checks to ensure our findings are not overly dependent on our specific model choices. We will test with an alternative outcome variable and run a placebo test with a fake event date."
   ]
}'

# Cell 8: Robustness Code (Alt Outcome)
robustness_code_alt='{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "print(\\"--- Robustness Check: DiD Model with 'Emergency_Visits' as Outcome ---\\")\n",
    "formula_robust = \\'Emergency_Visits ~ Treat * Post + C(Month) + C(zip_code)\\'\n",
    "res_did_robust = smf.ols(formula_robust, data=df).fit(cov_type=\\'cluster\\', cov_kwds={\\'groups\\': df[\\'zip_code\\']})\n",
    "print(res_did_robust.summary())"
   ]
}'

# Cell 9: Robustness Code (Placebo)
robustness_code_placebo='{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "print(\\"\\n--- Robustness Check: Placebo Test ---\\")\n",
    "df_placebo = df.copy()\n",
    "placebo_event_date = pd.to_datetime(df_placebo[\\'event_date\\']).median() - pd.DateOffset(years=1)\n",
    "df_placebo[\\'Post_Placebo\\'] = (pd.to_datetime(df_placebo[\\'Month\\']) >= placebo_event_date).astype(int)\n",
    "formula_placebo = \\'Primary_Visits ~ Treat * Post_Placebo + C(Month) + C(zip_code)\\'\n",
    "res_did_placebo = smf.ols(formula_placebo, data=df_placebo).fit(cov_type=\\'cluster\\', cov_kwds={\\'groups\\': df_placebo[\\'zip_code\\']})\n",
    "print(res_did_placebo.summary())"
   ]
}'

# Cell 10: Visualization Markdown
viz_md='{
   "cell_type": "markdown", "metadata": {},
   "source": [
    "# 11.1. Visualizing the Main Difference-in-Differences Effect\n",
    "\n",
    "Finally, we create a plot to visualize the main DiD effect, showing the trends for both groups and the point of intervention. This provides an intuitive understanding of the model's findings."
   ]
}'

# Cell 11: Visualization Code
viz_code='{
   "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": [
    "plot_data = df.groupby([\\'Month\\', \\'Treat\\'])[\\'Primary_Visits\\'].mean().reset_index()\n",
    "avg_event_date = pd.to_datetime(df[\\'event_date\\']).mean()\n",
    "\n",
    "plt.figure(figsize=(14, 8))\n",
    "ax = sns.lineplot(data=plot_data, x=\\'Month\\', y=\\'Primary_Visits\\', hue=\\'Treat\\', style=\\'Treat\\', markers=True, dashes=False)\n",
    "ax.axvline(x=avg_event_date, color=\\'r\\', linestyle=\\'--\\', lw=2, label=f\\'Avg. Event Date\\')\n",
    "ax.set_title(\\'Average Primary Visits Over Time (Treatment vs. Control)\\')\n",
    "ax.legend(title=\\'Group\\', labels=[\\'Control\\', \\'Treatment\\', \\'Avg. Event Date\\'])\n",
    "ax.grid(True)\n",
    "plt.show()"
   ]
}'


# --- Assemble the final notebook ---
# Combine the original cells with all the new cells.
# We will insert the new content after the data preparation cell (index 5)
# and the main DiD model after the pre-trend check.

# Split the original cells array
part1_cells=$(echo "$original_cells" | jq '.[0:6]') # Up to and including data prep
did_model_cell=$(echo "$original_cells" | jq '.[6:]') # DiD model and onward

# Create the final array of cells in the correct order.
final_cells_array=$(jq -n "
    \$p1 + [
        \$eda_md, \$eda_code,
        \$pre_trend_md, \$pre_trend_code
    ] + \$did_model + [
        \$diagnostics_md, \$diagnostics_code,
        \$robustness_md, \$robustness_code_alt, \$robustness_code_placebo,
        \$viz_md, \$viz_code
    ]
" --argjson p1 "$part1_cells" \
   --argjson eda_md "$eda_md" \
   --argjson eda_code "$eda_code" \
   --argjson pre_trend_md "$pre_trend_md" \
   --argjson pre_trend_code "$pre_trend_code" \
   --argjson did_model "$did_model_cell" \
   --argjson diagnostics_md "$diagnostics_md" \
   --argjson diagnostics_code "$diagnostics_code" \
   --argjson robustness_md "$robustness_md" \
   --argjson robustness_code_alt "$robustness_code_alt" \
   --argjson robustness_code_placebo "$robustness_code_placebo" \
   --argjson viz_md "$viz_md" \
   --argjson viz_code "$viz_code")

# Inject the final cell array into the top-level notebook structure.
echo "$base_notebook_json" | jq --argjson final_cells "$final_cells_array" '.cells = $final_cells' > notebooks/OSF_1109_all_sample_3M_1charR.ipynb

echo "Successfully rebuilt the notebook with all enhancements."
