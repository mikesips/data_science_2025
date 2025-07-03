## 🧪 2025-07-03: Your Hands-On Today

> 💡 **Don’t be afraid to revise and play with the code!**  
> These exercises are designed to encourage exploration.

You find the tasks for [hands-on](https://docs.google.com/presentation/d/1TKUMvhTUJr9g2E_e4xTpANnfKzc9T3gqB6T_k2zR_3I/edit?usp=sharing).

```sh
# Navigate into the project directory
cd data_science_2025

# Activate the virtual environment (Linux/macOS)
source data_science_2025/bin/activate

# On Windows (powershell), use:
data_science_2025\Scripts\Activate.ps1

# Navigate to the hands-on directory
cd hands-on

# Copy step5_extraction to step6_analysis
cp -r step5_extraction step6_analysis
```
Use the step6_analysis prototype for your hands-on exercise.

## 🧪 LA Wildfire January 2025
Check the internet for the wildfire information. Get the bounding box of affected counties. Revise the search configuration (search_parameters.yml).

```sh
# Navigate to the configuratiom directory
cd step6_analysis/config/eo_workflow
```

## 🧪 Calculation of BIC

```{python}
 	# Segment the data
    y1 = y[:cp]
    y2 = y[cp:]

    # Fit means to both segments
    mu1 = np.mean(y1)
    mu2 = np.mean(y2)

    # Compute residuals
    rss1 = np.sum((y1 - mu1) ** 2)
    rss2 = np.sum((y2 - mu2) ** 2)
    rss = rss1 + rss2

    n = len(y)
    k = 3  # two means + variance (assumed equal across segments)

    # BIC formula: BIC = n * log(RSS/n) + k * log(n)
    bic = n * np.log(rss / n) + k * np.log(n)
```  
see also wiki entry for Bayesian information criterion.

## 🧪 Interactive Visualization
For the interactive visualization, play around with [Bokeh] (http://bokeh.org/).  
