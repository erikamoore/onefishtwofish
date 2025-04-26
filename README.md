# One Fish, Two Fish
further explorations will take place soon-ish... :)


This repo contains code used for post processing + analysis in the SLEAP fish tracking project:

* **``models``**: folder with the `centroid` and `centered-instance` models used for tracking (not required for running analysis)
* **``allcsvs``**: folder of SLEAP csv output for all groups of tracked fish (each file labeled by `group_id`)
* **`constants.py`**: constants used throughout the project
* **`erikas_1min.csv`** & **`erikas_t2_4s.csv`**: sample datasets for testing code
* **`extras.ipynb`**: some explorations with DTW + some interactive `plotly` graphs
* **`find_center_demo.ipynb`**: walkthrough tutorial for finding center of tank (least squares + midpoint methods)
* **`fish_utils.py`**: all core utilities for preprocessing, data analysis, calculations + some extra functions
* **`total_dists.ipynb`**: main processing loop for calculating total distances traveled on full dataset (+ size estimation)
* **`total_dists_with_sizes_.csv`**: output of total_dists.ipynb (includes total dists + size est)
* **`visuals_demo.ipynb`** & **`visuals_demo_2.html/Rmd`**: contain all visuals from the project (+ some extra plots)
* **`visuals.py`** script for automating visualizations in `visuals_demo.ipynb`


Recommendations:
1. Run **`fish_utils.py`** and **`visuals.py`** for the core analysis + plotting functions.
2. See **`find_center_demo.ipynb`** and **`total_dists.ipynb`** for some analysis examples.
3. Generate visualizations with **`visuals_demo.ipynb`** and **`visuals_demo_2.Rmd`** (also has HTML).


---

**Repository URL**: [https://github.com/erikamoore/onefishtwofish.git](https://github.com/erikamoore/onefishtwofish.git)
