### Overall structure

* analytics: Holds the key analyses for the simulations.
* data: Holds the the data used by the analyses
* sim_jobs: Holds the scripts required to run the simulations on AWS
* sim_results: Holds the scripts to retrieve simulation results from AWS

### Key files

* [Description of the analysis](analytics/corona_analysis_description.ipynb)

* [Key results of the analyses](analytics/corona_sim_results.ipynb)
* [Simulation configurations used for analyses (run=2)](sim_jobs/schedulers/scheduler.py)

* [Sensitivity analyses](analytics/corona_sim_sensitivity.ipynb)
* [Simulation configuations used for sensitivity analyses (run>2)](sim_jobs/schedulers/scheduler_star.py)

* [Interactive visualization of simulation](analytics/sim_visualization.ipynb)
* [Simulation script](sim_jobs/worker/corona_simulation.py)


### Installation (More info on demand)

* [Download Anaconda] (https://www.anaconda.com/products/individual)
* pip install jupyter_plotly_dash
* Clone repository
* Run jupyter-notebook
* Requirements.txt will be added in the next update
