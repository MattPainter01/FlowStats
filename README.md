# About
There are a number of standard statistics that are reported in papers on optical flow. The main ones are the average end point error, the average angular error and the Fl score. This repository offers a simple Python class that calculates all these quantities alongside some simple analyses. 

# Metrics
We give the definitions of the metrics used:
- EPE: Euclidean norm of the difference between predicted and true flow fields.
- Angular Error: Mean angle between predicted and true flow vectors.
- Fl score: Ratio of pixels where flow estimate is wrong by both >= 3 pixels and >= 5%


