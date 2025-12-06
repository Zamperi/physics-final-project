# Physics – Final Project

## Task Description
This project analyzes simultaneously recorded smartphone acceleration and GPS data collected with Phyphox during walking and running. The goal is to prototype a simple sports-tracking application that computes step count, average speed, distance, and step length, and visualizes the results.

Measurements use the **Linear Acceleration** and **Location** sensors, producing two CSV files: `Accelerometer.csv` and `Location.csv`. The analysis includes filtering acceleration data, detecting steps (time-domain and Fourier-based), and deriving speed and distance from GPS data.

Required visualizations:
- filtered acceleration used for step counting  
- power spectral density of the selected acceleration component  
- route plotted on a map  

## How to View the Report

### Run directly from GitHub
```
streamlit run https://raw.githubusercontent.com/Zamperi/physics-final-project/main/app.py
```

### Or clone the repository
```
git clone https://github.com/Zamperi/physics-final-project.git
cd physics-final-project
streamlit run app.py
```

## Final Result

![Final result](/results/report-1.png)
![Final result](/results/report-2.png)
