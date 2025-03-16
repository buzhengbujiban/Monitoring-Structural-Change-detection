# 📊 Structural Breakpoint Detection with VAR Residuals

## 📌 Overview
Detecting structural breakpoints in time series panels is crucial for identifying systemic risks, especially in financial markets. This project introduces a novel method that eliminates temporal correlations by applying **Vector Autoregression (VAR) models** and sequential fluctuation monitoring on the residuals to detect breakpoints. 

## 📌 Overview
- ABSTRACT
- Detecting structural breakpoints among different sequences in time series panels holds significant practical value, particularly in the financial market area where it can effectively aid in identifying systemic risks. This paper presents a method for structural breakpoint detection that eliminates temporal correlations. The paper adopts the sequential fluctuate detection methods but also expanding them by proposing a procedure that involves fitting a vector autoregression (VAR) model to the time series panel and then conducting sequential Fluctuation Monitoring detection on the residuals to detect the breakpoints. The paper provides a theoretical proof of the rationality and effectiveness of using VAR model residuals for testing, and the results of Monte Carlo simulations also report on the practicality and quality of this method. Empirical analysis aimed at the financial market demonstrates the capability of this method in testing systemic risks in the financial market.


🔍 **Key Features:**
- Eliminates **temporal dependencies** in time series data.
- Uses **VAR residuals** for structural change detection.
- Monte Carlo simulations verify its robustness.
- Empirical tests on financial market data. 

## 🔬 Methodology
We propose a two-step approach:
1. **Fit a VAR Model** to the time series panel.
2. **Apply Sequential Fluctuation Monitoring** on the residuals to detect breakpoints.

### 🧪 Monte Carlo Simulation Setup
The Monte Carlo simulations assess the method’s effectiveness by generating time series under different **Data Generating Processes (DGPs)**:

| DGP Type  | Model | AR  | MA  | Error Term |
|-----------|--------|----|----|-----------|
| Case 0   | ARMA(1,1) | 0.5 | 0.6 | N(0,1) |
| Case 1   | ARMA(1,1) | 0.5 | 0.6 | t(20) |
| Case 2   | ARMA(1,1) | 0.5 | 0.2 | N(0,1) |
| Case 3   | ARMA(2,2) | 0.5 | 0.6 | N(0,1) |
| Case 4   | ARMA(1,1) | 0.5 | 0.6 | Garch(1,1) |
| Case 5   | ARMA(1,1) | 0.5 | 0.6 | Laplace(0,1) |
| Case 6   | ARMA(1,1) | 0.05 | 0.2 | N(0,1) |
| Case 7   | Long Memory | - | - | N(0,1) |
| Case 8   | ARMA(1,1) | 0.5 | 0.6 | E-Garch(1,1) |

## 📂 Repository Structure


📁 Structural-Break-Detection 
- │── 📜 README.md 
- │── 📂 src/ # Source code for breakpoint detection 
- │── 📂 data/ # Sample datasets 
- │── 📂 results/ # Monte Carlo simulation results 
- │── 📂 figures/ # Visualizations & application images 
- │── 📜 requirements.txt # Dependencies 



## 🚀 Getting Started
### 1️⃣ Install Dependencies
```bash
git clone https://github.com/buzhengbujiban/Monitoring-Structural-Change-detection.git
cd Monitoring-Structural-Change-detection-master
pip install -r requirements.txt
cd ./src
bash case00demo.sh  ## this will create the Empirical size , Empirical power and avg 1st hitting time results in the txt files
```

## 🎯 Applications
This method can be used in: ✅ Financial Market Risk Analysis 📉
✅ Economic Policy Assessment 📊
✅ Financial tail risks Structural Changes detection 🔄
✅ Algorithmic Trading Strategies 🚀

