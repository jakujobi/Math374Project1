"""
Project 1 - Numerical Differentiation Error Analysis Web App

Author: John Akujobi
Course: Math 374 - Scientific Computing
Date: Spring 2024

This Streamlit app analyzes and visualizes errors in numerical differentiation methods
for f(x) = sin(x) at x = 1, comparing forward and central difference formulas.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------
# Configuration & Constants
# --------------------------
def configure_page():
    """Set up page configuration and styling"""
    st.set_page_config(page_title="Numerical Differentiation Analysis", layout="wide")
    
    # Add MathJax support and custom CSS
    st.markdown("""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
    <style>
        .reportview-container { background: #f0f2f6; }
        .main .block-container { padding: 2rem; }
        h1 { color: #2a4a7d; }
        .st-expander { background: white; border: 1px solid #d6d6d6; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# Theory Section
# --------------------------
def show_theory():
    """Display theoretical background in expandable section"""
    with st.expander("📚 Theoretical Background", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(r"""
            **Forward Difference (Formula 1):**

            $f'(x) \approx \frac{f(x+h) - f(x)}{h}$

            - Truncation error: $O(h)$
            - Rounding error: $O(\epsilon/h)$
            """)
            
        with col2:
            st.markdown(r"""
            **Central Difference (Formula 2):**

            $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$

            - Truncation error: $O(h^2)$
            - Rounding error: $O(\epsilon/h)$
            """)

# --------------------------
# Controls & Inputs
# --------------------------
def get_user_inputs():
    """Create sidebar controls and return user inputs"""
    with st.sidebar:
        st.header("Controls")
        return {
            'h_min': st.slider("Minimum h (10^-k)", 1, 16, 1),
            'h_max': st.slider("Maximum h (10^-k)", 1, 16, 16),

            'num_points': st.slider("Number of points", 10, 100, 50),
            'eps': st.number_input("Machine epsilon (ε)", 1e-16, 1e-10, 2.22e-16, format="%e")
        }

# --------------------------
# Error Calculations
# --------------------------
@st.cache_data # Cache the results of expensive calculations for faster loading
def calculate_errors(h_values: np.ndarray, eps: float) -> dict:
    """
    Calculate errors and bounds for both differentiation methods
    
    Args:
        h_values: Array of step sizes to evaluate
        eps: Machine epsilon value
        
    Returns:
        Dictionary containing:
        - h values
        - Actual errors for both methods
        - Truncation error bounds
        - Rounding error bounds
    """
    try:
        exact = np.cos(1.0)  # Exact derivative of sin(x) at x=1
        results = {
            'h': h_values,
            'err1': [],
            'err2': [],
            'trunc1': [],
            'trunc2': [],
            'round1': [],
            'round2': []
        }
        
        for h in h_values:
            # Forward difference calculations
            f_plus = np.sin(1 + h)
            approx1 = (f_plus - np.sin(1)) / h
            results['err1'].append(abs(approx1 - exact))
            results['trunc1'].append(h / 2)
            results['round1'].append(2 * eps / h)
            
            # Central difference calculations
            f_minus = np.sin(1 - h)
            approx2 = (f_plus - f_minus) / (2 * h)
            results['err2'].append(abs(approx2 - exact))
            results['trunc2'].append(h**2 / 6)
            results['round2'].append(eps / h)
        return results
    except Exception as e:
        st.error(f"Error occurred during calculations: {e}")
        return None

# --------------------------
# Visualization
# --------------------------
def create_error_plot(data: dict, method: str) -> plt.Figure:
    """
    Create log-log error plot for a differentiation method
    
    Args:
        data: Results dictionary from calculate_errors
        method: 'forward' or 'central'
        
    Returns:
        Matplotlib figure object
    """
    method_key = '1' if method == 'forward' else '2'
    titles = {
        '1': "Forward Difference Formula (1)",
        '2': "Central Difference Formula (2)"
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(data['h'], data[f'err{method_key}'], 'b-', label='Actual Error')
    ax.loglog(data['h'], data[f'trunc{method_key}'], 'r--', label='Truncation Bound')
    ax.loglog(data['h'], data[f'round{method_key}'], 'g--', label='Rounding Bound')
    
    ax.set_title(titles[method_key])
    ax.set_xlabel("h (log scale)")
    ax.set_ylabel("Error (log scale)")
    ax.grid(True, which='both')
    ax.legend()
    
    return fig

# --------------------------
# Optimal Values Calculation
# --------------------------
def calculate_optimal_values(eps: float) -> dict:
    """
    Calculate optimal h values and minimum errors
    
    Args:
        eps: Machine epsilon value
        
    Returns:
        Dictionary with optimal values for both methods
    """
    return {
        'forward': {
            'h_opt': (2 * eps) ** 0.5,
            'min_error': np.sqrt(2 * eps) / 2
        },
        'central': {
            'h_opt': (3 * eps) ** (1/3),
            'min_error': (3 * eps) ** (2/3) / 6
        }
    }

# --------------------------
# Main App Flow
# --------------------------
def main():
    # Initialize application
    configure_page()
    st.title("Math 374 Project 1 - John Akujobi")
    st.markdown("""
## Numerical Differentiation Error Analysis
This 

""")
    
    # Show theory section
    show_theory()
    
    # Get user inputs
    inputs = get_user_inputs()
    eps = inputs['eps']
    
    # Generate h values and calculate errors
    h_values = np.logspace(-inputs['h_max'], -inputs['h_min'], inputs['num_points'])
    results = calculate_errors(h_values, inputs['eps'])
    
    # Show visualizations
    st.header("Error Analysis Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(create_error_plot(results, 'forward'))
    with col2:
        st.pyplot(create_error_plot(results, 'central'))
    
    # Show optimal values
    optimal = calculate_optimal_values(inputs['eps'])
    st.header("Detailed Error Analysis")
    # Optimal h calculations
    opt_h1 = (2*eps)**0.5
    opt_h2 = (3*eps)**(1/3)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(rf"""
        **Forward Difference Optimal h:**
        
        $h_{{\mathrm{{opt}}}} = \sqrt{{2\epsilon}} \approx {opt_h1:.2e}$
        
        - Minimum achievable error: {np.sqrt(2*eps)/2:.2e}
        """)

    with col2:
        st.markdown(rf"""
        **Central Difference Optimal h:**
        
        $h_{{\mathrm{{opt}}}} = \sqrt[3]{{3\epsilon}} \approx {opt_h2:.2e}$
        
        - Minimum achievable error: {(3*eps)**(2/3)/6:.2e}
        """)    

    # Show comparison table
    st.header("Comparison of Methods")
    st.markdown("""
    | Aspect                | Forward Difference | Central Difference |
    |-----------------------|--------------------|--------------------|
    | Truncation Error Order | O(h)               | O(h²)              |
    | Rounding Error Order   | O(ε/h)             | O(ε/h)             |
    | Optimal h             | ~√ε                | ~∛ε                |
    | Best Accuracy         | ~√ε                | ~ε^{2/3}           |
    | Stability             | Moderate           | Better             |

    **Key Observations:**
    1. Central difference provides better accuracy for same h
    2. Central difference maintains stability for larger h values
    3. Forward difference deteriorates faster for small h
    ---
    """)

   # Project Report
    st.header("Project Report")
    st.markdown(rf"""
# Project Report

## **Numerical Differentiation Error Analysis Web App**

## **Author**: John Akujobi

## **Course**: Math 374 – Scientific Computing

## Date: Spring 2024
""")
    st.markdown("""
---

## Project Question:

Here we consider two numerical differentiation formulas,
""")
    st.markdown(r"""
#### 1.$f'(x) \approx \frac{f(x + h) - f(x)}{h}$

#### 2. $f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$

Study and compare the two formulas for $f(x) = \sin x$ and $x = 1$ as $h \to 0$.

### Tasks:

1. Find truncation error bounds for (1) and (2).
  
2. Find rounding error bounds for (1) and (2).
  
3. On the two graphs for (1) and (2), plot truncation error bound, rounding error bound, and total error using a log-scale;  
  The axes in the plot should be $\log_{10} | \text{error} |$ versus $\log_{10} h$ as $h = 10^{-k}, k = 1, \dots, 16$.
  
4. Discuss the optimal values of \( h \) and the relations between errors.
  
5. Compare (1) and (2) for your conclusion.
  

---

## 1. Introduction

Through a DuckDuckGo search, I found out that the two numerical differentiation methods were called:

1. the forward difference formula
  
2. the central difference formula.
  

First we need to explore them and approximate the derivative of the function

f(x)=sin⁡(x) at x = 1 and to analyze the errors associated with each method.

Specifically, we calculate:

- **Truncation Error:** Error from neglecting higher order terms in the Taylor series.
- **Rounding Error:** Error due to the finite precision (machine epsilon) of computer arithmetic.
- **Total Error:** The combination of truncation and rounding errors.

#### How it is presented.

I presented the project as an interractive web app built with [Streamlit](https://streamlit.io/). THere, you can adjust parameters (like the range of h values and machine epsilon). And you can see log-log plots that show how the errors change as the step size hh varies.

---

## 2. Background and Theory

### 2.1. Numerical Differentiation

Numerical differentiation involves approximating the derivative of a function using its values at nearby points. Two widely used methods are:

- **Forward Difference Formula:**
""")
    st.markdown(r"""
  $f'(x) \approx \frac{f(x+h)-f(x)}{h}$
  
- **Central Difference Formula:**
  
  $f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}$
  

### 2.2. Error Analysis

Here are the errors in numerical differentiation

**Truncation Error:**

Derived by expanding f(x+h) (and f(x−h)f(x-h) for the central difference) in a Taylor series.

- For the **forward difference** , the expansion gives:
  
  $\frac{f(x+h)-f(x)}{h} = f'(x) + \frac{h}{2}f''(x) + O(h^2)$
  The leading error term is proportional to h (i.e., $O(h)$).
  
- For the **central difference** , the expansion gives us:
  
  $\frac{f(x+h)-f(x-h)}{2h} = f'(x) + \frac{h^2}{6}f'''(x) + O(h^4) $
  
  The truncation error is proportional to $h^2$ (i.e., $O(h^2)$). This means a higher accuracy.
  

**Rounding Error:**

Due to the limitations of finite precision arithmetic (quantified by machine epsilon, ϵ). When we subtract numbers that are nearly equal, (this happens in both formulas), the relative error is increased roughly by a factor of 1/h.

**Total Error:**

This is sum of the truncation error and rounding error.

As h decreases, truncation error reduces while rounding error increases. There is a sweet spot, an optimal value of h where these competing effects balance to minimize the total error.

---

## 3. Project Objectives

The key objectives of the project are:

1. Find the theoretical truncation and rounding error bounds for both differentiation formulas.
  
2. **Compute Errors Numerically:**
  
  Use Python to calculate the actual error (the difference between the numerical approximation and the true derivative), the truncation error, and the rounding error over a range of hh values.
  
3. **Visualization:**
  
  Plot the errors on a log-log scale to clearly observe the relationship between hh and the different types of errors.
  
4. **Interactivity:**(Self assigned)
  
  Build an interactive web app with Streamlit where users can adjust input parameters (such as the range of h values and the machine epsilon) and seeupdated plots and optimal hh values.
  
5. **Comparison and Conclusion:**
  
  Compare the forward and central difference methods. Then discuss the optimal h for each, and conclude which method is more accurate and stable.
  

---

## 4. Implementation Details

### 4.1. Configuration & Constants

- **Page Setup:**
  
  The `configure_page()` function sets up the Streamlit page. It includes the title, layout, and some custom CSS for styling. I also added MathJax support for properly rendering LaTeX formulas.
  

### 4.2. Theoretical Background

- **Theory Display:**
  
  The `show_theory()` function uses an expandable section to present the theoretical background of the forward and central difference methods. This includes the order of truncation and rounding errors for each method.
  

### 4.3. Controls & Inputs

- **User Input Controls:**
  
  `get_user_inputs()` function makesa sidebar in the Streamlit app where users can select:
  
  - The minimum and maximum exponents for h (e.g., $h = 10^{-k})$.
  - The number of points to generate between these values.
  - The machine epsilon (ϵ). This is is used in the rounding error calculation.

### 4.4. Error Calculations

- **Calculating Errors:**
  
  The `calculate_errors()` function iterates over an array of h values to compute:
  
  - **Actual error:** Difference between the numerical derivative and the exact derivative ($\cos(1)$).
  - **Truncation error bounds:** Based on the theoretical derivations:
    - For forward difference: h/2
    - For central difference: $h^2/6$
  - **Rounding error bounds:** Estimated as:
    - For forward difference: $2\epsilon/h$
    - For central difference: $\epsilon/h$

### 4.5. Visualization

- **Plotting the Data:**
  
  The `create_error_plot()` function makesa log-log plot for either the forward or central difference method. It plots:
  
  - The actual error.
  - The truncation error bound.
  - The rounding error bound.
  
  Both plots are integrated into the Streamlit app, allowing for side-by-side comparisons.
  

### 4.6. Optimal Values Calculation

- **Finding the Optimal hh:**
  
  The `calculate_optimal_values()` function computes the optimal step sizes for both methods:
  
  - **Forward difference optimal h:** $h_{\text{opt}} \approx \sqrt{2\epsilon}$
  - **Central difference optimal h:** $h_{\text{opt}} \approx \sqrt[3]{3\epsilon}$
  
  These values represent the balance point where the total error (truncation plus rounding) is minimized.
  

### 4.7. Main App Flow

- **Orchestrating the App:**
  
  The `main()` function ties together all of the components:
  
  - Configures the page and sets up the title.
  - Displays the theoretical background.
  - Collects user inputs.
  - Generates hh values, calculates errors, and displays the visualizations.
  - Presents the computed optimal hh values and a comparison table that summarizes the differences between the two methods.

---

## 5. Running the App

### Prerequisites

- **Python:** Version 3.13
  
- **Libraries:** NumPy, Matplotlib, and Streamlit
  
  (Install them with pip: `pip install numpy matplotlib streamlit`)
  

### Execution

1. Save the code in a file (e.g., `streamlit_app.py`).
2. In your terminal or command prompt, navigate to the directory containing the file.
3. Run the command:
  
  ```
  streamlit run streamlit_app.py
  ```
  
4. Your browser will open an interactive page where you can explore the error analysis.

---

## 6. Results & Discussion

- **Visualizations:**
  
  The app showstwo side-by-side plots (one for each differentiation method) on a log-log scale. This makes it easier to understand how the actual error, truncation error, and rounding error changewith h.
  
- **Error Behavior:**
  
  - For larger h, the truncation error dominates.
  - As h decreases, rounding error becomes more significant.
  - There is an optimal h where the total error (sum of truncation and rounding errors) is minimized.
- **Method Comparison:**
  
  The central difference method, with a truncation error of $O(h^2)$, generally offers better accuracy and stability than the forward difference method (with O(h) truncation error).
  
- **Optimal hh Values:**
  
  The app calculates and displays the optimal hh for both methods, demonstrating the balance between decreasing truncation error and increasing rounding error.
  

---

## 7. Conclusion

This project successfully demonstrates the key concepts in numerical differentiation error analysis. By:

- Finding the theoretical error bounds,
- Implementing the numerical computations in Python,
- Visualizing the results interactively with Streamlit,

I could easily understand the trade-offs between truncation and rounding errors.

The project shows that the central difference method is more accurate for the same hh values. THis shows why it is important to choosean optimal h to reduce totalerror.

---
""")
if __name__ == "__main__":
    main()