"""
Numerical Differentiation Error Analysis App

A Streamlit application that analyzes and compares the forward and central difference
methods for numerical differentiation of f(x) = sin(x) at x = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Constants
DEFAULT_EPS = np.finfo(float).eps
FUNCTION = lambda x: np.sin(x)
EXACT_DERIVATIVE = np.cos(1.0)  # f'(x) where f(x) = sin(x)

def configure_page() -> None:
    """Configure Streamlit page settings and styles."""
    st.set_page_config(page_title="Numerical Differentiation Analysis", layout="wide")
    st.markdown("""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
        <style>
            .reportview-container { background: #f0f2f6; }
            .main .block-container { padding: 2rem; }
            h1 { color: #2a4a7d; }
            .st-expander { background: white; border: 1px solid #d6d6d6; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

def show_theory() -> None:
    """Display theoretical background in expandable sections."""
    with st.expander("📚 Theoretical Background", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(r"""
            **Forward Difference:**
            \[ f'(x) \approx \frac{f(x+h) - f(x)}{h} \]
            - Truncation error: \( O(h) \)
            - Rounding error: \( O(\epsilon/h) \)
            """)
            
        with col2:
            st.markdown(r"""
            **Central Difference:**
            \[ f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} \]
            - Truncation error: \( O(h^2) \)
            - Rounding error: \( O(\epsilon/h) \)
            """)

def get_user_inputs() -> dict:
    """Collect user inputs from sidebar.
    
    Returns:
        dict: Dictionary containing:
            - h_values: Array of h values
            - eps: Machine epsilon value
    """
    with st.sidebar:
        st.header("Controls")
        h_min = st.slider("Minimum h (10^-k)", 1, 16, 1)
        h_max = st.slider("Maximum h (10^-k)", 1, 16, 16)
        num_points = st.slider("Number of points", 10, 100, 50)
        eps = st.number_input("Machine epsilon (ε)", 1e-16, 1e-10, DEFAULT_EPS, format="%e")
    
    return {
        'h_values': np.logspace(-h_max, -h_min, num_points),
        'eps': eps
    }

def calculate_errors(h_values: np.ndarray, eps: float) -> dict:
    """Calculate errors and bounds for both differentiation methods.
    
    Args:
        h_values: Array of step sizes
        eps: Machine epsilon value
    
    Returns:
        dict: Dictionary containing:
            - h: Step sizes
            - err1/err2: Actual errors for methods 1/2
            - trunc1/trunc2: Truncation error bounds
            - round1/round2: Rounding error bounds
    """
    results = {
        'h': [], 'err1': [], 'err2': [],
        'trunc1': [], 'trunc2': [],
        'round1': [], 'round2': []
    }
    
    for h in h_values:
        # Forward difference calculation
        f_plus = FUNCTION(1 + h)
        approx1 = (f_plus - FUNCTION(1)) / h
        err1 = abs(approx1 - EXACT_DERIVATIVE)
        
        # Central difference calculation
        f_minus = FUNCTION(1 - h)
        approx2 = (f_plus - f_minus) / (2 * h)
        err2 = abs(approx2 - EXACT_DERIVATIVE)
        
        # Error bounds calculation
        results['h'].append(h)
        results['err1'].append(err1)
        results['err2'].append(err2)
        results['trunc1'].append(h/2)          # Forward truncation bound
        results['trunc2'].append(h**2/6)        # Central truncation bound
        results['round1'].append(2*eps/h)       # Forward rounding bound
        results['round2'].append(eps/h)         # Central rounding bound
    
    return results

def plot_errors(ax: plt.Axes, h_values: list, 
                errors: list, trunc: list, round: list, 
                title: str) -> None:
    """Create a log-log error plot for a differentiation method.
    
    Args:
        ax: Matplotlib axes object
        h_values: List of step sizes
        errors: Actual error values
        trunc: Truncation error bounds
        round: Rounding error bounds
        title: Plot title
    """
    ax.loglog(h_values, errors, 'b-', label='Actual Error')
    ax.loglog(h_values, trunc, 'r--', label='Truncation Bound')
    ax.loglog(h_values, round, 'g--', label='Rounding Bound')
    ax.set_title(title)
    ax.set_xlabel("h (log scale)")
    ax.set_ylabel("Error (log scale)")
    ax.grid(True, which='both')
    ax.legend()

def show_results(results: dict, eps: float) -> None:
    """Display results and analysis sections.
    
    Args:
        results: Dictionary containing error data
        eps: Machine epsilon value
    """
    # Visualization
    st.header("Error Analysis Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        plot_errors(ax1, results['h'], results['err1'], 
                   results['trunc1'], results['round1'], 
                   "Forward Difference")
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_errors(ax2, results['h'], results['err2'], 
                   results['trunc2'], results['round2'], 
                   "Central Difference")
        st.pyplot(fig2)

    # Optimal h calculations
    opt_h1 = (2*eps)**0.5
    opt_h2 = (3*eps)**(1/3)
    
    st.header("Detailed Error Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(rf"""
        **Forward Difference Optimal h:**
        \[ h_{{\mathrm{{opt}}}} = \sqrt{{2\epsilon}} \approx {opt_h1:.2e} \]
        - Minimum achievable error: {np.sqrt(2*eps)/2:.2e}
        """)
    
    with col2:
        st.markdown(rf"""
        **Central Difference Optimal h:**
        \[ h_{{\mathrm{{opt}}}} = \sqrt[3]{{3\epsilon}} \approx {opt_h2:.2e} \]
        - Minimum achievable error: {(3*eps)**(2/3)/6:.2e}
        """)

    # Comparison table
    st.header("Method Comparison")
    st.markdown("""
    | Aspect                | Forward Difference | Central Difference |
    |-----------------------|--------------------|--------------------|
    | Truncation Error Order | O(h)               | O(h²)              |
    | Rounding Error Order   | O(ε/h)             | O(ε/h)             |
    | Optimal h             | ~√ε                | ~∛ε                |
    | Best Accuracy         | ~√ε                | ~ε²/³              |
    | Stability             | Moderate           | Better             |
    """)

def main():
    """Main application workflow."""
    configure_page()
    st.title("Numerical Differentiation Error Analysis")
    st.markdown("**Analyzing Forward and Central Difference Formulas for f(x) = sin(x) at x = 1**")
    
    show_theory()
    inputs = get_user_inputs()
    results = calculate_errors(inputs['h_values'], inputs['eps'])
    show_results(results, inputs['eps'])

if __name__ == "__main__":
    main()