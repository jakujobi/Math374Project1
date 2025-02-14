"""
Numerical Differentiation Error Analysis Web App

Author: John Akujobi
Course: Math 374 - Scientific Computing
Date: [Insert Date]

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
    st.title("John Akujobi - Math 374 Project 1 - Numerical Differentiation Error Analysis")
    st.markdown("""... (keep your project overview text here) ...""")
    
    # Show theory section
    show_theory()
    
    # Get user inputs
    inputs = get_user_inputs()
    
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
    # ... (keep your optimal values display code here) ...
    
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
    """)
if __name__ == "__main__":
    main()