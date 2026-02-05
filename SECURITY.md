# Security Policy

## Scope

This project is an academic web application for visualizing numerical differentiation errors. It does not:
- Store or transmit user data
- Connect to databases
- Process sensitive information
- Require authentication
- Accept file uploads

**Risk Level:** Low - Educational/demonstration tool with no data persistence

## Supported Versions

| Version | Supported | Status |
|---------|-----------|--------|
| Live (main) | ✅ Yes | Deployed on Streamlit Cloud |
| Older commits | ❌ No | Academic project, single version |

## Security Features

### Current Implementation

1. **No Data Persistence**
   - All calculations performed in-memory
   - No user data stored locally or remotely
   - Session state cleared on browser close

2. **Read-Only Operations**
   - Application only performs mathematical computations
   - No file system writes (except Streamlit cache)
   - No external API calls

3. **Input Validation**
   ```python
   # All inputs constrained by Streamlit widgets
   h_min = st.slider("Minimum h (10^-k)", 1, 16, 1)  # Range: 1-16
   h_max = st.slider("Maximum h (10^-k)", 1, 16, 16)  # Range: 1-16
   num_points = st.slider("Number of points", 10, 100, 50)  # Range: 10-100
   eps = st.number_input("Machine epsilon (ε)", 1e-16, 1e-10, 2.22e-16)  # Range: 1e-16 to 1e-10
   ```
   - Sliders prevent out-of-range values
   - Number inputs have min/max constraints
   - No free-text input fields

4. **Dependency Security**
   - Using vetted packages: `streamlit`, `numpy`, `matplotlib`
   - No deprecated or unmaintained dependencies
   - Regular updates via Streamlit Cloud auto-rebuild

### Known Limitations

1. **No Authentication**
   - Not required for public educational tool
   - Anyone can access the hosted app

2. **No Rate Limiting**
   - Streamlit Cloud provides infrastructure-level protection
   - No application-level rate limiting

3. **No HTTPS Enforcement**
   - Handled by Streamlit Cloud (https://math374p1.streamlit.app/)
   - Not configurable at application level

4. **Client-Side Rendering**
   - Mathematical formulas rendered via MathJax CDN
   - Dependency on third-party CDN: `cdnjs.cloudflare.com`

## Potential Security Considerations

### Mathematical Computation Safety

**Integer Overflow:** Not applicable - Python handles arbitrary precision integers

**Floating-Point Issues:**
- **Division by zero:** Not possible with current slider ranges (h > 0)
- **Numerical instability:** Expected and intentional for demonstration purposes
- **Infinite/NaN results:** Handled by NumPy and matplotlib gracefully

### Dependencies

**Current Dependencies (requirements.txt):**
```
streamlit        # Web framework
matplotlib       # Plotting library
```

**Transitive Dependencies:** Managed by pip, including:
- `numpy` (via streamlit)
- `pandas` (via streamlit)
- `pillow` (via matplotlib)

**Update Policy:**
- Monitor GitHub Dependabot alerts
- Update on security advisories
- Test updates in development before deploying

### Third-Party Resources

**MathJax CDN:**
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
```

**Risk:** CDN compromise could inject malicious JavaScript
**Mitigation:** 
- Using versioned CDN URL (2.7.5)
- Cloudflare CDN has strong security posture
- Consider: Add Subresource Integrity (SRI) hash in future

**Recommendation:**
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" 
        integrity="sha384-[HASH]" 
        crossorigin="anonymous" 
        async></script>
```

## Reporting a Vulnerability

### What to Report

Please report security issues if you discover:

1. **Code Execution:** Ability to execute arbitrary code
2. **Data Leakage:** Unintended data exposure
3. **DoS Vulnerabilities:** Ways to crash or overload the app
4. **Dependency Vulnerabilities:** CVEs in dependencies
5. **Injection Attacks:** XSS, code injection (though limited risk)

### What NOT to Report

These are expected behaviors or out of scope:
- Public access to the application
- Mathematical errors or numerical instability
- UI/UX issues (use normal issue tracker)
- Feature requests

### How to Report

**For Security Issues:**
1. **Do not** open a public GitHub issue
2. Email the maintainer via GitHub profile contact
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

**Example Report:**
```
Subject: Security: Potential XSS in user input field

Description:
Found that user-supplied values in [component] are rendered without sanitization.

Steps to Reproduce:
1. Navigate to [page]
2. Enter: <script>alert('XSS')</script>
3. Observe script execution

Impact:
Moderate - Could affect users who visit crafted URLs

Suggested Fix:
Use st.text() instead of st.markdown(unsafe_allow_html=True)
```

### Response Timeline

- **Acknowledgment:** Within 3 business days
- **Initial Assessment:** Within 1 week
- **Fix Development:** Depends on severity
  - Critical: Within 48 hours
  - High: Within 1 week
  - Medium: Within 1 month
  - Low: Next maintenance cycle

### Disclosure Policy

- **Coordinated Disclosure:** Fix developed privately
- **Public Disclosure:** After fix is deployed (minimum 7 days notice)
- **Credit:** Reporter credited in release notes (if desired)

## Security Best Practices for Users

### Running Locally

If running the application locally:

1. **Use Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Verify Dependencies:**
   ```bash
   pip list --outdated
   pip check
   ```

3. **Run on Localhost:**
   ```bash
   # Default: http://localhost:8501
   streamlit run streamlit_app.py
   
   # Explicitly bind to localhost only
   streamlit run streamlit_app.py --server.address localhost
   ```

4. **Firewall:**
   - Ensure port 8501 is not exposed to the internet
   - Use `localhost` binding, not `0.0.0.0`

### Using Hosted Version

When using https://math374p1.streamlit.app/:

- **No sensitive data:** Don't enter any confidential information
- **Public access:** Assume anyone can access your session
- **Ephemeral state:** All data is temporary and not logged

## Dependency Monitoring

### Automated Checks

- **GitHub Dependabot:** Enabled (monitors `requirements.txt`)
- **Streamlit Cloud:** Auto-rebuilds on dependency updates

### Manual Checks

```bash
# Check for outdated packages
pip list --outdated

# Check for known vulnerabilities
pip install safety
safety check

# Check for supply chain issues
pip install pip-audit
pip-audit
```

### Update Process

1. Review Dependabot PRs
2. Test in development
3. Update `requirements.txt`
4. Verify application functionality
5. Deploy to production

## Incident Response

### In Case of Security Incident

1. **Immediate:**
   - Take down hosted app if critical
   - Assess scope of impact
   - Document incident timeline

2. **Investigation:**
   - Identify root cause
   - Determine affected users (if any)
   - Develop fix

3. **Remediation:**
   - Deploy fix to production
   - Verify fix effectiveness
   - Update dependencies if needed

4. **Communication:**
   - Notify users via GitHub (if impact is significant)
   - Publish post-mortem
   - Update security documentation

### Contact

For security concerns:
- **GitHub:** Open a security advisory (preferred)
- **Email:** Contact maintainer via GitHub profile

## Security Considerations for Contributors

If contributing code:

1. **No Secrets in Code:**
   - No API keys, tokens, or passwords
   - Use environment variables if needed (none currently required)

2. **Input Sanitization:**
   - All user inputs must be validated
   - Use Streamlit widget constraints

3. **Safe Dependencies:**
   - Only add well-maintained packages
   - Check for known vulnerabilities before adding

4. **No Arbitrary Code Execution:**
   - No `eval()`, `exec()`, `__import__()`
   - No dynamic code generation

5. **Safe File Operations:**
   - Avoid file writes if possible
   - Use temporary directories if needed
   - Never write to user-specified paths

## Compliance

This project does not collect personal data and is therefore not subject to:
- GDPR (no personal data processing)
- COPPA (not directed at children, no data collection)
- HIPAA (no health information)
- PCI DSS (no payment processing)

## License Security Note

This project uses the [Hippocratic License 3.0](LICENSE.md), which includes ethical use restrictions. Using this software for harmful purposes is a license violation, not just an ethical concern.

## Acknowledgments

Security policy inspired by:
- OWASP Security Practices
- Streamlit Security Guidelines
- Academic Software Best Practices

Last Updated: 2026-02-05
