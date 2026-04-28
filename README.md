
## What I Learned

- Naive Bayes makes strong independence assumptions that don't hold in real medical data (BMI and skin thickness are correlated, for example)
- Feature selection matters — removing pregnancies didn't hurt accuracy much but made the model more inclusive
- Small imbalanced datasets need stratified splitting, otherwise metrics become unreliable
- A simple model with clear probabilities is sometimes more useful than an accurate but unexplainable one

## Limitations

- Trained on a specific population (Pima Indian women) — may not generalize well
- Zero-value handling is a rough approximation
- Not suitable for actual clinical use — this is a learning project

## Future Improvements

- Try other algorithms (Logistic Regression, Random Forest, SVM) for comparison
- Add cross-validation for more reliable metrics
- Include feature correlation analysis
- Build a Python backend for live model training

---

*Built as part of my data science learning journey.*