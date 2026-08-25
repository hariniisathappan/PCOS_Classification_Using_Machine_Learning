
---

## 🧠 Conclusion

- **Feature Engineering** grounded in Rotterdam clinical criteria improves mean F1 across both baseline (+2.3%) and tuned (+3.1%) pipelines, while raising the **EPV ratio from 4.02 to 16.09**, satisfying statistical validity requirements for parametric modelling.
- **AdaBoost** is the strongest baseline (default) model; **SVM (RBF)** is the strongest model once Bayesian-tuned with engineered features.
- **Logistic Regression** is remarkably consistent across all four conditions and offers the best-calibrated probability outputs (Brier = 0.069), making it ideal for clinical risk scoring.
- The **Four Archetype Framework** explains *why* some models need FE + tuning together (Bimodal Sensitivity), while others are largely self-sufficient (Default Optimality).

---

## 📚 References

Full reference lists are available in the original research manuscripts included in this repository (see `/notebooks` or contact for the complete papers). Key sources include the Rotterdam Consensus criteria, CHARMS/PROBAST guidelines for clinical prediction models, and the publicly available Kaggle PCOS dataset (Kottarathil, 2020; Pathak, S.V.).

---

## 👩‍💻 Author

**Harini S** — Division of Bioinformatics, SASTRA Deemed University, Thanjavur
Research collaboration with National Institute of Technology, Puducherry

---

## 📄 License

This project is licensed under the MIT License.
