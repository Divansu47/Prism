from mass_estimator import MassEstimator

estimator = MassEstimator()

result = estimator.estimate(
    "apple",
    159.64
)

print()
print("="*45)
print("Mass Estimation")
print("="*45)

for k, v in result.items():
    print(f"{k:10}: {v}")

print("="*45)