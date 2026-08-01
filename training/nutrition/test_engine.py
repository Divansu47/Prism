from engine import NutritionEngine

engine = NutritionEngine()

result = engine.predict(
    food_name="apple",
    estimated_mass=129.3
)

print()
print("=" * 70)
print("ESTIMATED NUTRITION")
print("=" * 70)

for key, value in result.items():
    print(f"{key:28}: {value}")