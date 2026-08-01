from lookup import NutritionLookup
from scaler import NutritionScaler


class NutritionEngine:

    def __init__(self):

        self.lookup = NutritionLookup()
        self.scaler = NutritionScaler()

    def predict(self, food_name, estimated_mass):

        nutrients = self.lookup.find_food(food_name)

        if nutrients is None:
            return None

        reference_mass = self.lookup.get_reference_mass(food_name)

        if reference_mass is None:
            raise ValueError(
                f"No serving reference found for '{food_name}'."
            )

        return self.scaler.scale(
            nutrients,
            estimated_mass,
            reference_mass
        )