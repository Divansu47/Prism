class NutritionScaler:

    def __init__(self):
        pass

    def scale(self, nutrients, estimated_mass, reference_mass):

        factor = estimated_mass / reference_mass

        scaled = {}

        for key, value in nutrients.items():

            if isinstance(value, (int, float)):
                scaled[key] = round(value * factor, 2)
            else:
                scaled[key] = value

        scaled["Estimated Mass (g)"] = round(estimated_mass, 2)
        scaled["Scaling Factor"] = round(factor, 3)

        return scaled