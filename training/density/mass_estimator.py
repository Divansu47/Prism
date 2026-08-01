from density_db import FOOD_DENSITY


class MassEstimator:

    def estimate(self,
                 food_name,
                 volume):

        density = FOOD_DENSITY.get(food_name.lower())

        if density is None:
            raise ValueError(f"Unknown food: {food_name}")

        mass = volume * density

        return {
            "food": food_name,
            "density": density,
            "volume": volume,
            "mass": mass
        }