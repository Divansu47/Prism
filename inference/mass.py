DENSITY_TABLE = {

    "bread": 0.25,

    "rice": 0.9,

    "noodle": 0.6,

    "pasta": 0.6,

    "potato": 0.65,

    "sauce": 1.0,

    "gravy": 1.0,

    "curry": 1.0,

    "soup": 1.0,

    "vegetable": 0.6,

    "salad": 0.4,

    "meat": 1.0,

    "chicken": 1.0,

    "fish": 1.0,

    "egg": 1.0,

    "protein": 1.0,

    "fruit": 0.85,

    "fig": 0.85,

    "banana": 0.95,

    "cheese": 1.1,

}

DEFAULT_DENSITY = 0.7

# Calibration constant converting relative_volume units
# (pixel_area * avg_depth) into cubic centimeters.
#
# Derived from a single real-world anchor point (no reference
# object or depth sensor available yet):
#   - Test image bread slice ~35g actual mass
#   - relative_volume = 84080.73, density(bread) = 0.25
#   - CALIBRATION_CONSTANT = 35 / (84080.73 * 0.25) ≈ 0.00166
#
# This is a rough global scale factor, not a true depth
# calibration. It will drift for images taken from different
# camera heights/angles. Proper fix later: detect a reference
# object (e.g. plate of known diameter) per image and compute
# a per-image pixel-to-cm scale instead of this fixed constant.
CALIBRATION_CONSTANT = 0.00166


def get_density(name):

    key = str(name).lower()

    for keyword, density in DENSITY_TABLE.items():

        if keyword in key:
            return density

    return DEFAULT_DENSITY


def estimate_volume_cm3(relative_volume):

    return relative_volume * CALIBRATION_CONSTANT


def estimate_mass(relative_volume, food_name):

    density = get_density(food_name)

    volume_cm3 = estimate_volume_cm3(relative_volume)

    mass_grams = volume_cm3 * density

    return mass_grams


def scale_nutrition(nutrition_per_100g, mass_grams):

    factor = mass_grams / 100.0

    scaled = {}

    for key, value in nutrition_per_100g.items():

        if isinstance(value, (int, float)):
            scaled[key] = value * factor
        else:
            scaled[key] = value

    return scaled