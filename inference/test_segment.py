from segment import FoodSegmenter

segmenter = FoodSegmenter()

result = segmenter.predict("assets/test.jpg")

print(result)