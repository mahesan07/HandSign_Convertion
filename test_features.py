from features import extract_features


# Create fake landmarks
class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# Create 21 fake landmarks
hand = []

for i in range(21):
    hand.append(
        Landmark(
            x=0.1 + i * 0.01,
            y=0.2 + i * 0.01,
            z=0.3 + i * 0.01
        )
    )


features = extract_features(hand)


print("Number of features:", len(features))
print("Features:")
print(features)