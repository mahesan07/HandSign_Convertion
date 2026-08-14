import math


def extract_features(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks
    into 63 normalized features.

    Returns:
        list of 63 values
    """

    # -----------------------------------
    # 1. Get wrist landmark
    # -----------------------------------

    wrist = hand_landmarks[0]

    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z


    # -----------------------------------
    # 2. Calculate normalized coordinates
    # -----------------------------------

    normalized_landmarks = []

    for landmark in hand_landmarks:

        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z

        normalized_landmarks.append(
            (x, y, z)
        )


    # -----------------------------------
    # 3. Calculate hand scale
    # -----------------------------------

    # Landmark 9 = middle finger MCP
    middle_mcp = hand_landmarks[9]

    dx = middle_mcp.x - wrist_x
    dy = middle_mcp.y - wrist_y
    dz = middle_mcp.z - wrist_z

    scale = math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )


    # -----------------------------------
    # 4. Prevent division by zero
    # -----------------------------------

    if scale == 0:
        scale = 1


    # -----------------------------------
    # 5. Scale-normalize the landmarks
    # -----------------------------------

    features = []

    for x, y, z in normalized_landmarks:

        x = x / scale
        y = y / scale
        z = z / scale

        features.extend([
            x,
            y,
            z
        ])


    return features