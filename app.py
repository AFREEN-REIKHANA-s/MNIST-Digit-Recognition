import os
import streamlit as st
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

model_path = os.path.join(
    BASE_DIR,
    "mnist_model.keras"
)


# Check model exists
if not os.path.exists(model_path):

    st.error(
        "❌ Model file not found."
    )

    st.info(
        "Make sure mnist_model.keras "
        "is inside the same folder as app.py."
    )

    st.stop()


# Load model
model = load_model("C:\\Users\\Dell\\mnist_model.keras")


# ============================================================
# TITLE
# ============================================================

st.title("🔢 Handwritten Digit Recognition")

st.write(
    "Upload an image containing a single digit "
    "or a number containing multiple digits."
)

st.info(
    "Examples: 7, 25, 123, 2026"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload your handwritten image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# FUNCTION: PREPARE ONE DIGIT
# ============================================================

def prepare_digit(digit):

    # Find non-zero pixels
    coords = cv2.findNonZero(digit)

    if coords is not None:

        x, y, w, h = cv2.boundingRect(coords)

        digit = digit[
            y:y + h,
            x:x + w
        ]

    # Get dimensions
    height, width = digit.shape

    # Make square
    size = max(height, width)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # Center digit
    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    square[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = digit

    # Resize to 20 x 20
    square = cv2.resize(
        square,
        (20, 20),
        interpolation=cv2.INTER_AREA
    )

    # Create 28 x 28 MNIST image
    final_image = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Put digit in center
    final_image[
        4:24,
        4:24
    ] = square

    # Normalize
    img_array = (
        final_image.astype("float32") / 255.0
    )

    # Add batch dimension
    img_array = img_array.reshape(
        1,
        28,
        28
    )

    return img_array, final_image


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("L")

    image_array = np.array(image)


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    st.subheader("📷 Original Image")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )


    # --------------------------------------------------------
    # INVERT IMAGE
    # --------------------------------------------------------

    # Convert white background + black digit
    # into black background + white digit

    if np.mean(image_array) > 127:

        image_array = 255 - image_array


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        image_array,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    # --------------------------------------------------------
    # REMOVE SMALL NOISE
    # --------------------------------------------------------

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )


    # ========================================================
    # REMOVE IMAGE BORDER
    # ========================================================

    height, width = binary.shape

    border_x = max(
        2,
        int(width * 0.02)
    )

    border_y = max(
        2,
        int(height * 0.02)
    )

    binary[:border_y, :] = 0
    binary[-border_y:, :] = 0
    binary[:, :border_x] = 0
    binary[:, -border_x:] = 0


    # --------------------------------------------------------
    # DISPLAY PROCESSED IMAGE
    # --------------------------------------------------------

    st.subheader("🔍 Detected Writing")

    st.image(
        binary,
        caption="Image used for digit detection",
        width=400
    )


    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # ========================================================
    # FIND DIGIT BOXES
    # ========================================================

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(
            contour
        )

        # Ignore small noise
        if area < 100:
            continue

        # Ignore short objects
        if h < 20:
            continue

        # Ignore very narrow objects
        if w < 5:
            continue

        # Ignore border objects
        if (
            x <= 2
            or y <= 2
            or x + w >= width - 2
            or y + h >= height - 2
        ):
            continue

        boxes.append(
            (x, y, w, h)
        )


    # ========================================================
    # SORT LEFT TO RIGHT
    # ========================================================

    boxes = sorted(
        boxes,
        key=lambda box: box[0]
    )


    # ========================================================
    # CHECK DETECTED DIGITS
    # ========================================================

    if len(boxes) == 0:

        st.error(
            "❌ No digit detected."
        )

        st.info(
            "Please upload a clearer handwritten image."
        )

        st.stop()


    if len(boxes) > 4:

        st.warning(
            f"⚠️ {len(boxes)} objects detected."
        )

        st.info(
            "Please upload an image containing "
            "1 to 4 digits."
        )

        st.stop()


    st.success(
        f"✅ Detected {len(boxes)} digit(s)"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    predicted_number = ""

    confidences = []

    processed_digits = []


    for box in boxes:

        x, y, w, h = box

        # Crop digit
        digit = binary[
            y:y + h,
            x:x + w
        ]

        # Prepare digit
        input_image, processed_image = (
            prepare_digit(digit)
        )

        # Model prediction
        prediction = model.predict(
            input_image,
            verbose=0
        )

        # Probability of digits 0-9
        probabilities = (
            prediction[0] * 100
        )

        # Predicted digit
        predicted_digit = int(
            np.argmax(probabilities)
        )

        # Confidence
        confidence = float(
            np.max(probabilities)
        )

        # Add predicted digit
        predicted_number += str(
            predicted_digit
        )

        confidences.append(
            confidence
        )

        # Store all information
        processed_digits.append(
            (
                processed_image,
                predicted_digit,
                confidence,
                probabilities
            )
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    st.subheader("🎯 Prediction")

    if len(boxes) == 1:

        st.success(
            f"Predicted Digit: {predicted_number}"
        )

    else:

        st.success(
            f"Predicted Number: {predicted_number}"
        )


    # ========================================================
    # AVERAGE CONFIDENCE
    # ========================================================

    average_confidence = (
        sum(confidences)
        / len(confidences)
    )

    st.info(
        f"📊 Average Confidence: "
        f"{average_confidence:.2f}%"
    )


    # ========================================================
    # INDIVIDUAL PREDICTIONS
    # ========================================================

    st.subheader(
        "🔍 Individual Digit Predictions"
    )

    for i, (
        processed_image,
        predicted_digit,
        confidence,
        probabilities
    ) in enumerate(processed_digits):

        st.write(
            f"Digit {i + 1}: "
            f"**{predicted_digit}** "
            f"— Confidence: "
            f"{confidence:.2f}%"
        )


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    st.subheader(
        "📊 Prediction Probability"
    )

    for i, (
        processed_image,
        predicted_digit,
        confidence,
        probabilities
    ) in enumerate(processed_digits):

        st.write(
            f"Probability Distribution "
            f"for Digit {i + 1}"
        )

        probability_data = {
            str(digit): float(
                probabilities[digit]
            )
            for digit in range(10)
        }

        st.bar_chart(
            probability_data
        )


    # ========================================================
    # PROCESSED DIGITS
    # ========================================================

    st.subheader(
        "🖼️ Processed Digits"
    )

    for i, (
        processed_image,
        predicted_digit,
        confidence,
        probabilities
    ) in enumerate(processed_digits):

        st.image(
            processed_image,
            caption=(
                f"Digit {i + 1} → "
                f"{predicted_digit}"
            ),
            width=100
        )