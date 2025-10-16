# camera_show.py
# Display live video feed from Raspberry Pi camera using picamera2

from picamera2 import Picamera2
import cv2 as cv

# Initialize camera
camera = Picamera2()

# Configure for preview at 1024x768
config = camera.create_preview_configuration(main={"size": (1024, 768)})
camera.configure(config)

# Start the camera
camera.start()

print("Displaying live video feed. Press 'q' to quit.")

try:
    while True:
        # Capture frame
        frame = camera.capture_array()

        # Display the frame
        cv.imshow('Live Camera Feed', frame)

        # Break loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping camera...")

finally:
    # Cleanup
    cv.destroyAllWindows()
    camera.stop()
    camera.close()
    print("Camera stopped.")
