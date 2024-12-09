import cv2

def test_camera(camera_index, backend=None):
    print(f"Testing camera at index {camera_index}...")
    # Open the camera using the specified backend if provided
    cap = cv2.VideoCapture(camera_index, backend) if backend else cv2.VideoCapture(cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Failed to open camera at index {camera_index}")
        return False

    print(f"Camera at index {camera_index} is working. Press 'q' to close the feed.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_index}. Retrying...")
            continue
        cv2.imshow(f"Camera {camera_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

# Main test
def main():
    backend = cv2.CAP_DSHOW  # Use DirectShow on Windows for better compatibility
    for camera_index in range(2):  # Test indices 0 and 1
        if test_camera(camera_index, backend=backend):
            print(f"Camera {camera_index} seems functional. Use this index in your application.")
        else:
            print(f"Camera {camera_index} is not functional.")

if __name__ == "__main__":
    main()
