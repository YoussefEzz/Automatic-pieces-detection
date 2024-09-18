import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

directory = "images"

j = 1
plt.figure(figsize=(100, 70))
for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            
            image_path = os.path.join(directory, filename)

            # Load image
            image = cv2.imread(image_path)
            height, width, channels = image.shape
            
            # Determine the aspect ratio of the image
            aspect_ratio = width / height
            new_width = 800
            new_height = int(new_width / aspect_ratio)
            image = cv2.resize(image, (new_width, new_height))
            resized_image = image.copy()  # Make a copy for displaying the original image

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and improve contour detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Define a convolution kernel (filter)
            kernel1 = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

            # Apply convolution using the filter
            convolved_image = cv2.filter2D(blurred, -1, kernel1)

            new_convolved_height, new_convolved_width = convolved_image.shape
            bottom_border = np.array([0, new_convolved_height, new_convolved_width, new_convolved_height ])
            left_border = np.array([0, 0, 0, new_convolved_height ])
            lines = []
            lines.append(left_border)
            lines.append(bottom_border)

            linesP = cv2.HoughLinesP(convolved_image, 1, np.pi / 180, 50, 1, 400, 0)


            # Draw the hough lines and borders
            if linesP is not None:
                leng = len(linesP)
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(resized_image, (l[0], l[1]), (l[2], l[3]), (0,255,0), 5, cv2.LINE_AA)
                    lines.append(l)
            cv2.line(resized_image, (bottom_border[0], bottom_border[1]), (bottom_border[2], bottom_border[3]), (0,255,0), 10, cv2.LINE_AA)
            cv2.line(resized_image, (left_border[0], left_border[1]), (left_border[2], left_border[3]), (0,255,0), 10, cv2.LINE_AA)


            # Use HoughCircles to detect circles
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=200, param1=50, param2=45, minRadius=100, maxRadius=200
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    x = i[0]
                    y = i[1]
                    r = i[2]
                    fontH=1
                    fontT=1
                    myFont=cv2.FONT_HERSHEY_DUPLEX
                    filtered = True
                    if lines is not None:
                        for line in lines:
                            x1 = line[0]
                            y1 = line[1]
                            x2 = line[2]
                            y2 = line[3]

                            # Calculate the perpendicular distance between the circle center and the line
                            distance = np.abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                            radius_str = str(round(r, 2))
                            cv2.putText(image,"radius = " +  radius_str, (x - 90,y), myFont, fontH, (0,255,255), fontT)
                        

                            # Check if the distance is within a threshold w.r.t radius
                            if distance - r < -10:
                                filtered = False
                                break
                            else:
                                continue

                    if filtered == True:
                        # Draw the outer circle
                        cv2.circle(image, (x, y), r, (255, 0, 0), 5)
                        # Draw the center of the circle
                        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

                            
            plt.subplot(7, 2, j)
            plt.imshow(resized_image, cmap='gray')
            #plt.title(filename)

            plt.subplot(7, 2, j + 1)
            plt.imshow(image, cmap='gray')
            #plt.title("Circle Detection")
                
            j = j + 2

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
