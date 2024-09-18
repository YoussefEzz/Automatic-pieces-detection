import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("images/image3.png")
original_image = image.copy()  # Make a copy for displaying the original image
height, width, channels = image.shape
if(height < 400 or width < 400):
                # Determine the aspect ratio of the image
                aspect_ratio = width / height
                new_width = 800
                new_height = int(new_width / aspect_ratio)
                image = cv2.resize(image, (new_width, new_height))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Define a convolution kernel (filter)
kernel1 = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

# Apply convolution using the filter
convolved_image = cv2.filter2D(blurred, -1, kernel1)

height, width = convolved_image.shape
bottom_border = np.array([0, height, width, height ])
left_border = np.array([0, 0, 0, height ])
lines = []
lines.append(left_border)
lines.append(bottom_border)

linesP = cv2.HoughLinesP(convolved_image, 1, np.pi / 180, 50, 1, 400, 0)


# Draw the hough lines and borders
if linesP is not None:
    leng = len(linesP)
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(original_image, (l[0], l[1]), (l[2], l[3]), (0,255,0), 5, cv2.LINE_AA)
        lines.append(l)
cv2.line(original_image, (bottom_border[0], bottom_border[1]), (bottom_border[2], bottom_border[3]), (0,255,0), 10, cv2.LINE_AA)
cv2.line(original_image, (left_border[0], left_border[1]), (left_border[2], left_border[3]), (0,255,0), 10, cv2.LINE_AA)


# Use HoughCircles to detect circles
circles = cv2.HoughCircles(
     blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=50, param2=40, minRadius=100, maxRadius=200
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        x = i[0]
        y = i[1]
        r = i[2]
        # Create a mask to extract the circular region
        mask = np.zeros_like(blurred)
        cv2.circle(mask, (x, y), r, 255, -1)
        # Calculate the circularity as the ratio of the area to the perimeter
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(image, contours, 0, 0, 0, 255, 2, 8)
        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        circularity =  4 * np.pi * area / (perimeter ** 2)
        fontH=1
        fontT=1
        perimeter_str= str(round(perimeter, 2))
        area_str = str(round(area, 2))
        circularity_str = str(round(circularity, 2))
        radius_str = str(round(r, 2))
        myFont=cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(image, radius_str, (x,y), myFont, fontH, (0,0,255), fontT)
        cv2.putText(image, area_str, (x - 60,y), myFont, fontH, (0,255,255), fontT)
        cv2.putText(image, perimeter_str, (x - 60,y-30), myFont, fontH, (0,255,255), fontT)
        cv2.putText(image, circularity_str, (x - 60,y-60), myFont, fontH, (0,255,255), fontT)
        filtered = True
        if lines is not None:
            for line in lines:
                x1 = line[0]
                y1 = line[1]
                x2 = line[2]
                y2 = line[3]

                # Calculate the distance between the circle center and the line
                distance = np.abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Check if the distance is within a threshold
                if distance < r:
                    filtered = False
                    break
                else:
                    continue
        else:
            # Draw the outer circle
            cv2.circle(image, (x, y), r, (255, 0, 0), 5)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        if filtered == True:
            # Draw the outer circle
            cv2.circle(image, (x, y), r, (255, 0, 0), 5)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        
        
# Display the original and LBP images
plt.figure(figsize=(10, 7))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
#plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(image, cmap='gray')
#plt.title("Circle Detection")

plt.subplot(1, 3, 3)
plt.imshow(convolved_image, cmap='gray')
#plt.title("Circle Detection")

plt.tight_layout()


plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()



    