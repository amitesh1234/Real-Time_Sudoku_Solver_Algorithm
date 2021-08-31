# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import keras

bottom_corner=""
image_width=""
image_height=""


def preprocess(img):
    
    bl_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (9,9), 0).astype('uint8')
    bl_img1=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    threshold_img=cv2.adaptiveThreshold(bl_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    invert_img=cv2.bitwise_not(threshold_img)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    final_img = cv2.dilate(invert_img, kernel)
    
    ################################lecture 2
    
    contours, hierarchy = cv2.findContours(final_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours,  key=cv2.contourArea, reverse=True)
    bounding_box=""
    for c in contours_sorted:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            # Here we are looking for the largest 4 sided contour
            bounding_box=approx
            break
    
    #################sorting corners: [top_left, bottom_left, bottom_right, top_right]
    corners=[]
    corners_temp=[(box[0][0], box[0][1]) for box in bounding_box]
    corners_sorted=sorted(corners_temp, key=lambda x:x[0])
    if corners_sorted[0][1] < corners_sorted[1][1]:
        corners.append(corners_sorted[0])
        corners.append(corners_sorted[1])
    else:
        corners.append(corners_sorted[1])
        corners.append(corners_sorted[0])
        
    if corners_sorted[2][1] < corners_sorted[3][1]:
        corners.append(corners_sorted[3])
        corners.append(corners_sorted[2])
    else:
        corners.append(corners_sorted[2])
        corners.append(corners_sorted[3])
        
    bottom_corner=corners[1]
    
    width_A = np.sqrt(((corners[3][0] - corners[2][0]) ** 2) + ((corners[3][1] - corners[2][1]) ** 2))
    width_B = np.sqrt(((corners[0][0] - corners[1][0]) ** 2) + ((corners[0][1] - corners[1][1]) ** 2))
    width = max(int(width_A), int(width_B))
    image_width=width
    
    height_A = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) + ((corners[0][1] - corners[3][1]) ** 2))
    height_B = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
    height = max(int(height_A), int(height_B))
    image_height=height
    
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")
    
    ordered_corners=[corners[0], corners[3], corners[2], corners[1]]
    ordered_corners = np.array(ordered_corners, dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    cropped_image= cv2.warpPerspective(invert_img, grid, (width, height))
    
    grid_h=np.shape(cropped_image)[0]
    grid_w=np.shape(cropped_image)[1]
    cell_h=grid_h//9
    cell_w=grid_w//9
    
    grid_cells=[]
    temp_grid=[]
    for i in range(cell_h, grid_h+1, cell_h):
        for j in range(cell_w, grid_w+1, cell_w):
            row=cropped_image[i-cell_h:i]
            temp_grid.append([row[k][j-cell_w:j] for k in range(len(row))])
    
    for i in range(0, len(temp_grid)-8, 9):
        grid_cells.append(temp_grid[i:i+9])
    
    for i in range(9):
        for j in range(9):
            grid_cells[i][j]=np.array(grid_cells[i][j])
    return grid_cells


def ocr(grid_cells):
    model = keras.models.load_model("model.h5")

    thresh=128
    sudoku_grid=np.zeros((9,9))
    
    for i in range(9):
        for j in range(9):
            num=grid_cells[i][j]
            
            gray=cv2.threshold(num, thresh, 255, cv2.THRESH_BINARY)[1]
            cnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                x,y,w,h=cv2.boundingRect(c)
                if (x < 3 or y < 3 or h < 10 or w < 10):
                    continue
                n_area = gray[y:y + h, x:x + w]
                n_area = cv2.resize(n_area, (28,28))
                n_area=n_area/255
                n_area=n_area.reshape(1,28,28,1)
                
                pred = model.predict(n_area, batch_size=1)
                print(pred.argmax())
                sudoku_grid[i][j]=pred.argmax()
    return sudoku_grid



N=9

def check(grid, row, col, num):
    for i in range(N):
        if grid[row][i]==num or grid[i][col]==num:
            return False
    startrow = row - row%3
    startcol = col - col%3
    for i in range(3):
        for j in range(3):
            if grid[startrow + i][startcol + j] == num:
                return False
    return True

def solver(grid, row, col):
    if row==N-1 and col==N:
        return True
    
    if col==N:
        col=0
        row+=1
    
    if grid[row][col]>0:
        return solver(grid, row, col+1)
    
    for i in range(1,N+1):
        if check(grid, row, col, i):
            grid[row][col]=i
            if solver(grid, row, col+1):
                return True
        grid[row][col]=0
    return False

def showtext(sudoku_grid, frame):
    bottom_x=bottom_corner[1]
    bottom_y=bottom_corner[0]
    cell_h=image_height/9
    cell_w=image_width/9
    print("DOne!!!!")
    

cap = cv2.VideoCapture(0)
flag=False
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.rectangle(frame, (140,5), (480,345), (0,255,0),1) 
    cv2.imshow('frame', frame)
    """
    if flag:
        showtext(sudoku_grid, frame)
    cv2.imshow('frame', frame)
    """
    img=frame[6:345,141:480]
    grid_cells=preprocess(img)
    sudoku_grid=ocr(grid_cells)
    if  sudoku_grid.any():
        if solver(sudoku_grid,0,0):
            flag=True
    else:
        flag=False
    print(flag)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

