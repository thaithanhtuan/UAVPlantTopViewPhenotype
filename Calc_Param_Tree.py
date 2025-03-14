import glob
import cv2
import numpy as np
import json
import math
import csv
import pickle
from FastLine import Line
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
# import matplotlib
# matplotlib.use("GTK3Agg")

### function to find slope
def slope(p1,p2):
    x1,y1=p1
    x2,y2=p2
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

### main function to draw lines between two points
def drawLine(image,p1,p2, color=(0, 255, 0), thickness = 2):
    x1,y1=p1
    x2,y2=p2
    ### finding slope
    m=slope(p1,p2)
    ### getting image shape
    h,w=image.shape[:2]

    if m!='NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px,py=x1,0
        qx,qy=x1,h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color, thickness)
    return image

def calculate_slopes(points):
    slopes = []
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i < j:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                slope = np.arctan2(dy, dx)  # Angle in radians
                slopes.append((i, j, slope))
    return slopes


def align_and_cluster(points, eps=0.3):
    slopes = calculate_slopes(points)
    angles = np.array([s[2] for s in slopes]).reshape(-1, 1)  # Use slope angles for clustering

    clustering = DBSCAN(eps=eps, min_samples=2).fit(angles)  # Cluster based on alignment
    return clustering.labels_


seg_dir = 'D:/Jeju/Thai/Dataset/Forest Dataset For Tree Instance Segmentation/Crawl Dataset/TreeUAV/'

Tree_list = []
Tree_ellipse_list = []

points_list = []
PIXEL2MICROMET = 0.17
out_data_file = 'Data_1.csv'
# open the file in the write mode
# with open('Data.csv', 'w') as csv_f:


with open(out_data_file, 'w', newline='') as csv_f:
    # create the csv writer
    writer = csv.writer(csv_f)
    # image path, image height, image width
    # Tree width, Tree length, Tree angle, Tree area, sto ellipse area
    # pore width, pore length, pore angle, pore area, pore ellipse area
    fields = ['No.', 'Image Path', 'Height', 'Width', 'Tree Width', 'Tree Length', 'Tree Angle', 'Tree Area', 'Tree Elip Area', 'Tree center X', 'Tree center Y']
    writer.writerow(fields)

    for file_path in glob.glob(seg_dir + '*.json'):
        if not "Screenshot 2024-10-20 231711.json" in file_path:
            continue
        line_data = ""
        img_path = file_path.replace(".json",".png")

        image = cv2.imread(img_path)
        image_ori = cv2.imread(img_path)

        h, w = image.shape[:2]
        # print (h , w)
        # cv2.imshow("Hello", image)
        # cv2.waitKey(0)
        Tree_list = []
        Tree_ellipse_list = []


        points_list = []

        with open(file_path, 'r') as file:
            data = json.load(file)
            print(data["imagePath"])
            # if(data["imagePath"] != "B (11).JPG"):
            #      continue
            line_data = data["imagePath"] + ", " + str(data["imageHeight"]) + ", " + str(data["imageWidth"])
            for item in data['shapes']:

                label = item["label"]
                if(label=="Tree"):
                    Tree_list.append(item["points"])
            print("Trees:", len(Tree_list))

            # continue

            ellipse_list = []

            index = 0
            mask = np.zeros(image.shape, dtype='uint8')
            maskID = np.zeros(image.shape, dtype='uint8')
            for Tree in Tree_list:
                mask_i = np.zeros(image.shape, dtype='uint8')
                index = index + 1
                # Convert list to list of tuples
                points_tuples = [tuple(point) for point in Tree]
                # print('points_tuples: ', points_tuples)

                # Make the list to array
                points_arr = np.array(points_tuples, dtype=np.float32)
                # print('points_arr: ', points_arr)

                # Normalized points array to 8bit one
                # points_arr = normalize_array_to_8bit(points_arr)
                # print('points_arr: ', points_arr)

                ## Fit an ellipse to the points
                ellipse = cv2.fitEllipse(points_arr)
                bx, by, bw, bh = cv2.boundingRect(points_arr)
                ellipse_list.append(ellipse)
                # print(ellipse)

                # Retrieve ellipse parameters
                (center, axes, angle) = ellipse
                Tree_ellipse_list.append(ellipse)
                flag = False
                i = 0

                #Building the mask from contour

                zero_image = np.zeros(image.shape, dtype='uint8')
                points_arr = np.array(points_arr, dtype=int)

                """
                contour_length = 0
                for x,y in zip (points_arr, points_arr[1:]):
                    d = math.sqrt((x[1] - y[1]) * (x[1] - y[1]) + (x[0] - y[0]) * (x[0] - y[0]))
                    contour_length = contour_length + d
                    print(x,y,d,contour_length)
                """


                cv2.fillPoly(mask, pts=[points_arr], color=(1, 1, 1))
                cv2.fillPoly(mask_i, pts=[points_arr], color=(1, 1, 1))

                cv2.fillPoly(maskID, pts=[points_arr], color=(index, index, index))


                # cv2.drawContours(mask, [points_arr], -1, (255, 255, 255), 1)

                """
                contour_length = cv2.arcLength(points_arr, True)

                perimeter = np.pi * np.sqrt((axes[0] * axes[0] + axes[1] * axes[1]) / 2)
                perimeter1 = np.pi * (3 * (axes[0] / 2 + axes[1] / 2) - np.sqrt((3 * axes[0] / 2 + axes[1] / 2)*(axes[0] / 2 + 3 * axes[1] / 2)))
                print ("contour_length:", contour_length, ", perimeter:", perimeter, ", perimeter/contour_length: ", perimeter/contour_length, ", perimeter1:", perimeter1)
                
                """
                # cv2.imshow("mask", mask)
                # cv2.waitKey(0)

                numpixelmask = np.count_nonzero(mask_i) / 3

                tree = image_ori.copy() * mask_i
                tree = tree.astype(np.uint8)

                # cv2.imshow("mask", mask)
                # cv2.waitKey(0)

                #color in OpenCV Blue; Green; Red
                CALCULATE_DEGREE_COLOR = False
                if (CALCULATE_DEGREE_COLOR == True):
                    tree_B = tree.copy()
                    tree_B[:, :, 1:3] = zero_image[:, :, 1:3]
                    sum_B = np.sum(tree_B[:, :, 0])
                    average_B = sum_B / numpixelmask
                    tree_G = tree.copy()
                    tree_G[:, :, 0] = zero_image[:, :, 0]
                    tree_G[:, :, 2] = zero_image[:, :, 2]
                    sum_G = np.sum(tree_G[:, :, 1])
                    average_G = sum_G / numpixelmask
                    tree_R = tree.copy()
                    tree_R[:, :, 0] = zero_image[:, :, 0]
                    tree_R[:, :, 1] = zero_image[:, :, 1]
                    sum_R = np.sum(tree_R[:, :, 2])
                    average_R = sum_R / numpixelmask
                    tree_Y = tree.copy()
                    tree_Y[:, :, 0] = zero_image[:, :, 0]
                    sum_Y = np.sum(tree_Y[:, :, 1:3])
                    average_Y = sum_Y / (numpixelmask * 2)

                    font = cv2.FONT_HERSHEY_SIMPLEX

                    fontScale = 1
                    fontColor = (255, 0, 0)
                    thickness = 2
                    lineType = 2

                    bottomLeftCornerOfText = (int(center[0] - 40), int(center[1] + 40))
                    string = "Yellow: " + str(int(average_Y))
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
                    bottomLeftCornerOfText = (int(center[0] - 40), int(center[1] + 80))
                    string = "Red: " + str(int(average_R))
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
                    bottomLeftCornerOfText = (int(center[0] - 40), int(center[1] + 120))
                    string = "Green: " + str(int(average_G))
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)


                # cv2.imshow("tree_Y", tree_Y)
                # cv2.waitKey(0)

                # tree_exg = 2 * tree_G - tree_R - tree_B
                # mask_green = np.zeros(image.shape, dtype='uint8')

                # mask_green[:, :, 1] = ( tree_exg > 0 ) * tree_exg


                # Displaying the image
                # cv2.imshow("mask_green", mask_green)
                # cv2.waitKey(0)

                cv2.ellipse(image, ellipse, (0, 255, 0), 3)
                cv2.drawContours(image, [points_arr], -1, (0, 0, 255), 3)

                cv2.circle(image, (int(center[0]), int(center[1])), int(4), (0,0,255), -1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2
                fontColor = (0, 255, 255)
                thickness = 3
                lineType = 2
                bottomLeftCornerOfText = (int(center[0] - 40), int(center[1] - 40))
                string = str(index)
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                # cv2.imshow("hello", image)
                # cv2.waitKey(0)
                """
                #Draw the minor and major line
                (xc,yc),(d1,d2),angle = ellipse
                rmajor = max(d1, d2) / 2
                if angle > 90:
                    angle = angle - 90
                else:
                    angle = angle + 90
                print(angle)
                x1 = xc + math.cos(math.radians(angle)) * rmajor
                y1 = yc + math.sin(math.radians(angle)) * rmajor
                x2 = xc + math.cos(math.radians(angle + 180)) * rmajor
                y2 = yc + math.sin(math.radians(angle + 180)) * rmajor
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

                # draw minor axis line in blue
                rminor = min(d1, d2) / 2
                if angle > 90:
                    angle = angle - 90
                else:
                    angle = angle + 90
                print(angle)
                x1 = xc + math.cos(math.radians(angle)) * rminor
                y1 = yc + math.sin(math.radians(angle)) * rminor
                x2 = xc + math.cos(math.radians(angle + 180)) * rminor
                y2 = yc + math.sin(math.radians(angle + 180)) * rminor
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                """


                #Schoelace formula.
                # x,y are arrays containing coordinates of the polygon vertices
                x = np.array(points_arr[:, 0])
                y = np.array(points_arr[:, 1])
                i = np.arange(len(x))
                # Area=np.sum(x[i-1]*y[i]-x[i]*y[i-1])*0.5 # signed area, positive if the vertex sequence is counterclockwise
                Area = np.abs(np.sum(x[i - 1] * y[i] - x[i] * y[i - 1]) * 0.5)  # one line of code for the shoelace formula
                ellipse_area = math.pi * axes[0] * axes[1] / 4
                # image path, image height, image width

                line_data = [str(index), data["imagePath"], str(data["imageHeight"]), str(data["imageWidth"])]
                # Tree width, Tree length, Tree angle, Tree area, sto ellipse area
                line_data = line_data + [str(axes[0]), str(axes[1]), str(angle), str(Area), str(ellipse_area), str(center[0]), str(center[1])]
                #, 'Height(micromet)', 'Width(micromet)', 'Sto Width(micromet)', 'Sto Length(micromet)', 'Sto Angle(micromet)', 'Sto Area(micromet)', 'Sto Elip Area(micromet)', 'Sto center X(micromet)', 'Sto center Y(micromet)', 'Pore Width(micromet)', 'Pore Length(micromet)', 'Pore Angle(micromet)', 'Pore Area(micromet)', 'Pore Elip Area(micromet)', 'Pore center X(micromet)', 'Pore center Y(micromet)']

                writer.writerow(line_data)

                PRINT_PHENO_TOPVIEW = False
                if(PRINT_PHENO_TOPVIEW == True):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 2
                    fontColor = (0, 255, 255)
                    thickness = 3
                    lineType = 2
                    bottomLeftCornerOfText = (int(center[0] - 40), int(center[1] - 40))
                    string = str(index)
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)

                    fontScale = 0.7
                    fontColor = (255, 0, 0)
                    thickness = 2
                    lineType = 2
                    bottomLeftCornerOfText = (int(center[0] - 110), int(center[1] + 0))
                    string = "(BBW:BBH) : (" + str(bw) + ":" + str(bh) + ")"
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)


                    bottomLeftCornerOfText = (int(center[0] - 110), int(center[1] + 40))
                    string = "Area: " + str(Area)
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)

                    bottomLeftCornerOfText = (int(center[0] - 110), int(center[1] + 80))
                    emajor = max(axes[0], axes[1])
                    eminor = min(axes[0], axes[1])
                    roundness = eminor/emajor
                    string = "(Em:EM:EO) : (" + str(int(eminor)) + ":" + str(int(emajor)) + ":" + str(int(
                        angle)) + ")"  # minor: major: orientation : (center, axes, angle) = ellipse
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)

                    bottomLeftCornerOfText = (int(center[0] - 110), int(center[1] + 120))
                    string = "Round : " + str(round(roundness,2))
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
                """
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (int(center[0] + 10), int(center[1] + 10))
                fontScale = 1
                fontColor = (0, 0, 255)
                thickness = 2
                lineType = 2
                fields = ['Tree Width', 'Tree Length', 'Tree Angle', 'Tree Area',
                          'Tree Elip Area', 'Tree center X', 'Tree center Y']
                string = "Tree Width: " + str(int(axes[0])) + ", " + "Tree Length: " + str(int(axes[1]))
                
                """
                """
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                bottomLeftCornerOfText = (int(center[0] + 10), int(center[1] + 40))
                string = "Tree Angle: " + str(int(angle)) + ", " + "Tree Area: " + str(int(Area))
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                bottomLeftCornerOfText = (int(center[0] - 80), int(center[1] - 80))
                string = str(index)
                fontScale = 3
                fontColor = (255, 0, 0)
                thickness = 5
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                """

            CALCULATE_DEGREE_COLOR_ALL = False
            if(CALCULATE_DEGREE_COLOR_ALL == True):
                numpixelmask_all = np.count_nonzero(mask) / 3
                tree_all = image_ori.copy()
                tree_all = tree_all * mask
                tree_all = tree_all.astype(np.uint8)
                if (CALCULATE_DEGREE_COLOR_ALL == True):
                    cv2.imwrite(file_path.replace(".json", "_TreeALL.jpg"), tree_all)
                # cv2.imshow("tree_all",tree_all)
                # cv2.waitKey(0)

                tree_all_B = tree_all.copy()
                tree_all_B[:, :, 1:3] = zero_image[:, :, 1:3]
                if (CALCULATE_DEGREE_COLOR_ALL == True):
                    cv2.imwrite(file_path.replace(".json", "_Tree_B_ALL.jpg"), tree_all_B)
                sum_all_B = np.sum(tree_all_B[:, :, 0])
                average_all_B = sum_all_B / numpixelmask_all
                tree_all_G = tree_all.copy()
                tree_all_G[:, :, 0] = zero_image[:, :, 0]
                tree_all_G[:, :, 2] = zero_image[:, :, 2]
                if (CALCULATE_DEGREE_COLOR_ALL == True):
                    cv2.imwrite(file_path.replace(".json", "_Tree_G_ALL.jpg"), tree_all_G)
                sum_all_G = np.sum(tree_all_G[:, :, 1])
                average_all_G = sum_all_G / numpixelmask_all
                tree_all_R = tree_all.copy()
                tree_all_R[:, :, 0] = zero_image[:, :, 0]
                tree_all_R[:, :, 1] = zero_image[:, :, 1]
                if (CALCULATE_DEGREE_COLOR_ALL == True):
                    cv2.imwrite(file_path.replace(".json", "_Tree_R_ALL.jpg"), tree_all_R)
                sum_all_R = np.sum(tree_all_R[:, :, 2])
                average_all_R = sum_all_R / numpixelmask_all
                tree_all_Y = tree_all.copy()
                tree_all_Y[:, :, 0] = zero_image[:, :, 0]
                if (CALCULATE_DEGREE_COLOR_ALL == True):
                    cv2.imwrite(file_path.replace(".json", "_Tree_Y_ALL.jpg"), tree_all_Y)
                sum_all_Y = np.sum(tree_all_Y[:, :, 1:3])
                average_all_Y = sum_all_Y / (numpixelmask_all * 2)
                string = "Yellow all: " + str(round(average_all_Y, 2))
                print(string)
                string = "Red all: " + str(round(average_all_R, 2))
                print(string)
                string = "Green all: " + str(round(average_all_G, 2))
                print(string)



            CALCULATE_BETWEEN_TREE = False
            if(CALCULATE_BETWEEN_TREE==True):
                list_tree_direct = []
                list_tree_inline = []
                index = 0
                for e in ellipse_list:

                    index = index + 1

                    if not index == 6:
                        continue
                    number_of_direct_neighbor = 0
                    # calculate the distance from ellipse to others
                    direct_tree = []
                    inline_tree = []
                    (center, axes, angle) = e
                    for e1 in ellipse_list:
                        (center1, axes1, angle1) = e1
                        mask_line = np.zeros(image.shape, dtype='uint8')
                        mask_line = cv2.line(mask_line, (int(center[0]), int(center[1])), (int(center1[0]),int(center1[1])), color= (1,1,1), thickness = 2)

                        numpixelmask_line = np.count_nonzero(mask_line) / 3

                        new_array = mask_line * mask
                        numpixelmask_tree = np.count_nonzero(new_array) / 3
                        new_array = np.concatenate(new_array[:,:,0])
                        set_array = set(new_array)
                        new_array = np.array(list(set_array))
                        print(new_array)
                        inline_tree.append(new_array)
                        if(len(set_array) == 3):
                            number_of_direct_neighbor = number_of_direct_neighbor + 1
                            direct_tree.append(e1)
                            print("numpixelmask_tree/numpixelmask_line: ", numpixelmask_tree / numpixelmask_line)
                            cv2.line(image, (int(center[0]), int(center[1])), (int(center1[0]), int(center1[1])),
                                     color=(0, 255, 255), thickness=2)

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            fontColor = (255, 0, 0)
                            thickness = 2
                            lineType = 2
                            bottomLeftCornerOfText = (int((center[0] + center1[0]) / 2), int((center[1] + center1[1]) / 2) - 10)
                            string = "ER: " + str((round(1 - (numpixelmask_tree / numpixelmask_line), 2))) #Empty ratio
                            cv2.putText(image, string,
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        thickness,
                                        lineType)


                        cv2.imwrite(file_path.replace(".json", "_mask_line_viz.jpg"), mask_line)
                        # cv2.imshow("mask_line",mask_line)
                        # cv2.waitKey(0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    fontColor = (255, 0, 0)
                    thickness = 2
                    lineType = 2

                    bottomLeftCornerOfText = (int(center[0] - 30), int(center[1] + 40))
                    string = "NDNT: " + str(int(number_of_direct_neighbor))
                    cv2.putText(image, string,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)

                    list_tree_direct.append(direct_tree)
                    list_tree_inline.append(inline_tree)

            CALCULATE_TREE_COVER_ALL = False
            if(CALCULATE_TREE_COVER_ALL == True):
                tree_cover = np.count_nonzero(mask[:,:,0])
                tree_cover_ratio = tree_cover/(h * w)
                print("tree_cover:", tree_cover, ", Image area: ", str(w* h), ", tree_cover_ratio:", tree_cover_ratio)


            CALCULATE_ORIENTATION_LINE_TREE = True
            if (CALCULATE_ORIENTATION_LINE_TREE == True):

                #load pkl list_tree_direct from file
                list_tree_direct = []


                mask_line = np.zeros(image.shape, dtype='uint8')
                mask_viz = image_ori.copy()
                point_list = []
                for e in ellipse_list:
                    (center, axes, angle) = e
                    point_list.append((int(center[0]), int(center[1])))

                    cv2.circle(mask_line, (int(center[0]), int(center[1])), int(0), (255, 255, 255), -1)

                point_list = np.array(point_list)
                slopes = calculate_slopes(point_list)
                angles = np.array([s[2] for s in slopes]).reshape(-1, 1)
                # clustering = DBSCAN(eps=0.1, min_samples=3).fit(angles)  # Cluster based on alignment
                slopes_degree = np.array([(s[0], s[1], int(s[2] * 180 / np.pi)) for s in slopes])

                hist_orientation = np.zeros(180, dtype='uint8')

                angle_list = []
                for i in range(len(slopes_degree)):
                    angle = slopes_degree[i][2]
                    #convert -180 ; 180 --> 0 180; 359
                    if(angle < 0):
                        angle = 180 + angle
                    # angle = angle + 180
                    angle_list.append(angle)
                    hist_orientation[angle] = hist_orientation[angle] + 1

                # print(hist_orientation)
                angle_list = np.array(angle_list)
                # n_bins = 90
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots( sharey=True, tight_layout=True)
                # .hist(angle_list, bins=n_bins)
                # plt.show()
                # plt.savefig(file_path.replace(".json", "_vizDemo10.jpg"))
                labels = align_and_cluster(point_list, 0.3)

                max_tree_pair = -1
                max_direction = -1
                for i in range(len(hist_orientation)):
                    if (max_tree_pair < hist_orientation[i]):
                        max_tree_pair = hist_orientation[i]
                        max_direction = i
                print(max_direction)

                with open(file_path.replace(".json","list_tree_direct.pkl"), 'rb') as f:
                    list_tree_direct = pickle.load(f)
                copy_ellipse = ellipse_list.copy()
                list_neighbor_index = []
                for i, neighbor_list in enumerate(list_tree_direct):
                    neighbor_index_list = []
                    center, axes, angle = ellipse_list[i]
                    for e_neighbor in neighbor_list:
                        center_nei, axes_nei, angle_nei = e_neighbor

                        for j, ej in enumerate(ellipse_list):
                            distance_jk = np.sqrt((center_nei[0] - ej[0][0]) * (center_nei[0] - ej[0][0]) + (center_nei[1] - ej[0][1]) * (center_nei[1] - ej[0][1]))
                            if (distance_jk < 10):
                                neightbor_index = j
                                break
                        neighbor_index_list.append(j)
                    list_neighbor_index.append(neighbor_index_list)

                ###---------------------------Calculate histogram of orientation from only neighbor nearest tree----------
                HISTOGRAM_FROM_DIRECT_NEIGHBOR = False
                if (HISTOGRAM_FROM_DIRECT_NEIGHBOR == True):
                    slopes = []
                    for i, p1 in enumerate(ellipse_list):
                        c, ax, an = ellipse_list[i]
                        for j, p2 in enumerate(list_tree_direct[i]):
                            c1, ax1, an1 = list_tree_direct[i][j]
                            dx = c1[0] - c[0]
                            dy = c1[1] - c[1]
                            slope = np.arctan2(dy, dx)  # Angle in radians
                            slopes.append((i, j, slope))

                    slopes_degree = np.array([(s[0], s[1], int(s[2] * 180 / np.pi)) for s in slopes])
                    hist_orientation = np.zeros(180, dtype='uint8')
                    angle_list = []
                    for i in range(len(slopes_degree)):
                        angle = slopes_degree[i][2]
                        # convert -180 ; 180 --> 0 180; 359
                        if (angle < 0):
                            angle = 180 + angle
                        angle_list.append(angle)
                        hist_orientation[angle] = hist_orientation[angle] + 1
                    angle_list = np.array(angle_list)
                    n_bins = 180
                    import matplotlib.pyplot as plt
                    fig, axs = plt.subplots( sharey=True, tight_layout=True)
                    axs.hist(angle_list, bins=n_bins)
                    for rect in axs.patches:
                        height = rect.get_height()
                        axs.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
                    plt.show()
                    # plt.savefig(file_path.replace(".json", "_vizDemo12.jpg"))

                    max_tree_pair = -1
                    max_direction = -1
                    for i in range(len(hist_orientation)):
                        if (max_tree_pair < hist_orientation[i]):
                            max_tree_pair = hist_orientation[i]
                            max_direction = i
                    print(max_direction)

                ###---------------------------END Calculate histogram of orientation from only neighbor nearest
                list_line_tree = []
                list_outline_tree = []
                image_viz = image.copy()
                for i in range(len(list_tree_direct)):
                    # if(i!=94):
                    #    continue
                    c, ax, an = ellipse_list[i]
                    flag = False
                    for j in range(len(list_tree_direct[i])):
                        c1, ax1, an1 = list_tree_direct[i][j]
                        dx = c[0] - c1[0]
                        dy = c[1] - c1[1]
                        slope = np.arctan2(dy, dx)
                        slope = int(slope * 180 / np.pi)
                        if (slope < 0):
                            slope = 180 + slope
                        THRESHOLD_LINE_CURVE = 15
                        if (slope < max_direction + THRESHOLD_LINE_CURVE and slope > max_direction - THRESHOLD_LINE_CURVE):
                            flag = True
                            # visualize the tree collection line
                            cv2.line(image_viz, (int(c[0]), int(c[1])), (int(c1[0]), int(c1[1])),
                                     color=(255, 255, 255), thickness=2)
                            #check whether tree i is in line or not.
                            #fine id for j
                            for k, e in enumerate(ellipse_list):
                                ck, axk, ank = e
                                distance_jk = np.sqrt((ck[0]-c1[0]) * (ck[0]-c1[0]) + (ck[1] - c1[1]) * (ck[1] - c1[1]))
                                if(distance_jk < 10):
                                    neightbor_index = k
                                    break

                            if(list_line_tree==[]):
                                newline = []
                                newline.append(i)
                                newline.append(neightbor_index)
                                list_line_tree.append(newline)
                            else:
                                flag_find = False
                                count = 0
                                for line in list_line_tree:
                                    count = count + 1
                                    if i in line and neightbor_index in line:
                                        break
                                    if i in line:
                                        if (not neightbor_index in line):
                                            line.append(neightbor_index)
                                            flag_find = True
                                            break
                                    else:
                                        if (neightbor_index in line):
                                            line.append(i)
                                            flag_find = True
                                            break


                                if(flag_find == False and count == len(list_line_tree)):
                                    newline = []
                                    newline.append(i)
                                    newline.append(neightbor_index)
                                    list_line_tree.append(newline)

                    if(flag == False): #Can not find neighor in neighbor --> outlier
                        list_outline_tree.append(i)

                new_list_line_tree = []
                for line in list_line_tree:
                    flag_exist = False
                    for k, new_line in enumerate(new_list_line_tree):
                        if (any(x in new_line for x in line)):
                            new_line = sorted(list(set(line)|set(new_line)))
                            new_list_line_tree[k] = new_line
                            flag_exist = True
                            break
                    if (flag_exist == False):
                        new_list_line_tree.append(line)
                for line in new_list_line_tree:
                    print([x+1 for x in line])
                print(list_outline_tree)
                print("Ratio outline/(numtree):", str(len(list_outline_tree)/len(ellipse_list)))

                #--------------------------find linear regression of each line
                image_viz = image.copy()
                for line in new_list_line_tree:
                    print("-------------LINE:", line)
                    x = []
                    y = []
                    for index_e in line:
                        center, axes, angle = ellipse_list[index_e]
                        x.append(center[0])
                        y.append(center[1])
                    x = np.array(x).reshape((-1, 1))
                    y = np.array(y)
                    model = LinearRegression().fit(x, y)
                    r_sq = model.score(x,y)
                    print(f"coefficient of determination: {r_sq}")
                    print(f"intercept: {model.intercept_}")
                    print(f"slope: {model.coef_}")
                    y_pred = model.predict(x)
                    print(f"x:\n{x}")
                    print(f"predicted response:\n{y_pred}")
                    bot_pred = model.intercept_ + model.coef_ * w
                    top_pred = model.intercept_ + model.coef_ * 0


                    #draw a line over image:

                    cv2.line(image_viz, (w, int(bot_pred)), (0, int(top_pred)), color=(255, 255, 255), thickness=2)

                    m_i = mask.copy()
                    mask_line = np.zeros(image.shape, dtype='uint8')
                    cv2.line(mask_line, (w, int(bot_pred)), (0, int(top_pred)), color=(1, 1, 1), thickness=1)
                    cv2.imwrite(file_path.replace(".json", "_vizDemo13.jpg"), m_i * 255)
                    line_length = np.sum(mask_line)/3
                    tree_cover_line = np.count_nonzero(mask_line * m_i) / 3
                    line_tree_cover_ratio = tree_cover_line / line_length
                    print("Line tree cover ratio: tree_cover_line / line_length = ",tree_cover_line, "/", line_length, " = ",  line_tree_cover_ratio)
                    cv2.imwrite(file_path.replace(".json", "_vizDemo13.jpg"), mask_line)


                cv2.imwrite(file_path.replace(".json", "_vizDemo12.jpg"), image_viz)

                #---------------END Linear regression

                #-------------find neighbor line of tree and calculate distance between
                image_viz = image.copy()
                neighbor_line_index = []
                neighbor_tree_to_line = []

                for i, linei in enumerate(new_list_line_tree):
                    neighbor_i = []
                    for j in linei:
                        neighbor_i = list(set(neighbor_i).union(list_neighbor_index[j]))
                    # print(neighbor_i)
                    neighbor_i = list(set(neighbor_i) - set(linei))
                    print(neighbor_i)
                    neighbor_tree_to_line.append(neighbor_i)
                # print(neighbor_tree_to_line)

                #find distance between two line:
                #find pair of tree in both line
                for i, linei in enumerate(new_list_line_tree):
                    j = i + 1
                    linej = new_list_line_tree[j]
                    direct_neighbor_ji = list(set(neighbor_tree_to_line[j]) & set(linei))
                    direct_neighbor_ij = list(set(neighbor_tree_to_line[i]) & set(linej))
                    distance_line = 0
                    if (len(direct_neighbor_ij) > len(direct_neighbor_ji)):
                        temp = direct_neighbor_ji
                        direct_neighbor_ji = direct_neighbor_ij
                        direct_neighbor_ij = temp
                    for k in direct_neighbor_ij:
                        min_dis = 999999999
                        ck,_,_ = ellipse_list[k]
                        for h in direct_neighbor_ji:
                            ch,_,_ = ellipse_list[h]
                            dist = np.sqrt((ck[0] - ch[0]) * (ck[0] - ch[0]) + (ck[1] - ch[1]) * (ck[1] - ch[1]))
                            if(min_dis > dist):
                                min_dis = dist
                        distance_line = distance_line + min_dis
                    print("distance between line ", i, " and ", j, ": ", str(distance_line/len(direct_neighbor_ij)))


                #----------------------END---------------

                """
                for i in range(len(slopes_degree)):
                    treei = slopes_degree[i][0]
                    treej = slopes_degree[i][1]
                    deg = slopes_degree[i][2]
                    if (deg < 0):
                        deg = 180 + deg
                    # if (treei == 57 and treej == 59):
                    #    print(deg)
                    THRESHOLD_LINE_CURVE = 4
                    if(deg < max_direction + THRESHOLD_LINE_CURVE) and (deg > max_direction - THRESHOLD_LINE_CURVE):
                        #treei and treej is in the same line
                        flag = False

                        center, axes, angle = ellipse_list[treei]
                        center1, axes1, angle1 = ellipse_list[treej]
                        #visualize the tree collection line
                        cv2.line(image_viz, (int(center[0]), int(center[1])), (int(center1[0]),int(center1[1])), color= (255,255,255), thickness = 2)
                        # cv2.imwrite(file_path.replace(".json", "_vizDemo11.jpg"), image_viz)





                for line in list_line_tree:
                    print(line)

                cv2.imwrite(file_path.replace(".json", "_vizDemo11.jpg"), image_viz)
                
                """
                

                # Step 4: Visualize the result

                """
                plt.figure(figsize=(8, 6))
                for i, label in enumerate(set(labels)):
                    if label == -1:  # Noise
                        color = 'k'
                    else:
                        color = plt.cm.jet(label / len(set(labels)))
                    cluster_points = point_list[np.array(labels) == label]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {label}')

                plt.title("Clustering Aligned Star Points")
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.legend()
                plt.show()
                # dst = cv2.Canny(mask_line, 50, 200, None, 3)
                lines = cv2.HoughLines(mask_line[:, :, 0], 1, np.pi / 180, 3, None, 0, 0)
                # Draw the lines
                if lines is not None:
                    for i in range(0, len(lines)):
                        rho = lines[i][0][0]
                        theta = lines[i][0][1]
                        a = math.cos(theta)
                        b = math.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                        cv2.line(mask_viz, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imwrite(file_path.replace(".json", "_vizDemo10.jpg"), mask_viz)
                """

                index = 0
                line_max_tree = []
                line_max_ellipse = []
                for e in ellipse_list:
                    index = index + 1
                    print(index)
                    # if not index == 6:
                    #    continue

                    (center, axes, angle) = e
                    max_tree = 0
                    tree_in_line_max = []
                    max_ellipse = e
                    for e1 in list_tree_direct[index - 1]:
                        (center1, axes1, angle1) = e1
                        mask_line = np.zeros(image.shape, dtype='uint8')

                        mask_line = drawLine(mask_line, (int(center[0]), int(center[1])),
                                             (int(center1[0]), int(center1[1])), color=(1, 1, 1), thickness=2)
                        img_viz = image_ori.copy()
                        img_viz = drawLine(img_viz, (int(center[0]), int(center[1])),
                                             (int(center1[0]), int(center1[1])), color=(255, 255, 255), thickness=2)

                        #build a line connect two ellipse center
                        line = Line(p1=(int(center[0]), int(center[1])), p2=(int(center1[0]), int(center1[1])))
                        #calculate the distance from each ellipse to the line
                        THESHOLD_DIS_2_LINE = (axes[0] + axes[1]) / 3
                        tree_in_line = []
                        for e2 in ellipse_list:
                            (center2, axes2, angle2) = e2
                            dis_to_line = line.distance_to((int(center2[0]), int(center2[1])))
                            if(dis_to_line < THESHOLD_DIS_2_LINE):
                                tree_in_line.append(e2)


                        if (max_tree < len(tree_in_line)):
                            max_tree = len(tree_in_line)
                            tree_in_line_max = tree_in_line
                            max_ellipse = e1

                    line_max_tree.append(tree_in_line_max)
                    line_max_ellipse.append(max_ellipse)
                    print(line_max_tree)
                # img_viz = image_ori.copy()
                for index in range(len(line_max_tree)):
                    img_viz = image.copy()
                    """for e3 in line_max_tree[index]:
                        (center3, axes3, angle3) = e3

                        cv2.circle(img_viz, (int(center3[0]), int(center3[1])), int(6), (255,255,255), -1)
                        # cv2.imshow("img_viz", img_viz)
                        # cv2.waitKey(0)
                    """
                    e = ellipse_list[index]
                    (center, axes, angle) = e
                    max_ellipse = line_max_ellipse[index]
                    (centermax_ellipse, axesmax_ellipse, anglemax_ellipse) = max_ellipse
                    drawLine(img_viz, (int(center[0]), int(center[1])),
                             (int(centermax_ellipse[0]), int(centermax_ellipse[1])), color=(255, 255, 255), thickness=int((axes[0] + axes[1])/3))

                    cv2.imwrite(file_path.replace(".json", "_vizDemo9.jpg"), img_viz)

            """
            #---------------------------------------Minimum distance to k neighbor trees------------------
            index = 0
            K_NEIGHBOR_TREE = 3
            minimum_distance_to_k_neighbor = []
            for e in ellipse_list:
                index = index + 1
                #calculate the distance from ellipse to others
                distance_tree = []
                (center, axes, angle) = e
                for e1 in ellipse_list:

                    (center1, axes1, angle1) = e1
                    d = np.sqrt((center[0] - center1[0]) * (center[0] - center1[0]) + (center[1] - center1[1]) * (center[1] - center1[1]))
                    distance_tree.append(d)
                distance_tree = sorted(distance_tree)
                minimum_distance_to_k_neighbor.append(distance_tree[K_NEIGHBOR_TREE])

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                fontColor = (0, 0, 255)
                thickness = 2
                lineType = 2

                bottomLeftCornerOfText = (int(center[0] - 60), int(center[1] + 40))
                # string = str(int(index)) + ": Min_dis_to_" + str(int(K_NEIGHBOR_TREE)) + "_tree: " + str(int(distance_tree[K_NEIGHBOR_TREE]))
                string = str(int(index)) + ": "
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                fontColor = (255, 0, 0)
                thickness = 2
                lineType = 2

                bottomLeftCornerOfText = (int(center[0] + 10), int(center[1] + 40))
                # string = str(int(index)) + ": Min_dis_to_" + str(int(K_NEIGHBOR_TREE)) + "_tree: " + str(int(distance_tree[K_NEIGHBOR_TREE]))
                string = str(int(distance_tree[K_NEIGHBOR_TREE]))
                cv2.putText(image, string,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                if( index % 7 == 0):
                    cv2.circle(image, (int(center[0]), int(center[1])), int(distance_tree[K_NEIGHBOR_TREE]), (0,255,255), 2)
                cv2.circle(image, (int(center[0]), int(center[1])), int(4), (0,0,255), -1)
            """
            """
            bottomLeftCornerOfText = (100, 100)
            string = "Number of Trees: " + str(index)
            fontScale = 3
            fontColor = (255, 0, 0)
            thickness = 5
            """
            """
            cv2.putText(image, string,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            """
            # Display the image
            """
            scale_percent = 40  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            """

            # resize image
            PRINT_CLEAN_TREE = False
            if (PRINT_CLEAN_TREE == True):
                cv2.imwrite(file_path.replace(".json", "_vizDemo6.jpg"), image)
            if (PRINT_PHENO_TOPVIEW == True):
                cv2.imwrite(file_path.replace(".json","_vizDemo4.jpg"), image)
            if (CALCULATE_BETWEEN_TREE == True):
                cv2.imwrite(file_path.replace(".json", "_vizDemo5.jpg"), image)
            if (CALCULATE_ORIENTATION_LINE_TREE == True):
                cv2.imwrite(file_path.replace(".json", "_vizDemo7.jpg"), image)

            if (CALCULATE_DEGREE_COLOR == True):
                cv2.imwrite(file_path.replace(".json", "_vizDemo8.jpg"), image)
            # cv2.imwrite(file_path.replace(".json","_mask_viz.jpg"), mask)
            # imgresized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            # cv2.imshow("Ellipse", imgresized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            """
            with open(file_path.replace(".json","list_tree_direct.pkl"), 'wb') as fp:
                pickle.dump(list_tree_direct, fp)
            with open(file_path.replace(".json", "list_tree_inline.pkl"), 'wb') as fp:
                pickle.dump(list_tree_inline, fp)
            """