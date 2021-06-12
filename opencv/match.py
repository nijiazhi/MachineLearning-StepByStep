import cv2
import os
import numpy as np

from matplotlib import pyplot as plt

import pdb
import time








def image_hist_demo(image):
    hists = []
    color = {"blue", "green", "red"}
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。
    for i, color in enumerate(color):
        tmp_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hists.append(tmp_hist)
        #plt.plot(hist, color=color)
        #plt.xlim([0, 256])
    #plt.show()

    hist = np.hstack((np.squeeze(hists[0]),np.squeeze(hists[1]),np.squeeze(hists[2])))
    return (hist-hist.min())/(hist.max()-hist.min())


def calc_hist(image):
    shape = image.shape
    
    hist = [[0]*256]*3
    #hist = [0]*(256*3)
    
    for y in range(shape[0]):
        for x in range(shape[1]):
            for c in range(shape[2]):
                hist[c][image[y,x,c]] += 1
                #hist[c*256+image[y,x,c]] += 1
                
    #for i,x in enumerate(np.nditer(image, order='C')):
    #    #print(i,x)
    #    hist[(i%3)*256+x] += 1

    return hist



def update_hist(hist, minus_line, add_line):
    shape = minus_line.shape
    for i in range(shape[0]):
        for c in range(shape[1]):
            hist[c][minus_line[i,c]] -= 1
            hist[c][add_line[i,c]] += 1
            


def calc_hist_dist(vector1, vector2):
    # cosine
    dist = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return dist





def main():
    base_dir='C:/Users/andyni/Desktop/ai-contest/第三期/'
    image_dir = base_dir+'dataset/GAME_DET_CONTEST_IMAGES/buluochongtu/'
    obj_dir = base_dir+'dataset/GAME_DET_CONTEST_OBJECTS/buluochongtu/'

    image_list = os.listdir(image_dir)
    obj_list = os.listdir(obj_dir)
    print('image_list len:', len(image_list))
    print('obj_list len:', len(obj_list))
    print('*'*50, '\n')


    # 计算object整张图的直方图
    t_start = time.time()
    obj_hist_list = []
    obj_shape_list = []
    for obj_name in obj_list:
        obj_image = cv2.imread(obj_dir+obj_name)
        
        hist = image_hist_demo(obj_image)
        #hist = calc_hist(obj_image)

        obj_shape_list.append(obj_image.shape)
        obj_hist_list.append(hist)
        


    print(time.time() - t_start)
    
    
    # 滑动窗口遍历计算image直方图, 匹配obj
    #'''
    for im_name in image_list:
        image = cv2.imread(image_dir+im_name)
        im_shape = image.shape
        
        t_start = time.time()
    
        win_size = obj_shape_list[0]
        obj_hist = obj_hist_list[0]
        max_dist = 0
        max_pos = [0,0]
        for y in range(im_shape[0]-win_size[0]):
            for x in range(im_shape[1]-win_size[1]):
                hist = image_hist_demo(image[y:y+win_size[0],x:x+win_size[1],:])
                dist = calc_hist_dist(hist, obj_hist)
                if dist > max_dist:
                    max_dist = dist
                    max_pos = [y, x]
                
        print(max_pos, max_dist)
        print(time.time() - t_start)
        
        #'''
        cv2.rectangle(image, (max_pos[1], max_pos[0]), (max_pos[1]+win_size[1], max_pos[0]+win_size[0]), (0,0,255), 2)
        cv2.imshow('obj', cv2.imread(obj_dir+obj_list[0]))
        cv2.imshow('image', image)
        k=cv2.waitKey()
        if k == 27:
            break
        #'''
    #'''
    
    '''
    for im_name in image_list:
        image = cv2.imread(image_dir+im_name)
        im_shape = image.shape
        
        last_line_first_hist = None
        last_hist = None
        for y in range(im_shape[0]-win_size[0]):
            for x in range(im_shape[1]-win_size[1]):
                if x == 0 and y == 0:
                    hist = calc_hist(image[:win_size[0],:win_size[1],:])
                    last_line_first_hist = hist
                    last_hist = hist
                elif x == 0:
                    update_hist(last_line_first_hist, image[y-1,x:x+win_size[1],:], image[y+win_size[0]-1,x:x+win_size[1],:])
                else:
                    update_hist(last_hist, image[y:y+win_size[0],x-1,:], image[y:y+win_size[0],x+win_size[1]-1,:])
                
        break
    #'''

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




