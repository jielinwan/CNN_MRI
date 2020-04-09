import nibabel as nib
import os
import numpy as np
file='D:/t1_2mm_affine_4tps'
import mat4py
mat_path='group1.mat'
label=mat4py.loadmat(mat_path)
label=label['group_subjects']
i=-1
id=0
for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        tag=label[i][0]
        year=4
        print(tag)
        for f in files:
            filename=os.path.join(root, f)
            if(filename[-3:]=='nii'):
                if(tag==0 or tag==4):
                    img = nib.load(filename).get_data()
                    for ni in range(5):
                        tempImg=img+np.random.randn(120,120,78)*ni*3
                        tempImg = (tempImg - np.average(tempImg)) / np.std(tempImg)
                        save_path = 'D:/Python/DataAddNoise/ImgData/Img_Values' + str(id) + '/'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        id += 1
                        np.save(save_path + 'x.npy', tempImg)
                        if (tag == 0):
                            np.save(save_path + 'y.npy', 0)
                        else:
                            np.save(save_path + 'y.npy', 1)
                    break
                else:
                    if(year==4 or year==1):
                        img = nib.load(filename).get_data()
                        for ni in range(5):
                            tempImg=img+np.random.randn(120,120,78)*ni*3
                            tempImg = (tempImg - np.average(tempImg)) / np.std(tempImg)
                            save_path = 'D:/Python/DataAddNoise/ImgData/Img_Values' + str(id) + '/'
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            id += 1
                            np.save(save_path + 'x.npy', img)
                            if (year == 4):
                                np.save(save_path + 'y.npy', 1)
                            else:
                                np.save(save_path + 'y.npy', 0)
                print(year,filename,tag)
                year -= 1
        # 遍历所有的文件夹
        i+=1
        print(i)


