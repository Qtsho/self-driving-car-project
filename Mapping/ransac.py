# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

# from sklearn import datasets, linear_model

# diabetes = datasets.load_diabetes()
# X_train = diabetes.data[:-20, (0,1,2)]

# y_train = diabetes.target[:-20]

# ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
# ransac.fit(X_train, y_train)


# # the plane equation
# z = lambda x,y: (-ransac.estimator_.intercept_ - ransac.estimator_.coef_[0]*x - ransac.estimator_.coef_[1]*y) / ransac.estimator_.coef_[2]

# tmp = np.linspace(-0.1,0.1,50)
# x,y = np.meshgrid(tmp,tmp)

# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')
# ax.plot3D(X_train[:,0], X_train[:,1], X_train[:,2], 'or')
# ax.plot_surface(x, y, z(x,y))
# ax.view_init(10, 60)
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# # Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# print('Coefficients test: \n', (diabetes_y_pred[0]- regr.intercept_) / diabetes_X_test[0] )
# # The mean squared error
# print('Mean squared error: %.2f'
#       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(diabetes_y_test, diabetes_y_pred))

# # Plot outputs

# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# x1 = np.linspace(-0.5,0.5,100)
# y1 = regr.coef_[0]*x1 + regr.intercept_

# plt.plot(x1, y1, '-r', label='line regression',color='red')
# plt.xticks(())
# plt.yticks(())
# plt.grid()
# plt.show()

import numpy as np
import numpy.linalg as la
from scipy.linalg import svd
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
#from svd_solve import svd, svd_solve
from tqdm import tqdm
class Ransac:
    def __init__(self):
        super().__init__()

    def fit_plane_LSE(self,points):
        # points: Nx4 homogeneous 3d points
        # return: 1d array of four elements [a, b, c, d] of
        # ax+by+cz+d = 0
        assert points.shape[0] >= 3 # number of rows at least 3 points needed
        U, S, Vt = svd(points)
        null_space = Vt[-1, :] 
        null_space   = null_space / np.linalg.norm(null_space)  #we want normal vectors normalized to unity (later added)
        return null_space
    
    def fit_plane_normal(self,points):
        assert points.shape[0] >= 3 # number of rows at least 3 points needed
        v1 = points[2]-points[0]
        v2 = points[1]-points[0]
        cp = np.cross(v1, v2)
        d = np.dot(cp, points[2])
        return cp,d

    def get_point_dist(self,points, plane):
        # return: 1d array of size N (number of points) the distance in the 
        dists = np.abs(points @ plane) / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
        return dists

    def fit_plane_ransac(self, points, iters=1000, inlier_thresh=0.05, return_outlier_list=False, plot=False):
        # points: Nx4 homogeneous 3d points
        # iters: number of interation
        # return: 
        #   plane: 1d array of four elements [a, b, c, d] of ax+by+cz+d = 0
        #   inlier_list: 1d array of size N of inlier points
        max_inlier_num = -1
        max_inlier_list = None
        
        N = points.shape[0] #number of points
        assert N >= 3#always choice from 3 more than 3 points

        for i in tqdm(range(iters)):
            chose_id = np.random.choice(N, 3, replace=True)
            chose_points = points[chose_id, :]
            for x in chose_points[:,2]:
                if x > -1.5:
                    print('Break')
                    break
                    
                else:
                    tmp_plane,d = self.fit_plane_normal(chose_points) #create a random plane frome choosing points        
                    dists = self.get_point_dist(points, tmp_plane)
            
                    tmp_inlier_list = np.where(dists < inlier_thresh)[0]
                    tmp_outlier_list = np.where(dists >= inlier_thresh)[0]
            
                    tmp_inliers = points[tmp_inlier_list, :]
                    num_inliers = tmp_inliers.shape[0]
                    if num_inliers > max_inlier_num:
                        max_inlier_num = num_inliers
                        max_inlier_list = tmp_inlier_list
                        
                        
                    if plot == True:
                        plot_surface= True
                        fig = plt.figure()
                        ax = Axes3D(fig)
                        if plot_surface == True:
                          
                            tmp_x = np.linspace(np.amin(points.T[0 ,tmp_inlier_list]),np.amax(points.T[0,tmp_inlier_list]),100)
                            tmp_y = np.linspace(np.amin(points.T[1 ,tmp_inlier_list]),np.amax(points.T[1,tmp_inlier_list]),100)
                            X,Y = np.meshgrid(tmp_x,tmp_y)
                            Z = (d - tmp_plane[0] * X - tmp_plane[1] * Y) / tmp_plane[2]
                            ax.plot_surface(X, Y, Z)
        
                      
                        #pl3d= ax.scatter(points.T[0],points.T[1],points.T[2],c = points.T[2], marker= ',')
                        #x.set_zlabel('m')
                        
                        #cbar=plt.colorbar(pl3d)
                        #cbar.set_label("Height (m)")
                        #ax.scatter(points.T[0,tmp_inlier_list],points.T[1,tmp_inlier_list],points.T[2,tmp_inlier_list],c = 'b', marker= ',')
                        #ax.scatter(points.T[0,tmp_outlier_list],points.T[1,tmp_outlier_list],points.T[2,tmp_outlier_list],c = 'g', marker= ',')
                        ax.scatter(points.T[0,chose_id],points.T[1,chose_id],points.T[2,chose_id],c = 'g', marker= ',')
                        
                        ax.set_xlim(-20, 20)
                        ax.set_ylim(-20, 20)
                        plt.show()

            
                    print('iter %d, %d inliers' % (i, max_inlier_num))
                    print(chose_id)

        final_points = points[max_inlier_list, :]
 
        plane = self.fit_plane_LSE(final_points)
        d = np.dot(final_points[0],plane)

        fit_variance = np.var(self.get_point_dist(final_points, plane))
        print('RANSAC fit variance: %f' % fit_variance)
        print(plane)

        dists = self.get_point_dist(points, plane)

        select_thresh = inlier_thresh * 1

        inlier_list = np.where(dists < select_thresh)[0]
        if not return_outlier_list:
            return plane, inlier_list ,d
        else:
            outlier_list = np.where(dists >= select_thresh)[0]
            return plane, inlier_list, d,  outlier_list