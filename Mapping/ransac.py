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
        null_space = Vt[-1, :] # find normal as 3rd column of matrtix U
        return null_space

    def get_point_dist(self,points, plane):
        # return: 1d array of size N (number of points) the distance
        dists = np.abs(points @ plane) / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
        return dists

    def fit_plane_ransac(self, points, iters=1000, inlier_thresh=0.05, return_outlier_list=False):
        # points: Nx4 homogeneous 3d points
        # iters: number of interation
        # return: 
        #   plane: 1d array of four elements [a, b, c, d] of ax+by+cz+d = 0
        #   inlier_list: 1d array of size N of inlier points
        max_inlier_num = -1
        max_inlier_list = None
        
        N = points.shape[0]
        assert N >= 3

        for i in tqdm(range(iters)):
            chose_id = np.random.choice(N, 3, replace=False)
            chose_points = points[chose_id, :]
            tmp_plane = self.fit_plane_LSE(chose_points) #create a random plane frome choosing points 
            
            dists = self.get_point_dist(points, tmp_plane)
            tmp_inlier_list = np.where(dists < inlier_thresh)[0]
            tmp_inliers = points[tmp_inlier_list, :]
            num_inliers = tmp_inliers.shape[0]
            if num_inliers > max_inlier_num:
                max_inlier_num = num_inliers
                max_inlier_list = tmp_inlier_list
            
            #print('iter %d, %d inliers' % (i, max_inlier_num))

        final_points = points[max_inlier_list, :]
        plane = self.fit_plane_LSE(final_points)
        
        fit_variance = np.var(self.get_point_dist(final_points, plane))
        print('RANSAC fit variance: %f' % fit_variance)
        print(plane)

        dists = self.get_point_dist(points, plane)

        select_thresh = inlier_thresh * 1

        inlier_list = np.where(dists < select_thresh)[0]
        if not return_outlier_list:
            return plane, inlier_list
        else:
            outlier_list = np.where(dists >= select_thresh)[0]
            return plane, inlier_list, outlier_list