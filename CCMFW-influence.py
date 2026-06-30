import numpy as np
import matplotlib.pyplot as plt
from lib import Extract_Map_I2B, Map_I2B
from itertools import combinations
from scipy.linalg import null_space

def HyperPlaneShiftingMethod(A,Imin,Imax):
    #Create permutation matrix for the selection of unitary actuation fields
    def CreatePermuationMatrix(A):
        # A: Jacobian matrix
        # M: permutation matrix
        d = np.shape(A)[0] #dimension of output space (if field, this is 3)
        n = np.shape(A)[1] #number of coils
        comb = combinations(np.arange(n), d-1) 
        M = np.asarray(list(comb))  
        return M
    #Create combination matrix to test combination of field
    def CreateFieldCombinationMatrix(n):
        # n: dimension of the combination matrix
        # M: combination matrix
        nums = np.arange(2**n)
        M = ((nums.reshape(-1,1) & (2**np.arange(n))) != 0).astype(int)
        return M
    # Imin: minimum current (scalar) in A
    # Imax: maximum current (scalar) in A
    # J: Jacobian matrix of the eMNS
    # N, d_vec: Hyperplane representation of the zonotope
    # Imin = -10
    # Imax = 10
    dI = Imax - Imin
    M = CreatePermuationMatrix(A)
    nb_comb = np.shape(M)[0] #number of combination
    
    d = np.shape(A)[0] #dimension of output space (if field, this is 3)
    n_coils = np.shape(A)[1] #number of coils
    
    #Initialize matrix and vector for hyperplane representation
    N = np.zeros((2*nb_comb,d))
    d_vec = np.zeros((2*nb_comb,1))
    bmin = np.matmul(A,Imin*np.ones((n_coils,1)))
    
    #Iterate on the combination of unitary fields
    for i in range(nb_comb):
        # Step 1: define initial hyperplane
        #Define the set of vectors to be orthogonal with
        W = A[:,M[i,:]]
        
        #Get the orthogonal vector using the nullspace of W^T
        Wns = null_space(np.transpose(W))
        v = Wns[:,0]

        # Step 2: shift intial hyperplane
        temp = v / np.linalg.norm(v)
        n = temp.reshape((-1,1))
        
        # Step 3: build projections   
        lj_arr = np.zeros((n_coils-(d-1),1))
        k = 0
        h = 0. 
        for j in range(n_coils):
            if not(j in M[i,:]):
                lj = np.dot(np.transpose(A[:,j]),n)
                # lj_arr[k,0] = lj
                lj_arr[k,0] = lj if np.isscalar(lj) else lj[0]
                k += 1

        C = CreateFieldCombinationMatrix(n_coils-(d-1))
        h = np.matmul(C,dI*lj_arr)
        hp = np.max(h)
        hm = np.min(h)
        
        #Step 4: compute hyperplane support
        pp = hp*n + bmin
        pm = hm*n + bmin
        
        # Step 5: build hyperplane representation
        N[i,:] = n.T
        N[i+nb_comb,:] = -n.T
        d_vec[i,:] = np.dot(n.T,pp)
        d_vec[i+nb_comb,:] = np.dot(-n.T,pm)
    return N, d_vec

xx = np.linspace(-0.03, 0.03, 50)
yy = np.linspace(-0.06, 0, 50)

# Create a meshgrid from x and y
X, Y = np.meshgrid(xx, yy)

# Create a mask for points inside the circle x^2 + y^2 <= 0.06^2
R = 0.06
circle_mask = (X**2 + Y**2) <= R **2

I_max = 17
I_min = - I_max

# Define the amplitude of the magnetic field
B_amp = 0.055 # Units of Tesla

# Initialize P with NaN values
P = np.full_like(X, np.nan) 

# Evaluate the function only at points inside the circle
for i in range(len(xx)):
    for j in range(len(yy)):
        if circle_mask[i, j]:  # Only process points inside the circle
            target_points = [
                {'X': X[i, j], 'Y': Y[i, j], 'Z': -0.05, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
            ]
            # print("X,Y:", X[i, j], Y[i, j])
            A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
            G, K = HyperPlaneShiftingMethod(A, I_min, I_max)
            # print("K:", K)
            d = min(K)
            # print("d:", d)
            P[i, j] = 1 if d >= B_amp else -1

# Create a 2D plot - only plot points inside the circle
valid_mask = ~np.isnan(P)
in_workspace = (P == 1) & valid_mask
out_workspace = (P == -1) & valid_mask

plt.scatter(X[in_workspace], Y[in_workspace], c='green', s=10, alpha=1, label='In workspace')
plt.scatter(X[out_workspace], Y[out_workspace], c='salmon', s=10, alpha=1, label='Out workspace')

# Draw the circle boundary for reference
theta = np.linspace(0, 2*np.pi, 100)
circle_x = R * np.cos(theta)
circle_y = R * np.sin(theta)
plt.plot(circle_x, circle_y, 'k--', linewidth=1, alpha=0.5)
R = R + 0.01
circle_x = R * np.cos(theta)
circle_y = R * np.sin(theta)
plt.plot(circle_x, circle_y, 'k--', linewidth=1, alpha=0.5)

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')

# Add a legend
plt.legend(loc='upper right')

plt.xlim(-0.03, 0.03)
plt.ylim(-0.06, 0.0)

# Show the plot
plt.grid()
plt.gca().set_facecolor('lightgray')
plt.show()


