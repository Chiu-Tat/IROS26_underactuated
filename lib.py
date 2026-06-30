import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations
from scipy.linalg import null_space
import math

# 10 coils data 3D
params_list = [
    # np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),  # coil 1
    # np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),  # coil 2
    # np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),  # coil 3
    # np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]), # coil 4
    # np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),  # coil 5
    # np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),  # coil 6
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),   # coil 7
    np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),   # coil 8
    np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),   # coil 9
    np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979])   # coil 10
]

num_coils = len(params_list)

# Define the symbols
m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z = sp.symbols('m0 m1 m2 r0_0 r0_1 r0_2 X Y Z')

# Constants
mu0 = 4 * sp.pi * 1e-7

# Calculate displacement vector
dx = X - r0_0
dy = Y - r0_1
dz = Z - r0_2

# Calculate distance to the coordinate point
r = sp.sqrt(dx**2 + dy**2 + dz**2) + 1e-9  # Add a small constant to avoid division by zero

# Calculate dot product of displacement vector and magnetic dipole moment
dot_product = m0 * dx + m1 * dy + m2 * dz

# Calculate magnetic field components
model_Bx = (mu0 / (4 * sp.pi)) * (3 * dx * dot_product / r**5 - m0 / r**3)
model_By = (mu0 / (4 * sp.pi)) * (3 * dy * dot_product / r**5 - m1 / r**3)
model_Bz = (mu0 / (4 * sp.pi)) * (3 * dz * dot_product / r**5 - m2 / r**3)

# Convert the symbolic functions to numerical functions
dipole_model_Bx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx, 'numpy')
dipole_model_By = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By, 'numpy')
dipole_model_Bz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bz, 'numpy')

# Calculate the partial derivatives
model_Bx_dx = sp.diff(model_Bx, X)
model_Bx_dy = sp.diff(model_Bx, Y)
model_Bx_dz = sp.diff(model_Bx, Z)
model_By_dy = sp.diff(model_By, Y)
model_By_dz = sp.diff(model_By, Z)
model_Bz_dz = sp.diff(model_Bz, Z)

# Convert the symbolic functions to numerical functions
dipole_model_Bx_dx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dx, 'numpy')
dipole_model_Bx_dy = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dy, 'numpy')
dipole_model_Bx_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dz, 'numpy')
dipole_model_By_dy = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By_dy, 'numpy')
dipole_model_By_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By_dz, 'numpy')
dipole_model_Bz_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bz_dz, 'numpy')

# Calculate the magnetic field and its partial derivatives at given points
def calculate_B_and_derivatives(currents, X, Y, Z):
    Bx_total = 0
    By_total = 0
    Bz_total = 0
    Bx_dx_total = 0
    Bx_dy_total = 0
    Bx_dz_total = 0
    By_dy_total = 0
    By_dz_total = 0
    Bz_dz_total = 0

    # Loop over the coils
    for i in range(num_coils):
        # Get the coil parameters
        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        # Calculate the magnetic field produced by this coil
        Bx = dipole_model_Bx(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By = dipole_model_By(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bz = dipole_model_Bz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]

        # Add to the total magnetic field
        Bx_total += Bx
        By_total += By
        Bz_total += Bz

        # Calculate the partial derivatives
        Bx_dx = dipole_model_Bx_dx(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bx_dy = dipole_model_Bx_dy(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bx_dz = dipole_model_Bx_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By_dy = dipole_model_By_dy(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By_dz = dipole_model_By_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bz_dz = dipole_model_Bz_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]

        # Add to the total partial derivatives
        Bx_dx_total += Bx_dx
        Bx_dy_total += Bx_dy
        Bx_dz_total += Bx_dz
        By_dy_total += By_dy
        By_dz_total += By_dz
        Bz_dz_total += Bz_dz

    return np.array([Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total])

# Create the mapping matrix between the currents and the magnetic field
# target_points = [{'X': , 'Y': , 'Z': , 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': }] <<-- the first three are required parameters, the rest are optional depending on the need for workspace analysis
def Map_I2B(target_points):

    # Initialize the matrix as a zero matrix
    A = np.zeros((8 * len(target_points), num_coils))

    # Loop over the target points
    for i, target_point in enumerate(target_points): 
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']

        # Loop over the coils
        for j in range(num_coils):
            # Calculate the magnetic field and its derivatives for a unit current in the j-th coil
            currents = np.zeros(num_coils)
            currents[j] = 1
            Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total = calculate_B_and_derivatives(currents, X, Y, Z)

            # Set the corresponding rows in the A matrix
            A[8*i:8*(i+1), j] = [Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total]
    return A

# Create the extraction matrix between the currents and the magnetic field
# target_points = [{'X': , 'Y': , 'Z': , 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': }] <<-- the first three are required parameters, the rest are optional depending on the need for workspace analysis
def Extract_Map_I2B(target_points):
    row_selection_matrix = np.zeros((8 * len(target_points), 8 * len(target_points)))

    for i, target_point in enumerate(target_points): 
        if target_point['Bx'] is not None:
            row_selection_matrix[8*i, 8*i] = 1
        if target_point['By'] is not None:
            row_selection_matrix[8*i+1, 8*i+1] = 1
        if target_point['Bz'] is not None:
            row_selection_matrix[8*i+2, 8*i+2] = 1
        if target_point['Bx_dx'] is not None:
            row_selection_matrix[8*i+3, 8*i+3] = 1
        if target_point['Bx_dy'] is not None:
            row_selection_matrix[8*i+4, 8*i+4] = 1
        if target_point['Bx_dz'] is not None:
            row_selection_matrix[8*i+5, 8*i+5] = 1
        if target_point['By_dy'] is not None:
            row_selection_matrix[8*i+6, 8*i+6] = 1
        if target_point['By_dz'] is not None:
            row_selection_matrix[8*i+7, 8*i+7] = 1
    return row_selection_matrix[np.sum(row_selection_matrix, axis=1) != 0]

def remove_duplicate_rows(N, d):
    """
    Remove duplicate rows in matrices N and d.
    
    Parameters:
    N (numpy.ndarray): The first matrix with shape (m, n).
    d (numpy.ndarray): The second matrix with shape (m, p).
    
    Returns:
    tuple: Two matrices with duplicate rows removed.
    """
    
    if N.shape[0] != d.shape[0]:
        raise ValueError("Both matrices must have the same number of rows")
    
    # Combine N and d along columns to compare rows jointly
    combined = np.hstack((N, d))
    
    # Use numpy unique function to find unique rows and their indices
    unique_combined, indices = np.unique(combined, axis=0, return_index=True)
    
    # Sort indices to maintain the original order of rows
    sorted_indices = np.sort(indices)
    
    # Retrieve the unique rows based on the sorted indices
    unique_N = N[sorted_indices]
    unique_d = d[sorted_indices]
    
    return unique_N, unique_d

# Transform a polytope and then extract its facets
# A is the mapping matrix. G and K are the constraint matrix and vector, respectively: G * I <= K.
def Transform_and_Extract_Facets(A, G, K):
    # Step 1: Enumerate the vertices of the polytope defined by G * I <= K
    vertices = compute_polytope_vertices(G, K)
    
    # Step 2: Transform vertices using the Jacobian matrix A
    vertices = np.array(vertices)
    transformed_vertices = np.dot(vertices, A.T)
    
    # Step 3: Compute the convex hull of the transformed vertices
    hull = ConvexHull(transformed_vertices)
    
    # Step 4: Extract the hyperplanes (facets) from the convex hull
    N = -hull.equations[:, :-1]
    d_vec = -hull.equations[:, -1].reshape(-1, 1)
    
    return N, d_vec

#Compute the hyperplane representation of the zonotope
# def HyperPlaneShiftingMethod(A,Imin,Imax):
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

# Compute the multi-target magnetic-feasible workspace (MMFW)
# target_points = [{'X': , 'Y': , 'Z': , 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': }]
def Get_MMFW(target_points, I_min, I_max):
    # target_points: list of dictionaries with target points and their magnetic field values
    # I_min: minimum current (scalar) in A
    # I_max: maximum current (scalar) in A
    # P_max: maximum power (scalar) in W
    # R: resistance (vector) in Ohm

    A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
    
    # se HyperPlane method
    # print("Using HyperPlane method")
    G, k = HyperPlaneShiftingMethod(A, I_min, I_max)
    k = k.reshape(-1, 1)
    # Remove duplicate rows
    # G, k = remove_duplicate_rows(G, k)
    return G, k

# Create the mapping matrix between the currents and the magnetic field in 2D
# target_points = [{'X': , 'Y': , 'Z': 0, 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': }] <<-- the first three are required parameters, the rest are optional depending on the need for workspace analysis
def Map_I2B_2D(target_points):
    # Initialize the matrix as a zero matrix
    A = np.zeros((5 * len(target_points), num_coils))
    # Loop over the target points
    for i, target_point in enumerate(target_points): 
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']

        # Loop over the coils
        for j in range(num_coils):
            # Calculate the magnetic field and its derivatives for a unit current in the j-th coil
            currents = np.zeros(num_coils)
            currents[j] = 1
            Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total = calculate_B_and_derivatives(currents, X, Y, Z)

            # Set the corresponding rows in the A matrix
            A[5*i:5*(i+1), j] = [Bx_total, By_total, Bx_dx_total, Bx_dy_total, By_dy_total]
    return A

# Create the extraction matrix between the currents and the magnetic field in 2D
# target_points = [{'X': , 'Y': , 'Z': , 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': }] <<-- the first three are required parameters, the rest are optional depending on the need for workspace analysis
def Extract_Map_I2B_2D(target_points):
    row_selection_matrix = np.zeros((5 * len(target_points), 5 * len(target_points)))

    for i, target_point in enumerate(target_points): 
        if target_point['Bx'] is not None:
            row_selection_matrix[5*i, 5*i] = 1
        if target_point['By'] is not None:
            row_selection_matrix[5*i+1, 5*i+1] = 1
        if target_point['Bx_dx'] is not None:
            row_selection_matrix[5*i+2, 5*i+2] = 1
        if target_point['Bx_dy'] is not None:
            row_selection_matrix[5*i+3, 5*i+3] = 1
        if target_point['By_dy'] is not None:
            row_selection_matrix[5*i+4, 5*i+4] = 1
    return row_selection_matrix[np.sum(row_selection_matrix, axis=1) != 0]

# # Transform a polytope and then extract its facets
# # A is the mapping matrix. G and K are the constraint matrix and vector, respectively: G * I <= K.
def Transform_and_Extract_Facets(A, G, K):
    # Step 1: Enumerate the vertices of the polytope defined by G * I <= K
    vertices = compute_polytope_vertices(G, K)
    
    # Step 2: Transform vertices using the Jacobian matrix A
    vertices = np.array(vertices)
    transformed_vertices = np.dot(vertices, A.T)

    # Step 3: Compute the convex hull of the transformed vertices
    hull = ConvexHull(transformed_vertices)

    # Step 4: Extract the hyperplanes (facets) from the convex hull
    N = -hull.equations[:, :-1]
    d_vec = -hull.equations[:, -1].reshape(-1, 1)

    return N, d_vec

# Plot the convex hull based on the hyperplane representation
def Plot_Hull_H(N, d):
    # Generate the convex hull
    vertices = np.array(compute_polytope_vertices(N, d))
    hull = ConvexHull(vertices)

    # Plot the convex hull
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the vertices
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])

    # Plot the edges
    # for simplex in hull.simplices:
    #     ax.plot(vertices[simplex, 0], vertices[simplex, 1], vertices[simplex, 2], 'k-')

        # Create a list of polygons for the faces
    faces = [vertices[simplex] for simplex in hull.simplices]

    # Create a Poly3DCollection object for the faces
    face_collection = Poly3DCollection(faces, alpha=0.25, linewidths=0.5, edgecolors='black')
    
    # Optionally set the color of the faces. Here we use a single color, but you can customize it as needed.
    face_collection.set_facecolor('lightgreen')  # You can also use a list of colors to color each face differently.

    # Add the collection to the axes
    ax.add_collection3d(face_collection)
    
    plt.show()

# Plot the 2D convex hull based on the hyperplane representation
def Plot_Hull_H_2D(N, d, color='lightblue', alpha=0.5, radius=0.03):
    # Generate the convex hull
    vertices = np.array(compute_polytope_vertices(N, d))
    hull = ConvexHull(vertices)

    # Plot the convex hull
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    # show the grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    # Set axis labels and limits so that x and y have equal length
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    ax.set_xlim(x_center - max_range / 2 - 0.01, x_center + max_range / 2 + 0.01)
    ax.set_ylim(y_center - max_range / 2 - 0.01, y_center + max_range / 2 + 0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Fill the hull region with color
    hull_vertices = vertices[hull.vertices]
    hull_vertices = hull_vertices[ConvexHull(hull_vertices).vertices]  # Ensure proper ordering
    ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=color, alpha=alpha, label='Feasible Region')

    # Plot the vertices
    # ax.scatter(vertices[:,0], vertices[:,1], s=30, zorder=5)

    # # Plot the edges
    for simplex in hull.simplices:
        ax.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-', linewidth=1.5)

    # Plot a circle with a radius of 0.03
    circle = plt.Circle((0, 0), radius, fill=False, ec='red', linewidth=2, label='Minimal actuation circle')
    ax.add_patch(circle)

    ax.legend(loc='upper right', fontsize=15)
    plt.show()


if __name__ == "__main__":
    # Interval example usage
    target_points = [
        # {'X': 0.01, 'Y': 0.01, 'Z': 0.0, 'm': 0.1, 'alpha': np.pi/2, 'beta': 0, 'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': 0.02, 'Y': 0.02, 'Z': 0.0, 'm': 0.1,'alpha': np.pi/3, 'beta': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None}
        # {'X': 0.01, 'Y': -0.03, 'Z': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None}
        ]

    # A = Extract_Map_I2H(target_points) @ Map_I2H(target_points)
    A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
    print("Jacobian matrix A:", A)

    # Singular value decomposition of A
    U, s, Vt = np.linalg.svd(A)
    print("Singular values:", s)
    
    # Get the corresponding directions
    print("Left singular vectors (U):", U)
    print("Right singular vectors (Vt):", Vt)
    
    # For each singular value, show its corresponding directions
    for i, singular_value in enumerate(s):
        print(f"\nSingular value {i+1}: {singular_value:.6f}")
        print(f"Output direction (left singular vector): {U[:, i]}")
        print(f"Input direction (right singular vector): {Vt[i, :]}")
        
    # You can also visualize the directions
    print("\nSingular value analysis:")
    print("- Largest singular value corresponds to the direction of maximum 'gain'")
    print("- Smallest singular value corresponds to the direction of minimum 'gain'")
    
    # Check condition number
    condition_number = s[0] / s[-1] if s[-1] != 0 else np.inf
    print(f"Condition number: {condition_number:.2f}")

    G, k = HyperPlaneShiftingMethod(A, -15, 15)

    print("Minimal k:", min(k))

    Plot_Hull_H_2D(G, k)


