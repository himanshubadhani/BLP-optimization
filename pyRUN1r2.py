import numpy as np
from numpy.random import rand
from numpy.linalg import eig
from scipy.linalg import sqrtm
#..........................................................................
#..........................................................................
# Definitions
def Metric(th1, th2, m, gamma, t):
    # Function generates the metric operator for the walk Pwalk1d
    W = Pwalk1dPT(th1, th2, m, gamma)
    D, V = np.linalg.eig(W)
    X = np.outer(V[: ,0], V[:, 0].conj()) * (np.abs(D[0]) ** (2 * t)) + np.outer(V[:, 1], V[:, 1].conj()) * (np.abs(D[1]) ** (2 * t))
    G = np.linalg.inv(X)
    return G
#..........................................................................
#..........................................................................
def randomstate(st1, st2):
    # Creates two new states around the given states, in a radius of e
    e = 10**-2
    pauli1 = np.array([[0, 1], [1, 0]])
    pauli2 = np.array([[0, -1j], [1j, 0]])
    pauli3 = np.array([[1, 0], [0, -1]])

    while True:
        a = -1 + 2 * np.random.rand(3)
        b = -1 + 2 * np.random.rand(3)
        Ra = np.sqrt(np.dot(a, a))
        Rb = np.sqrt(np.dot(b, b))
        da = e * a / Ra
        db = e * b / Rb
        state1 = st1 + (da[0] * pauli1 + da[1] * pauli2 + da[2] * pauli3) / 2
        state2 = st2 + (db[0] * pauli1 + db[1] * pauli2 + db[2] * pauli3) / 2

        # Check if both new states have non-negative eigenvalues
        if np.sum(np.linalg.eigvals(state1) >= 0) == 2 and np.sum(np.linalg.eigvals(state2) >= 0) == 2:
            break
    return state1, state2
#..........................................................................
#..........................................................................
def randomstate3(st1, e):
    # Creates a new state around the given state st1, in a radius of e
    pauli1 = np.array([[0, 1], [1, 0]])
    pauli2 = np.array([[0, -1j], [1j, 0]])
    pauli3 = np.array([[1, 0], [0, -1]])

    while True:
        a = -1 + 2 * np.random.rand(3)
        Ra = np.sqrt(np.dot(a, a))
        da = e * a / Ra
        state1 = st1 + (da[0] * pauli1 + da[1] * pauli2 + da[2] * pauli3) / 2

        # Check if the new state has non-negative eigenvalues
        if np.sum(np.linalg.eigvals(state1) >= 0) == 2:
            break

    return state1
#..........................................................................
#..........................................................................
def Pwalk1dPT(th1, th2, k, gamma):
    # Function generates the walk operator in momentum space
    C1 = np.array([[np.cos(np.deg2rad(th1 / 2)), 1j * np.sin(np.deg2rad(th1 / 2))],
                   [1j * np.sin(np.deg2rad(th1 / 2)), np.cos(np.deg2rad(th1 / 2))]])

    C2 = np.array([[np.cos(np.deg2rad(th2)), 1j * np.sin(np.deg2rad(th2))],
                   [1j * np.sin(np.deg2rad(th2)), np.cos(np.deg2rad(th2))]])

    Shift = np.array([[np.exp(1j * k), 0], [0, np.exp(-1j * k)]])
    Coin1 = C1
    Coin2 = C2
    Gain1 = np.array([[1 / gamma, 0], [0, gamma]],dtype=float)
    Gain2 = np.array([[gamma, 0], [0, 1 / gamma]],dtype=float)

    M = Coin1 @ Shift @ Gain2 @ Coin2 @ Gain1 @ Shift @ Coin1
    return M
#..........................................................................
# Main code
#..........................................................................
T = 25  # number of steps
N = 2 * T + 1  # resolution of pi
X = 2  # size of each sweep
dx = 2 * np.pi / N
e = 10**-2  #step size of the walker
# e = 10**-3  # for run2
#......................................................
#Cooling
tmax = 8000
tem = np.zeros((tmax, 1))
tem[0] = 1
for i in range(1, tmax):
    tem[i] = tem[0] * np.exp(-i * 0.001)
#.......................................................
# Coin and gamma parameters
Xn = 30
cn=1
if cn==1:
    th1 = 45  # Coin parameter
    th2 = -25.7143
    t1 = 45
    t2 = 25
    G1 = np.linspace(1.0, 1.3, 10)  # 45 and -25.7 (EP at 1.34)
    G2 = np.linspace(1.31, 1.36, Xn - 10)  # X=30
    G = np.concatenate((G1, G2)).reshape(-1, 1)
if cn==2:
    th1 = 45  # Coin parameter
    th2 = -30
    t1 = 45
    t2 = 30
    G1 = np.linspace(1.0, 1.2, 10)  # 45 and -30 (EP at 1.25)
    G2 = np.linspace(1.21, 1.26, Xn - 10)  # X=30
    G = np.concatenate((G1, G2)).reshape(-1, 1)
if cn==3:
    th1 = 60  # Coin parameter
    th2 = -30
    t1 = 60
    t2 = 30
    G1 = np.linspace(1.0, 1.4, 10)  # 60 and -30 (EP at 1.468)
    G2 = np.linspace(1.41, 1.5, Xn - 10)  # X=30
    G = np.concatenate((G1, G2)).reshape(-1, 1)
#........................................................
eye_matrix = np.eye(2)
eye_matrix = np.eye(2)
for g in range(6, 11):
    gamma = G[g - 1]
    rhota0 = eye_matrix / 2
    rhotb0 = eye_matrix / 2
    Memory = np.zeros((X * tmax, 1))
    # Loop for tmax
    for t in range(1,tmax + 1):
        # Loop for X
        for x in range(1, X+1):
            npmc1 = 0
            npmc2 = 0
            nTraceDist = np.zeros((T, 1))
            for k in range(1, T + 1):
                c1 = np.zeros((2, 2))
                d1 = np.zeros((2, 2))
                Wm=np.zeros((2,2))
                nY1= np.zeros((2, 2))
                eta0 = sqrtm(Metric(th1, th2, 0, gamma, k))

                for m in np.arange(-np.pi, np.pi, dx):
                    Wm = Pwalk1dPT(th1, th2, m, gamma)
                    eta = sqrtm(Metric(th1, th2, m, gamma, k))

                    c1 =c1+ np.linalg.inv(eta0) @ eta @ np.linalg.matrix_power(Wm, k) @ rhota0 @ np.linalg.matrix_power(Wm.conj().T, k) @ eta @ eta0
                    d1 =d1+ np.linalg.inv(eta0) @ eta @ np.linalg.matrix_power(Wm, k) @ rhotb0 @ np.linalg.matrix_power(Wm.conj().T, k) @ eta @ eta0
                
                nc1 = c1/np.trace(c1)
                nd1 = d1/np.trace(d1)

                nY1 = nc1-nd1
                nTraceDist[k-1] = np.sum(np.abs(np.linalg.eigvals(nY1)))/2

                if k > 1:
                    if nTraceDist[k - 1] > nTraceDist[k - 2]:
                        npmc1 += nTraceDist[k - 1] - nTraceDist[k - 2]  # for normalized density state

            count = X * (t - 1) + x
            Memory[count - 1] = npmc1  # BLP measure of the metric normalized state

            if np.max(Memory) == Memory[count - 1]:
                rhomaxa = rhota0.copy()
                rhomaxb = rhotb0.copy()

            statep = np.random.rand(1, 1)
            if count == 1:
                rhota0, rhotb0 = randomstate(rhota0, rhotb0)
                rhota2 = rhota0.copy()
                rhotb2 = rhotb0.copy()

            if count > 1:
                if Memory[count - 1] >= Memory[count - 2]:
                    if statep < 0.5:
                        rhota0 = randomstate3(rhota1, e)
                    else:
                        rhotb0 = randomstate3(rhotb1, e)
                    rhota2 = rhota1.copy()
                    rhotb2 = rhotb1.copy()
                if Memory[count - 1] < Memory[count - 2]:
                    p = np.random.rand(1, 1)
                    if p < np.exp((Memory[count - 1] - Memory[count - 2]) / tem[t - 1]):
                        if statep < 0.5:
                            rhota0 = randomstate3(rhota1, e)
                        else:
                            rhotb0 = randomstate3(rhotb1, e)
                    else:
                        if statep < 0.5:
                            rhota0 = randomstate3(rhota2, e)
                        else:
                            rhotb0 = randomstate3(rhotb2, e)
            rhota1 = rhota0.copy()
            rhotb1 = rhotb0.copy()
        if t % 100 == 0:
            print([g, t])
    print(g)
    if cn==1:
        np.savetxt('PmetMaxBLPmemory4525G{0}.dat'.format(g),np.column_stack(Memory))
        np.savetxt('PmetMaxBLPrhoa4525G{0}.dat'.format(g),np.column_stack(rhomaxa))
        np.savetxt('PmetMaxBLPrhob4525G{0}.dat'.format(g),np.column_stack(rhomaxb))
    if cn==2:
        np.savetxt('PmetMaxBLPmemory4530G{0}.dat'.format(g),np.column_stack(Memory))
        np.savetxt('PmetMaxBLPrhoa4530G{0}.dat'.format(g),np.column_stack(rhomaxa))
        np.savetxt('PmetMaxBLPrhob4530G{0}.dat'.format(g),np.column_stack(rhomaxb))
    if cn==3:
        np.savetxt('PmetMaxBLPmemory6030G{0}.dat'.format(g),np.column_stack(Memory))
        np.savetxt('PmetMaxBLPrhoa6030G{0}.dat'.format(g),np.column_stack(rhomaxa))
        np.savetxt('PmetMaxBLPrhob6030G{0}.dat'.format(g),np.column_stack(rhomaxb))