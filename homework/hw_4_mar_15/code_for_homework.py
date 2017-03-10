def smooth_initial_condition(x):
    return (1/2)*np.sin(2*np.pi*x)+(1/2)

def discon_initial_condition(x):
    return np.piecewise(x, [abs(x-0.5)<0.25, abs(x-0.5)>=0.25], [1, 0])

# Define Nt, Nx, dt, and dx
Nt = 10*(2**5)
Nx = int(0.9*Nt)
dt = 1/Nt
dx = 1/Nx
transport_coef = 1

# Define an update rule.
step = make_upwinding_method(Nx,dx,dt,transport_coef)

# Define initial condition.
xaxis = np.linspace(0,1-dx,Nx)
u_0 = discon_initial_condition(xaxis)

# Loop through time and apply the update rule.
for t in tqdm(xrange(1,Nt+1)):
    u_1 = step(u_0)
    u_0 = u_1 + 0