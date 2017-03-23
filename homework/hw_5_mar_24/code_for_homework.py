def IC(x,IC_choice):
    # Get the initial condition, given the choice
    if IC_choice == 1:
        # cos(16 pi x)exp(-50(x-.5)^2)
        IC = np.cos(16*np.pi*x)*np.exp(-50*(x-0.5)**2)
    elif IC_choice == 2:
        # sin(2 pi x)sin(4 pi x)
        IC = np.sin(2*np.pi*x)*np.sin(4*np.pi*x)
    elif IC_choice == 3:
        # step function, up at 1/4, down at 3/2
        IC = np.piecewise(x,[abs(x-0.5)<0.25,abs(x-0.5)>=0.25],[1,0])
    return IC

# Set N_t = 10*(2^4)
power = 4
Nt = 10*2**power
a = 1
# Since a = 1, set N_x = .9*(N_t - 1)
#   It is (N_t - 1) since we are using finite volume methods.
Nx = int(0.9*(Nt-1))
# Get dt and dx from N_t and N_x
dt = 1/(Nt-1)
dx = 1/Nx
# Set the final time
final_time = 5

# Define IC choice
IC_choice = 1
# Define phi choice
phi_choice = 2

# Define the recursion
FV_step = make_FV_method(phi_choice,a,dt,dx)

# Get initial condition
xaxis = np.linspace(dx/2,1-dx/2,Nx)
u_initial = IC(xaxis,IC_choice)
u_0 = u_initial + 0

# Do recursion
for t in xrange(int((Nt-1)*final_time)):
    u_1 = FV_step(u_0)
    u_0 = u_1 + 0