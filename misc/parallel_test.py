import ipyparallel as ipp
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

#
NCPU = 2
cluster = ipp.Cluster(engines="mpi", n=NCPU)
client = cluster.start_and_connect_sync()
#
client.ids

#
# Start Intracommunicator and get what is out there i.t.o. size and rank
# Note that all these variables are known to all ranks where size is equal
# for all and rank is specific to the rank number of the process.
#
comm = MPI.COMM_WORLD      # start the communicator assign to comm
size = comm.Get_size()     # get the size and assign to size
rank = comm.Get_rank()     # get the rank and assign to rank
#
# This is just for checking if it worked and should give sth. like this:
# [stdout:3] Rank/Size 3/16
#
# [stdout:5] Rank/Size 5/16
#
# [stdout:10] Rank/Size 10/16
#
# [stdout:0] Rank/Size 0/16
#
# [stdout:1] Rank/Size 1/16
#
print('Rank/Size {}/{}'.format(rank, size))

nx = 1000
dx = 0.1
nt = 100000
dt = 0.001
D = 1  # diffusion constant

# Domain decomposition: set up domain boundaries
nx1 = rank * nx // size
nx2 = (rank + 1) * nx // size

print('{}, Domain boundaries: {}-{}'.format(rank, nx1, nx2))

# We include one additional cell at the boundaries for communication purposes
x = np.arange(nx1 - 1, nx2 + 1) * dx

sigma0 = 20 * dx
c = np.exp(-(x - nx * dx / 2)**2 / (2 * sigma0**2)) / (np.sqrt(2 * np.pi) * sigma0)

# plt.title('rank ${}$'.format(rank))
# plt.xlim(0, nx * dx)
# plt.ylim(0, 0.3)
# plt.plot(x, c)

fig, ax = plt.subplots()
x_full_range = np.arange(nx) * dx
c_full_range = np.zeros(nx)
comm.Gather(c[1:-1], c_full_range, root=0)

for t in range(nt + 1):
    # Send to right, receive from left
    comm.Sendrecv(c[-2:-1], (rank + 1) % size, recvbuf=c[:1], source=(rank - 1) % size)
    # Send to left, receive from right
    comm.Sendrecv(c[1:2], (rank - 1) % size, recvbuf=c[-1:], source=(rank + 1) % size)
    if t % (nt // 5) == 0:
        comm.Gather(c[1:-1], c_full_range, root=0)
        if rank == 0:
            ax.plot(x_full_range, c_full_range, label='t = {}'.format(t))
    d2c_dx2 = (np.roll(c, 1) - 2 * c + np.roll(c, -1)) / (dx**2)
    c += D * d2c_dx2 * dt
if rank == 0:
    ax.legend(loc='best')

plt.show()
