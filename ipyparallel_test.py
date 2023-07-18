import ipyparallel as ipp

cluster = ipp.Cluster(engines="mpi", n=8)
client = cluster.start_and_connect_sync()
print(client.ids)
