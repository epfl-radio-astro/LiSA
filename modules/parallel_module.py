'''
# super simple example
from multiprocessing import shared_memory
import numpy as np
a = np.array([1, 1, 2, 3, 5, 8])
shm_a = shared_memory.SharedMemory(create=True, size=a.nbytes, name = "psm_test")
b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm_a.buf)
b[:] = a[:]  # Copy the original data into shared memory
shm_a.close()
shm_a.unlink()'''

import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import tracemalloc

def check_noise_in_slice(q, f, domain_name, shape, dtype):

    # Locate the shared memory by its name
    domain_shm = shared_memory.SharedMemory(name=domain_name)

    # Create the array from the buffer of the shared memory
    domain = np.ndarray(shape=shape, dtype=dtype, buffer=domain_shm.buf)

    q.put( np.max(domain))

    domain_shm.close()

def foo(q):
    q.put('hello')

def run_basic(domain):
    try:
        print("Setting up shared memory")
        # Create a shared memory of size np_arry.nbytes
        shm = shared_memory.SharedMemory(create = True, size = domain.nbytes, name = "psm_test1")

        # Create a np.recarray using the buffer of shm
        shm_np_array = np.ndarray(shape=domain.shape, dtype=domain.dtype, buffer=shm.buf)

        # Copy the data into the shared memory
        np.copyto(shm_np_array, domain)

        # Spawn some processes to do some work
        #with ProcessPoolExecutor(nprocesses) as exe:
        #    fs = [exe.submit(check_noise_in_slice_mp, i, shm.name, domain.shape, domain.dtype) for i in range(nprocesses)]
        #    print([f.result() for f in fs])
        #    for _ in as_completed(fs):
        #        pass

        print("Setting up processes")
        mp.set_start_method('spawn')
        q = mp.Queue()
        p = mp.Process(target=check_noise_in_slice, args=(q,0,shm.name, domain.shape, domain.dtype,))
        p.start()
        print(q.get())
        p.join()
    except FileExistsError:
        print("Shared memory setup failed")
        shm = shared_memory.SharedMemory(name = "psm_test1")

    finally: 
        print("DEBUG: in finally statement")
        shm.close()
        shm.unlink()

def run_manager(domain):
    with SharedMemoryManager() as smm:
        # Create a shared memory of size np_arry.nbytes
        shm = smm.SharedMemory(domain.nbytes)
        # Create a np.recarray using the buffer of shm
        shm_np_array = np.ndarray(shape=domain.shape, dtype=domain.dtype, buffer=shm.buf)
        # Copy the data into the shared memory
        np.copyto(shm_np_array, domain)

        print("Setting up processes")
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        q = mp.Queue()
        p = mp.Process(target=check_noise_in_slice, args=(q,0,shm.name, domain.shape, domain.dtype,))
        p.start()
        print(q.get())
        p.join()

if __name__ == '__main__':
    #tracemalloc.start()
    start_time = time.process_time()
    nprocesses=mp.cpu_count()
    domain  = np.array([1, 1, 2, 3, 5, 8])
    #run_basic(domain)
    run_manager(domain)

# Check memory usage
#current, peak = tracemalloc.get_traced_memory()
#print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
#print(f'Time elapsed: {time.process_time()-start_time:.2f}s')
#tracemalloc.stop()