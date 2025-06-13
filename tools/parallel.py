# Use multiprocess instead of multiprocessing because of pickle's limitations
import multiprocess as mp
import time
import random

"""
Creates a large pool doing the task f
	f 	(fun): function
	pll_itr (iter): iterator (for parallel processes)
	pll_chk (int): chunk size (for parallel processes)
	pll_lzy	(float): waiting time between reader tries (in seconds) (for parallel processes)
	num_pro (int): number of processors
"""


def default_f(x=0, *_, **__):
	return -x


pll_itr = iter(int, 1)  # Infinite iterator
pll_chk = 100  # Chunk size
pll_lzy = 300  # Waiting time


def worker(f, shared_list, lock):
	"""Extends the list with a few results."""
	for _ in pll_itr:
		random.seed()
		result = [f() for _ in range(pll_chk)]
		with lock:
			# print("worker loops with lock  !")
			shared_list.extend(result)


def reader(shared_list, lock, output_file, working):
	"""Adds the minimal value to output_file while workers are working."""
	while working.value:
		read_loop(shared_list, lock, output_file)
		time.sleep(pll_lzy)  # Check every pll_lzy (lazyness) seconds
	read_loop(shared_list, lock, output_file)  # Last check


def read_loop(shared_list, lock, output_file):
	"""Adds the minimal value to output_file."""
	with lock:
		# print("reader loops with lock  !")
		if shared_list:
			min_value, min_group = shared_list[0]
			for i in range(1, len(shared_list)):
				if shared_list[i][0] < min_value:
					min_value, min_group = shared_list[i]
			shared_list[:] = [(min_value, min_group)]  # Reset the list to only contain the min value
			with open(output_file, "a") as fil:
				fil.write(f"{min_value} -> {list(min_group)}\n")


def imap_lake(f, num_pro=4, file_name="optimization_results.txt"):
	"""
	Creates a large pool doing the task f
		f 	(fun): function
		num_pro (int): number of processors
	"""
	# Shared list and lock for synchronization
	manager = mp.Manager()
	shared_list = manager.list()
	working = manager.Value('b', True)  # Shared boolean variable
	lock = manager.Lock()  # Semaphore

	# Output file
	output_file = file_name

	workers = []  # Create (num_pro - 1) worker processes
	for _ in range(num_pro - 1):
		p = mp.Process(target=worker, args=(f, shared_list, lock))
		workers.append(p)
		p.start()

	reader_process = mp.Process(target=reader, args=(shared_list, lock, output_file, working))
	reader_process.start()
	# print("The reader starts...")

	# Wait for all worker processes to finish (they won't if they run forever)
	for p in workers:
		p.join()

	working.value = False  # Signal the reader to stop
	# print("Every worker has stopped working !")

	# Wait for the reader process to finish (it won't if it runs forever)
	reader_process.join()

# Entry point
if __name__ == "__main__":
	imap_lake(default_f, num_pro=4)