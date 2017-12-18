def formatter(line):
	return None

def draw_out(in_path,out_path):
	inf = open(in_path,"r")
	ouf = open(out_path,"w")

	for line in inf:
		try:
			result = formatter(line)
		except Exception as e:
			print(e)
			gg


def work(from_id,to_id):
	gg


num_file = 30
num_process = 10

if __name__ == "__main__":
	import multiprocessing

	process_pool = []

	for a in range(0,num_process):
