import json

accusation_file = open("../../data_processor/accusation_list2.txt","r")
accusation_list = json.loads(accusation_file.readline())

f = open("count_data/total.txt","r")
nowb = ""

for line in f:
	line = line[:-1]
	if line == "law":
		nowb = "law"
		ouf = open("law_result1.txt","w")
	elif line == "crit":
		nowb = "crit"
		ouf = open("crit_result.txt","w")
	elif line == "time":
		break
	else:
		if line == "":
			continue
		if nowb == "crit":
			data = line.split(" ")
			a = int(data[0])
			b = int(data[1])
			print(accusation_list[a],a,b,file=ouf)
		else:
			data = line.split(" ")
			c = int(data[1])
			b = int(int(data[0])%10)
			a = int(int(data[0])//10)
			print(a,b,c,file=ouf)
