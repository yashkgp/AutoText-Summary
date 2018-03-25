from req_files import clusterrank
import sys

summarizer = clusterrank.ClusterRank()

num_of_sent = int(sys.argv[1])
pathToFile = "./"+sys.argv[2]
summary = summarizer.summarizeFile(pathToFile,num_of_sent)
save = open(sys.argv[3],"w+")

for line in summary:
	save.write(line+'\n')

save.close()
