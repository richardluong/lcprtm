fen = open("corpus.1.en")
fzh = open("corpus.1.zh")
fw = open("corpus.enzh", "w+")

lines_en = fen.read().splitlines()
lines_zh = fzh.read().splitlines()

print len(lines_en)
print len(lines_zh)

for i in range(len(lines_en)):
	print "Writing line " + str(i)
	fw.write(lines_en[i] + " " + lines_zh[i] + "\n")
