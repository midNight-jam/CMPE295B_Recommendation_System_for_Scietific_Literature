def average():
  sum = 0.0
  total = 5551.0
  f = open(file)
  for line in f:
    val = float(line)
    sum += val
  f.close()
  avg = (sum / total)
  print('Average : {0}'.format(avg))


file = 'auto_enco_svd/auto_enco_svd_recall___M_@_10___readings__2018_05_09T01_35_59_028229.dat'
print(file)
average()