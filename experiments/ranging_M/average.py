def average():
  sum1 = 0.0
  sum2 = 0.0
  total = 5551.0
  i = 1 
  count1 = 1834.0
  count2 = 3717.0

  f = open(file)
  for line in f:
    val = float(line)

    if(i >= count1):
      sum2 += val
    else:
      sum1 += val
    i += 1
  f.close()

  avg1 = (sum1 / count1)
  avg2 = (sum2 / count2)

  print('Average1 : {0} = {1} / {2}'.format(avg1, sum1, count1))
  print('Average2 : {0} = {1} / {2}'.format(avg2, sum2, count2))

file = 'cdl/recall___M_@_10___readings__2018_05_09T13_59_06_811622.dat'
print(file)
average()