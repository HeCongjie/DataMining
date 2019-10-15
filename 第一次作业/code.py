#!/usr/bin/env python

def find_len(str_arr, point):
	end = point+1
	start = point
	max_len = 0
	while start>=0 and end<=len(str_arr)-1:
		if str_arr[start] == str_arr[end]:
			start -=1
			end +=1
			max_len +=2
		else:
			return max_len
	return max_len

if __name__ == '__main__':
	str_ = input("Input string:")
	str_arr = list(str_)
	sum_ = len(str_arr)
	lens = []
	for x in range(sum_-1):
		lens.append(find_len(str_arr, x))
	print(max(lens))