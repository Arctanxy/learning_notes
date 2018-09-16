'''Given an increasing sequence S of N integers, the median is the number
 at the middle position. For example, the median of S1 = { 11, 12, 13, 14 }
  is 12, and the median of S2 = { 9, 10, 15, 16, 17 } is 15. The median of
   two sequences is defined to be the median of the nondecreasing sequence
    which contains all the elements of both sequences. For example, the
     median of S1 and S2 is 13.

Given two increasing sequences of integers, you are asked to find their median.

Input Specification:
Each input file contains one test case. Each case occupies 2 lines, each gives
 the information of a sequence. For each sequence, the first positive integer N (≤2×10
​5
​​ ) is the size of that sequence. Then N integers follow, separated by a space.
 It is guaranteed that all the integers are in the range of long int.

Output Specification:
For each test case you should output the median of the two given sequences in a line.

Sample Input:
4 11 12 13 14
5 9 10 15 16 17
Sample Output:
13'''

nums1 = [int(t) for t in input().split(' ')]
M = nums1.pop(0)
nums2 = [int(t) for t in input().split(' ')]
N = nums2.pop(0)

def search(nums1,nums2):
    count = 0
    j = 0
    for i in range(M):

        while j < N:
            if nums2[j] < nums1[i]:
                count += 1
                if count == int((M + N + 1) / 2):
                    return nums2[j]
                j += 1
            else:
                break
        count += 1
        if count == int((M + N + 1) / 2):
            return nums1[i]

median = search(nums1,nums2)
print(median)