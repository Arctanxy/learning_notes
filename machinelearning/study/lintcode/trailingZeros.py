class Solution:
    """
    @param: n: An integer
    @return: An integer, denote the number of trailing zeros in n!
    """
    def trailingZeros(self, n):
        # write your code here, try to do it without arithmetic operators.
        count = 0
        for m in range(n):
            c = 0
            while m >= 5:
                if m%5 == 0:
                    m = int(m/5)
                    c += 1

                
            count += c
            
        return count

if __name__ == "__main__":
    s = Solution()
    k = s.trailingZeros(11)
    print(k)