class Solution:
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        morse_dict = dict(zip(letters,morse))
        morse_list = []
        for i in range(len(words)):
            morse_list.append("".join(morse_dict[c] for c in words[i]))
        return len(set(morse_list))

if __name__ == "__main__":
    s = Solution()
    words = ["gin", "zen", "gig", "msg"]
    d = s.uniqueMorseRepresentations(words)
    print(d)