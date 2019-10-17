'''
A Digital Library contains millions of books, stored according to their titles, authors, key words of their abstracts, publishers, and published years.
 Each book is assigned an unique 7-digit number as its ID. Given any query from a reader, you are supposed to output the resulting books, sorted in
 increasing order of their ID's.

Input Specification:
Each input file contains one test case. For each case, the first line contains a positive integer N (≤10
​4
​​ ) which is the total number of books. Then N blocks follow, each contains the information of a book in 6 lines:

Line #1: the 7-digit ID number; 编号
Line #2: the book title -- a string of no more than 80 characters; 书名
Line #3: the author -- a string of no more than 80 characters; 作者
Line #4: the key words -- each word is a string of no more than 10 characters without any white space, and the keywords are separated by exactly one space; 关键词
Line #5: the publisher -- a string of no more than 80 characters; 出版社
Line #6: the published year -- a 4-digit number which is in the range [1000, 3000]. 出版年份
每本书一个作者
It is assumed that each book belongs to one author only, and contains no more than 5 key words; there are no more than
1000 distinct key words in total; and there are no more than 1000 distinct publishers.

After the book information, there is a line containing a positive integer M (≤1000) which is the number of user's search queries.
 Then M lines follow, each in one of the formats shown below:

1: a book title
2: name of an author
3: a key word
4: name of a publisher
5: a 4-digit number representing the year
Output Specification:
For each query, first print the original query in a line, then output the resulting book ID's in increasing order, each occupying a line. If no book is found, print Not Found instead.

Sample Input:
3
1111111
The Testing Book
Yue Chen
test code debug sort keywords
ZUCS Print
2011
3333333
Another Testing Book
Yue Chen
test code sort keywords
ZUCS Print2
2012
2222222
The Testing Book
CYLL
keywords debug book
ZUCS Print2
2011
6
1: The Testing Book
2: Yue Chen
3: keywords
4: ZUCS Print
5: 2011
3: blablabla
Sample Output:
1: The Testing Book
1111111
2222222
2: Yue Chen
1111111
3333333
3: keywords
1111111
2222222
3333333
4: ZUCS Print
1111111
5: 2011
1111111
2222222
3: blablabla
Not Found
'''
N = int(input())
information = []
for i in range(N):
    information.append([input() for i in range(6)])

qn = int(input())
for j in range(qn):
    _input = input()
    print(_input)
    query = _input.split(': ')
    got = False
    if query[0] == '1':
        for info in information:
            if query[1] in info[1]:
                print(info[0])
                got = True
    elif query[0] == '2':
        for info in information:
            if query[1] in info[2]:
                print(info[0])
                got = True
    elif query[0] == '3':
        for info in information:
            if query[1] in info[3]:
                print(info[0])
                got = True
    elif query[0] == '4':
        for info in information:
            if query[1] in info[4]:
                print(info[0])
                got = True
    elif query[0] == '5':
        for info in information:
            if query[1] in info[5]:
                print(info[0])
                got = True
    if got == False:
        print('Not Found')
